import collections

import numpy as np
from marllib.marl.algos.utils.episode_replay_buffer import EpisodeBasedReplayBuffer
from ray.rllib import SampleBatch
from ray.rllib.execution import PrioritizedReplayBuffer
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.typing import SampleBatchType
from scipy.ndimage import convolve1d

from niql.utils import get_lds_kernel_window, LDS_WEIGHTS


class LDSEpisodeReplayBuffer(EpisodeBasedReplayBuffer):

    def __init__(
            self,
            num_shards: int = 1,
            learning_starts: int = 1000,
            capacity: int = 10000,
            replay_batch_size: int = 32,
            prioritized_replay_alpha: float = 0.6,
            prioritized_replay_beta: float = 0.4,
            prioritized_replay_eps: float = 1e-6,
            replay_mode: str = "independent",
            replay_sequence_length: int = 1,
            replay_burn_in: int = 0,
            replay_zero_init_states: bool = True,
            buffer_size=DEPRECATED_VALUE,
    ):
        super().__init__(num_shards, learning_starts, capacity, replay_batch_size,
                         prioritized_replay_alpha, prioritized_replay_beta,
                         prioritized_replay_eps, replay_mode, replay_sequence_length, replay_burn_in,
                         replay_zero_init_states,
                         buffer_size)
        self.num_bins = 50
        self.lds_kernel = "gaussian"
        self.lds_ks = 5
        self.lds_sigma = 2

        # buffer creation
        def new_buffer():
            return LDSPriorityReplayBuffer(
                capacity=self.capacity,
                alpha=prioritized_replay_alpha,
                lds_kernel=self.lds_kernel,
                lds_ks=self.lds_ks,
                lds_sigma=self.lds_sigma,
                num_bins=self.num_bins,
            )
        self.replay_buffers = collections.defaultdict(new_buffer)


class LDSPriorityReplayBuffer(PrioritizedReplayBuffer):
    """
    Implements Label Density Smoothing for the replay buffer.
    """

    def __init__(self, capacity, alpha, lds_kernel, lds_ks, lds_sigma, num_bins):
        super().__init__(capacity, alpha)
        self.lds_kernel = lds_kernel
        self.lds_ks = lds_ks
        self.lds_sigma = lds_sigma
        self.num_bins = num_bins

    def sample(self, num_items: int, beta: float) -> SampleBatchType:
        """Sample a batch of experiences and return priority weights, indices.

        Args:
            num_items (int): Number of items to sample from this buffer.
            beta (float): To what degree to use importance weights
                (0 - no corrections, 1 - full correction).

        Returns:
            SampleBatchType: Concatenated batch of items including "weights"
                and "batch_indexes" fields denoting IS of each sampled
                transition and original idxes in buffer of sampled experiences.
        """
        assert beta >= 0.0

        lds_weights = self._update_weights()

        idxes = self._sample_proportional(num_items)

        weights = []
        batch_indexes = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            count = self._storage[idx].count
            # If zero-padded, count will not be the actual batch size of the
            # data.
            if isinstance(self._storage[idx], SampleBatch) and \
                    self._storage[idx].zero_padded:
                actual_size = self._storage[idx].max_seq_len
            else:
                actual_size = count
            weights.extend([weight / max_weight] * actual_size)
            batch_indexes.extend([idx] * actual_size)
            self._num_timesteps_sampled += count
        batch = self._encode_sample(idxes)

        # Note: prioritization is not supported in lockstep replay mode.
        if isinstance(batch, SampleBatch):
            batch["weights"] = np.array(weights)
            batch["batch_indexes"] = np.array(batch_indexes)

        # Add LDS weights of sampled instances
        if lds_weights is not None:
            lds_wts = lds_weights[idxes]
            batch[LDS_WEIGHTS] = lds_wts.ravel()

        return batch

    def _update_weights(self):
        # consider all data in buffer
        data = SampleBatch.concat_samples(self._storage)
        rewards = data[SampleBatch.REWARDS]

        # create bins
        hist, bins = np.histogram(rewards, bins=self.num_bins)
        bin_index_per_label = np.digitize(rewards, bins, right=True)
        Nb = max(bin_index_per_label) + 1
        num_samples_of_bins = dict(collections.Counter(bin_index_per_label))
        emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]

        # compute effective label distribution
        lds_kernel_window = get_lds_kernel_window(self.lds_kernel, self.lds_ks, self.lds_sigma)
        eff_label_dist = convolve1d(emp_label_dist, weights=lds_kernel_window, mode='constant')

        # Use re-weighting based on effective label distribution, sample-wise weights: [Ns,]
        eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
        weights = [np.float32(1 / (x + 1e-6)) for x in eff_num_per_label]
        scaling = len(weights) / np.sum(weights)
        lds_weights = np.array([scaling * x for x in weights]).reshape(len(self._storage), -1)
        return lds_weights

