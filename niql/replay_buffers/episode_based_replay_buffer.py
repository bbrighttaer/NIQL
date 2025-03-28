import collections
import random

import numpy as np
import torch
import torch.nn.functional as F
from marllib.marl.algos.utils.episode_replay_buffer import EpisodeBasedReplayBuffer as EpBasedReplayBuffer
from ray.rllib import SampleBatch
from ray.rllib.execution import ReplayBuffer, PrioritizedReplayBuffer
from ray.rllib.execution.replay_buffer import warn_replay_capacity
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.typing import SampleBatchType

from niql.utils import to_numpy


class EpisodeBasedReplayBuffer(EpBasedReplayBuffer):

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
            enable_stochastic_eviction=False,
            buffer_size=DEPRECATED_VALUE,
    ):
        super().__init__(num_shards, learning_starts, capacity, replay_batch_size,
                         prioritized_replay_alpha, prioritized_replay_beta,
                         prioritized_replay_eps, replay_mode, replay_sequence_length, replay_burn_in,
                         replay_zero_init_states,
                         buffer_size)
        if enable_stochastic_eviction:
            def new_buffer():
                return PrioritizedReplayBufferWithStochasticEviction(self.capacity, alpha=prioritized_replay_alpha)

            self.replay_buffers = collections.defaultdict(new_buffer)

    def plot_statistics(self, summary_writer, timestep):
        for policy_id, replay_buffer in self.replay_buffers.items():
            samples = SampleBatch.concat_samples(replay_buffer._storage)
            rewards = samples[SampleBatch.REWARDS]
            if summary_writer is not None:
                summary_writer.add_histogram(
                    policy_id + "/exp_buffer_dist", rewards, timestep
                )


class SimpleReplayBuffer(ReplayBuffer):

    def __init__(self, capacity):
        super().__init__(capacity)
        self._num_sampled = 0

    def sample(self, num_items: int, *args, **kwargs) -> SampleBatchType:
        batch = super().sample(num_items)
        batch["weights"] = np.ones_like(batch[SampleBatch.REWARDS])
        return batch


class PrioritizedReplayBufferWithStochasticEviction(PrioritizedReplayBuffer):

    def __init__(self, capacity: int = 10000, alpha: float = 1.0):
        super().__init__(capacity, alpha)

    def add(self, item: SampleBatchType, weight: float) -> None:
        idx = self._next_idx
        self._add(item, weight)
        if weight is None:
            weight = self._max_priority
        self._it_sum[idx] = weight ** self._alpha
        self._it_min[idx] = weight ** self._alpha

    def _add(self, item: SampleBatchType, weight: float) -> None:
        assert item.count > 0, item
        warn_replay_capacity(item=item, num_items=self.capacity / item.count)

        self._num_timesteps_added += item.count

        if self._next_idx < self.capacity and not self._eviction_started:  # buffer is not full
            self._storage.append(item)
            self._est_size_bytes += item.size_bytes()
            self._next_idx += 1
        elif self._eviction_started:
            self._storage[self._next_idx] = item
            self._num_timesteps_added_wrap += 1
            self._select_next_index()

        # if next_idx is out of range update it via stochastic eviction
        if self._next_idx >= self.capacity:
            self._eviction_started = True
            self._select_next_index()

        if self._eviction_started:
            self._evicted_hit_stats.push(self._hit_count[self._next_idx])
            self._hit_count[self._next_idx] = 0

    def _select_next_index(self):
        # get current priorities
        priorities = np.array([float(self._it_sum[i]) for i in np.arange(len(self._storage))])

        # normalize
        priorities /= np.max(priorities)
        priorities = 1. - priorities

        # calc props
        probs = to_numpy(F.softmax(torch.from_numpy(priorities), dim=-1))

        # weighted sampling
        self._next_idx = np.random.choice(np.arange(self.capacity), p=probs)

    def sample(self, num_items: int, beta: float) -> SampleBatchType:
        batch = super().sample(num_items, beta)
        uniform_batch = self.uniform_sample(min(num_items * 3, len(self._storage)))
        batch["uniform_batch"] = uniform_batch
        return batch

    def uniform_sample(self, num_items: int) -> SampleBatchType:
        """Sample a batch of experiences.

        Args:
            num_items (int): Number of items to sample from this buffer.

        Returns:
            SampleBatchType: concatenated batch of items.
        """
        idxes = [
            random.randint(0,
                           len(self._storage) - 1) for _ in range(num_items)
        ]
        return self._encode_sample(idxes)
