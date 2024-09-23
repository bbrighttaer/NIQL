import collections
import random

import numpy as np
from marllib.marl.algos.utils.episode_replay_buffer import EpisodeBasedReplayBuffer
from ray.rllib import SampleBatch
from ray.rllib.execution import ReplayBuffer
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.typing import SampleBatchType


class BasicEpisodeBasedReplayBuffer(EpisodeBasedReplayBuffer):

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

        def new_buffer():
            return SimpleReplayBuffer(self.capacity)

        self.replay_buffers = collections.defaultdict(new_buffer)


class SimpleReplayBuffer(ReplayBuffer):

    def __init__(self, capacity):
        super().__init__(capacity)
        self._num_sampled = 0

    def sample(self, num_items: int, *args, **kwargs) -> SampleBatchType:
        """Sample a batch of experiences.

        Args:
            num_items (int): Number of items to sample from this buffer.

        Returns:
            SampleBatchType: concatenated batch of items.
        """
        # Get all_indexes
        pop_size = len(self._storage)
        all_indexes = list(range(pop_size))

        # Sample without replacement
        sample_size = min(pop_size, num_items)
        idxes = random.sample(all_indexes, sample_size)
        self._num_sampled += num_items
        batch = self._encode_sample(idxes)

        # Add dummy weights to keep a consistent set of keys in sample batch
        batch["weights"] = np.ones_like(batch[SampleBatch.REWARDS])
        return batch
