import collections
import random

from marllib.marl.algos.utils.episode_replay_buffer import EpisodeBasedReplayBuffer
from ray.rllib import SampleBatch
from ray.rllib.execution import PrioritizedReplayBuffer
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.typing import SampleBatchType


class AugmentedEpisodeReplayBuffer(EpisodeBasedReplayBuffer):

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
            return StateValueAugmentedPrioritizedReplayBuffer(self.capacity, alpha=prioritized_replay_alpha)

        self.replay_buffers = collections.defaultdict(new_buffer)


class StateValueAugmentedPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    """
    Adds shared state-values support to agent buffer.
    """

    def __init__(self, capacity, alpha):
        super().__init__(capacity, alpha)

        # state-value tuples storage.
        self._state_value_storage = []

    def clear_state_value_buffer(self):
        self._state_value_storage.clear()

    def sample_local_experiences(self, num_items):
        """Sample a batch of experiences.

        Args:
            num_items (int): Number of items to sample from this buffer.

        Returns:
            SampleBatchType: concatenated batch of items.
        """
        idxes = [
            random.randint(0, len(self._storage) - 1) for _ in range(num_items)
        ]
        return self._encode_sample(idxes)

    def add_shared_state_value_batch(self, data: SampleBatch):
        self._state_value_storage.append(data)

    def sample(self, num_items: int, beta: float) -> SampleBatchType:
        batch = super().sample(num_items, beta)
        if self._state_value_storage:
            state_val_cat = SampleBatch.concat_samples(self._state_value_storage)
            batch["state_value_batch"] = state_val_cat
        return batch
