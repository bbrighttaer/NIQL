import collections

from marllib.marl.algos.utils.episode_replay_buffer import EpisodeBasedReplayBuffer
from ray.rllib.execution import PrioritizedReplayBuffer
from ray.rllib.utils.deprecation import DEPRECATED_VALUE


class JointEpisodeReplayBuffer(EpisodeBasedReplayBuffer):

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

        # change experience replay buffer setup to joint for all agents
        joint_replay_buffer = PrioritizedReplayBuffer(self.capacity, alpha=prioritized_replay_alpha)

        def new_buffer():
            return joint_replay_buffer

        self.replay_buffers = collections.defaultdict(new_buffer)
