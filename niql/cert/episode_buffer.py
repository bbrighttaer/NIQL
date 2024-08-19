import collections
import random

import numpy as np
from marllib.marl.algos.utils.episode_replay_buffer import EpisodeBasedReplayBuffer as EpBasedReplayBuffer
from ray.rllib import SampleBatch
from ray.rllib.policy.sample_batch import MultiAgentBatch, DEFAULT_POLICY_ID
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.typing import SampleBatchType

from niql.cert.cert_replay_buffer import CERTReplayBuffer

# Constant that represents all policies in lockstep replay mode.
_ALL_POLICIES = "__all__"


class CERTEpisodeBasedReplayBuffer(EpBasedReplayBuffer):

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
            enable_joint_buffer=False,
            enable_stochastic_eviction=False,
            buffer_size=DEPRECATED_VALUE,
    ):
        super().__init__(num_shards, learning_starts, capacity, replay_batch_size,
                         prioritized_replay_alpha, prioritized_replay_beta,
                         prioritized_replay_eps, replay_mode, replay_sequence_length, replay_burn_in,
                         replay_zero_init_states,
                         buffer_size)

        def new_buffer():
            return CERTReplayBuffer(self.capacity)

        self.replay_buffers = collections.defaultdict(new_buffer)

    def replay(self) -> SampleBatchType:
        if self._fake_batch:
            fake_batch = SampleBatch(self._fake_batch)
            return MultiAgentBatch({
                DEFAULT_POLICY_ID: fake_batch
            }, fake_batch.count)

        if self.num_added < self.replay_starts:
            return None
        with self.replay_timer:
            # Lockstep mode: Sample from all policies at the same time an
            # equal amount of steps.
            if self.replay_mode == "lockstep":
                return self.replay_buffers[_ALL_POLICIES].sample(
                    self.replay_batch_size, beta=self.prioritized_replay_beta)
            else:
                # determine the experiences to be sampled
                idxes = [
                    random.randint(
                        0,
                        len(list(self.replay_buffers.values())[0]._storage) - 1,
                    ) for _ in range(self.replay_batch_size)
                ]
                samples = {}
                for policy_id, replay_buffer in self.replay_buffers.items():
                    samples[policy_id] = replay_buffer.sample(idxes)
                return MultiAgentBatch(samples, self.replay_batch_size)
