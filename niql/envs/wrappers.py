from typing import Tuple

import numpy as np
from ray.rllib import MultiAgentEnv
from gym.spaces import Dict as GymDict, Box
from ray.rllib.utils.typing import MultiAgentDict

from niql.config import FINGERPRINT_SIZE


def create_fingerprint_env_wrapper_class(parent_env_class: type) -> type:
    """
    Wraps a multi-agent environment by dynamically subclassing it.

    :param parent_env_class: Parent env
    :return: a child class of the parent env with some overriden methods.
    """

    def _zero_pad_obs(obs):
        for agent, obs_dict in obs.items():
            padded_obs_dict = {}
            for _obs_key, _obs_val in obs_dict.items():
                _obs_val = np.array(_obs_val)
                pad = np.zeros_like(_obs_val)
                padded = np.concatenate([_obs_val, pad[:FINGERPRINT_SIZE]])
                padded_obs_dict[_obs_key] = padded.tolist()
            obs[agent] = padded_obs_dict
        return obs

    class FingerPrintWrapper(parent_env_class):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # modify observation space
            if isinstance(self.observation_space, GymDict):
                obs_space = self.observation_space["obs"]
            else:
                obs_space = self.observation_space
            self.observation_space = GymDict({"obs": Box(
                low=-100.0,
                high=100.0,
                shape=(obs_space.shape[0] + FINGERPRINT_SIZE,),
                dtype=obs_space.dtype)}
            )

        def step(
                self, action_dict: MultiAgentDict
        ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
            obs, rewards, dones, info = super().step(action_dict)
            obs = _zero_pad_obs(obs)
            return obs, rewards, dones, info

        def reset(self):
            obs = _zero_pad_obs(super().reset())
            return obs

    return FingerPrintWrapper
