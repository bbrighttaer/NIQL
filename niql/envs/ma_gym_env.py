import copy

import numpy as np
import gym
import ma_gym  # noqa
from gym.spaces import Dict as GymDict, Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from niql import seed

policy_mapping_dict = {
    "all_scenario": {
        "description": "one team cooperate",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}


class MAGymEnv(MultiAgentEnv):
    """
    Wrapper around the ma_gym envs.
    """

    def __init__(self, env_config):
        env_config = copy.deepcopy(env_config)
        map_name = env_config.pop("map_name")
        env_id = env_config.pop("gym_id")
        self.env = gym.make(id=env_id, **env_config)
        self.action_space = self.env.action_space[0]
        observation_space = self.env.observation_space[0]
        self._dtype = observation_space.dtype
        self.observation_space = GymDict({
            "obs": observation_space,
            "terminal": Box(low=0., high=1., shape=(1,))
        })
        self.agents = [f'agent_{i}' for i in range(self.env.n_agents)]
        self.num_agents = self.env.n_agents
        env_config["map_name"] = map_name
        self.env.seed(seed)

    def reset(self):
        raw_obs = self.env.reset()
        obs = {}
        for agent, r_obs in zip(self.agents, raw_obs):
            obs[agent] = {
                "obs": np.array(r_obs, dtype=self._dtype),
                "terminal": np.array([0.], dtype=self._dtype)
            }
        return obs

    def step(self, action_dict):
        raw_obs, raw_rew, raw_done, raw_info = self.env.step(action_dict.values())
        obs = {}
        rew = {}
        done = {"__all__": all(raw_done)}
        info = {}

        for agent, r_obs, r_rew, r_done in zip(self.agents, raw_obs, raw_rew, raw_done):
            obs[agent] = {
                "obs": np.array(r_obs, dtype=self._dtype),
                "terminal": np.array([r_done], dtype=self._dtype)
            }
            rew[agent] = r_rew
            done[agent] = r_done
            info[agent] = dict(raw_info)

        return obs, rew, done, info

    def render(self, mode="human"):
        return self.env.render(mode)

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.env._max_steps,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
