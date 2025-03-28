# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import time
import warnings

import numpy as np
import supersuit as ss
from gym.spaces import Dict as GymDict, Box
from gym.utils import colorize
from pettingzoo.mpe import simple_v2, simple_spread_v2
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from niql.envs import mpe_simple_reference
from niql import seed
from niql.utils import unwrap_multi_agent_actions, apply_coop_reward

# from pettingzoo 1.12.0

policy_mapping_dict = {
    "simple": {
        "description": "one team cooperate",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
    "simple_spread": {
        "description": "one team cooperate",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
    "simple_reference": {
        "description": "one team cooperate",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}

REGISTRY = {
    "simple": simple_v2.parallel_env,
    "simple_spread": simple_spread_v2.parallel_env,
    "simple_reference": mpe_simple_reference.parallel_env,
}


class MPEEnv(MultiAgentEnv):

    def __init__(self, env_config):
        map_name = env_config["map_name"]
        env_config.pop("map_name", None)
        env = REGISTRY[map_name](**env_config)
        self.max_cycles = env_config["max_cycles"]

        # keep obs and action dim same across agents
        # pad_action_space_v0 will auto mask the padding actions
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)

        self.env = ParallelPettingZooEnv(env)
        self.action_space = self.env.action_space
        self.observation_space = GymDict({
            "obs": Box(
                low=-100.0,
                high=100.0,
                shape=(self.env.observation_space.shape[0],),
                dtype=self.env.observation_space.dtype),
            "terminal": Box(low=0., high=1., shape=(1,))
        })
        self._dtype = self.env.observation_space.dtype
        self.agents = self.env.agents
        self.num_agents = len(self.agents)
        env_config["map_name"] = map_name
        self.env_config = env_config
        self.env.seed(seed=seed)

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for i in self.agents:
            obs[i] = {
                "obs": original_obs[i],
                "terminal": np.array([0.], dtype=self._dtype)
            }
        return obs

    def step(self, action_dict):
        action_dict = unwrap_multi_agent_actions(action_dict)
        o, r, d, info = self.env.step(action_dict)
        rewards = {}
        obs = {}

        # Check for cooperation env reward
        raw_rew = list(r.values())
        # if np.mean(raw_rew) != raw_rew[0]:
        #     warnings.warn(
        #         colorize("%s: %s" % ("WARN", "Agent rewards are not the same: " + str(raw_rew)), "yellow")
        #     )

        for key in action_dict.keys():
            rewards[key] = r[key]
            obs[key] = {
                "obs": o[key],
                "terminal": np.array([d[key]], dtype=self._dtype)
            }
        dones = {"__all__": d["__all__"]}
        rewards = apply_coop_reward(rewards)
        return obs, rewards, dones, info

    def close(self):
        self.env.close()

    def render(self, mode=None):
        self.env.render()
        return True

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.max_cycles,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
