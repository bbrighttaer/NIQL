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

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from smac.env.starcraft2.starcraft2 import StarCraft2Env
import numpy as np
from gym.spaces import Dict as GymDict, Discrete, Box

from niql import seed
from niql.utils import unwrap_multi_agent_actions, apply_coop_reward

policy_mapping_dict = {
    "all_scenario": {
        "description": "smac all scenarios",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
        # be careful using one_agent_one_policy when training in maps like 27m_vs_30m, which has relatively large
        # number of agents
    },
}


class RLlibSMAC(MultiAgentEnv):

    def __init__(self, map_name):
        map_name = map_name if isinstance(map_name, str) else map_name["map_name"]
        self.env = StarCraft2Env(map_name, seed=seed)

        env_info = self.env.get_env_info()
        self.num_agents = self.env.n_agents
        self.agents = ["agent_{}".format(i) for i in range(self.num_agents)]
        obs_shape = env_info['obs_shape']
        n_actions = env_info['n_actions']
        state_shape = env_info['state_shape']
        self.observation_space = GymDict({
            "obs": Box(-2.0, 2.0, shape=(obs_shape,)),
            "state": Box(-2.0, 2.0, shape=(state_shape,)),
            "action_mask": Box(-2.0, 2.0, shape=(n_actions,)),
            "terminal": Box(low=0., high=1., shape=(1,))
        })
        self.action_space = Discrete(n_actions)
        self._last_info = None

    @property
    def death_tracker_ally(self):
        return self.env.death_tracker_ally

    @property
    def death_tracker_enemy(self):
        return self.env.death_tracker_enemy

    @property
    def info(self):
        return self._last_info

    def reset(self):
        self.env.reset()
        self._last_info = None
        obs_smac = self.env.get_obs()
        state_smac = self.env.get_state()
        obs_dict = {}
        for agent_index in range(self.num_agents):
            obs_one_agent = obs_smac[agent_index]
            state_one_agent = state_smac
            action_mask_one_agent = np.array(self.env.get_avail_agent_actions(agent_index)).astype(np.float32)
            agent_index = "agent_{}".format(agent_index)
            obs_dict[agent_index] = {
                "obs": obs_one_agent,
                "state": state_one_agent,
                "action_mask": action_mask_one_agent,
                "terminal": np.array([0.], dtype=np.float32)
            }

        return obs_dict

    def step(self, action_dict):
        action_dict = unwrap_multi_agent_actions(action_dict)
        actions_ls = [int(action_dict[agent_id]) for agent_id in action_dict.keys()]

        reward, terminated, info = self.env.step(actions_ls)
        self._last_info = info

        obs_smac = self.env.get_obs()
        state_smac = self.env.get_state()

        obs_dict = {}
        reward_dict = {}
        for agent_index in range(self.num_agents):
            obs_one_agent = obs_smac[agent_index]
            state_one_agent = state_smac
            action_mask_one_agent = np.array(self.env.get_avail_agent_actions(agent_index)).astype(np.float32)
            agent_index = "agent_{}".format(agent_index)
            obs_dict[agent_index] = {
                "obs": obs_one_agent,
                "state": state_one_agent,
                "action_mask": action_mask_one_agent,
                "terminal": np.array([terminated], dtype=np.float32)
            }
            reward_dict[agent_index] = reward

        dones = {"__all__": terminated}
        reward_dict = apply_coop_reward(reward_dict)
        return obs_dict, reward_dict, dones, {}

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.env.episode_limit,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info

    def close(self):
        self.env.close()
