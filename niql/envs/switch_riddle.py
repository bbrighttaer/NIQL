"""
Logic attempts to follow: https://github.com/minqi/learning-to-communicate-pytorch/blob/master/switch/switch_game.py
"""

import numpy as np
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from niql.utils import DotDict, unwrap_multi_agent_actions, apply_coop_reward

policy_mapping_dict = {
    "all_scenario": {
        "description": "one team cooperate",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}


class SwitchRiddle(MultiAgentEnv):

    def __init__(self, env_config):
        self.agents = [f"agent_{i}" for i in range(env_config["n_agents"])]
        self.num_agents = len(self.agents)
        self.env_config = env_config
        self.max_steps = 4 * self.num_agents - 6
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "state": spaces.Box(low=0, high=1, shape=(self.num_agents,), dtype=np.float32),
        })

        self.game_actions = DotDict({
            'NOTHING': 0,
            'TELL': 1,
        })

        self.game_states = DotDict({
            'OUTSIDE': 0,
            'INSIDE': 1,
        })

        self.action_masks = DotDict({
            'NONE_ACTION_ONLY': [1, 0],
            'NONE_AND_TELL': [1, 1],
        })

        self.reward_all_live = 1
        self.reward_all_die = -1

        self.reset()

    def reset(self):
        # step count
        self.step_count = 0

        # who has been in the room?
        self.has_been = np.zeros((self.max_steps, self.num_agents))

        # active agent
        self.active_agent = np.zeros((self.max_steps,), dtype=np.int32)
        for step in range(self.max_steps):
            agent_id = np.random.randint(self.num_agents)
            self.active_agent[step] = agent_id
            self.has_been[step][agent_id] = 1

        obs = {}
        for agent in self.agents:
            obs[agent] = self.get_obs(agent)
        return obs

    def get_obs(self, agent_id):
        state = self.get_state()
        if f"agent_{self.active_agent[self.step_count]}" == agent_id:
            return {
                "obs": np.array([self.game_states.INSIDE], dtype=np.float32),
                "action_mask": np.array(self.action_masks.NONE_AND_TELL, dtype=np.float32),
                "state": state,
            }
        else:
            return {
                "obs": np.array([self.game_states.OUTSIDE], dtype=np.float32),
                "action_mask": np.array(self.action_masks.NONE_ACTION_ONLY, dtype=np.float32),
                "state": state,
            }

    def step(self, action_dict: MultiAgentDict):
        action_dict = unwrap_multi_agent_actions(action_dict)
        terminal = False
        reward = 0
        active_agent_idx = self.active_agent[self.step_count]
        if action_dict[f"agent_{active_agent_idx}"] == self.game_actions.TELL and not terminal:
            has_been = (self.has_been[:self.step_count + 1].sum(0) > 0).sum(0)
            if has_been == self.num_agents:
                reward = self.reward_all_live
            else:
                reward = self.reward_all_die
            terminal = True

        if self.step_count == self.max_steps - 1:
            terminal = True

        # Multi-agent obs data
        obs = {}
        rew = {}
        done = {'__all__': terminal}
        info = {}
        for agent in self.agents:
            obs[agent] = self.get_obs(agent)
            rew[agent] = reward
            done[agent] = terminal
            info[agent] = {"god_reward": self.god_strategy_reward()}

        self.step_count += 1
        rew = apply_coop_reward(rew)
        return obs, rew, done, info

    def get_state(self):
        state = np.zeros((self.num_agents,), dtype=np.float32)

        # Get the state of the game
        for a in range(self.num_agents):
            if self.active_agent[self.step_count] == a:
                state[a] = self.game_states.INSIDE

        return state

    def god_strategy_reward(self):
        reward = 0
        has_been = (self.has_been[:self.max_steps + 1].sum(0) > 0).sum(0)
        if has_been == self.num_agents:
            reward = self.reward_all_live

        return reward

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.max_steps,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
