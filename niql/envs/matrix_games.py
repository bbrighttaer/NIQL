"""
Based on https://github.com/semitable/matrix-games/blob/master/matrixgames/games.py
"""
import random

import gym
import numpy as np
from ray.rllib import MultiAgentEnv

from niql import seed
from niql.utils import unwrap_multi_agent_actions, apply_coop_reward


# penalty game
def create_penalty_game(ep_length, penalty):
    assert penalty <= 0
    payoff = np.array([
        [10, 0, penalty],
        [0, 2, 0],
        [penalty, 0, 10],
    ])
    game = MatrixGame(payoff, ep_length)
    return game


# climbing game
def create_climbing_game(ep_length):
    payoff = np.array([
        [11, -30, 0],
        [-30, 7, 6],
        [0, 0, 5],
    ])
    game = MatrixGame(payoff, ep_length)
    return game


# one-step cooperative matrix game in QTran (https://arxiv.org/pdf/1905.05408)
def create_one_step_cooperative_matrix_game():
    payoff = np.array([
        [8, -12, -12],
        [-12, 0, 0],
        [-12, 0, 0],
    ])
    game = MatrixGame(payoff, 1,)
    return game


# two-step cooperative matrix game in QMIX (https://arxiv.org/pdf/1803.11485)
def create_two_step_cooperative_matrix_game(linear_payoff=False, monotonic_payoff=False):
    # Get payoff matrices
    payoff_matrix_1 = np.array([
        [7, 7],
        [7, 7],
    ])
    if linear_payoff and monotonic_payoff:
        payoff_matrix_2 = np.array([
            [0, 1],
            [1, 2],
        ])
    elif not linear_payoff and monotonic_payoff:
        payoff_matrix_2 = np.array([
            [0, 1],
            [1, 8],
        ])
    else:  # non-linear and non-monotonic
        payoff_matrix_2 = np.array([
            [2, 1],
            [1, 8],
        ])
    game = TwoStepGame(payoff_matrix_1, payoff_matrix_2)
    return game


policy_mapping_dict = {
    "one_step": {
        "description": "one team cooperate",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
    "two_step": {
        "description": "one team cooperate",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
    "climbing": {
        "description": "one team cooperate",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
    "penalty": {
        "description": "one team cooperate",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}

REGISTRY = {
    "one_step": create_one_step_cooperative_matrix_game,
    "two_step": create_two_step_cooperative_matrix_game,
    "climbing": create_climbing_game,
    "penalty": create_penalty_game
}


class MultiAgentMatrixGameEnv(MultiAgentEnv):
    """
    Wrapper around the Matrix Games
    """

    def __init__(self, env_config):
        map_name = env_config.pop("map_name", None)
        self.env = REGISTRY[map_name](**env_config)
        self.num_agents = self.env.n_agents
        self.agents = [f"agent_{i}" for i in range(self.env.n_agents)]
        self.action_space = self.env.action_space.spaces[0]
        self.observation_space = gym.spaces.Dict({
            "obs": self.env.observation_space.spaces[0],
            "terminal": gym.spaces.Box(low=0., high=1., shape=(1,))
        })
        env_config["map_name"] = map_name
        self.env_config = env_config
        self.max_cycles = self.env.ep_length
        self._dtype = np.float32

        random.seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

    def reset(self):
        raw_obs, _ = self.env.reset()
        obs = {}
        for agent, r_obs in zip(self.agents, raw_obs):
            obs[agent] = {
                "obs": np.array(r_obs, dtype=self._dtype),
                "terminal": np.array([0.], dtype=self._dtype)
            }
        return obs

    def step(self, action_dict):
        action_dict = unwrap_multi_agent_actions(action_dict)
        raw_obs, raw_rew, raw_done, truncated, raw_info = self.env.step(list(action_dict.values()))
        obs = {}
        rew = {}
        done = {"__all__": raw_done}
        info = {}

        # Check for cooperation env reward
        # if np.mean(raw_rew) != raw_rew[0]:
        #     warnings.warn(
        #         colorize("%s: %s" % ("WARN", "Agent rewards are not the same: " + str(raw_rew)), "yellow")
        #     )

        for agent, r_obs, r_rew in zip(self.agents, raw_obs, raw_rew):
            obs[agent] = {
                "obs": np.array(r_obs, dtype=self._dtype),
                "terminal": np.array([raw_done], dtype=self._dtype)
            }
            rew[agent] = r_rew
            done[agent] = raw_done
            info[agent] = dict(raw_info)
        rew = apply_coop_reward(rew)
        return obs, rew, done, info

    def close(self):
        self.env.close()

    def render(self, mode=None):
        return self.env.render()

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.max_cycles,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info


class MatrixGame(gym.Env):
    def __init__(self, payoff_matrix, ep_length):
        """
        Create matrix game
        :param payoff_matrix: list of lists or numpy array for payoff matrix of all agents
        :param ep_length: length of episode (before done is True)
        """
        self.n_agents = 2
        self.num_actions = payoff_matrix.shape
        self.payoff = [payoff_matrix] * len(self.num_actions)

        self.ep_length = ep_length
        self._states = np.eye(ep_length + 1)

        self.last_actions = [-1 for _ in range(self.n_agents)]
        self.t = 0

        self.action_space = gym.spaces.Tuple(
            [gym.spaces.Discrete(num_action) for num_action in self.num_actions]
        )

        shape = (ep_length + 1,)
        low = np.zeros(shape)
        high = np.ones(shape)
        obs_space = gym.spaces.Box(shape=shape, low=low, high=high)
        self.observation_space = gym.spaces.Tuple(
            [obs_space for _ in range(self.n_agents)]
        )

    def _make_obs(self):
        return [self._states[self.t] for _ in range(self.n_agents)]

    def reset(self, seed=None, options=None):
        self.t = 0
        return self._make_obs(), {}

    def step(self, action):
        self.t += 1
        self.last_actions = action

        rewards = [0 for _ in range(len(action))]
        for i in range(len(action)):
            temp = self.payoff[i]
            for j in range(len(action)):
                temp = temp[action[j]]
            rewards[i] = temp

        done = self.t >= self.ep_length
        truncated = False

        return self._make_obs(), rewards, done, truncated, {}

    def render(self):
        print(f"Step {self.t}:")
        for i in range(self.n_agents):
            print(f"\tAgent {i + 1} action: {self.last_actions[i]}")


class TwoStepGame(gym.Env):
    def __init__(self, payoff_matrix1, payoff_matrix2):
        self.payoff1 = payoff_matrix1
        self.payoff2 = payoff_matrix1

        self.matrix1 = MatrixGame(payoff_matrix1, ep_length=1)
        self.matrix2 = MatrixGame(payoff_matrix2, ep_length=1)
        self.n_agents = self.matrix1.n_agents

        self.ep_length = 2

        self.action_space = self.matrix1.action_space
        shape = (3,)
        low = np.zeros(shape)
        high = np.ones(shape)
        obs_space = gym.spaces.Box(shape=shape, low=low, high=high)
        self.observation_space = gym.spaces.Tuple(
            [obs_space for _ in range(self.n_agents)]
        )

        self.t = 0
        self.state = "A"

    def _make_obs(self):
        if self.state == "A":
            x = [1, 0, 0]
        elif self.state == "2A":
            x = [0, 1, 0]
        else:
            x = [0, 0, 1]
        return tuple([np.array(x)] * self.n_agents)

    def reset(self, seed=None, options=None):
        self.state = "A"
        self.t = 0
        # self.last_actions = actions_to_onehot(self.num_actions, [0] * self.n_agents)
        self.matrix1.reset()
        self.matrix2.reset()

        return self._make_obs(), {}

    def step(self, action):
        if self.t == 0:
            if action[0] == 0:
                self.state = "2A"
            else:
                self.state = "2B"
            rewards = self.n_agents * [0]
        elif self.t == 1:
            if self.state == "2A":
                _, rewards, _, _, _ = self.matrix1.step(action)
            elif self.state == "2B":
                _, rewards, _, _, _ = self.matrix2.step(action)
        else:
            rewards = self.n_agents * [0]

        done = self.t != 0
        truncated = False
        self.t += 1

        return self._make_obs(), rewards, done, truncated, {}
