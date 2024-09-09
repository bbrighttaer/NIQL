import time

import numpy as np
from gym import spaces
from ray.rllib import MultiAgentEnv

from niql import seed

policy_mapping_dict = {
    "all_scenario": {
        "description": "one team cooperate",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}


class OneStepMultiAgentCoopMatrixGame(MultiAgentEnv):
    """
    Implements a simple one-step cooperative matrix game following QTran paper
    """

    def __init__(self, env_config):
        self.map_name = env_config["map_name"]
        self.action_space = spaces.Discrete(3)  # Two actions for Agent 1: 0 or 1
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(
                low=0,
                high=1,
                shape=(1,),
                dtype=np.int32,
            ),
            "terminal": spaces.Box(low=0., high=1., shape=(1,))
        })
        self.agents = ["agent_0", "agent_1"]
        self.num_agents = len(self.agents)
        self.step_count = 0
        self.max_steps = 1
        # Payoff matrix for game 1
        self.payoff = np.array([
            [8, -12, -12],
            [-12, 0, 0],
            [-12, 0, 0],
        ])
        self.seed(seed)

    def reset(self):
        self.step_count = 0
        obs = {}
        for i in self.agents:
            obs[i] = {
                "obs": np.array([0], dtype=np.int32),  # Initial observation for each agent
                "terminal": np.array([0.], dtype=np.float32)
            }
        return obs

    def step(self, actions):
        self.step_count += 1
        if self.step_count > self.max_steps:
            raise ValueError("All steps already taken")

        if len(actions) != len(self.agents):
            raise ValueError("Number of actions must match the number of agents")

        # Calculate payoff based on the chosen game and actions of both agents
        payoffs = [
            self.payoff[actions["agent_0"]][actions["agent_1"]],
            self.payoff[actions["agent_0"]][actions["agent_1"]],
        ]

        # Observations after taking actions
        obs = {}
        for i in self.agents:
            obs[i] = {
                "obs": np.array([1], dtype=np.int32),
                "terminal": np.array([1.], dtype=np.float32)
            }

        # Return observations, global payoff, done flag (always False for this game), and info dictionary
        rewards = {agent: reward for agent, reward in zip(self.agents, payoffs)}
        info = {agent: {} for agent in self.agents}
        return obs, rewards, {"__all__": True}, info

    def render(self, mode='human'):
        print("This is a one-step multi-agent cooperative matrix game.")
        time.sleep(0.05)
        return True

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 1,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
