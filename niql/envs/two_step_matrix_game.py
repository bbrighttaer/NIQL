import time

import numpy as np
from gym import spaces
from ray.rllib import MultiAgentEnv

policy_mapping_dict = {
    "all_scenario": {
        "description": "one team cooperate",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}


def _get_payoff2(is_monotonic: bool, is_linear: bool) -> np.array:
    if is_linear and is_monotonic:
        return np.array([
            [0, 1],
            [1, 2],
        ])
    elif not is_linear and is_monotonic:
        return np.array([
            [0, 1],
            [1, 8],
        ])
    else:  # non-linear and non-monotonic
        return np.array([
            [2, 1],
            [1, 8],
        ])


class TwoStepMultiAgentCoopMatrixGame(MultiAgentEnv):
    """
    Implements a simple two-step cooperative matrix game following https://arxiv.org/abs/1803.11485
    """

    def __init__(self, env_config):
        self.map_name = env_config["map_name"]
        self.action_space = spaces.Discrete(2)  # Two actions for Agent 1: 0 or 1
        self.observation_space = spaces.Dict({"obs": spaces.Box(low=0, high=1, shape=(3,), dtype=np.int32)})
        self.agents = ["agent_0", "agent_1"]
        self.num_agents = len(self.agents)
        self.step_count = 0
        self.max_steps = 2
        # Payoff matrix for game 1
        self.payoff1 = np.array([
            [7, 7],
            [7, 7],
        ])
        # Payoff matrix for game 2
        self.payoff2 = _get_payoff2(env_config["monotonic_payoff"], env_config["linear_payoff"])
        self.current_game = 0  # Current game being played
        self.seed(321)


    def reset(self):
        self.step_count = 0
        self.current_game = 0
        obs = {}
        for i in self.agents:
            obs[i] = {
                "obs": [0, 0, 0]  # Initial observation for each agent
            }
        return obs

    def step(self, actions):
        self.step_count += 1
        if self.step_count > self.max_steps:
            raise ValueError("All steps already taken")

        if len(actions) != len(self.agents):
            raise ValueError("Number of actions must match the number of agents")

        if self.step_count == 1:
            # Agent 1 chooses which game to play next
            self.current_game = actions["agent_0"]

            # Placeholder observations for each agent in the first step
            obs = {}
            for i in self.agents:
                obs[i] = {
                    "obs": [0, 1, 0] if self.current_game == 0 else [0, 0, 1]
                }

            # No reward in the first step
            return obs, {agent: 0 for agent in self.agents}, {"__all__": False}, {agent: {} for agent in self.agents}

        elif self.step_count == 2:
            # Calculate payoff based on the chosen game and actions of both agents
            if self.current_game == 0:
                payoffs = [
                    self.payoff1[actions["agent_0"]][actions["agent_1"]],
                    self.payoff1[actions["agent_0"]][actions["agent_1"]],
                ]
            else:
                payoffs = [
                    self.payoff2[actions["agent_0"]][actions["agent_1"]],
                    self.payoff2[actions["agent_0"]][actions["agent_1"]],
                ]

            # Observations after taking actions
            obs = {}
            for i in self.agents:
                obs[i] = {
                    "obs": [1, 1, 1]
                }

            # Return observations, global payoff, done flag (always False for this game), and info dictionary
            rewards = {agent: reward for agent, reward in zip(self.agents, payoffs)}
            info = {agent: {} for agent in self.agents}
            return obs, rewards, {"__all__": True}, info

    def render(self, mode='human'):
        print("This is a two-step multi-agent cooperative matrix game.")
        time.sleep(0.05)
        return True

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 2,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
