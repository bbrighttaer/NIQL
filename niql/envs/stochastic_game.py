import random

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


class CooperativeStochasticGame(MultiAgentEnv):
    def __init__(self, env_config):
        # Game parameters
        self.num_agents = env_config["num_agents"]
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.num_states = env_config["num_states"]
        self.num_actions_per_agent = env_config["num_actions_per_agent"]
        self.joint_action_space = self.num_actions_per_agent ** env_config["num_agents"]
        self.current_step = 0  # Initialize step count
        self.max_steps = env_config["max_steps"]
        self.start_state = -1
        self.normalised_episode_reward = -1
        self.sum_episode_rewards = 0
        self.sum_optimal_episode_rewards = 0

        # Define state and action spaces
        self._dtype = np.float32
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(.0, 1.0, shape=(self.num_states,), dtype=self._dtype),
            "state": spaces.Box(.0, 1.0, shape=(self.num_states,), dtype=self._dtype),
            "terminal": spaces.Box(low=0., high=1., shape=(1,), dtype=self._dtype)
        })
        self.action_space = spaces.Discrete(self.num_actions_per_agent)

        # Apply random seed
        random.seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

        # Randomly generate transition probabilities and reward function
        self.transition_matrix = self._generate_transition_matrix()
        self.reward_function = self._generate_reward_function()

        # Compute optimal value table and policy for this game
        self.V, self.optimal_policy = self.value_iteration()

        # Initial state
        self.state = None
        self.state_one_hot = np.eye(self.num_states).astype(self._dtype)

    def _generate_transition_matrix(self):
        """
        Randomly generate the transition matrix. It will have the shape:
        (num_states, joint_action_space, num_states), where the value at
        [state, joint_action, next_state] is the probability of transitioning
        from 'state' to 'next_state' given 'joint_action'.
        """
        transition_matrix = np.zeros((self.num_states, self.joint_action_space, self.num_states))
        for s in range(self.num_states):
            for a in range(self.joint_action_space):
                probabilities = self.rng.dirichlet(np.ones(self.num_states))  # Stochastic transition
                transition_matrix[s, a, :] = probabilities
        return transition_matrix

    def _generate_reward_function(self):
        """
        Randomly generate the reward function. This will have the shape:
        (num_states, joint_action_space), where the value at
        [state, joint_action] is the reward for taking 'joint_action' in 'state'.
        """
        return self.rng.uniform(low=-1.0, high=1.0, size=(self.num_states, self.joint_action_space))

    def _get_current_obs(self):
        obs = {}
        game_state = self.state_one_hot[self.state]
        for agent in self.agents:
            obs[agent] = {
                "obs": game_state,
                "state": game_state,
                "terminal": np.array([1. if self.current_step >= self.max_steps else 0.], dtype=self._dtype)
            }
        return obs

    def reset(self):
        """
        Reset the environment to the initial state and return the initial observation.
        """
        # Reset to a random initial state
        self.state = self.rng.integers(low=0, high=self.num_states)
        self.start_state = self.state
        self.current_step = 0  # Reset step count
        self.normalised_episode_reward = -1
        self.sum_episode_rewards = 0
        self.sum_optimal_episode_rewards = 0
        obs = self._get_current_obs()
        return obs

    def step(self, actions_dict):
        """
        Take a step in the environment with the joint action of all agents.

        Args:
        - actions: dictionary of actions taken by each agent (length = num_agents)

        Returns:
        - next_state: the next state after applying the joint action
        - reward: the collective reward for this step (fully cooperative)
        - done: whether the episode has ended (True if max_steps is reached)
        - info: additional information (empty dictionary here)
        """
        actions = list(actions_dict.values())

        # Convert individual actions to joint action index
        joint_action = np.ravel_multi_index(actions, [self.num_actions_per_agent] * self.num_agents)

        # Sample next state based on transition probabilities
        transition_probabilities = self.transition_matrix[self.state, joint_action]
        next_state = self.rng.choice(self.num_states, p=transition_probabilities)

        # Get reward for the joint action in the current state (fully cooperative reward)
        reward = self.reward_function[self.state, joint_action]
        self.sum_episode_rewards += reward
        rewards = {a: reward for a in self.agents}

        # Determine the optimal reward
        optimal_joint_action = self.optimal_policy[self.state]
        optimal_reward = self.reward_function[self.state, optimal_joint_action]
        self.sum_optimal_episode_rewards += optimal_reward

        # Update the state and step count
        self.state = next_state
        self.current_step += 1

        # Check if max steps has been reached
        done = self.current_step >= self.max_steps

        # Get current observation encoding
        obs = self._get_current_obs()

        # Add optimal expected return as extra info
        if done:
            self.normalised_episode_reward = self.sum_episode_rewards / self.sum_optimal_episode_rewards

        return obs, rewards, {"__all__": done}, {}

    def render(self, mode="human"):
        """
        Simple rendering method to print the current state of the environment.
        """
        print(f"Step: {self.current_step}, Current State: {self.state}")

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.max_steps,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info

    def value_iteration(self, epsilon=1e-3, discount_factor=0.99):
        """
        Perform value iteration to compute the optimal value function and return the optimal policy.
        This will be used to compute the optimal return.

        Args:
        - epsilon: convergence threshold for value iteration

        Returns:
        - V: optimal value function for each state
        - optimal_policy: the optimal policy (joint action) for each state
        """
        print("Computing the optimal value function...")
        V = np.zeros(self.num_states)  # Initialize value function to zero for all states
        optimal_policy = np.zeros(self.num_states, dtype=int)  # Initialize policy (best joint action)

        while True:
            delta = 0  # Measure of improvement in value function
            for s in range(self.num_states):
                # For each state, find the joint action that maximizes the expected reward + discounted future value
                action_values = np.zeros(self.joint_action_space)
                for a in range(self.joint_action_space):
                    action_values[a] = (
                        np.sum(self.transition_matrix[s, a, :] * (self.reward_function[s, a] + discount_factor * V))
                    )

                # Get the best action and update the value function for this state
                best_action_value = np.max(action_values)
                best_action = np.argmax(action_values)

                delta = max(delta, np.abs(best_action_value - V[s]))
                V[s] = best_action_value
                optimal_policy[s] = best_action  # Store the best action for this state

            # Check for convergence
            if delta < epsilon:
                break

        return V, optimal_policy
