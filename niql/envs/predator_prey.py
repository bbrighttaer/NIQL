from gym.spaces import Dict as GymDict, Box
from ma_gym.envs.predator_prey import PredatorPrey as Env
from ray.rllib import MultiAgentEnv

policy_mapping_dict = {
    "all_scenario": {
        "description": "one team cooperate",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}


class PredatorPrey(MultiAgentEnv):
    """
    Wrapper around the PredatorPrey environment of ma-gym.
    """

    def __init__(self, env_config):
        map_name = env_config["map_name"]
        env_config.pop("map_name", None)
        self.env = Env(**env_config)
        self.action_space = self.env.action_space[0]
        observation_space = self.env.observation_space[0]
        self.observation_space = GymDict({"obs": Box(
            low=observation_space.low[0],
            high=observation_space.high[0],
            shape=observation_space.shape,
            dtype=observation_space.dtype)})
        self.agents = [f'agent_{i}' for i in range(self.env.n_agents)]
        self.num_agents = self.env.n_agents
        env_config["map_name"] = map_name
        self.env.seed(321)

    def reset(self):
        raw_obs = self.env.reset()
        obs = {}
        for agent, r_obs in zip(self.agents, raw_obs):
            obs[agent] = r_obs
        return obs

    def step(self, action_dict):
        raw_obs, raw_rew, raw_done, raw_info = self.env.step(action_dict.values())
        obs = {}
        rew = {}
        done = {}
        info = {}

        for agent, r_obs, r_rew, r_done in zip(self.agents, raw_obs, raw_rew, raw_done):
            obs[agent] = r_obs
            rew[agent] = r_rew
            done[agent] = r_done
            info[agent] = dict(raw_info)

        return obs, rew, done, info

    def render(self, mode='human'):
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
