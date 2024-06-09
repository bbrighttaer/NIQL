import numpy as np
from pettingzoo.mpe._mpe_utils.core import World, Agent, Landmark
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario


class Scenario(BaseScenario):

    def __init__(self, num_agents):
        super().__init__()
        self.num_agents = num_agents

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 10
        world.collaborative = True  # whether agents share rewards
        # add agents
        world.agents = [Agent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent_{i}'
            agent.collide = False
        # add landmarks
        world.landmarks = [Landmark() for i in range(int(1.5 * self.num_agents))]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        return world

    def reset_world(self, world, np_random):
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None

        # Randomly assign unique landmarks to agents as goals
        landmarks = np_random.choice(world.landmarks, size=len(world.agents), replace=False)
        for agent, landmark in zip(world.agents, landmarks):
            other_agents = [a for a in world.agents if a is not agent]
            agent.goal_a = np_random.choice(other_agents)
            agent.goal_b = landmark

        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])

        # Random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np_random.uniform(0, 1, 3)  # Random color for each landmark

        # Special colors for goals
        for agent in world.agents:
            agent.goal_a.color = agent.goal_b.color

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        if agent.goal_a is None or agent.goal_b is None:
            agent_reward = 0.0
        else:
            agent_reward = np.sqrt(np.sum(np.square(agent.goal_a.state.p_pos - agent.goal_b.state.p_pos)))
        return -agent_reward

    def global_reward(self, world):
        all_rewards = sum(self.reward(agent, world) for agent in world.agents)
        return all_rewards / len(world.agents)

    def observation(self, agent, world):
        # goal color
        goal_color = [np.zeros(world.dim_color), np.zeros(world.dim_color)]
        if agent.goal_b is not None:
            goal_color[1] = agent.goal_b.color

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
        return np.concatenate([agent.state.p_vel] + entity_pos + [goal_color[1]] + comm)
