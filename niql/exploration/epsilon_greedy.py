"""
Based on epsilon-greedy in RLLib.
"""
import random
from typing import Union
import torch
import tree  # pip install dm_tree
from ray.rllib.models import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchMultiActionDistribution
from ray.rllib.utils import PiecewiseSchedule
from ray.rllib.utils.torch_ops import FLOAT_MIN
from ray.rllib.utils.typing import TensorType


class AnnealSchedule:

    def __init__(self, initial_value, min_value, decay_steps):
        self.initial_value = initial_value
        self.min_value = min_value
        self._anneal_value = (initial_value - min_value) / decay_steps
        self._val = initial_value

    def value(self, t):
        return self._val

    def update(self):
        self._val = self._val - self._anneal_value if self._val > self.min_value else self._val

    def __call__(self, timestep):
        return self.value(timestep)


class EpsilonGreedy:

    ANNEAL = "anneal"
    PIECEWISE = "piecewise"

    def __init__(self, *, action_space, epsilon=1., min_epsilon=0.05, epsilon_decay_steps=50000, decay_type="piecewise",
                 device="cpu"):
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.device = device
        self.last_timestep = 0
        self.action_space = action_space
        if decay_type == "piecewise":
            self.epsilon_schedule = PiecewiseSchedule(
                endpoints=[(0, epsilon), (epsilon_decay_steps, min_epsilon)],
                outside_value=min_epsilon,
                framework="torch",
            )
        else:
            self.epsilon_schedule = AnnealSchedule(epsilon, min_epsilon, epsilon_decay_steps)

    def get_exploration_action(self, *,
                               action_distribution: ActionDistribution,
                               timestep: Union[int, TensorType],
                               explore: bool = True):
        q_values = action_distribution.inputs
        self.last_timestep = timestep
        exploit_action = action_distribution.deterministic_sample()
        batch_size = q_values.size()[0]
        action_logp = torch.zeros(batch_size, dtype=torch.float)

        # Explore.
        if explore:
            # Get the current epsilon.
            epsilon = self.epsilon_schedule(self.last_timestep)

            # update epsilon
            if isinstance(self.epsilon_schedule, AnnealSchedule):
                self.epsilon_schedule.update()

            if isinstance(action_distribution, TorchMultiActionDistribution):
                exploit_action = tree.flatten(exploit_action)
                for i in range(batch_size):
                    if random.random() < epsilon:
                        random_action = tree.flatten(
                            self.action_space.sample())
                        for j in range(len(exploit_action)):
                            exploit_action[j][i] = torch.tensor(
                                random_action[j])
                exploit_action = tree.unflatten_as(
                    action_distribution.action_space_struct, exploit_action)

                return exploit_action, action_logp

            else:
                # Mask out actions, whose Q-values are -inf, so that we don't
                # even consider them for exploration.
                random_valid_action_logits = torch.where(
                    q_values <= FLOAT_MIN,
                    torch.ones_like(q_values) * 0.0, torch.ones_like(q_values))
                # A random action.
                random_actions = torch.squeeze(
                    torch.multinomial(random_valid_action_logits, 1), axis=1)

                # Pick either random or greedy.
                action = torch.where(
                    torch.empty(
                        (batch_size,)).uniform_().to(self.device) < epsilon,
                    random_actions, exploit_action)

                return action, action_logp
        # Return the deterministic "sample" (argmax) over the logits.
        else:
            return exploit_action, action_logp

    def get_state(self, *args, **kwargs):
        eps = self.epsilon_schedule(self.last_timestep)
        return {
            "cur_epsilon": eps,
            "last_timestep": self.last_timestep,
        }

    def set_state(self, state: dict, *args, **kwargs) -> None:
        self.last_timestep = state["last_timestep"]

    def on_episode_start(self, policy, *, environment=None, episode=None, tf_sess=None):
        """Handles necessary exploration logic at the beginning of an episode.

        Args:
            policy (Policy): The Policy object that holds this Exploration.
            environment (BaseEnv): The environment object we are acting in.
            episode (int): The number of the episode that is starting.
            tf_sess (Optional[tf.Session]): In case of tf, the session object.
        """
        pass

    def on_episode_end(self, policy, *, environment=None, episode=None, tf_sess=None):
        """Handles necessary exploration logic at the end of an episode.

        Args:
            policy (Policy): The Policy object that holds this Exploration.
            environment (BaseEnv): The environment object we are acting in.
            episode (int): The number of the episode that is starting.
            tf_sess (Optional[tf.Session]): In case of tf, the session object.
        """
        pass

    def postprocess_trajectory(self, policy, sample_batch, tf_sess=None):
        """Handles post-processing of done episode trajectories.

        Changes the given batch in place. This callback is invoked by the
        sampler after policy.postprocess_trajectory() is called.

        Args:
            policy (Policy): The owning policy object.
            sample_batch (SampleBatch): The SampleBatch object to post-process.
            tf_sess (Optional[tf.Session]): An optional tf.Session object.
        """
        return sample_batch
