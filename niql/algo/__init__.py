from marllib.marl.algos.utils.episode_execution_plan import episode_execution_plan
from ray.rllib.agents.dqn import DEFAULT_CONFIG
from ray.rllib.agents.trainer_template import build_trainer

from .dqn_policy import IQLPolicy
from .q_learning import QLearningPolicy

DQNTrainer = build_trainer(
    name="DQNTrainer",
    get_policy_class=lambda c: QLearningPolicy,
    default_config=DEFAULT_CONFIG,
    execution_plan=episode_execution_plan,
)
