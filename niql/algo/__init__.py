from marllib.marl.algos.utils.episode_execution_plan import episode_execution_plan
from ray.rllib.agents.dqn import DEFAULT_CONFIG
from ray.rllib.agents.trainer_template import build_trainer

from .iql import IQLPolicy
from .imix import IMIX

IQLTrainer = build_trainer(
    name="IQLTrainer",
    get_policy_class=lambda c: IQLPolicy,
    default_config=DEFAULT_CONFIG,
    execution_plan=episode_execution_plan,
)
