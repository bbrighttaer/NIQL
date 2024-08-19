from marllib.marl.algos.utils.episode_execution_plan import episode_execution_plan as basic_execution_plan  # noqa
from ray.rllib.agents.dqn import DEFAULT_CONFIG
from ray.rllib.agents.trainer_template import build_trainer

from .bql import BQLPolicy
from .iql import IQLPolicy
from .wbql import WBQLPolicy
from ..execution_plans import episode_execution_plan, cert_episode_execution_plan

IQLTrainer = build_trainer(
    name="IQLTrainer",
    get_policy_class=lambda c: IQLPolicy,
    default_config=DEFAULT_CONFIG,
    execution_plan=episode_execution_plan,
)

BQLTrainer = build_trainer(
    name="BQLTrainer",
    get_policy_class=lambda c: WBQLPolicy,
    default_config=DEFAULT_CONFIG,
    execution_plan=episode_execution_plan,
)
