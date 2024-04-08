from marllib.marl.algos.utils.episode_execution_plan import episode_execution_plan
from ray.rllib.agents.dqn import DEFAULT_CONFIG
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents.dqn import DEFAULT_CONFIG as IQL_TRAINER_Config

from .iql import IQLPolicy

IQLTrainer = build_trainer(
    name="IQLTrainer",
    get_policy_class=lambda c: IQLPolicy,
    default_config=DEFAULT_CONFIG,
    execution_plan=episode_execution_plan,
)

NIQL_TRAINERS = {
    'iql': ('IL', IQLTrainer, IQL_TRAINER_Config),
}
