from marllib.marl.algos.utils.episode_execution_plan import episode_execution_plan
from ray.rllib.agents.dqn import DEFAULT_CONFIG
from ray.rllib.agents.trainer_template import build_trainer

from .iql import IQLPolicy
from .imix import IMIX
from .bql import BQLPolicy
from .dueling_bql import DuelingBQLPolicy

from ..execution_plans import imix_episode_execution_plan, joint_episode_execution_plan

IQLTrainer = build_trainer(
    name="IQLTrainer",
    get_policy_class=lambda c: IQLPolicy,
    default_config=DEFAULT_CONFIG,
    execution_plan=episode_execution_plan,
)

IMIXTrainer = build_trainer(
    name="IMIXTrainer",
    get_policy_class=lambda c: IMIX,
    default_config=DEFAULT_CONFIG,
    execution_plan=imix_episode_execution_plan,
)

BPQTrainer = build_trainer(
    name="BPQTrainer",
    get_policy_class=lambda c: DuelingBQLPolicy,
    default_config=DEFAULT_CONFIG,
    execution_plan=joint_episode_execution_plan,
)
