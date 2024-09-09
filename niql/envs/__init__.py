import numpy as np
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY

from .ma_gym_env import MAGymEnv
from .mpe_env import MPEEnv
from .one_step_matrix_game import OneStepMultiAgentCoopMatrixGame
from .one_step_matrix_game import OneStepMultiAgentCoopMatrixGame
from .smac import RLlibSMAC
from .switch_riddle import SwitchRiddle
from .two_step_matrix_game import TwoStepMultiAgentCoopMatrixGame
from .two_step_matrix_game import TwoStepMultiAgentCoopMatrixGame
from .utils import make_local_env
from ..config import SMAC, MPE, MATRIX_GAME, ma_gym_env_conf
from ..config.switch_game_conf import SWITCH_RIDDLE

DEBUG_ENVS = ["TwoStepsCoopMatrixGame", "OneStepCoopMatrixGame"]


def get_active_env(**kwargs):
    return make_predator_prey_env(**kwargs)


def make_mpe_simple_spread_env(**kwargs):
    # register new env
    ENV_REGISTRY["mpe"] = MPEEnv
    COOP_ENV_REGISTRY["mpe"] = MPEEnv

    env = make_local_env(
        environment_name="mpe",
        map_name="simple_spread",
        force_coop=True,
        max_cycles=25,
        **kwargs,
    )
    return env, MPE


def make_mpe_simple_reference(**kwargs):
    env = marl.make_env(
        environment_name="mpe",
        map_name="simple_reference",
        force_coop=True,
        max_cycles=25,
        **kwargs,
    )
    return env, MPE


def make_mpe_extended_simple_reference(**kwargs):
    # register new env
    ENV_REGISTRY["mpe"] = MPEEnv
    COOP_ENV_REGISTRY["mpe"] = MPEEnv

    env = make_local_env(
        environment_name="mpe",
        map_name="simple_reference",
        force_coop=True,
        max_cycles=25,
        **kwargs,
    )
    return env, MPE


def make_mpe_simple_speaker_listener(**kwargs):
    env = marl.make_env(
        environment_name="mpe",
        map_name="simple_speaker_listener",
        force_coop=True,
        max_cycles=25,
        **kwargs,
    )
    return env, MPE


def make_predator_prey_env(**kwargs):
    return make_ma_gym_env(env_name="PredatorPrey", **kwargs)


def make_checkers_env(**kwargs):
    return make_ma_gym_env(env_name="Checkers", **kwargs)


def make_combat_env(**kwargs):
    return make_ma_gym_env(env_name="Combat", **kwargs)


def make_lumber_jack(**kwargs):
    return make_ma_gym_env(env_name="LumberJack", **kwargs)


def make_switch_env(**kwargs):
    return make_ma_gym_env(env_name="Switch", **kwargs)


def make_ma_gym_env(env_name, **kwargs):
    # register new env
    COOP_ENV_REGISTRY[env_name] = MAGymEnv
    ENV_REGISTRY[env_name] = MAGymEnv

    # choose environment + scenario
    env = make_local_env(
        environment_name=env_name,
        map_name="all_scenario",
        force_coop=True,
        **kwargs,
    )
    config = ma_gym_env_conf.REGISTRY.get(env_name.lower(), ma_gym_env_conf.default_config)
    return env, config


def make_smac_env(**kwargs):
    # register new env
    ENV_REGISTRY["smac"] = RLlibSMAC
    COOP_ENV_REGISTRY["smac"] = RLlibSMAC
    env = make_local_env(
        environment_name="smac",
        map_name=kwargs.get("map_name", "3m"),
    )
    return env, SMAC


def make_mpe_simple_env(**kwargs):
    # register new env
    ENV_REGISTRY["mpe"] = MPEEnv
    COOP_ENV_REGISTRY["mpe"] = MPEEnv

    # choose environment + scenario
    env = marl.make_env(
        environment_name="mpe",
        map_name="simple",
        # force_coop=True,
        max_cycles=25,
        **kwargs,
    )
    return env, MPE


def make_two_step_matrix_game_env(**kwargs):
    # register new env
    ENV_REGISTRY["TwoStepsCoopMatrixGame"] = TwoStepMultiAgentCoopMatrixGame
    COOP_ENV_REGISTRY["TwoStepsCoopMatrixGame"] = TwoStepMultiAgentCoopMatrixGame

    # choose environment + scenario
    env = make_local_env(
        environment_name="TwoStepsCoopMatrixGame",
        map_name="all_scenario",
        **kwargs,
    )
    return env, MATRIX_GAME


def make_one_step_matrix_game_env(**kwargs):
    # register new env
    ENV_REGISTRY["OneStepCoopMatrixGame"] = OneStepMultiAgentCoopMatrixGame
    COOP_ENV_REGISTRY["OneStepCoopMatrixGame"] = OneStepMultiAgentCoopMatrixGame

    # choose environment + scenario
    env = make_local_env(
        environment_name="OneStepCoopMatrixGame",
        map_name="all_scenario",
        **kwargs,
    )
    return env, MATRIX_GAME


def pad_obs_space(obs_space):
    for prop in ['bounded_above', 'bounded_below', 'high', 'low']:
        if hasattr(obs_space, prop):
            val = obs_space.bounded_above[0]
            setattr(obs_space, prop, np.pad(getattr(obs_space, prop), (0, 2), 'constant', constant_values=(val,)))
    if hasattr(obs_space, 'original_space'):
        pad_obs_space(obs_space.original_space.spaces['obs'])
    if hasattr(obs_space, '_shape'):
        obs_space._shape = obs_space.bounded_above.shape
    return obs_space


def make_switch_riddle_env(**kwargs):
    # register new env
    ENV_REGISTRY["SwitchRiddle"] = SwitchRiddle
    COOP_ENV_REGISTRY["SwitchRiddle"] = SwitchRiddle

    # choose environment + scenario
    env = make_local_env(
        environment_name="SwitchRiddle",
        map_name="all_scenario",
        **kwargs,
    )
    return env, SWITCH_RIDDLE
