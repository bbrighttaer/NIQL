import numpy as np
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY

from .two_step_matrix_game import TwoStepMultiAgentCoopMatrixGame
from .one_step_matrix_game import OneStepMultiAgentCoopMatrixGame
from .mpe_simple import MPESimple
from .predator_prey import PredatorPrey
from .utils import make_local_env
from ..config import PREDATOR_PREY, SMAC, MPE, MATRIX_GAME

DEBUG_ENVS = ["TwoStepsCoopMatrixGame", "OneStepCoopMatrixGame"]


def get_active_env(**kwargs):
    return make_predator_prey_env(**kwargs)


def make_mpe_simple_spread_env(**kwargs):
    # choose environment + scenario
    env = marl.make_env(
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
    # register new env
    ENV_REGISTRY["PredatorPrey"] = PredatorPrey
    COOP_ENV_REGISTRY["PredatorPrey"] = PredatorPrey

    # choose environment + scenario
    env = make_local_env(
        environment_name="PredatorPrey",
        map_name="all_scenario",
        **kwargs,
    )
    return env, PREDATOR_PREY


def make_smac_env(**kwargs):
    env = marl.make_env(
        environment_name="smac",
        map_name=kwargs.get("map_name", "3m"),
        difficulty=kwargs.get("difficulty", "7"),
        reward_scale_rate=kwargs.get("reward_scale_rate", 20),
    )
    return env, SMAC


def make_mpe_simple_env(**kwargs):
    # register new env
    ENV_REGISTRY["mpe"] = MPESimple
    COOP_ENV_REGISTRY["mpe"] = MPESimple

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
