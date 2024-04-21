import numpy as np
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY

from .two_step_matrix_game import TwoStepMultiAgentCoopMatrixGame
from .one_step_matrix_game import OneStepMultiAgentCoopMatrixGame
from .mpe_simple import MPESimple
from .utils import make_local_env


def make_mpe_simple_spread_env(**kwargs):
    # choose environment + scenario
    env = marl.make_env(
        environment_name="mpe",
        map_name="simple_spread",
        force_coop=True,
        max_cycles=25,
        **kwargs,
    )
    return env


def make_mpe_simple_env(**kwargs):
    # register new env
    ENV_REGISTRY["mpe"] = MPESimple
    COOP_ENV_REGISTRY["mpe"] = MPESimple

    # choose environment + scenario
    env = marl.make_env(
        environment_name="mpe",
        map_name="simple_spread",
        # force_coop=True,
        max_cycles=25,
        **kwargs,
    )
    return env


def make_two_step_matrix_game_env(**kwargs):
    # register new env
    ENV_REGISTRY["CoopMatrixGame"] = TwoStepMultiAgentCoopMatrixGame
    COOP_ENV_REGISTRY["CoopMatrixGame"] = TwoStepMultiAgentCoopMatrixGame

    # choose environment + scenario
    env = make_local_env(
        environment_name="CoopMatrixGame",
        map_name="all_scenario",
        **kwargs,
    )
    return env


def make_one_step_matrix_game_env(**kwargs):
    # register new env
    ENV_REGISTRY["CoopMatrixGame"] = OneStepMultiAgentCoopMatrixGame
    COOP_ENV_REGISTRY["CoopMatrixGame"] = OneStepMultiAgentCoopMatrixGame

    # choose environment + scenario
    env = make_local_env(
        environment_name="CoopMatrixGame",
        map_name="all_scenario",
        **kwargs,
    )
    return env


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
