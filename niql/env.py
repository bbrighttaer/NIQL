import numpy as np
from marllib import marl


def make_mpe_env(**kwargs):
    # choose environment + scenario
    env = marl.make_env(
        environment_name="mpe",
        map_name="simple_spread",
        force_coop=True,
        max_cycles=25,
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
