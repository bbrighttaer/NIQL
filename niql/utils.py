import tree
from ray.rllib.execution.replay_buffer import *
from ray.rllib.models.modelv2 import _unpack_obs
from ray.rllib.models.preprocessors import get_preprocessor


# -----------------------------------------------------------------------------------------------
# Adapted from Adapted from https://github.com/Morphlng/MARLlib/blob/main/examples/eval.py
class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


# ----------------------------------------------------------------------------------------------


def notify_wrap(f, cb):
    def wrapped(*args, **kwargs):
        f(*args, **kwargs)
        cb(*args, **kwargs)

    return wrapped


def unpack_observation(policy, obs_batch):
    """Unpacks the observation, action mask, and state (if present)
    from agent grouping.

    Returns:
        obs (np.ndarray): obs tensor of shape [B, n_agents, obs_size]
        mask (np.ndarray): action mask, if any
        state (np.ndarray or None): state tensor of shape [B, state_size]
            or None if it is not in the batch
    """
    unpacked = _unpack_obs(
        np.array(obs_batch, dtype=np.float32),
        policy.observation_space.original_space,
        tensorlib=np)

    if not isinstance(unpacked, tuple):
        unpacked = [unpacked]

    if isinstance(unpacked[0], dict):
        assert "obs" in unpacked[0]
        unpacked_obs = [
            np.concatenate(tree.flatten(u["obs"]), 1) for u in unpacked
        ]
    else:
        unpacked_obs = unpacked

    obs = np.concatenate(
        unpacked_obs,
        axis=1).reshape([len(obs_batch), policy.n_agents, policy.obs_size])

    if policy.has_action_mask:
        action_mask = np.concatenate(
            [o["action_mask"] for o in unpacked], axis=1).reshape(
            [len(obs_batch), policy.n_agents, policy.n_actions])
    else:
        action_mask = np.ones(
            [len(obs_batch), policy.n_agents, policy.n_actions],
            dtype=np.float32)

    if policy.has_env_global_state:
        state = np.concatenate(tree.flatten(unpacked[0]["state"]), 1)
    else:
        state = None
    return obs, action_mask, state


def get_size(obs_space):
    return get_preprocessor(obs_space)(obs_space).size


def get_group_rewards(n_agents, info_batch):
    group_rewards = np.array([
        info.get("_group_rewards", [0.0] * n_agents)
        for info in info_batch
    ])
    return group_rewards
