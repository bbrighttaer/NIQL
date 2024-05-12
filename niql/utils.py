import tree
from ray.rllib.execution.replay_buffer import *
from ray.rllib.models.modelv2 import _unpack_obs
from ray.rllib.models.preprocessors import get_preprocessor
import torch
from ray.rllib.policy.rnn_sequencing import chop_into_sequences


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


def get_size(obs_space):
    return get_preprocessor(obs_space)(obs_space).size


def get_group_rewards(n_agents, info_batch):
    group_rewards = np.array([
        info.get("_group_rewards", [0.0] * n_agents)
        for info in info_batch
    ])
    return group_rewards


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


def preprocess_trajectory_batch(policy, samples: SampleBatch, has_neighbour_data=False):
    shared_obs_batch, shared_next_obs_batch = None, None
    # preprocess training data
    obs_batch, action_mask, env_global_state = unpack_observation(
        policy,
        samples[SampleBatch.CUR_OBS],
    )
    next_obs_batch, next_action_mask, next_env_global_state = unpack_observation(
        policy,
        samples[SampleBatch.NEXT_OBS],
    )

    # if has_neighbour_data:
    #     n_samples = samples[NEIGHBOUR_OBS]
    #     n_shape = n_samples.shape
    #     n_samples = n_samples.reshape(-1, n_shape[-1])
    #     shared_obs_batch, _, _ = unpack_observation(
    #         policy,
    #         n_samples,
    #     )
    #     shared_obs_batch = shared_obs_batch.reshape(*n_shape[:2], -1)
    #
    #     n_next_samples = samples[NEIGHBOUR_NEXT_OBS]
    #     n_next_samples = n_next_samples.reshape(-1, n_shape[-1])
    #     shared_next_obs_batch, _, _ = unpack_observation(
    #         policy,
    #         n_next_samples,
    #     )
    #     shared_next_obs_batch = shared_next_obs_batch.reshape(*n_shape[:2], -1)

    input_list = [
        samples[SampleBatch.REWARDS], action_mask, next_action_mask,
        samples[SampleBatch.ACTIONS], samples[SampleBatch.DONES],
        obs_batch, next_obs_batch
    ]

    if policy.has_env_global_state:
        input_list.extend([env_global_state, next_env_global_state])

    if has_neighbour_data:
        input_list.extend([samples[NEIGHBOUR_OBS], samples[NEIGHBOUR_NEXT_OBS]])

    n_obs = None
    n_next_obs = None

    output_list, _, seq_lens = chop_into_sequences(
        episode_ids=samples[SampleBatch.EPS_ID],
        unroll_ids=samples[SampleBatch.UNROLL_ID],
        agent_indices=samples[SampleBatch.AGENT_INDEX],
        feature_columns=input_list,
        state_columns=[],  # RNN states not used here
        max_seq_len=policy.config["model"]["max_seq_len"],
        dynamic_max=True,
    )
    # These will be padded to shape [B * T, ...]
    if policy.has_env_global_state and has_neighbour_data:
        (rew, action_mask, next_action_mask, act, dones, obs, next_obs,
         env_global_state, next_env_global_state, n_obs, n_next_obs) = output_list
    elif policy.has_env_global_state:
        (rew, action_mask, next_action_mask, act, dones, obs, next_obs,
         env_global_state, next_env_global_state) = output_list
    elif has_neighbour_data:
        (rew, action_mask, next_action_mask, act, dones, obs, next_obs,
         n_obs, n_next_obs) = output_list
    else:
        (rew, action_mask, next_action_mask, act, dones, obs,
         next_obs) = output_list
    B, T = len(seq_lens), max(seq_lens)

    def to_batches(arr, dtype, squeeze=True):
        new_shape = [B, T] + list(arr.shape[1:])
        tensor = torch.as_tensor(
            np.reshape(arr, new_shape),
            dtype=dtype,
            device=policy.device,
        )
        if squeeze:
            return tensor.squeeze(2)
        else:
            return tensor

    # reduce the scale of reward for small variance. This is also
    # because we copy the global reward to each agent in rllib_env
    rewards = to_batches(rew.reshape(-1, 1), torch.float)
    if policy.reward_standardize:
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

    obs = to_batches(obs, torch.float)
    next_obs = to_batches(next_obs, torch.float)
    actions = to_batches(act.reshape(-1, 1), torch.long)
    action_mask = to_batches(action_mask, torch.float)
    next_action_mask = to_batches(next_action_mask, torch.float)
    terminated = to_batches(dones.reshape(-1, 1), torch.float)

    if policy.has_env_global_state:
        env_global_state = to_batches(env_global_state, torch.float)
        next_env_global_state = to_batches(next_env_global_state, torch.float)

    if has_neighbour_data:
        n_obs = to_batches(n_obs, torch.float, squeeze=False)  # squeeze=False means maintain agent dim
        n_next_obs = to_batches(n_next_obs, torch.float, squeeze=False)

    # Create mask for where index is < unpadded sequence length
    filled = np.reshape(np.tile(np.arange(T, dtype=np.float32), B), [B, T]) < np.expand_dims(seq_lens, 1)
    mask = torch.as_tensor(filled, dtype=torch.float, device=policy.device)

    return (action_mask, actions, env_global_state, mask, next_action_mask,
            next_env_global_state, next_obs, obs, rewards, terminated, n_obs, n_next_obs)


NEIGHBOUR_NEXT_OBS = "n_next_obs"
NEIGHBOUR_OBS = "n_obs"
