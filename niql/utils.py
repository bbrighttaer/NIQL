import numpy as np
import pandas as pd
import tree
from ray.rllib.agents.qmix.qmix_policy import _mac
from ray.rllib.execution.replay_buffer import *
from ray.rllib.models.modelv2 import _unpack_obs
from ray.rllib.models.preprocessors import get_preprocessor
import torch
from ray.rllib.policy.rnn_sequencing import chop_into_sequences
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from scipy.ndimage import gaussian_filter1d, convolve1d
from scipy.stats import triang


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


def preprocess_trajectory_batch(policy, samples: SampleBatch, has_neighbour_data=False, **kwargs):
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
            next_env_global_state, next_obs, obs, rewards, terminated, n_obs, n_next_obs, seq_lens)


NEIGHBOUR_NEXT_OBS = "n_next_obs"
NEIGHBOUR_OBS = "n_obs"
LDS_WEIGHTS = "lds_weights"


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
        # kernel = gaussian(ks)
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
            map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


def standardize(r):
    return (r - r.mean()) / (r.std() + 1e-5)


def get_lds_weights(
        samples: SampleBatch,
        lds_kernel="gaussian",
        lds_ks=50,
        lds_sigma=2,
        num_bins=50,
) -> np.array:
    # consider all data in buffer
    rewards = samples[SampleBatch.REWARDS]

    # create bins
    hist, bins = np.histogram(a=np.array([], dtype=np.float32), bins=num_bins, range=(-10, 10))
    bin_index_per_label = np.digitize(rewards, bins, right=True)
    Nb = max(bin_index_per_label) + 1
    num_samples_of_bins = dict(collections.Counter(bin_index_per_label))
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]

    # compute effective label distribution
    lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
    eff_label_dist = convolve1d(emp_label_dist, weights=lds_kernel_window, mode='constant')

    # Use re-weighting based on effective label distribution, sample-wise weights: [Ns,]
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    weights = [np.float32(1 / (x + 1e-6)) for x in eff_num_per_label]
    scaling = len(weights) / np.sum(weights)
    weights = [scaling * x for x in weights]
    lds_weights = np.array(weights).reshape(len(samples), -1)
    lds_weights = standardize(lds_weights)
    return lds_weights, bin_index_per_label


def unroll_mac(model, obs_tensor):
    """Computes the estimated Q values for an entire trajectory batch"""
    B = obs_tensor.size(0)
    T = obs_tensor.size(1)
    n_agents = obs_tensor.size(2)

    mac_out = []
    mac_h_out = []
    h = [s.expand([B, n_agents, -1]) for s in model.get_initial_state()]
    for t in range(T):
        q, h = _mac(model, obs_tensor[:, t], h)
        mac_out.append(q)
        mac_h_out.extend(h)
    mac_out = torch.stack(mac_out, dim=1)  # Concat over time
    mac_h_out = torch.stack(mac_h_out, dim=1)

    return mac_out, mac_h_out


def unroll_mac_squeeze_wrapper(model_outputs):
    pred, hs = model_outputs
    return pred.squeeze(2), hs.squeeze(2)


def soft_update(target_net, source_net, tau):
    """
    Soft update the parameters of the target network with those of the source network.

    Args:
    - target_net: Target network.
    - source_net: Source network.
    - tau: Soft update parameter (0 < tau <= 1).

    Returns:
    - target_net: Updated target network.
    """
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    return target_net


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def save_representations(obs, latent_rep, model_out, target, reward, filename_prefix=""):
    """
    Saves the observation and model predictions for analysis.

    :param obs: Tensor of shape [B,T,obs_dim]
    :param latent_rep: Tensor of shape [B,T,latent_dim]
    :param model_out:  Tensor of shape [B,T]
    :param target:  Tensor of shape [B,T]
    :param reward:  Tensor of shape [B,T]
    """
    # prep data
    obs = to_numpy(obs).reshape(-1, obs.shape[-1])
    latent_rep = to_numpy(latent_rep).reshape(-1, latent_rep.shape[-1])
    model_out = to_numpy(model_out).reshape(-1, 1)
    target = to_numpy(target).reshape(-1, 1)
    reward = to_numpy(reward).reshape(-1, 1)
    data = np.concatenate([obs, latent_rep, model_out, target, reward], axis=-1)

    # prepare column labels
    obs_col_labels = [f"obs_{i + 1}" for i in range(obs.shape[1])]
    latent_rep_col_labels = [f"latent_{i + 1}" for i in range(latent_rep.shape[1])]
    model_out_col = "prediction"
    target_col = "target"
    reward = "reward"
    cols = obs_col_labels + latent_rep_col_labels
    cols.append(model_out_col)
    cols.append(target_col)
    cols.append(reward)

    # save data to csv
    df = pd.DataFrame(data=data, columns=cols)
    df.to_csv(f"{filename_prefix}representation_analysis_data.csv", index=False)


def get_priority_update_func(local_replay_buffer, config):
    """
    Returns the function for updating priorities in a prioritised experience replay buffer.
    Adapted from rllib.
    """

    def update_prio(item):
        samples, info_dict = item
        if config.get("prioritized_replay"):
            prio_dict = {}
            for policy_id, info in info_dict.items():
                td_error = info.get("td_error", info[LEARNER_STATS_KEY].get("td_error"))
                samples.policy_batches[policy_id].set_get_interceptor(None)
                batch_indices = samples.policy_batches[policy_id].get("batch_indexes")

                # In case the buffer stores sequences, TD-error could already
                # be calculated per sequence chunk.
                if len(batch_indices) != len(td_error):
                    T = local_replay_buffer.replay_sequence_length
                    assert len(batch_indices) > len(
                        td_error) and len(batch_indices) % T == 0
                    full_seq = td_error.shape[-1]
                    batch_indices = batch_indices.reshape(-1, )
                    td_error = td_error.reshape(-1, )

                    if len(batch_indices) != len(td_error):
                        # fallback on sequence lengths
                        seq_lens = info.get("seq_lens")
                        td_error = td_error.reshape(-1, full_seq)
                        if seq_lens is not None:
                            td_error_list = []
                            for traj, s_len in zip(td_error, seq_lens):
                                td_error_list.extend(traj[:s_len].tolist())
                            td_error = np.array(td_error_list)

                        assert len(batch_indices) == len(td_error)

                prio_dict[policy_id] = (batch_indices, td_error)
            local_replay_buffer.update_priorities(prio_dict)
        return info_dict

    return update_prio
