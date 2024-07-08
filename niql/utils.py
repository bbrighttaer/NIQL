import time
from typing import Callable

import pandas as pd
import torch
import torch.nn.functional as F
import tree
from ray.rllib.agents.qmix.qmix_policy import _drop_agent_dim
from ray.rllib.execution.replay_buffer import *
from ray.rllib.models.modelv2 import _unpack_obs
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.policy.rnn_sequencing import chop_into_sequences
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import KernelDensity

from niql.torch_kde import TorchKernelDensity


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

    input_list = [
        samples[SampleBatch.REWARDS], samples["weights"], action_mask, next_action_mask,
        samples[SampleBatch.ACTIONS], samples[SampleBatch.PREV_ACTIONS], samples[SampleBatch.DONES],
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
        (rew, wts, action_mask, next_action_mask, act, prev_act, dones, obs, next_obs,
         env_global_state, next_env_global_state, n_obs, n_next_obs) = output_list
    elif policy.has_env_global_state:
        (rew, wts, action_mask, next_action_mask, act, prev_act, dones, obs, next_obs,
         env_global_state, next_env_global_state) = output_list
    elif has_neighbour_data:
        (rew, wts, action_mask, next_action_mask, act, prev_act, dones, obs, next_obs,
         n_obs, n_next_obs) = output_list
    else:
        (rew, wts, action_mask, next_action_mask, act, prev_act, dones, obs,
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
    prev_actions = to_batches(prev_act.reshape(-1, 1), torch.long)
    action_mask = to_batches(action_mask, torch.float)
    next_action_mask = to_batches(next_action_mask, torch.float)
    terminated = to_batches(dones.reshape(-1, 1), torch.float)
    weights = to_batches(wts.reshape(-1, 1), torch.float)

    if policy.has_env_global_state:
        env_global_state = to_batches(env_global_state, torch.float)
        next_env_global_state = to_batches(next_env_global_state, torch.float)

    if has_neighbour_data:
        n_obs = to_batches(n_obs, torch.float, squeeze=False)  # squeeze=False means maintain agent dim
        n_next_obs = to_batches(n_next_obs, torch.float, squeeze=False)

    # Create mask for where index is < unpadded sequence length
    filled = np.reshape(np.tile(np.arange(T, dtype=np.float32), B), [B, T]) < np.expand_dims(seq_lens, 1)
    mask = torch.as_tensor(filled, dtype=torch.float, device=policy.device)

    return (action_mask, actions, prev_actions, env_global_state, mask, next_action_mask,
            next_env_global_state, next_obs, obs, rewards, weights, terminated, n_obs, n_next_obs, seq_lens)


NEIGHBOUR_NEXT_OBS = "n_next_obs"
NEIGHBOUR_OBS = "n_obs"
LDS_WEIGHTS = "lds_weights"


def standardize(r):
    return (r - r.mean()) / (r.std() + 1e-5)


def tb_add_scalar(policy, label, value):
    if hasattr(policy, "summary_writer") and hasattr(policy, "policy_id"):
        policy.summary_writer.add_scalar(policy.policy_id + "/" + label, value, policy.global_timestep)


def tb_add_histogram(policy, label, data):
    if hasattr(policy, "summary_writer") and hasattr(policy, "policy_id"):
        policy.summary_writer.add_histogram(policy.policy_id + "/" + label, data.reshape(-1, ), policy.global_timestep)


def tb_add_scalars(policy, label, values_dict):
    if hasattr(policy, "summary_writer") and hasattr(policy, "policy_id"):
        policy.summary_writer.add_scalars(
            policy.policy_id + "/" + label, {str(k): v for k, v in values_dict.items()}, policy.global_timestep
        )


def estimate_density(data, bandwidth=1):
    data = standardize(data)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data)
    return np.exp(kde.score_samples(data)).reshape(-1, 1)


def target_distribution_weighting(policy, q_targets, p_targets):
    q_targets_flat = q_targets.view(-1, 1)
    p_targets_flat = p_targets.view(-1, 1)

    # kde for q(x)
    q_density = estimate_density(to_numpy(q_targets_flat))
    q_x = convert_to_torch_tensor(q_density, q_targets.device)

    # kde for approximating p(x)
    p_density = estimate_density(to_numpy(p_targets_flat))
    p_density = convert_to_torch_tensor(p_density, p_targets.device)
    p_density = 1. / (p_density + 1e-7)
    p_density = apply_scaling(p_density)
    # p_weights = torch.sigmoid(standardize(p_targets_flat))
    p_weights = shift_and_scale(p_density)
    p_x = p_weights * p_density

    # compute weights
    weights = p_x / (q_x + 1e-7)
    weights = apply_scaling(weights)

    tb_add_scalars(policy, "tdw_stats", {
        # "scaling": scaling,
        "max_weight": weights.max(),
        "min_weight": weights.min(),
        "mean_weight": weights.mean(),
    })

    return weights


def apply_scaling(vector):
    # Compute scaling factor
    scaling = len(vector) / vector.sum()

    # Scale the vector
    vector *= scaling

    # Normalise to [0, 1]
    vector /= (vector.max() + 1e-7)
    return vector


def shift_and_scale(x):
    # Find the minimum value
    x_min = x.min()

    # Shift the vector to make all elements non-negative
    x_shifted = (x - x_min) + 1e-4

    # Normalise to [0, 1]
    x_scaled = x_shifted / (x_shifted.max() + 1e-7)

    return x_scaled


# def target_distribution_weighting(policy, targets, sim_threshold):
#     targets_flat = targets.reshape(-1, 1)
#     if random.random() < policy.tdw_schedule.value(policy.global_timestep):
#         lds_weights = get_target_dist_weights_torch(
#             targets_flat, sim_threshold
#         )
#         lds_weights = convert_to_torch_tensor(lds_weights, policy.device).reshape(*targets.shape)
#         # min_w = max(1e-2, lds_weights.min())
#         # lds_weights = torch.clamp(torch.log(lds_weights), min_w, max=2 * min_w)
#
#         tb_add_scalars(policy, "tdw_stats", {
#             # "scaling": scaling,
#             "max_weight": lds_weights.max(),
#             "min_weight": lds_weights.min(),
#             "mean_weight": lds_weights.mean(),
#         })
#     else:
#         lds_weights = torch.ones_like(targets)
#     return lds_weights


def normalize_zero_mean_unit_variance(X):
    """
    Normalize the input tensor to have zero mean and unit variance.

    Parameters:
    X (torch.Tensor): The input tensor of shape (M, D).

    Returns:
    torch.Tensor: The normalized tensor with zero mean and unit variance.
    """
    # Calculate the mean and standard deviation along the feature dimension (D)
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, unbiased=False, keepdim=True)

    # Normalize the input tensor
    X_normalized = (X - mean) / (std + 1e-6)

    return X_normalized


def get_target_dist_weights_torch(targets, kernel, bandwidth=1) -> np.array:
    targets = standardize(targets)
    kde = TorchKernelDensity(kernel=TorchKernelDensity.gaussian, bandwidth=5.)
    kde.fit(targets)
    densities = kde.score_samples(targets)
    weights = 1. / (densities + 1e-7)
    scaling = len(weights) / weights.sum()
    weights *= scaling
    weights /= (weights.max() + 1e-7)
    return weights


def get_target_dist_weights_sk(targets, kernel, bandwidth=1.) -> np.array:
    targets = standardize(targets)
    kde = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(to_numpy(targets))
    log_density = kde.score_samples(to_numpy(targets))
    densities = np.exp(log_density)
    weights = 1. / (densities + 1e-7)
    weights /= (weights.max() + 1e-7)
    return weights


def get_target_dist_weights_l2(data, sim_threshold=0.01) -> np.array:
    data = standardize(data)
    distances = pairwise_l2_distance(data)
    distances = normalize_zero_mean_unit_variance(distances)
    context = distances < sim_threshold
    densities = context.float().mean(dim=-1)
    weights = 1. / (densities + 1e-7)
    weights /= (weights.max() + 1e-7)
    return weights


def get_target_dist_weights_cl(targets, *args, **kwargs) -> np.array:
    data = to_numpy(targets)

    # clustering
    bin_index_per_label = cluster_labels(data)
    Nb = max(bin_index_per_label) + 1
    num_samples_of_bins = dict(collections.Counter(bin_index_per_label))
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]

    # Use re-weighting based on empirical cluster distribution, sample-wise weights: [Ns,]
    eff_num_per_label = [emp_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    weights = [1. / (x + 1e-6) for x in eff_num_per_label]
    tdw_weights = np.array(weights).reshape(len(data), -1)
    scaling = len(tdw_weights) / (tdw_weights.sum() + 1e-7)
    tdw_weights *= scaling
    tdw_weights /= (tdw_weights.max() + 1e-7)
    return tdw_weights


def cluster_labels(data, min_samples_in_cluster=2, eps=10., n_clusters=100):
    data = standardize(data)
    # n_clusters = min(n_clusters, len(np.unique(labels)))
    # clustering = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit(labels.reshape(-1, 1))
    clustering = DBSCAN(min_samples=min_samples_in_cluster, eps=eps).fit(data)
    bin_index_per_label = clustering.labels_
    # create bins
    # num_bins = 1000
    # bounds = (math.floor(np.min(targets)), math.ceil(np.max(targets)))
    # hist, bins = np.histogram(a=np.array([], dtype=np.float32), bins=num_bins, range=bounds)
    # bin_index_per_label = np.digitize(targets, bins, right=True)
    # bin_index_per_label = np.array(bin_index_per_label)
    return bin_index_per_label


def pairwise_l2_distance(tensor):
    # Calculate pairwise L2 distances
    diff = tensor.unsqueeze(1) - tensor.unsqueeze(0)  # Broadcasting to compute differences
    l2_distances = torch.sqrt(torch.sum(diff ** 2, dim=2))  # L2 distance calculation
    return l2_distances


# def get_lds_kernel_window(kernel, ks, sigma):
#     assert kernel in ['gaussian', 'triang', 'laplace']
#     half_ks = (ks - 1) // 2
#     if kernel == 'gaussian':
#         base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
#         gaussian_kernel = gaussian_filter1d(base_kernel, sigma=sigma)
#         kernel_window = gaussian_kernel / max(gaussian_kernel)
#     elif kernel == 'triang':
#         kernel_window = np.bartlett(ks)  # equivalent to triang in scipy
#     else:
#         laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
#         kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
#             map(laplace, np.arange(-half_ks, half_ks + 1)))
#
#     return kernel_window


def mac(model, obs, h, **kwargs):
    """Forward pass of the multi-agent controller.

    Args:
        model: TorchModelV2 class
        obs: Tensor of shape [B, n_agents, obs_size]
        h: List of tensors of shape [B, n_agents, h_size]

    Returns:
        q_vals: Tensor of shape [B, n_agents, n_actions]
        h: Tensor of shape [B, n_agents, h_size]
    """
    B, n_agents = obs.size(0), obs.size(1)
    if not isinstance(obs, dict):
        obs = {"obs": obs}
    obs_agents_as_batches = {k: _drop_agent_dim(v) for k, v in obs.items()}
    h_flat = [s.reshape([B * n_agents, -1]) for s in h]
    q_flat, h_flat = model(obs_agents_as_batches, h_flat, None, **kwargs)
    return q_flat.reshape(
        [B, n_agents, -1]), [s.reshape([B, n_agents, -1]) for s in h_flat]


def unroll_mac(model, obs_tensor, comm_net=None, shared_messages=None, aggregation_func=None, **kwargs):
    """Computes the estimated Q values for an entire trajectory batch"""
    B = obs_tensor.size(0)
    T = obs_tensor.size(1)
    n_agents = obs_tensor.size(2)

    mac_out = []
    mac_h_out = []
    h = [s.expand([B, n_agents, -1]) for s in model.get_initial_state()]
    is_recurrent = len(h) > 0
    prev_actions = kwargs.get("prev_actions")

    # forward propagation through time
    for t in range(T):
        # get input data for this time step
        obs = obs_tensor[:, t]

        # if comm is enabled, process comm messages
        if comm_net is not None:
            # get local message sent to other agents
            local_msg = comm_net(obs)

            # put together local and received messages
            msgs = torch.cat([local_msg, shared_messages[:, t]], dim=1)

            # get query for message aggregation
            query = h[0] if is_recurrent else obs

            # aggregate received messages
            msg = aggregation_func(query, msgs)

            # update input data with messages
            obs = torch.cat([obs, msg], dim=-1)

        if prev_actions is not None:
            actions = prev_actions[:, t:t + 1]
            actions_enc = torch.eye(kwargs["n_actions"]).float().to(obs.device)[actions]
            obs = torch.cat([obs, actions_enc], dim=-1)

        q, h = mac(model, obs, h, **kwargs)
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


def get_priority_update_func(local_replay_buffer: LocalReplayBuffer, config: Dict[str, Any]):
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


def pairwise_cosine_similarity(x, batch_size=1000):
    """
    Computes pairwise cosine similarity in a scalable manner using batch processing.

    Args:
        x (torch.Tensor): Input tensor of shape (N, D), where N is the number of samples and D is the feature dimension.
        batch_size (int): The size of each batch for processing.

    Returns:
        torch.Tensor: Pairwise cosine similarity matrix of shape (N, N).
    """
    N, D = x.shape
    similarity_matrix = torch.zeros(N, N, device=x.device)

    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        x_i = x[i:end_i]

        for j in range(0, N, batch_size):
            end_j = min(j + batch_size, N)
            x_j = x[j:end_j]

            # Compute cosine similarity between batches
            norm_i = F.normalize(x_i, p=2, dim=1)
            norm_j = F.normalize(x_j, p=2, dim=1)
            similarity_matrix[i:end_i, j:end_j] = torch.mm(norm_i, norm_j.t())

    return similarity_matrix


def add_comm_msg(model, obs, next_obs, neighbour_obs, neighbour_next_obs, msg_aggregator: Callable):
    """
    Adds batch neighbour messages to observation batch after passing local observation through comm net.
    """
    B, T = obs.shape[:2]
    local_msg = model(obs).unsqueeze(2)
    msgs = torch.cat([local_msg, neighbour_obs], dim=2)
    agg_msg = msg_aggregator(obs, msgs.view(B * T, *msgs.shape[2:])).view(B, T, -1)
    obs = torch.cat([obs, agg_msg], dim=-1)

    next_local_msg = model(next_obs).unsqueeze(2)
    next_msgs = torch.cat([next_local_msg, neighbour_next_obs], dim=2)
    agg_next_msg = msg_aggregator(next_obs, next_msgs.view(B * T, *msgs.shape[2:]), is_target=True).view(B, T, -1)
    next_obs = torch.cat([next_obs, agg_next_msg], dim=-1)

    return obs, next_obs


def batch_message_inter_agent_sharing(sample_batch, other_agent_batches):
    if len(other_agent_batches) > 0:
        sample_batch = sample_batch.copy()
        n_obs = []
        n_next_obs = []
        for n_id, (policy, batch) in other_agent_batches.items():
            n_obs.append(
                policy.get_message(batch[SampleBatch.OBS])
            )
            n_next_obs.append(
                policy.get_message(batch[SampleBatch.NEXT_OBS])
            )
        sample_batch[NEIGHBOUR_OBS] = np.array(n_obs).reshape(
            len(sample_batch), len(other_agent_batches), -1)
        sample_batch[NEIGHBOUR_NEXT_OBS] = np.array(n_next_obs).reshape(
            len(sample_batch), len(other_agent_batches), -1)
    return sample_batch


def add_evaluation_config(config: dict) -> dict:
    config = dict(config)
    config.update({
        "evaluation_interval": 1,
        "evaluation_num_episodes": 20,
        "evaluation_num_workers": 1,
        # "evaluation_unit": "timesteps", # not supported in ray 1.8.0
    })
    return config


def gaussian_density(z, mu, logvar):
    """
    Compute the Gaussian density of z given a Gaussian defined by mu and logvar.

    Parameters:
    z (tensor): Input tensor of shape (N, D).
    mu (tensor): Mean tensor of shape (N, D).
    logvar (tensor): Log variance tensor of shape (N, D).

    Returns:
    tensor: Gaussian density of shape (N, D).
    """
    std = torch.exp(0.5 * logvar)
    var = std ** 2
    normalization = torch.sqrt(2 * np.pi * var)

    # Compute exponent
    x = -0.5 * ((z - mu) ** 2 / var)

    # Compute density
    exponent = torch.exp(x)
    density = exponent / normalization

    return density


def compute_gaussian_densities(Z, logvars, mus):
    N, D = Z.shape

    # Expand dimensions for broadcasting
    Z_expanded = Z.detach().unsqueeze(1)
    mus_expanded = mus.detach().unsqueeze(0)
    logvars_expanded = logvars.detach().unsqueeze(0)

    # Compute pairwise Gaussian densities
    pairwise_densities = gaussian_density(Z_expanded, mus_expanded, logvars_expanded)

    # Compute product of densities across dimensions
    pairwise_densities_prod = pairwise_densities.prod(dim=2)

    # Sum densities excluding self
    mask = 1 - torch.eye(N, device=Z.device)
    densities_sum = (pairwise_densities_prod * mask).sum(dim=1)

    # Normalize by N-1
    densities = densities_sum / (N - 1)
    densities = densities.view(-1, 1)

    return densities


def nystroem_gaussian_density(z, mu, logvar, num_samples):
    """
    Compute the Gaussian density of z given a Gaussian defined by mu and logvar.

    Parameters:
    z (tensor): Input tensor of shape (N, D).
    mu (tensor): Mean tensor of shape (N, D).
    logvar (tensor): Log variance tensor of shape (N, D).
    num_samples (int): Number of samples for the Nystroem approximation.

    Returns:
    tensor: Gaussian density of shape (N, 1).
    """
    N, D = z.shape
    std = torch.exp(0.5 * logvar)
    var = std ** 2

    # Sample selection
    indices = torch.randperm(N)[:num_samples]
    z_sampled = z[indices]
    mu_sampled = mu[indices]
    var_sampled = var[indices]

    # Compute normalization factors
    normalization = torch.sqrt(2 * np.pi * var_sampled)

    # Compute kernel sub-matrix K_m
    diff = z_sampled.unsqueeze(1) - mu_sampled.unsqueeze(0)
    K_m = torch.exp(-0.5 * (diff ** 2 / var_sampled.unsqueeze(0))) / normalization.unsqueeze(0)
    K_m = K_m.view(num_samples, num_samples, D).prod(dim=2)

    # Compute cross-kernel sub-matrix K_Nm
    diff = z.unsqueeze(1) - mu_sampled.unsqueeze(0)
    K_Nm = torch.exp(-0.5 * (diff ** 2 / var_sampled.unsqueeze(0))) / normalization.unsqueeze(0)
    K_Nm = K_Nm.view(N, num_samples, D).prod(dim=2)

    # Compute the approximate kernel matrix
    K_m_inv = torch.linalg.pinv(K_m)
    K_approx = K_Nm @ K_m_inv @ K_Nm.T

    # Compute density
    mask = 1 - torch.eye(N, device=z.device)
    densities_sum = (K_approx * mask).sum(dim=1) / (N - 1)
    density = torch.clamp(densities_sum.view(-1, 1), min=1e-7)

    return density


def kde_density(Z, mus, logvars, approx=True, num_samples=100):
    """
    Compute the density of each sample z_i in Z by merging all individual Gaussian distributions.

    Parameters:
    Z (tensor): NxD tensor of samples.
    mus (tensor): NxD tensor of means.
    logvars (tensor): NxD tensor of log variances.
    approx: whether to use an approximation method for KDE
    num_samples: only applicable when approx is set to True

    Returns:
    tensor: Nx1 tensor of densities for each sample.
    """
    if approx:
        densities = nystroem_gaussian_density(Z, mus, logvars, num_samples)
    else:
        densities = compute_gaussian_densities(Z, logvars, mus)

    return densities
