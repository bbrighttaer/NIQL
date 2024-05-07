import numpy as np
import torch
import torch.nn.functional as F


def batch_cosine_similarity_reward_update_np(obs, actions, rewards, threshold=0.999):
    """
    Computes the cosine similarity of the observation and uses that to optimistically reconcile the rewards.
    Numpy version.

    :param obs: observation.
    :param actions: corresponding actions.
    :param rewards: received rewards.
    :param threshold: threshold for determining observations that are similar.
    :return: rewards.
    """
    eps = 1e-9
    max_num = 1e9

    # prep data
    obs = obs + eps
    actions = actions.reshape(-1, 1)
    rewards = rewards.reshape(1, -1)
    # reward shaping
    rewards += np.random.uniform(low=0., high=eps, size=rewards.shape)
    obs = np.concatenate([obs, actions], axis=1)

    # cosine similarity
    obs_norm = np.linalg.norm(obs, axis=1, keepdims=True)
    obs_div_norm = obs / np.maximum(obs_norm, eps * np.ones_like(obs_norm))
    obs_sim_mat = obs_div_norm @ obs_div_norm.T

    # construct new rewards array based on similarity matrix
    r_mat = np.tile(rewards, (obs.shape[0], 1))
    flags = obs_sim_mat >= threshold
    mask = ~flags * (~flags - max_num)
    masked_rewards = r_mat + mask
    opt_rewards = np.max(masked_rewards, axis=1)

    # update rewards
    rewards = np.array(opt_rewards)
    return rewards


def batch_cosine_similarity_reward_update_torch(obs, actions, rewards, threshold=0.999):
    """
    Computes the cosine similarity of the observation and uses that to optimistically reconcile the rewards.
    Pytorch version.

    :param obs: observation.
    :param actions: corresponding actions.
    :param rewards: received rewards.
    :param threshold: threshold for determining observations that are similar.
    :return: rewards.
    """
    eps = 1e-9
    device = obs.device

    # prep data
    obs = obs + eps
    actions = actions.reshape(-1, 1)
    rewards = rewards.reshape(1, -1)
    # reward shaping
    rewards += torch.zeros_like(rewards).uniform_(0., eps).to(device)
    obs = torch.cat([obs, actions], dim=1)

    # cosine similarity
    n_obs = F.normalize(obs)
    obs_sim_mat = n_obs @ n_obs.T

    # construct new rewards array based on similarity matrix
    r_mat = torch.tile(rewards, (obs.shape[0], 1))
    r_mat[obs_sim_mat < threshold] = -np.inf
    rewards = torch.max(r_mat, dim=1, keepdim=True)[0]
    return rewards
