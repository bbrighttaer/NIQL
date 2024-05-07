import numpy as np
import torch
import torch.nn.functional as F
from ray.rllib import SampleBatch


def batch_cosine_similarity_reward_update(obs, actions, rewards, threshold=0.999):
    """
    Computes the cosine similarity of the observation and uses that to optimistically reconcile the rewards.

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
    obs_sim_mat = F.cosine_similarity(obs[None, :, :], obs[:, None, :], dim=-1)

    # construct new rewards array based on similarity matrix
    r_mat = torch.tile(rewards, (obs.shape[0], 1))
    r_mat[obs_sim_mat < threshold] = -np.inf
    rewards = torch.max(r_mat, dim=1, keepdim=True)[0]
    return rewards
