from functools import partial

import numpy as np
import torch
from ray.rllib import SampleBatch
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from torch.nn.functional import mse_loss
from torch.utils.data import TensorDataset, DataLoader

from niql.models import VAEEncoder
from niql.utils import tb_add_scalar, normalize_zero_mean_unit_variance, tb_add_scalars, apply_scaling, shift_and_scale, \
    standardize


def get_tdw_weights(policy, targets, samples: SampleBatch):
    assert hasattr(policy, "vae_model"), "Policy has no VAE model"

    # construct data from given properties
    data = construct_tdw_dataset(policy, samples)

    vae = policy.vae_model
    vae_target = policy.vae_model_target

    # set training and eval properties
    eps = 1e-7
    n_epochs = 2
    ns = min(policy.config.get("kde_subset_size") or 100, data.shape[0] // 3)

    # train
    fit_vae(policy, data, num_epochs=n_epochs)

    with torch.no_grad():
        # # convert targets to weights for KDE
        targets_flat = targets.detach().view(-1, 1)
        # targets_flat_norm = standardize(targets_flat)
        # kde_weights = torch.sigmoid(targets_flat_norm)

        # encode data
        # samples, mu, log_var = vae.encode(data)
        # samples_target, mu_target, log_var_target = vae_target.encode(data)

        # densities
        # p_densities = kde_density(samples, mu_target, log_var_target, ns, weights=kde_weights)
        # q_densities = kde_density(samples, mu, log_var, ns)

        # q densities
        outputs, mu, logvar = vae.encode(data)
        q_densities = kde_density(outputs, mu, logvar, ns)
        q_densities = apply_scaling(q_densities)

        # p densities
        targets_scaled = shift_and_scale(targets_flat)
        target_outputs, target_mu, target_logvar = vae_target.encode(data)
        p_densities = targets_scaled / (kde_density(target_outputs, target_mu, target_logvar, ns) + eps)
        p_densities /= (p_densities.max() + eps)

        # compute weights
        weights = p_densities / (q_densities + eps)
        weights = apply_scaling(weights)

        tb_add_scalars(policy, "tdw_stats", {
            # "scaling": scaling,
            "max_weight": weights.max(),
            "min_weight": weights.min(),
            "mean_weight": weights.mean(),
        })
    return weights.view(*targets.shape)


def vae_loss_function(recon_x, x, mu, logvar):
    mse = mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld, mse, kld


def compute_vae_loss(policy, data):
    vae = policy.model.encoder
    if isinstance(vae, VAEEncoder):
        outputs, mu, logvar = vae.encode_decode(data)
        loss = policy.vae_loss_function(outputs, data, mu, logvar)
    else:
        loss = 0.
    return loss


def fit_vae(policy, training_data, num_epochs=2):
    dataset = TensorDataset(training_data)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    policy.vae_model.train()
    training_loss = []
    mse_losses = []
    kld_losses = []

    for epoch in range(num_epochs):
        ep_loss = 0
        ep_mse_loss = 0
        ep_kld_loss = 0

        for batch in data_loader:
            batch = batch[0]
            policy.vae_optimiser.zero_grad()
            recon_batch, mu, log_var = policy.vae_model(batch)
            loss, mse_, kld_ = vae_loss_function(recon_batch, batch, mu, log_var)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.vae_model.parameters(), policy.config["grad_clip"])
            ep_loss += loss.item()
            ep_mse_loss += mse_.item()
            ep_kld_loss += kld_.item()
            policy.vae_optimiser.step()

        training_loss.append(ep_loss / len(dataset))
        mse_losses.append(ep_mse_loss / len(dataset))
        kld_losses.append(ep_kld_loss / len(dataset))
    policy.vae_model.eval()

    tb_add_scalar(policy, "vae_loss", np.mean(training_loss))
    tb_add_scalar(policy, "vae_mse_loss", np.mean(mse_losses))
    tb_add_scalar(policy, "vae_kld_loss", np.mean(kld_losses))


# def construct_tdw_dataset(policy, samples):
#     # construct training data
#     samples.set_get_interceptor(
#         partial(convert_to_torch_tensor, device=policy.device)
#     )
#     sample_size = samples.count
#     obs = samples[SampleBatch.OBS].reshape(sample_size, -1)
#     actions = samples[SampleBatch.ACTIONS].reshape(sample_size, )
#     actions = torch.eye(policy.n_actions).to(policy.device).float()[actions]
#     # next_obs = samples[SampleBatch.NEXT_OBS].reshape(sample_size, -1)
#     # rewards = samples[SampleBatch.REWARDS].reshape(sample_size, -1)
#     data = torch.cat([obs, actions], dim=-1)
#
#     data = normalize_zero_mean_unit_variance(data)
#     return data

def construct_tdw_dataset(self, samples: SampleBatch):
    # construct training data
    samples.set_get_interceptor(
        partial(convert_to_torch_tensor, device=self.device)
    )
    sample_size = samples.count
    obs = samples[SampleBatch.OBS].reshape(sample_size, -1)
    next_obs = samples[SampleBatch.NEXT_OBS].reshape(sample_size, -1)
    actions = samples[SampleBatch.ACTIONS].reshape(sample_size, )
    rewards = samples[SampleBatch.REWARDS].reshape(sample_size, -1)
    # if self.add_action_to_obs:
    actions = torch.eye(self.n_actions, self.n_actions).to(self.device).float()[actions]
        # data = torch.cat([obs, actions, rewards], dim=-1)
    # else:
    #     data = torch.cat([obs, rewards], dim=-1)
    data = torch.cat([obs, actions], dim=-1)
    data = normalize_zero_mean_unit_variance(data)
    return data


def adaptive_gamma(policy, alpha=0.01, beta=10000):
    """
    Compute the adaptive gamma for importance weighting in DQN.

    Parameters:
    n_tr (int): Current number of training iterations.
    alpha (float): Controls the steepness of the sigmoid curve.
    beta (float): Shifts the midpoint of the sigmoid curve.

    Returns:
    float: Computed gamma value.
    """
    return 1 / (1 + np.exp(-alpha * (policy.global_timestep - beta)))


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

    # Sum densities excluding policy
    mask = 1 - torch.eye(N, device=Z.device)
    densities_sum = (pairwise_densities_prod * mask).sum(dim=1)

    # Normalize by N-1
    densities = densities_sum / (N - 1)
    densities = densities.view(-1, 1)

    return densities


def nystroem_gaussian_density(z, mu, log_var, num_samples, weights=None):
    """
    Compute the Gaussian density of z given a Gaussian defined by mu and logvar.

    Parameters:
    z (tensor): Input tensor of shape (N, D).
    mu (tensor): Mean tensor of shape (N, D).
    log_var (tensor): Log variance tensor of shape (N, D).
    num_samples (int): Number of samples for the Nystroem approximation.

    Returns:
    tensor: Gaussian density of shape (N, 1).
    """
    N, D = z.shape
    std = torch.exp(0.5 * log_var)
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
    K_approx_mask = K_approx * mask
    if weights is not None:
        K_approx_mask *= weights.view(1, -1).expand_as(K_approx_mask)
    densities_sum = K_approx_mask.sum(dim=1) / (N - 1)
    density = torch.clamp(densities_sum.view(-1, 1), min=1e-7)

    return density


def kde_density(Z, mus, log_vars, num_samples, weights=None, approx=True):
    """
    Compute the density of each sample z_i in Z by merging all individual Gaussian distributions.

    Parameters:
    Z (tensor): NxD tensor of samples.
    mus (tensor): NxD tensor of means.
    logvars (tensor): NxD tensor of log variances.
    approx: whether to use an approximation method for KDE
    num_samples: only applicable when approx is set to True
    weights: weights of datapoints. Must be same dimensions as Z

    Returns:
    tensor: Nx1 tensor of densities for each sample.
    """
    if approx:
        densities = nystroem_gaussian_density(Z, mus, log_vars, num_samples, weights=weights)
    else:
        densities = compute_gaussian_densities(Z, log_vars, mus)

    return densities
