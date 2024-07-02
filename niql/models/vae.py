from typing import List

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss


class VAE(nn.Module):
    """
    Implement a variational autoencoder for density estimation.
    """

    def __init__(self, input_dim: int, hidden_layer_dims: List[int], latent_dim: int):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        prev_dim = input_dim
        encoder_layers = []
        for hdim in hidden_layer_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.ReLU(),
            ])
            prev_dim = hdim
        bottle_neck = nn.Linear(prev_dim, latent_dim * 2)  # Mean and log-variance
        encoder_layers.append(bottle_neck)

        prev_dim = latent_dim
        decoder_layers = []
        for hdim in reversed(hidden_layer_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.ReLU(),
            ])
            prev_dim = hdim

        output_layer = nn.Linear(prev_dim, input_dim)
        decoder_layers.append(output_layer)

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        params = self.encoder(x)
        mu, logvar = params[:, : self.latent_dim], params[:, self.latent_dim:]
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def estimate_density(self, x):
        with torch.no_grad():
            # Encode the input to obtain the latent parameters
            params = self.encoder(x)
            mu, logvar = params[:, :self.latent_dim], params[:, self.latent_dim:]

            # Reparameterize to get z
            z = self.reparameterize(mu, logvar)

            # Decode z to get reconstructed x
            recon_x = self.decoder(z)

            # Compute the reconstruction loss (log-likelihood)
            recon_loss = mse_loss(recon_x, x, reduction='none')
            recon_loss = recon_loss.sum(dim=1)

            # Compute the KL divergence
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

            # Compute the ELBO
            elbo = recon_loss + KLD

            # Convert ELBO to density estimate
            density_estimate = torch.exp(-elbo)

            return density_estimate

