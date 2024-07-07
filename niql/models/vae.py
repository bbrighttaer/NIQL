from typing import List

import torch
import torch.nn as nn


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
        z, mu, logvar = self.encode(x)
        return self.decoder(z), mu, logvar

    def encode(self, x):
        params = self.encoder(x)
        mu, logvar = params[:, : self.latent_dim], params[:, self.latent_dim:]
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode_decode(self, x):
        return self.forward(x)
