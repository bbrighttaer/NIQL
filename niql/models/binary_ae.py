from typing import List

import torch
import torch.nn as nn


class BinaryAutoEncoder(nn.Module):
    """
    Learns a hash function
    """

    def __init__(self, input_dim: int, hidden_layer_dims: List[int], latent_dim: int, k):
        super(BinaryAutoEncoder, self).__init__()
        self.register_buffer("A", torch.randn((k, latent_dim)))
        self.latent_dim = latent_dim
        prev_dim = input_dim
        encoder_layers = []
        for hdim in hidden_layer_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.ReLU(),
            ])
            prev_dim = hdim
        bottle_neck = nn.Sequential(nn.Linear(prev_dim, latent_dim), nn.Sigmoid())
        encoder_layers.append(bottle_neck)

        prev_dim = latent_dim
        decoder_layers = []
        for hdim in reversed(hidden_layer_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.ReLU(),
            ])
            prev_dim = hdim

        output_layer = nn.Sequential(nn.Linear(prev_dim, input_dim), nn.Softmax())
        decoder_layers.append(output_layer)

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        noise = torch.empty_like(z).uniform_(-0.25, 0.25)
        z_noise = z + noise
        x = self.decoder(z_noise)
        return z, x

    def encode(self, x):
        z = self.encoder(x)
        z = torch.round(z)
        z = (z @ self.A.T > 0).float()
        return z
