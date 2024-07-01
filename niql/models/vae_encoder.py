import torch
import torch.nn as nn
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from torch.nn.functional import mse_loss


class VAEEncoder(nn.Module):

    def __init__(self, model_config, obs_space):
        nn.Module.__init__(self)

        # decide the model arch
        self.custom_config = model_config["custom_model_config"]
        self.activation = model_config.get("fcnet_activation")

        # gather model info
        comm_dim = model_config.get("comm_dim", 0)
        msg_agg_dim = model_config.get("comm_aggregator_dim", 0)
        action_dim = model_config.get("action_dim", 0)
        input_dim = obs_space['obs'].shape[0] + comm_dim + msg_agg_dim + action_dim
        encode_layer = self.custom_config["model_arch_args"]["encode_layer"]
        encoder_layer_dim = encode_layer.split("-")
        encoder_layer_dim = [int(i) for i in encoder_layer_dim]
        self.latent_dim = encoder_layer_dim[-1]
        hidden_layer_dims = encoder_layer_dim[:-1]

        # create encoder
        prev_dim = input_dim
        encoder_layers = []
        for hdim in hidden_layer_dims:
            encoder_layers.append(
                SlimFC(
                    in_size=prev_dim,
                    out_size=hdim,
                    initializer=normc_initializer(1.0),
                    activation_fn=self.activation
                )
            )
            prev_dim = hdim

        # Mean and log-variance
        bottle_neck = SlimFC(
                    in_size=prev_dim,
                    out_size=self.latent_dim * 2,
                    initializer=normc_initializer(1.0),
                    activation_fn=self.activation
                )
        encoder_layers.append(bottle_neck)

        # create decoder
        prev_dim = self.latent_dim
        decoder_layers = []
        for hdim in reversed(hidden_layer_dims):
            decoder_layers.append(
                SlimFC(
                    in_size=prev_dim,
                    out_size=hdim,
                    initializer=normc_initializer(1.0),
                    activation_fn=self.activation
                )
            )
            prev_dim = hdim
        output_layer = SlimFC(
                    in_size=prev_dim,
                    out_size=input_dim,
                    initializer=normc_initializer(1.0),
                    activation_fn=None
                )
        decoder_layers.append(output_layer)

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    @property
    def output_dim(self):
        return self.latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        params = self.encoder(x)
        mu, logvar = params[:, : self.latent_dim], params[:, self.latent_dim:]
        z = self.reparameterize(mu, logvar)
        return z

    def encode_decode(self, x):
        params = self.encoder(x)
        mu, logvar = params[:, : self.latent_dim], params[:, self.latent_dim:]
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def estimate_density(self, x):
        # self.eval()
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
