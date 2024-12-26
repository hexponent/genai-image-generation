import torch
from torch import nn

from .ae import Autoencoder


def VLB(beta=1):
    def inner(gt, prediction):
        """
        VAE loss is Variational lower bound
        reconstruction loss - BCE
        regularizer for latent space - KL Divergence

        https://discuss.pytorch.org/t/correct-implementation-of-vae-loss/146750
        """
        predicted, mu, var = prediction

        BCE = torch.nn.MSELoss()(gt, predicted)
        KLD = -0.5 * beta * torch.mean(1 + var - mu.pow(2) - var.exp())

        return BCE + KLD

    return inner


class VariationalAutoencoder(Autoencoder):
    def __init__(self, name, latent_dim=1024, beta=1):
        super().__init__(name, latent_dim)

        self.loss = VLB(beta)

        del self.e_fc
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)  # Mean for latent space
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)  # Log variance for latent space

    def forward(self, x):
        features = self.encoder(x)

        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        x_reconstructed = self.decode(z)

        return x_reconstructed, mu, logvar
