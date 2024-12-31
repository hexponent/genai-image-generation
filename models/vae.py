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
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
