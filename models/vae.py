import torch
from torch import nn

from .base import TrainableModule


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


class VariationalAutoencoder(TrainableModule):
    def __init__(self, name, beta=1):
        super().__init__(name)

        self.loss = VLB(beta)

        h_dim = 512
        z_dim = 512
        # 3 x 32 x 32
        # https://github.com/darleybarreto/vae-pytorch/blob/master/models/normal_vae.py
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.fc_e = nn.Sequential(
            nn.Linear(8 * 8 * 16, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
        )

        self.mu = nn.Linear(h_dim, z_dim)
        self.var = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus(),
        )

        # Decoder
        self.fc_d = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 8 * 8 * 16),
            nn.BatchNorm1d(8 * 8 * 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def encode(self, x):
        encoded = self.encoder(x)
        fc = self.fc_e(encoded.view(-1, 8 * 8 * 16))

        return self.mu(fc), self.var(fc)

    def decode(self, z):
        fc = self.fc_d(z).view(-1, 16, 8, 8)

        conv8 = self.decoder(fc)
        return conv8.view(-1, 3, 32, 32)

    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps
        return z

    def forward(self, x):
        mu_hat, logvar_hat = self.encode(x)
        z = self.reparameterize(mu_hat, logvar_hat)
        return self.decode(z), mu_hat, logvar_hat
