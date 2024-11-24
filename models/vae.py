import torch

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
    def __init__(self, name, beta=1):
        super().__init__(name)

        self.loss = VLB(beta)

        self.var = torch.nn.Linear(1024, 1024)
        self.mu = torch.nn.Linear(1024, 1024)

    def forward(self, inp):
        encoded = self.encoder(inp)

        flat_encoded = torch.flatten(encoded, start_dim=1)
        mu, var = self.mu(flat_encoded), self.var(flat_encoded)
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        reconstructed = eps.mul(std).add_(mu)
        reconstructed = torch.reshape(reconstructed, (*reconstructed.shape, 1, 1))

        out = self.decoder(reconstructed)
        return out, mu, var
