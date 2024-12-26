import torch

from .base import TrainValidationTrainableModule


class Autoencoder(TrainValidationTrainableModule):
    loss = torch.nn.MSELoss()

    def __init__(self, name, latent_dim=1024):
        super().__init__(name)

        # 3 x 32 x 32
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # Downsample to 16x16
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Downsample to 8x8
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Downsample to 4x4
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Flatten()  # Flatten the 4x4 spatial dimensions
        )
        self.e_fc = torch.nn.Linear(256 * 4 * 4, latent_dim)  # Project to latent space

        self.d_fc = torch.nn.Linear(latent_dim, 256 * 4 * 4)  # Project latent vector to feature map
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample to 8x8
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample to 16x16
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample to 32x32
            torch.nn.Sigmoid()  # Final activation for pixel values in [0, 1]
        )

    def decode(self, z):
        features = self.d_fc(z)
        features = features.view(-1, 256, 4, 4)  # Reshape to 4x4 feature map
        x_reconstructed = self.decoder(features)
        return x_reconstructed

    def forward(self, x):
        features = self.encoder(x)

        z = self.e_fc(features)  # Latent representation

        x_reconstructed = self.decode(z)

        return x_reconstructed
