from torch import nn

from .base import TrainValidationTrainableModule, ResidualBlock


class Autoencoder(TrainValidationTrainableModule):
    loss = nn.MSELoss()

    def __init__(self, name, latent_dim=1024):
        super().__init__(name)

        # 3 x 32 x 32
        self.encoder = nn.Sequential(
            ResidualBlock(3, 64, stride=2),  # 32x32 → 16x16
            ResidualBlock(64, 128, stride=2),  # 16x16 → 8x8
            ResidualBlock(128, 256, stride=2),  # 8x8 → 4x4
            nn.Flatten(),
        )
        self.e_fc = nn.Linear(256 * 4 * 4, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),  # Map latent_dim to a feature map
            nn.Unflatten(1, (512, 4, 4)),  # Reshape to feature map

            # Add refinement layers after each upsampling
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4x4 → 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Refinement layer
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8 → 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Refinement layer
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16 → 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Refinement layer
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 3, kernel_size=3, padding=1),  # Final output
            nn.Sigmoid()  # Normalize pixel values to [0, 1]
        )

    def forward(self, x):
        features = self.encoder(x)

        z = self.e_fc(features)  # Latent representation

        x_reconstructed = self.decoder(z)

        return x_reconstructed
