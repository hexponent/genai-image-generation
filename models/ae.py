import torch

from .base import TrainableModule


class Autoencoder(TrainableModule):
    loss = torch.nn.MSELoss()

    def __init__(self, name):
        super().__init__(name)

        # 3 x 32 x 32
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1),  # 16x16
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 8x8
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 512, kernel_size=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 1024, kernel_size=5)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(1024, 512, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(512, 256, kernel_size=4),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, inp):
        encoded = self.encoder(inp)
        out = self.decoder(encoded)
        return out
