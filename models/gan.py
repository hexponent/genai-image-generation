"""
GAN implementation is based on https://github.com/Ksuryateja/DCGAN-CIFAR10-pytorch/blob/master/gan_cifar.py
But adapted to be used in existing pipeline
"""

import torch

from .base import AdversarialTrainableModule


class Generator(torch.nn.Module):

    def __init__(self, latent_dim, num_filters=64):
        super().__init__()

        self.decoder = torch.nn.Sequential(
            # input is Z, going into a convolution
            torch.nn.ConvTranspose2d(latent_dim, num_filters * 8, 2, 1, 0, bias=False),
            torch.nn.BatchNorm2d(num_filters * 8),
            torch.nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            torch.nn.ConvTranspose2d(num_filters * 8, num_filters * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(num_filters * 4),
            torch.nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            torch.nn.ConvTranspose2d(num_filters * 4, num_filters * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(num_filters * 2),
            torch.nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            torch.nn.ConvTranspose2d(num_filters * 2, num_filters, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(num_filters),
            torch.nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            torch.nn.ConvTranspose2d(num_filters, 3, 4, 2, 1, bias=False),
            torch.nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.decoder(input)
        return output


class Discriminator(torch.nn.Module):
    def __init__(self, num_filters=64):
        super().__init__()

        self.main = torch.nn.Sequential(
            # input is (nc) x 32 x 32
            torch.nn.Conv2d(3, num_filters, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            torch.nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(num_filters * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            torch.nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(num_filters * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            torch.nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(num_filters * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            torch.nn.Conv2d(num_filters * 8, 1, 2, 1, 0, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


class GAN(AdversarialTrainableModule):
    loss = torch.nn.BCELoss()

    def __init__(self, name, latent_dim=100):
        super().__init__(name)

        self.latent_dim = latent_dim
        self.gen = Generator(self.latent_dim)
        self.dis = Discriminator()

        self.gen.apply(self.weights_init)
        self.dis.apply(self.weights_init)

        self.optim_g = torch.optim.Adam(self.gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optim_d = torch.optim.Adam(self.dis.parameters(), lr=0.0002, betas=(0.5, 0.999))

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def generate(self, noise):
        return self.gen(noise)

    def state_dict(self):
        return {
            'generator': self.gen.state_dict(),
            'discriminator': self.dis.state_dict(),
        }

    def load_state_dict(self, data):
        self.gen.load_state_dict(data['generator'])
        self.dis.load_state_dict(data['discriminator'])
