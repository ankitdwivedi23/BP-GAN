import torch.nn as nn
import torch.nn.functional as F
import torch

class Generator(nn.Module):
    def __init__(self, num_channels, latent_dim, num_classes, ngf):
        super(Generator, self).__init__()

        self.convt1_1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf*4, 4, 1, 0),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.convt1_2 = nn.Sequential(
            nn.ConvTranspose2d(num_classes, ngf*4, 4, 1, 0),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.convt2 = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.convt3 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.convt4 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.convt5 = nn.Sequential(
            nn.ConvTranspose2d(ngf, num_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, noise, label):
        x = self.convt1_1(noise)
        y = self.convt1_2(label)
        x = torch.cat([x, y], 1)
        x = self.convt2(x)
        x = self.convt3(x)
        x = self.convt4(x)
        x = self.convt5(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, num_channels, num_classes, ndf):
        super(Discriminator, self).__init__()        
        
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(num_channels, ndf//2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv1_2 = nn.Sequential(
            nn.Conv2d(num_classes, ndf//2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, 4, 2, 1),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(ndf*8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, img, label):
        x = self.conv1_1(img)
        y = self.conv1_2(label)
        x = torch.cat([x, y], 1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x
