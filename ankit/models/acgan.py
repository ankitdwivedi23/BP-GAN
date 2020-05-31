import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self, latent_dim, num_classes):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim

        self.label_emb = nn.Embedding(num_classes, latent_dim)
        self.fc = nn.Linear(latent_dim, 768)

        self.conv_layers = nn.Sequential(
            # ConvT-BN-ReLU-1 (Input => 1 * 1 * 768, Output => 4 * 4 * 384)
            nn.ConvTranspose2d(in_channels=768, out_channels=384, kernel_size=4, stride=1),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.2, inplace=True),
            # ConvT-BN-ReLU-2 (Input => 4 * 4 * 384, Output => 8 * 8 * 192)
            nn.ConvTranspose2d(in_channels=384, out_channels=192, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2, inplace=True),
            # ConvT-BN-ReLU-3 (Input => 8 * 8 * 192, Output => 16 * 16 * 96)
            nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2, inplace=True),
            # ConvT-BN-ReLU-4 (Input => 16 * 16 * 96, Output => 32 * 32 * 48)
            nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2, inplace=True),
            # ConvT-Tanh (Input => 32 * 32 * 48, Output => 64 * 64 * 3)
            nn.ConvTranspose2d(in_channels=48, out_channels=3, kernel_size=4, stride=2, padding=1)
        )
    
    def forward(self, noise, labels):
        z = torch.mul(self.label_emb(labels), noise)
        z = self.fc(z)
        z = z.view(-1, 768, 1, 1)
        z = self.conv_layers(z)
        return z


class Discriminator(nn.Module):

    def __init__(self, num_classes):
        super(Discriminator, self).__init__()

        self.conv_layers = nn.Sequential(
            # Conv-LeakyReLU-Dropout (Input => 64 * 64 * 3, Output => 32 * 32 * 16)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            # Conv-BN-LeakyReLU-Dropout-1 (Input => 32 * 32 * 16, Output => 32 * 32 * 32)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            # Conv-BN-LeakyReLU-Dropout-2 (Input => 32 * 32 * 32, Output => 16 * 16 * 64)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            # Conv-BN-LeakyReLU-Dropout-3 (Input => 16 * 16 * 64, Output => 16 * 16 * 128)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            # Conv-BN-LeakyReLU-Dropout-4 (Input => 16 * 16 * 128, Output => 8 * 8 * 256)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            # Conv-BN-LeakyReLU-Dropout-5 (Input => 8 * 8 * 256, Output => 8 * 8 * 512)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5)
        )

        self.fc_source=nn.Linear(8*8*512,1)
        self.fc_class=nn.Linear(8*8*512, num_classes)
        self.sig=nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv_layers(x)
        x=x.view(-1,8*8*512)
        validity = self.sig(self.fc_source(x)) # real or fake score
        class_scores = self.fc_class(x) # logit scores for each class

        return validity, class_scores







