import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils.spectral_norm as spectral_norm


"""
    ResNet and Inception Net based Generator:
        I 
"""

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()

        self.branch1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out

class Generator(nn.Module):
    def __init__(self, latent_dim, image_channels, num_state_of_art_blocks=3):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        
        self.initial = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(latent_dim, 256*3, kernel_size=4, stride=1, padding=0)),
            nn.BatchNorm2d(256*3),
            nn.ReLU(inplace=True)
        )
        
        self.state_of_art_blocks = nn.Sequential(
            ResidualBlock(256*3,256*3),
            spectral_norm(nn.ConvTranspose2d(256*3, 256*3, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(256*3),
            nn.ReLU(inplace=True),
            InceptionBlock(256*3, 256),
        )
        
        self.state_of_art_blocks = nn.Sequential(
            *[self.state_of_art_blocks for _ in range(num_state_of_art_blocks)]
        )
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(256*3, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
        self.initialize_weights()

    def forward(self, x):
        x = self.initial(x)
        x = self.state_of_art_blocks(x)
        x = self.final(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight, mean=1.0, std=0.02)
                init.zeros_(m.bias)
