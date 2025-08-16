import torch
import torch.nn as nn
from monai.networks.nets import ViT

# Encoder: MedViT-like 3D Vision Transformer
class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        img_size=(128, 128, 128),
        vit_hidden_size: int = 384,
        latent_dim: int = 256,
        patch_size: int = 16,
        num_heads: int = 12,
        num_layers: int = 12,
    ):
        super().__init__()
        # ViT configuration
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=vit_hidden_size,
            mlp_dim=vit_hidden_size * 4,
            num_heads=num_heads,
            num_layers=num_layers,
            classification=False,
        )
        # Determine actual hidden size from model if available
        actual_hidden = getattr(self.vit, "hidden_size", vit_hidden_size)
        self.fc = nn.Linear(actual_hidden, latent_dim)

    def forward(self, x):
        # x: (batch, C, D, H, W)
        features = self.vit(x)
        # MONAI ViT may return a tuple (features, aux) depending on version/config
        if isinstance(features, tuple):
            features = features[0]
        # If features is not a vector, pool it to (batch, hidden_size)
        if features.dim() > 2:
            features = features.mean(dim=tuple(range(2, features.dim())))
        z = self.fc(features)   # z: (batch, latent_dim)
        return z

# Generator: 3D Deconvolutional Network
class Generator(nn.Module):
    def __init__(self, latent_dim=256, phase_dim=32, output_shape=(128, 128, 128, 1)):
        super().__init__()
        self.latent_dim = latent_dim
        self.phase_dim = phase_dim
        # Initial dense layer to project to 3D feature map
        self.fc = nn.Linear(latent_dim + phase_dim, 8 * 8 * 8 * 256)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8x8 -> 16x16x16
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),   # 16x16x16 -> 32x32x32
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),    # 32x32x32 -> 64x64x64
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),     # 64x64x64 -> 128x128x128
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

    def forward(self, z, phase_emb):
        # z: (batch, latent_dim), phase_emb: (batch, phase_dim)
        x = torch.cat([z, phase_emb], dim=1)  # (batch, latent_dim + phase_dim)
        x = self.fc(x)  # (batch, 8*8*8*256)
        x = x.view(-1, 256, 8, 8, 8)  # Reshape to 3D
        return self.decoder(x)  # (batch, 1, 128, 128, 128)

# Discriminator: PatchGAN-style 3D CNN
class Discriminator(nn.Module):
    def __init__(self, input_shape=(128, 128, 128, 1)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1),  # 128x128x128 -> 64x64x64
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),  # 64x64x64 -> 32x32x32
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),  # 32x32x32 -> 16x16x16
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            nn.Conv3d(256, 1, kernel_size=4, stride=1, padding=0),   # 16x16x16 -> 13x13x13
            nn.Sigmoid()  # PatchGAN output
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)  # (batch, 1)

# Phase Detector
class PhaseDetector(nn.Module):
    def __init__(self, latent_dim=256, num_phases=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_phases),
            nn.Softmax(dim=1)
        )

    def forward(self, z):
        return self.model(z)  # (batch, num_phases)