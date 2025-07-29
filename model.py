import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from dataset import ExGANDataset
from torch.utils.data import DataLoader
from losses import ExGANLosses, get_optimizer_generator
from torchvision import models


import torch
import torch.nn as nn
import torch.nn.functional as F


class ExGANGenerator(nn.Module):
    def __init__(self, img_channels=3, feature_channels=512, base_channels=64):
        super(ExGANGenerator, self).__init__()

        # Input processing (3x128x128 -> 64x64x64)
        self.down1 = nn.Sequential(
            nn.Conv2d(img_channels, base_channels, 4, 2, 1),
            nn.InstanceNorm2d(base_channels),
            nn.LeakyReLU(0.2),
        )

        # Downsample to 128x32x32
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2),
        )

        # Downsample to 256x16x16 (matches exemplar feature map size)
        self.down3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.InstanceNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2),
        )

        # Exemplar attention module at 16x16 resolution
        self.attention = ExemplarAttention(base_channels * 4, feature_channels)

        # Upsample to 128x32x32
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(),
        )

        # Upsample to 64x64x64
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(),
        )

        # Final output 3x128x128
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, img_channels, 4, 2, 1), nn.Tanh()
        )

        # Residual connections
        self.res1 = ResidualBlock(base_channels * 2)
        self.res2 = ResidualBlock(base_channels)

    def forward(self, x, exemplar_features):
        # Encoder path
        d1 = self.down1(x)  # 64x64x64
        d2 = self.down2(d1)  # 128x32x32
        d3 = self.down3(d2)  # 256x16x16

        # Apply exemplar attention at bottleneck
        attn = self.attention(d3, exemplar_features)

        # Decoder path with residual connections
        u1 = self.up1(attn)  # 128x32x32
        u1 = self.res1(u1 + d2)  # Add skip connection

        u2 = self.up2(u1)  # 64x64x64
        u2 = self.res2(u2 + d1)  # Add skip connection

        output = self.up3(u2)  # 3x128x128

        return output


class ExemplarAttention(nn.Module):
    def __init__(self, in_channels, feat_channels):
        super(ExemplarAttention, self).__init__()

        # Query from input features
        self.query = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.InstanceNorm2d(in_channels // 8),
        )

        # Key from exemplar features
        self.key = nn.Sequential(
            nn.Conv2d(feat_channels, in_channels // 8, 1),
            nn.InstanceNorm2d(in_channels // 8),
        )

        # Value from exemplar features
        self.value = nn.Sequential(
            nn.Conv2d(feat_channels, in_channels, 1), nn.InstanceNorm2d(in_channels)
        )

        # Output projection
        self.proj = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, exemplar):
        batch_size, _, height, width = x.size()

        # Project queries
        q = self.query(x).view(batch_size, -1, height * width)  # [B, C', N]

        # Project keys and values
        k = self.key(exemplar).view(batch_size, -1, height * width)  # [B, C', N]
        v = self.value(exemplar).view(batch_size, -1, height * width)  # [B, C, N]

        # Attention scores
        energy = torch.bmm(q.permute(0, 2, 1), k)  # [B, N, N]
        attention = F.softmax(energy, dim=-1)

        # Apply attention to values
        out = torch.bmm(v, attention.permute(0, 2, 1))  # [B, C, N]
        out = out.view(batch_size, -1, height, width)

        # Final projection and residual
        out = self.proj(out)
        return x + self.gamma * out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


# Discriminator Network
class ExGANDiscriminator(nn.Module):
    def __init__(self, feature_channels=512, base_channels=64):
        super(ExGANDiscriminator, self).__init__()

        self.model = nn.Sequential(
            # Input: 512x16x16
            nn.Conv2d(feature_channels, base_channels * 8, 4, 2, 1),  # 256x8x8
            nn.InstanceNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 8, base_channels * 4, 4, 2, 1),  # 128x4x4
            nn.InstanceNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2),
            # Changed last layer to handle 4x4 input properly
            nn.Conv2d(base_channels * 4, 1, 4, 1, 0),  # 1x1x1
            nn.Sigmoid(),
        )

        # PatchGAN output (alternative approach)
        self.patch_gan = nn.Sequential(
            nn.Conv2d(feature_channels, base_channels * 8, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 8, base_channels * 16, 4, 2, 1),
            nn.InstanceNorm2d(base_channels * 16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 16, base_channels * 32, 4, 2, 1),
            nn.InstanceNorm2d(base_channels * 32),
            nn.LeakyReLU(0.2),
            # Outputs a 2x2 patch (instead of 1x1)
            nn.Conv2d(base_channels * 32, 1, 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Choose which discriminator to use
        x = self.model(x)  # Standard discriminator
        # x = self.patch_gan(x)  # PatchGAN discriminator

        return x.squeeze(
            -1,
        ).squeeze(-1)
        # return


# Feature Extractor (for exemplars)
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg = models.vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *list(vgg.features.children())[:20]
        )  # Up to conv4_2

    def forward(self, x):
        # VGG normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        return self.feature_extractor(x)


# Full ExGAN Model
class ExGAN(nn.Module):
    def __init__(self, z_dim=128, img_channels=3):
        super(ExGAN, self).__init__()
        self.generator = ExGANGenerator(img_channels=3, feature_channels=512)
        self.discriminator = ExGANDiscriminator(feature_channels=512)
        self.feature_extractor = FeatureExtractor()

        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, z, exemplar_img):
        # Extract features from exemplar
        exemplar_features = self.feature_extractor(exemplar_img)

        # Generate image
        generated_img = self.generator(z, exemplar_features)

        generated_img_exemplar_features = self.feature_extractor(generated_img)

        return exemplar_features, generated_img, generated_img_exemplar_features


# Initialize models and optimizers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exgan = ExGAN().to(device)
losses = ExGANLosses(device)

# Get optimizers
optimizer_g, scheduler_g = get_optimizer_generator(exgan.generator)
optimizer_d, scheduler_d = get_optimizer_generator(exgan.discriminator)  # Similar setup

num_epochs = 100

dataset = ExGANDataset(
    root_dir="../data/celeb_id_raw", exemplar_dir="../data/celeb_id_aligned"
)

print(len(dataset))
dataloader = DataLoader(
    dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
)


import os
import numpy as np
import wandb

wandb.init(
    config={
        "project": os.environ.get("WANDB_PROJECT", "test"),
        "entity": os.environ.get("WANDB_ENTITY", "uipath"),
    }
)

wandb.watch(exgan, log_freq=100)

for epoch in range(num_epochs):
    D_epoch_losses = []
    G_epoch_losses = []
    for real_imgs, masked_imgs, exemplar_imgs in dataloader:
        real_imgs = real_imgs.to(device)
        masked_imgs = masked_imgs.to(device)
        exemplar_imgs = exemplar_imgs.to(device)

        # Generate fake images
        exemplar_features, fake_imgs, fake_imgs_exemplar_features = exgan(
            masked_imgs, exemplar_imgs
        )
        # Update discriminator
        optimizer_d.zero_grad()
        d_loss = losses.compute_adversarial_loss(
            exgan.discriminator, exemplar_features, fake_imgs_exemplar_features
        )

        d_loss.backward()
        optimizer_d.step()

        # Update generator
        optimizer_g.zero_grad()
        g_losses = losses.compute_perceptual_loss(
            exemplar_features, fake_imgs_exemplar_features
        )
        r_losses = losses.compute_reconstruction_loss(fake_imgs, real_imgs)
        g_losses = g_losses + r_losses
        g_losses.backward()
        optimizer_g.step()

        D_epoch_losses.append(d_loss.item())
        G_epoch_losses.append(g_losses.item())

    # Print losses at the end of each epoch
    print(
        f"Epoch [{epoch}/{num_epochs}] "
        f"D_loss: {np.mean(D_epoch_losses):.4f}, "
        f"G_loss: {np.mean(G_epoch_losses):.4f}, "
    )

    wandb.log({"D_loss": np.mean(D_epoch_losses), "G_loss": np.mean(G_epoch_losses)})

    # Update learning rates
    scheduler_g.step()
    scheduler_d.step()


import os

model_dir = "../models"
os.makedirs(model_dir, exist_ok=True)
# Save model states
torch.save(exgan.generator.state_dict(), os.path.join(model_dir, "generator.pth"))
torch.save(
    exgan.discriminator.state_dict(), os.path.join(model_dir, "discriminator.pth")
)
