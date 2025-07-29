import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ExGANLosses:
    def __init__(
        self, device, lambda_adv=1.0, lambda_fm=10.0, lambda_rec=5.0, lambda_style=2.0
    ):
        self.device = device
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
        self.lambda_rec = lambda_rec
        self.lambda_style = lambda_style

        # Initialize feature extractor for perceptual losses
        self.feature_extractor = self._build_feature_extractor().to(device)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Loss functions
        self.adversarial_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def _build_feature_extractor(self):
        vgg = models.vgg19(pretrained=True)
        feature_extractor = nn.Sequential(
            *list(vgg.features.children())[:20]
        )  # Up to conv4_2
        return feature_extractor

    def compute_adversarial_loss(
        self, discriminator, exemplar_features, fake_imgs_exemplar_features
    ):
        """Standard GAN adversarial loss"""
        batch_size = exemplar_features.size(0)

        # Real images loss
        pred_real = discriminator(exemplar_features)
        real_labels = torch.ones(batch_size, 1, device=self.device)
        loss_real = self.adversarial_loss(pred_real, real_labels)

        # Fake images loss
        pred_fake = discriminator(fake_imgs_exemplar_features.detach())
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        loss_fake = self.adversarial_loss(pred_fake, fake_labels)

        # Total discriminator loss
        d_loss = (loss_real + loss_fake) * 0.5

        return d_loss

    def compute_feature_matching_loss(self, real_imgs, fake_imgs):
        """Perceptual loss that matches features between real and fake images"""
        with torch.no_grad():
            real_features = self.feature_extractor(real_imgs)
        fake_features = self.feature_extractor(fake_imgs)
        return self.l1_loss(fake_features, real_features)

    def compute_exemplar_reconstruction_loss(self, fake_imgs, exemplar_imgs):
        """Encourages generated images to reconstruct exemplar characteristics"""
        return self.l1_loss(fake_imgs, exemplar_imgs)

    def compute_style_loss(self, fake_imgs, exemplar_imgs):
        """Style loss based on Gram matrices of features"""

        def gram_matrix(x):
            batch, channels, height, width = x.size()
            features = x.view(batch * channels, height * width)
            gram = torch.mm(features, features.t())
            return gram.div(batch * channels * height * width)

        with torch.no_grad():
            exemplar_features = self.feature_extractor(exemplar_imgs)
            exemplar_gram = gram_matrix(exemplar_features)

        fake_features = self.feature_extractor(fake_imgs)
        fake_gram = gram_matrix(fake_features)

        return self.mse_loss(fake_gram, exemplar_gram)

    def compute_perceptual_loss(self, exemplar_features, fake_imgs_exemplar_features):
        return self.l1_loss(exemplar_features, fake_imgs_exemplar_features)

    def compute_reconstruction_loss(self, geneated_imgs, actual_imgs):
        return self.mse_loss(actual_imgs, geneated_imgs)

    def compute_discriminator_loss(self, discriminator, real_imgs, fake_imgs):
        """Discriminator loss function"""
        d_loss, _ = self.compute_adversarial_loss(discriminator, real_imgs, fake_imgs)
        return d_loss


import torch.optim as optim
from torch.optim import lr_scheduler


def get_optimizer_generator(
    generator, lr=0.0002, beta1=0.5, beta2=0.999, weight_decay=1e-4
):
    """
    Returns the optimizer for the generator with recommended settings

    Args:
        generator: The generator model
        lr: Learning rate (default: 0.0002)
        beta1: Adam beta1 parameter (default: 0.5)
        beta2: Adam beta2 parameter (default: 0.999)
        weight_decay: L2 regularization (default: 1e-4)

    Returns:
        optimizer: Configured Adam optimizer
        scheduler: Learning rate scheduler
    """
    # Adam optimizer is typically used for GANs
    optimizer_g = optim.Adam(
        generator.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
    )

    # Learning rate scheduler (optional but recommended)
    scheduler_g = lr_scheduler.StepLR(
        optimizer_g,
        step_size=30,  # Reduce LR every 30 epochs
        gamma=0.5,  # Reduce by half
    )

    return optimizer_g, scheduler_g
