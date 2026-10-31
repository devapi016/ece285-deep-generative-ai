"""Convolutional variational autoencoder and its ELBO loss."""

import torch
import torch.nn.functional as F
from torch import nn

from config import ENCODER_CHANNELS, IMAGE_SIZE, LATENT_DIM

IMAGE_CHANNELS = 1
KERNEL_SIZE = 4
STRIDE = 2
PADDING = 1
LEAKY_SLOPE = 0.2


def _down_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, KERNEL_SIZE, STRIDE, PADDING),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(LEAKY_SLOPE),
    )


def _up_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, KERNEL_SIZE, STRIDE, PADDING),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(LEAKY_SLOPE),
    )


class VAE(nn.Module):
    """Gaussian-latent VAE with a Bernoulli decoder over grayscale pixels.

    The encoder halves the resolution at every stage of ENCODER_CHANNELS; the decoder
    mirrors it. The decoder returns logits so the loss can use the numerically
    stable BCE-with-logits.
    """

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        bottleneck_channels = ENCODER_CHANNELS[-1]
        self.bottleneck_size = IMAGE_SIZE // STRIDE ** len(ENCODER_CHANNELS)
        bottleneck_features = bottleneck_channels * self.bottleneck_size**2

        encoder_widths = (IMAGE_CHANNELS, *ENCODER_CHANNELS)
        self.encoder = nn.Sequential(
            *(_down_block(cin, cout) for cin, cout in zip(encoder_widths, encoder_widths[1:])),
            nn.Flatten(),
        )
        self.to_mean = nn.Linear(bottleneck_features, latent_dim)
        self.to_log_variance = nn.Linear(bottleneck_features, latent_dim)

        decoder_widths = ENCODER_CHANNELS[::-1]
        self.from_latent = nn.Linear(latent_dim, bottleneck_features)
        self.decoder = nn.Sequential(
            *(_up_block(cin, cout) for cin, cout in zip(decoder_widths, decoder_widths[1:])),
            nn.ConvTranspose2d(decoder_widths[-1], IMAGE_CHANNELS, KERNEL_SIZE, STRIDE, PADDING),
        )

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(images)
        return self.to_mean(features), self.to_log_variance(features)

    def decode_logits(self, latents: torch.Tensor) -> torch.Tensor:
        features = self.from_latent(latents).view(
            -1, ENCODER_CHANNELS[-1], self.bottleneck_size, self.bottleneck_size
        )
        return self.decoder(features)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encode(images)
        # Reparameterisation trick keeps sampling differentiable w.r.t. mean/variance.
        latents = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
        return self.decode_logits(latents), mean, log_variance

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Draws images in [0, 1] by decoding latents from the N(0, I) prior."""
        latents = torch.randn(num_samples, self.latent_dim, device=device)
        return torch.sigmoid(self.decode_logits(latents))


def elbo_terms(
    logits: torch.Tensor, images: torch.Tensor, mean: torch.Tensor, log_variance: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (reconstruction, KL) per image, each summed over its dimensions."""
    batch_size = images.shape[0]
    reconstruction = F.binary_cross_entropy_with_logits(logits, images, reduction="sum")
    kl_divergence = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())
    return reconstruction / batch_size, kl_divergence / batch_size
