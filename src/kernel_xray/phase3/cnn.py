"""Compact, same-length counter-only CNN and explicitly masked L1 objectives."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .targets import valid_primary_logits


@dataclass(frozen=True)
class CNNConfig:
    input_channels: int = 24
    classes: int = 12
    width: int = 64
    kernel_size: int = 5
    dilations: tuple = (1, 2, 4)
    convolutions_per_block: int = 2
    dropout: float = 0.1

    @property
    def receptive_field(self):
        return 1 + self.convolutions_per_block * (self.kernel_size - 1) * sum(self.dilations)


class ChannelLayerNorm(nn.Module):
    """Normalize channels independently at each timestep, never across time/traces."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, config, dilation):
        super().__init__()
        layers = []
        for _ in range(config.convolutions_per_block):
            layers.extend([
                ChannelLayerNorm(config.width), nn.GELU(),
                nn.Conv1d(config.width, config.width, config.kernel_size,
                          dilation=dilation, padding=dilation * (config.kernel_size - 1) // 2),
                nn.Dropout(config.dropout),
            ])
        self.temporal = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.temporal(x)


class L1TemporalCNN(nn.Module):
    """Only accepts float32 [B,24,T]; output channel IDs retain the original vocabulary."""
    def __init__(self, config=None):
        super().__init__()
        self.config = CNNConfig() if config is None else config
        c = self.config
        if c.kernel_size % 2 != 1 or c.convolutions_per_block < 1 or any(d < 1 for d in c.dilations):
            raise ValueError('Use positive dilations/convolution count and odd same-length kernels')
        self.projection = nn.Conv1d(c.input_channels, c.width, 1)
        self.blocks = nn.Sequential(*(ResidualTemporalBlock(c, d) for d in c.dilations))
        self.final_norm = ChannelLayerNorm(c.width)
        self.head = nn.Conv1d(c.width, c.classes, 1)

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1] != self.config.input_channels:
            raise ValueError('CNN input must be the counter tensor X [B,24,T] only')
        if x.dtype != torch.float32:
            raise ValueError('CNN input must be float32')
        return self.head(F.gelu(self.final_norm(self.blocks(self.projection(x)))))

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def masked_cross_entropy(logits, batch, weights=None):
    """Weighted/unweighted sum divided by valid-bin count; None means skip optimization."""
    selected, targets = valid_primary_logits(logits, batch, 'l1')
    if not targets.numel():
        return None
    return F.cross_entropy(selected, targets, weight=weights, reduction='sum') / targets.numel()


def bounded_class_weights(counts, maximum_ratio=4.0):
    """Square-root inverse frequency capped relative to the most frequent class.

    Positive-class weights have max/min <= maximum_ratio and training-bin
    expectation one. Zero-support classes have zero positive weight. The softmax
    still includes all 12 outputs, including zero-support classes as competitors.
    """
    counts = torch.as_tensor(counts, dtype=torch.float64)
    positive = counts > 0
    if not positive.any() or maximum_ratio < 1 or (counts < 0).any():
        raise ValueError('Need nonnegative training frequencies and ratio >=1')
    weights = torch.zeros_like(counts)
    weights[positive] = (counts.max() / counts[positive]).sqrt().clamp(max=maximum_ratio)
    weights *= counts.sum() / (counts * weights).sum()
    return weights.float()
