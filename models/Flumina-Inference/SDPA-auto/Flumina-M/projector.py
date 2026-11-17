from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn

from .config import FEATURE_CHANNELS, POOL_FACTORS
from .transformer import ChannelRMSNorm


class ScaleProjector(nn.Module):
    """Strided projection and channel RMS normalization for one CNN scale."""

    def __init__(self, in_channels: int, embed_dim: int, pool_factor: int) -> None:
        """
        Args:
            in_channels (int): number of input CNN channels
            embed_dim (int): channel dimension of the output residual stream
            pool_factor (int): convolution kernel size and stride along each spatial axis
        """
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=pool_factor,
            stride=pool_factor,
            padding=0,
            bias=True,
        )
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(cast(torch.Tensor, self.proj.bias))
        self.norm = ChannelRMSNorm(embed_dim, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): CNN feature tensor with shape (B, C_in, D, H, W)

        Returns:
            torch.Tensor: normalized stream with shape (B, embed_dim, D//f, H//f, W//f), where f is pool_factor
        """
        return self.norm(self.proj(x))


class Projector(nn.Module):
    """Independent CNN projections onto a shared grid of four residual streams."""

    def __init__(
        self, embed_dim: int, in_channels: Sequence[int] = FEATURE_CHANNELS
    ) -> None:
        """
        Args:
            embed_dim (int): channel dimension of each output residual stream
            in_channels (Sequence[int]): four CNN channel dimensions from finest to coarsest scale
        """
        super().__init__()
        self.projectors = nn.ModuleList(
            ScaleProjector(width, embed_dim, factor)
            for width, factor in zip(in_channels, POOL_FACTORS)
        )

    def forward(self, pyramid: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            pyramid (Sequence[torch.Tensor]): four CNN tensors with shapes (B, C_i, D/2**i, H/2**i, W/2**i) for i from 0 to 3

        Returns:
            torch.Tensor: stacked residual streams with shape (B, 4, embed_dim, D/8, H/8, W/8)
        """
        projected = tuple(
            projector(feature) for projector, feature in zip(self.projectors, pyramid)
        )
        return torch.stack(projected, dim=1)
