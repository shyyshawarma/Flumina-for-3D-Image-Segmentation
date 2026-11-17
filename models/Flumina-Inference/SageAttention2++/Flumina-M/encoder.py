from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from .config import Config
from .projector import Projector
from .tokenizer import Tokenizer
from .transformer import Transformer


FeaturePyramid = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True, slots=True)
class EncoderOutput:
    """
    CNN features and residual streams from the encoder.

    Args:
        pyramid (FeaturePyramid): tuple of four CNN tensors with shapes (B, C_i, D/2**i, H/2**i, W/2**i) for i from 0 to 3
        projected (torch.Tensor): initial streams with shape (B, 4, C, D/8, H/8, W/8), where C is embed_dim
        streams (torch.Tensor): final Transformer streams with the same shape as projected
    """

    pyramid: FeaturePyramid
    projected: torch.Tensor
    streams: torch.Tensor


class Encoder(nn.Module):
    """CNN pyramid encoder followed by a shared multi-stream Transformer."""

    def __init__(self, config: Config) -> None:
        """
        Args:
            config (Config): model architecture, input channels, and crop dimensions
        """
        super().__init__()
        self.config = config
        self.tokenizer = Tokenizer(
            config.in_channels, config.patch_embed_channels, config.use_checkpoint
        )
        self.projectors = Projector(config.embed_dim, config.feature_channels)
        self.vit = Transformer(config.transformer_config())

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """
        Args:
            x (torch.Tensor): floating image tensor with shape (B, C_in, D, H, W), matching the configured channels and crop

        Returns:
            EncoderOutput: four CNN maps and the projected and final stream tensors,
                each stream tensor having shape (B, 4, C, D/8, H/8, W/8), where C is embed_dim
        """
        expected = (self.config.in_channels, *self.config.crop_shape)
        if x.ndim != 5 or x.shape[0] <= 0 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"Expected image trailing shape {expected}, got {tuple(x.shape)}"
            )
        if not torch.is_floating_point(x):
            raise ValueError("image must be a floating-point tensor")
        pyramid = cast(FeaturePyramid, self.tokenizer.forward_features(x))
        projected = self.projectors(pyramid)
        streams = self.vit(projected)
        return EncoderOutput(pyramid=pyramid, projected=projected, streams=streams)
