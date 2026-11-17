from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from .config import DECODER_CHANNELS, POOL_FACTORS, SpatialShape
from .tokenizer import ConvBlock
from .triton_resize import resize2x_for_bf16_conv


FeaturePyramid = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class ScaleRestorer(nn.Module):
    """Learned spatial upsampling for one residual stream."""

    def __init__(
        self, in_channels: int, out_channels: int, upsample_factor: int
    ) -> None:
        """
        Args:
            in_channels (int): channel dimension of the input residual stream
            out_channels (int): number of restored feature channels
            upsample_factor (int): transposed-convolution kernel size and stride along each spatial axis
        """
        super().__init__()
        self.projection = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=upsample_factor,
            stride=upsample_factor,
            padding=0,
            output_padding=0,
            bias=True,
        )
        nn.init.trunc_normal_(self.projection.weight, std=0.02)
        nn.init.zeros_(cast(torch.Tensor, self.projection.bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): residual stream with shape (B, C_in, d, h, w)

        Returns:
            torch.Tensor: restored features with shape (B, C_out, f*d, f*h, f*w), where f is upsample_factor
        """
        return self.projection(x)


class Restorer(nn.Module):
    """Independent restoration of four residual streams to CNN pyramid grids."""

    def __init__(self, embed_dim: int) -> None:
        """
        Args:
            embed_dim (int): channel dimension of each input residual stream
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.restorers = nn.ModuleList(
            ScaleRestorer(embed_dim, channels, factor)
            for channels, factor in zip(DECODER_CHANNELS, POOL_FACTORS)
        )

    def forward(
        self, processed_streams: torch.Tensor, target_sizes: Sequence[SpatialShape]
    ) -> FeaturePyramid:
        """
        Args:
            processed_streams (torch.Tensor): final residual streams with shape (B, 4, embed_dim, d, h, w)
            target_sizes (Sequence[SpatialShape]): four expected grids (f*d, f*h, f*w), with factors f = 8, 4, 2, 1

        Returns:
            FeaturePyramid: tuple of four restored tensors with shapes (B, C_i, f_i*d, f_i*h, f_i*w),
                channel counts (32, 64, 128, 256), and factors (8, 4, 2, 1)
        """
        if processed_streams.ndim != 6 or tuple(processed_streams.shape[1:3]) != (
            4,
            self.embed_dim,
        ):
            raise ValueError(
                f"Expected states shaped (B, 4, {self.embed_dim}, d, h, w)"
            )
        expected = tuple(
            tuple(size * factor for size in processed_streams.shape[3:])
            for factor in POOL_FACTORS
        )
        if tuple(tuple(size) for size in target_sizes) != expected:
            raise ValueError(f"target_sizes must match the restored grids {expected}")
        return tuple(
            restorer(processed_streams[:, index])
            for index, restorer in enumerate(self.restorers)
        )


class DecoderStage(nn.Module):
    """Trilinear upsampling, additive skip fusion, and convolutional refinement."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Args:
            in_channels (int): number of channels in the coarse input
            out_channels (int): number of channels in the finer skip and output
        """
        super().__init__()
        self.use_triton_resize = True
        self.channel_reduction = ConvBlock(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.refinement = ConvBlock(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): coarse features with shape (B, C_in, d, h, w)
            skip (torch.Tensor): finer restored features with shape (B, C_out, D, H, W)

        Returns:
            torch.Tensor: refined features with shape (B, C_out, D, H, W)
        """
        if self.use_triton_resize:
            upsampled = resize2x_for_bf16_conv(
                x, size=cast(SpatialShape, tuple(skip.shape[2:]))
            )
        else:
            upsampled = F.interpolate(
                x, size=tuple(skip.shape[2:]), mode="trilinear", align_corners=False
            )
        upsampled = self.channel_reduction(upsampled)
        return self.refinement(upsampled + skip)


class Decoder(nn.Module):
    """Three additive decoder stages and a full-resolution segmentation head."""

    def __init__(self, num_classes: int) -> None:
        """
        Args:
            num_classes (int): number of output segmentation channels
        """
        super().__init__()
        self.num_classes = num_classes
        self.stages = nn.ModuleList(
            (DecoderStage(256, 128), DecoderStage(128, 64), DecoderStage(64, 32))
        )
        self.segmentation_head = nn.Conv3d(
            32, num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, features: FeaturePyramid) -> torch.Tensor:
        """
        Args:
            features (FeaturePyramid): tuple of four restored tensors with shapes (B, C_i, D/2**i, H/2**i, W/2**i),
                where C_i is 32, 64, 128, or 256

        Returns:
            torch.Tensor: raw segmentation logits with shape (B, num_classes, D, H, W)
        """
        x0, x1, x2, x3 = features
        decoded_x2 = self.stages[0](x3, x2)
        decoded_x1 = self.stages[1](decoded_x2, x1)
        decoded_x0 = self.stages[2](decoded_x1, x0)
        return self.segmentation_head(decoded_x0)


class SegmentationDecoder(nn.Module):
    """Residual stream restoration followed by multi-scale segmentation decoding."""

    def __init__(self, embed_dim: int, num_classes: int) -> None:
        """
        Args:
            embed_dim (int): channel dimension of each input residual stream
            num_classes (int): number of output segmentation channels
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.restorer = Restorer(embed_dim)
        self.decoder = Decoder(num_classes)

    def forward(
        self, processed_streams: torch.Tensor, target_sizes: Sequence[SpatialShape]
    ) -> torch.Tensor:
        """
        Args:
            processed_streams (torch.Tensor): final residual streams with shape (B, 4, embed_dim, d, h, w)
            target_sizes (Sequence[SpatialShape]): four expected grids (f*d, f*h, f*w), with factors f = 8, 4, 2, 1

        Returns:
            torch.Tensor: raw segmentation logits with shape (B, num_classes, 8*d, 8*h, 8*w)
        """
        return self.decoder(self.restorer(processed_streams, target_sizes))
