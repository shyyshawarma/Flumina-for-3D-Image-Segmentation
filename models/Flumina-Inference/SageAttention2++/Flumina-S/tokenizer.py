from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class LayerNorm(nn.Module):
    """Channel LayerNorm for tensors with channels in dimension one."""

    def __init__(
        self, num_channels: int, eps: float = 1e-5, elementwise_affine: bool = True
    ) -> None:
        """
        Args:
            num_channels (int): number of channels to normalize
            eps (float): stability constant added to the variance
            elementwise_affine (bool): learn a separate scale and bias for each channel
        """
        super().__init__()
        self.num_channels = num_channels
        self.norm = nn.LayerNorm(
            num_channels, eps=float(eps), elementwise_affine=elementwise_affine
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape (B, C, ...)

        Returns:
            torch.Tensor: normalized tensor with shape (B, C, ...)
        """
        return self.norm(x.movedim(1, -1)).movedim(-1, 1)


NormFactory = Callable[[int], nn.Module]


class ConvBlock(nn.Module):
    """3D convolution with reflection padding, channel normalization, and GELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm: NormFactory = LayerNorm,
    ) -> None:
        """
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int): convolution kernel size along each spatial axis
            stride (int): convolution stride along each spatial axis
            padding (int): reflection padding on each side of every spatial axis
            norm (NormFactory): callable taking a channel count and returning a normalization module
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode="reflect",
                bias=False,
            ),
            norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape (B, C_in, D, H, W)

        Returns:
            torch.Tensor: output tensor with shape (B, C_out, D_out, H_out, W_out),
                where each spatial extent is (size + 2*padding - kernel_size)//stride + 1
        """
        return self.block(x)


class ResBlock(nn.Module):
    """Residual convolution block with optional pooling and channel projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm: NormFactory = LayerNorm,
    ) -> None:
        """
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int): convolution kernel size; use 2*padding + 1 to preserve the residual grid
            stride (int): stride of the first convolution and residual average pooling
            padding (int): reflection padding on each side of every spatial axis
            norm (NormFactory): callable taking a channel count and returning a normalization module
        """
        super().__init__()
        self.main = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size, stride, padding, norm),
            ConvBlock(out_channels, out_channels, kernel_size, 1, padding, norm),
        )
        skip: list[nn.Module] = []
        if stride != 1:
            skip.append(nn.AvgPool3d(kernel_size=stride, stride=stride))
        if in_channels != out_channels:
            skip.append(nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False))
        self.skip = nn.Sequential(*skip) if skip else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape (B, C_in, D, H, W);
                spatial dimensions must make the main and residual branches match

        Returns:
            torch.Tensor: output tensor with shape (B, C_out, D//stride, H//stride, W//stride)
        """
        return self.main(x) + self.skip(x)


class Tokenizer(nn.Module):
    """Convolutional feature pyramid with successive spatial downsampling."""

    def __init__(
        self,
        in_channels: int,
        feature_channels: tuple[int, ...] = (32, 64, 128, 256),
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            in_channels (int): number of input image channels
            feature_channels (tuple[int, ...]): CNN channel dimensions from finest to coarsest scale; four levels by default
            use_checkpoint (bool): recompute downsampling stages during backward while training with gradients enabled
        """
        super().__init__()
        self.in_channels = in_channels
        self.feature_channels = tuple(feature_channels)
        self.use_checkpoint = use_checkpoint
        self.stem = ResBlock(in_channels, self.feature_channels[0])
        self.stages = nn.ModuleList(
            ResBlock(cin, cout, stride=2)
            for cin, cout in zip(self.feature_channels[:-1], self.feature_channels[1:])
        )
        self.proj = nn.Identity()

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Args:
            x (torch.Tensor): image tensor with shape (B, C_in, D, H, W)

        Returns:
            tuple[torch.Tensor, ...]: CNN tensors with shape (B, C_i, D/2**i, H/2**i, W/2**i),
                ordered from finest to coarsest scale
        """
        x = self.stem(x)
        features = [x]
        for stage in self.stages:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                x = checkpoint(stage, x, use_reentrant=False, preserve_rng_state=False)
            else:
                x = stage(x)
            features.append(x)
        return tuple(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): image tensor with shape (B, C_in, D, H, W)

        Returns:
            torch.Tensor: coarsest CNN features with shape (B, C_last, D/2**k, H/2**k, W/2**k),
                where k is the number of downsampling stages (three by default)
        """
        return self.proj(self.forward_features(x)[-1])
