from __future__ import annotations

from dataclasses import dataclass
import math
from typing import cast

import torch


SpatialShape = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class RotaryCache:
    """
    Rotary embedding table cached for one spatial grid.

    Args:
        spatial_shape (SpatialShape): token grid dimensions (d, h, w)
        dtype (torch.dtype): dtype of the cached table
        device (torch.device): device holding the cached table
        embed (torch.Tensor): sine and cosine table with shape (d*h*w, 3*rope_dim)
    """

    spatial_shape: SpatialShape
    dtype: torch.dtype
    device: torch.device
    embed: torch.Tensor


def axial_embed(
    spatial_shape: SpatialShape, rope_dim: int, rope_base: float
) -> torch.Tensor:
    """
    Build rotary sine and cosine tables for a flattened 3D grid.

    Args:
        spatial_shape (SpatialShape): token grid dimensions (d, h, w)
        rope_dim (int): positive multiple of four; the table encodes a head dimension of 3*rope_dim/2
        rope_base (float): geometric frequency base

    Returns:
        torch.Tensor: float32 table with shape (d*h*w, 3*rope_dim), with all sine values followed by all cosine values
    """
    num_bands = rope_dim // 4
    exponent = torch.arange(num_bands, dtype=torch.float32) / num_bands
    bands = 1.0 / torch.pow(torch.tensor(float(rope_base)), exponent)
    coordinates = [torch.arange(size, dtype=torch.float32) for size in spatial_shape]
    grid = torch.stack(torch.meshgrid(*coordinates, indexing="ij"), dim=-1)
    phase = grid.unsqueeze(-1) * bands
    sequence_length = math.prod(spatial_shape)
    sine = phase.sin().reshape(sequence_length, -1).repeat_interleave(2, -1)
    cosine = phase.cos().reshape(sequence_length, -1).repeat_interleave(2, -1)
    return torch.cat((sine, cosine), dim=-1)


class RoPE3D:
    """Cached axial rotary embeddings for spatial attention tokens."""

    def __init__(self, rope_dim: int, rope_base: float = 10000.0) -> None:
        """
        Args:
            rope_dim (int): positive multiple of four equal to two thirds of the attention head dimension
            rope_base (float): geometric frequency base
        """
        if rope_dim <= 0 or rope_dim % 4:
            raise ValueError("rope_dim must be a positive multiple of four")
        self.rope_dim = rope_dim
        self.rope_base = float(rope_base)
        self._cache: RotaryCache | None = None

    def get_embed(
        self,
        spatial_shape: SpatialShape,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Retrieve or build a rotary table for the requested grid.

        Args:
            spatial_shape (SpatialShape): token grid dimensions (d, h, w)
            device (torch.device | str): destination device
            dtype (torch.dtype): destination floating-point dtype

        Returns:
            torch.Tensor: sine and cosine table with shape (d*h*w, 3*rope_dim) on the requested device and dtype
        """
        shape = cast(SpatialShape, tuple(spatial_shape))
        target_device = torch.device(device)
        cache = self._cache
        if cache is None or cache.spatial_shape != shape:
            embed = axial_embed(shape, self.rope_dim, self.rope_base).to(
                device=target_device, dtype=dtype
            )
            self._cache = RotaryCache(shape, dtype, target_device, embed)
        elif cache.device != target_device or cache.dtype != dtype:
            embed = cache.embed.to(device=target_device, dtype=dtype)
            self._cache = RotaryCache(shape, dtype, target_device, embed)
        return cast(RotaryCache, self._cache).embed

    @staticmethod
    def prepend_register_tokens(
        rope_embed: torch.Tensor,
        num_register_tokens: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """
        Prepend zero placeholders for tokens excluded from rotary embedding.

        Args:
            rope_embed (torch.Tensor): spatial rotary table with shape (N, 2*head_dim)
            num_register_tokens (int): number R of prefix rows; zero returns the input unchanged
            device (torch.device | str | None): destination device for a nonempty prefix; None keeps the input device
            dtype (torch.dtype | None): destination dtype for a nonempty prefix; None keeps the input dtype

        Returns:
            torch.Tensor: table with shape (R + N, 2*head_dim); zero prefix rows are skipped by apply_rope
        """
        if num_register_tokens == 0:
            return rope_embed
        target_device = rope_embed.device if device is None else torch.device(device)
        target_dtype = rope_embed.dtype if dtype is None else dtype
        embed = rope_embed.to(device=target_device, dtype=target_dtype)
        prefix = torch.zeros(
            num_register_tokens,
            embed.shape[-1],
            device=target_device,
            dtype=target_dtype,
        )
        return torch.cat((prefix, embed), dim=0)


def rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate adjacent channel pairs by a quarter turn.

    Args:
        x (torch.Tensor): input tensor with shape (..., head_dim), where head_dim is even

    Returns:
        torch.Tensor: rotated tensor with shape (..., head_dim)
    """
    paired = x.reshape(*x.shape[:-1], -1, 2)
    rotated = torch.stack((-paired[..., 1], paired[..., 0]), dim=-1)
    return rotated.flatten(start_dim=-2)


def apply_rotary_embed(x: torch.Tensor, rope_embed: torch.Tensor) -> torch.Tensor:
    """
    Apply a rotary table to queries or keys.

    Args:
        x (torch.Tensor): query or key tensor with shape (B, num_heads, N, head_dim)
        rope_embed (torch.Tensor): sine and cosine table with shape (N, 2*head_dim)

    Returns:
        torch.Tensor: rotated query or key tensor with shape (B, num_heads, N, head_dim)
    """
    sine, cosine = rope_embed.to(device=x.device, dtype=x.dtype).chunk(2, dim=-1)
    return x * cosine + rotate_pairs(x) * sine


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    rope_embed: torch.Tensor,
    num_register_tokens: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rotate spatial queries and keys while preserving register tokens.

    Args:
        q (torch.Tensor): query tensor with shape (B, num_heads, N, head_dim), including any register prefix
        k (torch.Tensor): key tensor with shape (B, num_heads, N, head_dim)
        rope_embed (torch.Tensor): rotary table with shape (N, 2*head_dim), including any register prefix rows
        num_register_tokens (int): number of leading tokens excluded from rotation

    Returns:
        tuple[torch.Tensor, torch.Tensor]: rotated queries and keys, each with shape (B, num_heads, N, head_dim)
    """
    if num_register_tokens == 0:
        return apply_rotary_embed(q, rope_embed), apply_rotary_embed(k, rope_embed)
    return (
        torch.cat(
            (
                q[..., :num_register_tokens, :],
                apply_rotary_embed(
                    q[..., num_register_tokens:, :], rope_embed[num_register_tokens:]
                ),
            ),
            dim=-2,
        ),
        torch.cat(
            (
                k[..., :num_register_tokens, :],
                apply_rotary_embed(
                    k[..., num_register_tokens:, :], rope_embed[num_register_tokens:]
                ),
            ),
            dim=-2,
        ),
    )
