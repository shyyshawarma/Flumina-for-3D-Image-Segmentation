from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from torch.nn import functional as F
import triton
import triton.language as tl


@triton.jit
def _resize2x_bf16_kernel(
    X,
    Y,
    N: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    S4: tl.constexpr,
    CHANNELS_LAST_OUT: tl.constexpr,
    LERP: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_P: tl.constexpr,
    WARPS: tl.constexpr,
):
    """
    Interpolate a 3D feature map at exactly twice its spatial resolution and store BF16 convolution inputs.

    Args:
        X (pointer): float32 input storage with logical shape (N, C, D, H, W)
        Y (pointer): output storage with logical shape (N, C, 2*D, 2*H, 2*W)
        N (tl.constexpr): batch size
        C (tl.constexpr): channel count
        D (tl.constexpr): input depth
        H (tl.constexpr): input height
        W (tl.constexpr): input width
        S0 (tl.constexpr): input batch stride in elements
        S1 (tl.constexpr): input channel stride in elements
        S2 (tl.constexpr): input depth stride in elements
        S3 (tl.constexpr): input height stride in elements
        S4 (tl.constexpr): input width stride in elements
        CHANNELS_LAST_OUT (tl.constexpr): store channels contiguously when True
        LERP (tl.constexpr): interpolate using differences when True, otherwise use weighted sums
        BLOCK_C (tl.constexpr): channels processed by each program
        BLOCK_P (tl.constexpr): output spatial positions processed by each program
        WARPS (tl.constexpr): launch warp count, retained as a specialization parameter

    Returns:
        None: writes the interpolated BF16 values into Y
    """
    # Axes are [channel, output spatial position]. The compiler exchanges BF16 values through shared memory to coalesce I/O.
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    n = tl.program_id(2)
    spatial: tl.constexpr = 8 * D * H * W
    valid = (c[:, None] < C) & (p[None, :] < spatial)
    oz = p // (4 * H * W)
    oy = (p // (2 * W)) % (2 * H)
    ox = p % (2 * W)
    z0 = tl.maximum(oz - 1, 0) // 2
    y0 = tl.maximum(oy - 1, 0) // 2
    x0 = tl.maximum(ox - 1, 0) // 2
    z1 = tl.minimum(z0 + 1, D - 1)
    y1 = tl.minimum(y0 + 1, H - 1)
    x1 = tl.minimum(x0 + 1, W - 1)
    wz = tl.where(oz == 0, 0.0, tl.where(oz % 2 == 0, 0.75, 0.25))[None, :]
    wy = tl.where(oy == 0, 0.0, tl.where(oy % 2 == 0, 0.75, 0.25))[None, :]
    wx = tl.where(ox == 0, 0.0, tl.where(ox % 2 == 0, 0.75, 0.25))[None, :]
    base = n * S0 + c[:, None] * S1
    v000 = tl.load(X + base + (z0 * S2 + y0 * S3 + x0 * S4)[None, :], valid, 0)
    v001 = tl.load(X + base + (z0 * S2 + y0 * S3 + x1 * S4)[None, :], valid, 0)
    v010 = tl.load(X + base + (z0 * S2 + y1 * S3 + x0 * S4)[None, :], valid, 0)
    v011 = tl.load(X + base + (z0 * S2 + y1 * S3 + x1 * S4)[None, :], valid, 0)
    v100 = tl.load(X + base + (z1 * S2 + y0 * S3 + x0 * S4)[None, :], valid, 0)
    v101 = tl.load(X + base + (z1 * S2 + y0 * S3 + x1 * S4)[None, :], valid, 0)
    v110 = tl.load(X + base + (z1 * S2 + y1 * S3 + x0 * S4)[None, :], valid, 0)
    v111 = tl.load(X + base + (z1 * S2 + y1 * S3 + x1 * S4)[None, :], valid, 0)
    if LERP:
        a00 = v000 + (v001 - v000) * wx
        a01 = v010 + (v011 - v010) * wx
        a10 = v100 + (v101 - v100) * wx
        a11 = v110 + (v111 - v110) * wx
        a0 = a00 + (a01 - a00) * wy
        a1 = a10 + (a11 - a10) * wy
        result = a0 + (a1 - a0) * wz
    else:
        a00 = (1.0 - wx) * v000 + wx * v001
        a01 = (1.0 - wx) * v010 + wx * v011
        a10 = (1.0 - wx) * v100 + wx * v101
        a11 = (1.0 - wx) * v110 + wx * v111
        a0 = (1.0 - wy) * a00 + wy * a01
        a1 = (1.0 - wy) * a10 + wy * a11
        result = (1.0 - wz) * a0 + wz * a1
    # This is the original immediate Conv3d's autocast rounding point.
    result_bf16 = result.to(tl.bfloat16)
    if CHANNELS_LAST_OUT:
        tl.store(Y + n * C * spatial + p[None, :] * C + c[:, None], result_bf16, valid)
    else:
        out_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        out_p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
        out_mask = (out_c[:, None] < C) & (out_p[None, :] < spatial)
        tl.store(
            Y + (n * C + out_c[:, None]) * spatial + out_p[None, :],
            result_bf16,
            out_mask,
        )


def resize2x_for_bf16_conv(
    x: torch.Tensor,
    size: tuple[int, int, int],
    *,
    block_c: int = 32,
    block_p: int = 64,
    num_warps: int = 4,
    channels_last_out: bool = False,
    lerp: bool = True,
) -> torch.Tensor:
    """
    Fuse exact 2x interpolation with the input cast of the immediately following BF16-autocast Conv3d.

    Other inputs retain PyTorch interpolation, including the first BF16-input decoder stage. This is not a general
    interpolate replacement: the fast path returns BF16 because its caller immediately applies a BF16-autocast convolution.
    Positive strides and storage offsets are supported. Inputs requiring backward use PyTorch. Difference-based interpolation
    can overflow for opposite-sign extreme float32 values; preceding convolutions, normalization, and GELU are not reordered.

    Args:
        x (torch.Tensor): input features with shape (N, C, D, H, W); the fast path requires CUDA float32 and BF16 autocast
        size (tuple[int, int, int]): output spatial shape; the fast path requires (2*D, 2*H, 2*W)
        block_c (int): channels processed by each Triton program
        block_p (int): output spatial positions processed by each Triton program
        num_warps (int): warps assigned to each Triton program
        channels_last_out (bool): use channels-last output strides on the fast path
        lerp (bool): use difference-based interpolation rather than weighted sums on the fast path

    Returns:
        torch.Tensor: resized features with shape (N, C, *size); BF16 on the fast path, otherwise the input dtype
    """
    if (
        x.device.type != "cuda"
        or x.ndim != 5
        or x.dtype != torch.float32
        or not torch.is_autocast_enabled("cuda")
        or torch.get_autocast_dtype("cuda") != torch.bfloat16
        or tuple(size) != tuple(2 * d for d in x.shape[2:])
        or any(s <= 0 for s in x.stride())
        or (torch.is_grad_enabled() and x.requires_grad)
    ):
        return F.interpolate(x, size=size, mode="trilinear", align_corners=False)
    n, c, d, h, w = x.shape
    od, oh, ow = size
    spatial = od * oh * ow
    if channels_last_out:
        strides = (c * spatial, 1, oh * ow * c, ow * c, c)
    else:
        strides = (c * spatial, spatial, oh * ow, ow, 1)
    # Explicit strides prevent Inductor from copying an undefined torch.empty(memory_format=...) buffer before the full write.
    y = torch.empty_strided(
        (n, c, od, oh, ow), strides, device=x.device, dtype=torch.bfloat16
    )
    cast(
        Callable[..., None],
        _resize2x_bf16_kernel[
            (triton.cdiv(spatial, block_p), triton.cdiv(c, block_c), n)
        ],
    )(
        x,
        y,
        n,
        c,
        d,
        h,
        w,
        *x.stride(),
        channels_last_out,
        lerp,
        block_c,
        block_p,
        num_warps,
        num_warps=num_warps,
        enable_fp_fusion=True,
    )
    return y
