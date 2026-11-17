from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import Any, Callable

import torch
from torch.nn import functional as F

from ._common import cuda_environment, package_version, validate_inputs, validate_output
from ._tuning import selected_options, tuning_metadata


@lru_cache(maxsize=1)
def _load_kernel() -> Callable[..., Any]:
    """
    Load the optional attention kernel or report the missing dependency.

    Returns:
        Callable[..., Any]: FlashAttention-4 public attention function.
    """
    try:
        return import_module("flash_attn.cute").flash_attn_func
    except (ImportError, AttributeError, OSError) as error:
        raise RuntimeError(
            "FlashAttention-4 requires the flash-attn-4 package and its CuTe DSL dependencies "
            "in the active environment; the requested backend cannot run"
        ) from error


def _padded_head_dim(head_dim: int) -> int:
    """
    Compute the head width required by the selected kernel.

    Args:
        head_dim (int): Original positive attention head dimension.

    Returns:
        int: Head dimension rounded up to the multiple of eight.
    """
    return ((head_dim + 7) // 8) * 8


def _prepare_qkv(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad head dimensions and expose the BNHD layout consumed by CuTe.

    Args:
        q (torch.Tensor): BF16 queries with shape (B, H, N, D).
        k (torch.Tensor): BF16 keys with the same shape and device as q.
        v (torch.Tensor): BF16 values with the same shape and device as q.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Query, key, and value in padded (B, N, H, D) layout.
    """
    padding = _padded_head_dim(q.shape[-1]) - q.shape[-1]
    if padding:
        q, k, v = (F.pad(t, (0, padding)) for t in (q, k, v))
    # CuTe consumes BNHD; strided sequence/head axes are supported.
    return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)


def _run_kernel(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    """
    Execute FlashAttention-4 attention while preserving the original head-dimension scale.

    Args:
        q (torch.Tensor): BF16 queries with shape (B, H, N, D).
        k (torch.Tensor): BF16 keys with the same shape and device as q.
        v (torch.Tensor): BF16 values with the same shape and device as q.
        softmax_scale (float): Attention scale derived from the unpadded head dimension.

    Returns:
        torch.Tensor: BF16 attention output with the same shape and device as q.
    """
    options = selected_options(q.shape, str(q.device))
    if options is None or options["implementation"] == "public_default":
        query, key, value = _prepare_qkv(q, k, v)
        result = _load_kernel()(
            query,
            key,
            value,
            softmax_scale=softmax_scale,
            causal=False,
            num_splits=1,
            return_lse=False,
        )
    else:
        # Preparation inside the measured opaque operation fixes kernel-facing strides across eager and compiled layouts.
        padding = _padded_head_dim(q.shape[-1]) - q.shape[-1]
        prepared = (tensor.transpose(1, 2) for tensor in (q, k, v))
        # Padding after transposing allocates contiguous BNHD storage without an intermediate BHND copy.
        query, key, value = (
            F.pad(tensor, (0, padding)) if padding else tensor.contiguous()
            for tensor in prepared
        )
        tile = options["tile_mn"]
        result = import_module("flash_attn.cute.interface")._flash_attn_fwd(
            query,
            key,
            value,
            softmax_scale=softmax_scale,
            causal=False,
            num_splits=1,
            return_lse=False,
            tile_mn=None if tile is None else tuple(tile),
        )
    # 4.0.0b30 returns (output, lse); some API revisions return output alone.
    output = result[0] if isinstance(result, tuple) else result
    head_dim = q.shape[-1]
    output = output[..., :head_dim].transpose(1, 2)
    return validate_output(output, q)


@torch.library.custom_op("flumina_fa4::attention", mutates_args=(), device_types="cuda")
def _compiled_kernel(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    """
    Expose the eager inference kernel as an opaque compiler operation.

    Args:
        q (torch.Tensor): BF16 queries with shape (B, H, N, D).
        k (torch.Tensor): BF16 keys with the same shape and device as q.
        v (torch.Tensor): BF16 values with the same shape and device as q.
        softmax_scale (float): Attention scale derived from the unpadded head dimension.

    Returns:
        torch.Tensor: BF16 attention output with the same shape and device as q.
    """
    return _run_kernel(q, k, v, softmax_scale)


@_compiled_kernel.register_fake
def _fake_kernel(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    """
    Describe the eager kernel output shape, dtype, device, and strides for compilation.

    Args:
        q (torch.Tensor): BF16 queries with shape (B, H, N, D).
        k (torch.Tensor): BF16 keys with the same shape and device as q.
        v (torch.Tensor): BF16 values with the same shape and device as q.
        softmax_scale (float): Attention scale derived from the unpadded head dimension.

    Returns:
        torch.Tensor: Uninitialized output matching the runtime kernel layout without running attention.
    """
    batch, heads, length, head_dim = q.shape
    output = torch.empty(
        (batch, length, heads, _padded_head_dim(head_dim)),
        dtype=q.dtype,
        device=q.device,
    )
    return output[..., :head_dim].transpose(1, 2)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Apply FlashAttention-4 attention under the shared BF16 inference contract.

    Args:
        q (torch.Tensor): BF16 queries with shape (B, H, N, D).
        k (torch.Tensor): BF16 keys with the same shape and device as q.
        v (torch.Tensor): BF16 values with the same shape and device as q.
        dropout_p (float): Dropout probability; inference requires zero.
        is_causal (bool): Causal masking flag; inference requires False.

    Returns:
        torch.Tensor: BF16 attention output with the same shape and device as q.
    """
    validate_inputs(q, k, v, dropout_p, is_causal)
    return validate_output(_compiled_kernel(q, k, v, q.shape[-1] ** -0.5), q)


def metadata() -> dict[str, Any]:
    """
    Describe the attention backend, arithmetic, padding, and compiler integration.

    Returns:
        dict[str, Any]: Backend configuration and optional dependency or tuning metadata.
    """
    tuned = tuning_metadata()
    return {
        "backend": "FlashAttention-4",
        "implementation": (
            "per-shape measured choice of flash_attn.cute.interface._flash_attn_fwd or flash_attn.cute.flash_attn_func"
            if tuned["tuning_enabled"]
            else "flash_attn.cute.flash_attn_func"
        ),
        "kernel_selection": "FlashAttention-4 CuTe BF16, num_splits=1; choices recorded in tuning metadata",
        "input_dtype": "bfloat16",
        "output_dtype": "bfloat16",
        "attention_arithmetic": "BF16 attention; no FP8/FP4 quantization",
        "layout": "BHND converted to BNHD for CuTe",
        "causal": False,
        "dropout_p": 0.0,
        "softmax_scale": "explicit original_head_dim ** -0.5",
        "head_dimension_padding": {"48": 48, "66": 72, "72": 72},
        "compile_integration": "inference-only torch.library.custom_op with fake implementation",
        "custom_op": "flumina_fa4::attention",
        "optional_package": "flash-attn-4",
        "package_version": package_version("flash-attn-4"),
    } | tuned


def preflight(device: str | torch.device = "cuda:0") -> dict[str, Any]:
    """
    Verify hardware and dependency support for the selected attention backend.

    Args:
        device (str | torch.device): CUDA device used to inspect hardware support.

    Returns:
        dict[str, Any]: Backend metadata combined with the inspected CUDA environment.
    """
    result = metadata() | cuda_environment(device)
    capability = tuple(result["compute_capability"])
    if capability[0] not in (9, 10, 11, 12):
        raise RuntimeError(
            f"FlashAttention-4 CuTe requires supported Hopper/Blackwell hardware, got SM{capability}"
        )
    _load_kernel()
    result["kernel_module_file"] = import_module("flash_attn.cute").__file__
    result.update(tuning_metadata(str(torch.device(device))))
    return result
