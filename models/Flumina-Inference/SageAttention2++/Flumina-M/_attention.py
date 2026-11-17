from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import Any, Callable

import torch

from ._common import cuda_environment, package_version, validate_inputs, validate_output


@lru_cache(maxsize=1)
def _load_kernel() -> Callable[..., torch.Tensor]:
    """
    Load the optional attention kernel or report the missing dependency.

    Returns:
        Callable[..., torch.Tensor]: SageAttention2++ CUDA attention function.
    """
    try:
        core = import_module("sageattention.core")
        if not core.SM89_ENABLED:
            raise ImportError(
                "The SageAttention SM89/SM120 kernel wrapper is unavailable"
            )
        return core.sageattn_qk_int8_pv_fp8_cuda
    except (ImportError, AttributeError, OSError) as error:
        raise RuntimeError(
            "SageAttention2++ requires the SageAttention CUDA extensions built for this GPU "
            "in the active environment; the requested backend cannot run"
        ) from error


def _padded_head_dim(head_dim: int) -> int:
    """
    Compute the head width required by the selected kernel.

    Args:
        head_dim (int): Original positive attention head dimension.

    Returns:
        int: Head dimension rounded up to the supported width of 64 or 128.
    """
    return 64 if head_dim <= 64 else 128


def _run_kernel(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    # This FP8 PV path selects SageAttention2++ accumulation; upstream pads/truncates heads with the original QK scale.
    """
    Execute SageAttention2++ attention while preserving the original head-dimension scale.

    Args:
        q (torch.Tensor): BF16 queries with shape (B, H, N, D).
        k (torch.Tensor): BF16 keys with the same shape and device as q.
        v (torch.Tensor): BF16 values with the same shape and device as q.
        softmax_scale (float): Attention scale derived from the unpadded head dimension.

    Returns:
        torch.Tensor: BF16 attention output with the same shape and device as q.
    """
    output = _load_kernel()(
        q,
        k,
        v,
        tensor_layout="HND",
        is_causal=False,
        qk_quant_gran="per_warp",
        sm_scale=softmax_scale,
        pv_accum_dtype="fp32+fp16",
        smooth_k=True,
        smooth_v=False,
        return_lse=False,
    )
    return validate_output(output, q)


@torch.library.custom_op(
    "flumina_sage2pp::attention", mutates_args=(), device_types="cuda"
)
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
    output = torch.empty(
        (*q.shape[:-1], _padded_head_dim(q.shape[-1])),
        dtype=q.dtype,
        device=q.device,
    )
    head_dim = q.shape[-1]
    return output[..., :head_dim]


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Apply SageAttention2++ attention under the shared BF16 inference contract.

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
    return {
        "backend": "SageAttention2++",
        "implementation": "sageattention.core.sageattn_qk_int8_pv_fp8_cuda",
        "kernel_selection": "qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf",
        "input_dtype": "bfloat16",
        "output_dtype": "bfloat16",
        "attention_arithmetic": "INT8 Q/K, FP8 P/V, fp32+fp16 accumulation",
        "qk_quant_gran": "per_warp",
        "pv_accum_dtype": "fp32+fp16",
        "v_quant_scale_max": 2.25,
        "smooth_k": True,
        "smooth_v": False,
        "layout": "BHND",
        "causal": False,
        "dropout_p": 0.0,
        "softmax_scale": "explicit original_head_dim ** -0.5",
        "head_dimension_padding": {"48": 64, "66": 128, "72": 128},
        "compile_integration": "inference-only torch.library.custom_op with fake implementation",
        "custom_op": "flumina_sage2pp::attention",
        "optional_package": "sageattention",
        "package_version": package_version("sageattention"),
    }


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
    if capability not in ((8, 9), (12, 0), (12, 1)):
        raise RuntimeError(
            f"This explicit SageAttention2++ FP8 PV kernel supports SM89/SM120/SM121, got {capability}"
        )
    _load_kernel()
    result["kernel_module_file"] = import_module("sageattention.core").__file__
    return result
