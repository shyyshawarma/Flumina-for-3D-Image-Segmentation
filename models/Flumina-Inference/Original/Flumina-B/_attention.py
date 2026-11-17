from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from ._common import cuda_environment, validate_inputs, validate_output


def _run_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # Force Math SDPA; the opaque operation prevents Inductor from replacing it with fused attention.
    """
    Execute Original attention while preserving the original head-dimension scale.

    Args:
        q (torch.Tensor): BF16 queries with shape (B, H, N, D).
        k (torch.Tensor): BF16 keys with the same shape and device as q.
        v (torch.Tensor): BF16 values with the same shape and device as q.

    Returns:
        torch.Tensor: BF16 attention output with the same shape and device as q.
    """
    with sdpa_kernel(SDPBackend.MATH):
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        )
    return validate_output(output, q)


@torch.library.custom_op(
    "flumina_original::attention",
    mutates_args=(),
    device_types=("cpu", "cuda"),
)
def _compiled_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Expose the eager inference kernel as an opaque compiler operation.

    Args:
        q (torch.Tensor): BF16 queries with shape (B, H, N, D).
        k (torch.Tensor): BF16 keys with the same shape and device as q.
        v (torch.Tensor): BF16 values with the same shape and device as q.

    Returns:
        torch.Tensor: BF16 attention output with the same shape and device as q.
    """
    return _run_kernel(q, k, v)


@_compiled_kernel.register_fake
def _fake_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Describe the eager kernel output shape, dtype, device, and strides for compilation.

    Args:
        q (torch.Tensor): BF16 queries with shape (B, H, N, D).
        k (torch.Tensor): BF16 keys with the same shape and device as q.
        v (torch.Tensor): BF16 values with the same shape and device as q.

    Returns:
        torch.Tensor: Uninitialized output matching the runtime kernel layout without running attention.
    """
    return torch.empty(q.shape, dtype=q.dtype, device=q.device)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Apply Original attention under the shared BF16 inference contract.

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
    return validate_output(_compiled_kernel(q, k, v), q)


def metadata() -> dict[str, Any]:
    """
    Describe the attention backend, arithmetic, padding, and compiler integration.

    Returns:
        dict[str, Any]: Backend configuration and optional dependency or tuning metadata.
    """
    return {
        "backend": "Original",
        "implementation": "torch.nn.functional.scaled_dot_product_attention",
        "kernel_selection": "SDPBackend.MATH only; fused attention backends disabled",
        "baseline_definition": "Original source SDPA operation with Math dispatch forced",
        "historical_manual_attention_claim": False,
        "input_dtype": "bfloat16",
        "output_dtype": "bfloat16",
        "attention_arithmetic": "PyTorch Math SDPA; FP32 intermediates for BF16 input by default",
        "layout": "BHND",
        "causal": False,
        "dropout_p": 0.0,
        "softmax_scale": "original_head_dim ** -0.5 (original SDPA default)",
        "head_dimension_padding": "none",
        "compile_integration": "opaque inference-only custom operation; Math attention executes eagerly",
        "custom_op": "flumina_original::attention",
        "optional_package": None,
        "runtime_policy": "matched BF16 AMP + outer fullgraph compile",
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
    reduced_math = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
    if reduced_math:
        raise RuntimeError(
            "Original requires PyTorch Math SDPA's default FP32 intermediates; "
            "disable allow_fp16_bf16_reduction_math_sdp before running this baseline"
        )
    result["fp16_bf16_reduction_math_sdp_allowed"] = reduced_math
    result["math_dispatch_forced_inside_custom_op"] = True
    return result
