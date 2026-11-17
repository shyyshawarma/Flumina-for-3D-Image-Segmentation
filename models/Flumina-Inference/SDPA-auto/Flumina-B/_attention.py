from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from ._common import cuda_environment, validate_inputs, validate_output


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Apply SDPA-auto attention under the shared BF16 inference contract.

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
    return validate_output(
        F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        ),
        q,
    )


def metadata() -> dict[str, Any]:
    """
    Describe the attention backend, arithmetic, padding, and compiler integration.

    Returns:
        dict[str, Any]: Backend configuration and optional dependency or tuning metadata.
    """
    return {
        "backend": "SDPA-auto",
        "implementation": "torch.nn.functional.scaled_dot_product_attention",
        "kernel_selection": "PyTorch automatic SDPA dispatch",
        "input_dtype": "bfloat16",
        "output_dtype": "bfloat16",
        "attention_arithmetic": "BF16 Q/K/V; PyTorch-selected SDPA arithmetic",
        "layout": "BHND",
        "causal": False,
        "dropout_p": 0.0,
        "softmax_scale": "original_head_dim ** -0.5",
        "head_dimension_padding": "PyTorch internal backend-specific padding",
        "compile_integration": "native PyTorch SDPA",
        "optional_package": None,
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
    result["enabled_sdpa_backends"] = {
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
        "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
    }
    if not any(result["enabled_sdpa_backends"].values()):
        raise RuntimeError("All PyTorch SDPA backends are disabled")
    return result
