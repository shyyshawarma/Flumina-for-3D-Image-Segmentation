from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

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
    Apply SDPA-Flash attention under the shared BF16 inference contract.

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
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        )
    return validate_output(output, q)


def metadata() -> dict[str, Any]:
    """
    Describe the attention backend, arithmetic, padding, and compiler integration.

    Returns:
        dict[str, Any]: Backend configuration and optional dependency or tuning metadata.
    """
    return {
        "backend": "SDPA-Flash",
        "implementation": "torch.nn.functional.scaled_dot_product_attention",
        "kernel_selection": "SDPBackend.FLASH_ATTENTION only",
        "input_dtype": "bfloat16",
        "output_dtype": "bfloat16",
        "attention_arithmetic": "BF16 Q/K/V; PyTorch Flash SDPA arithmetic",
        "layout": "BHND",
        "causal": False,
        "dropout_p": 0.0,
        "softmax_scale": "original_head_dim ** -0.5",
        "head_dimension_padding": "PyTorch internally pads to a multiple of 8 with original scale",
        "compile_integration": "native PyTorch SDPA with sdpa_kernel context",
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
    if not torch.backends.cuda.is_flash_attention_available():
        raise RuntimeError("This PyTorch build has no Flash SDPA kernel")
    return result
