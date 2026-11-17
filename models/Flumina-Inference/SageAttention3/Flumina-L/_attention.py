from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from types import ModuleType
from typing import Any

import torch
from torch.nn import functional as F

from ._common import cuda_environment, package_version, validate_inputs, validate_output


@lru_cache(maxsize=1)
def _load_api() -> ModuleType:
    """
    Load the SageAttention3 preprocessing, quantization, and native FP4 API.

    Returns:
        ModuleType: Imported API module with all required kernel entry points.
    """
    try:
        api = import_module("sageattn3.api")
        for name in (
            "preprocess_qkv",
            "scale_and_quant_fp4",
            "scale_and_quant_fp4_permute",
            "scale_and_quant_fp4_transpose",
            "fp4attn_cuda",
        ):
            getattr(api, name)
        getattr(api.fp4attn_cuda, "fwd")
        return api
    except (ImportError, AttributeError, OSError) as error:
        raise RuntimeError(
            "SageAttention3 requires sageattn3 and its FP4 CUDA extensions built for SM120/SM121 "
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


def _prepare_qkv(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad Q/K/V and allocate private key storage for in-place mean subtraction.

    Args:
        q (torch.Tensor): BF16 queries with shape (B, H, N, D).
        k (torch.Tensor): BF16 keys with the same shape and device as q.
        v (torch.Tensor): BF16 values with the same shape and device as q.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Query, key, and value in padded BHND layout; K never aliases input.
    """
    padding = _padded_head_dim(q.shape[-1]) - q.shape[-1]
    if padding:
        # Padding allocates private K storage because upstream preprocessing subtracts its sequence mean in place.
        q, k, v = (F.pad(t, (0, padding)) for t in (q, k, v))
    else:
        k = k.clone(memory_format=torch.contiguous_format)
    return q, k, v


def _run_kernel(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    """
    Execute SageAttention3 attention while preserving the original head-dimension scale.

    Args:
        q (torch.Tensor): BF16 queries with shape (B, H, N, D).
        k (torch.Tensor): BF16 keys with the same shape and device as q.
        v (torch.Tensor): BF16 values with the same shape and device as q.
        softmax_scale (float): Attention scale derived from the unpadded head dimension.

    Returns:
        torch.Tensor: BF16 attention output with the same shape and device as q.
    """
    api = _load_api()
    query, key, value = _prepare_qkv(q, k, v)
    query, key, value, delta_s = api.preprocess_qkv(
        query,
        key,
        value,
        per_block_mean=True,
    )
    query_fp4, query_scale = api.scale_and_quant_fp4(query)
    key_fp4, key_scale = api.scale_and_quant_fp4_permute(key)
    value_fp4, value_scale = api.scale_and_quant_fp4_transpose(value)
    # Pass the original scale to the native kernel: the public wrapper ignores sm_scale and uses padded_head_dim.
    # Rescaling BF16 Q to compensate would add rounding error.
    result = api.fp4attn_cuda.fwd(
        query_fp4,
        key_fp4,
        value_fp4,
        query_scale,
        key_scale,
        value_scale,
        delta_s,
        k.shape[2],
        None,
        softmax_scale,
        False,
        True,
        True,
    )
    length, head_dim = q.shape[2], q.shape[-1]
    output = result[0][:, :, :length, :head_dim]
    return validate_output(output, q)


@torch.library.custom_op(
    "flumina_sage3::attention", mutates_args=(), device_types="cuda"
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
    batch, heads, length, head_dim = q.shape
    output = torch.empty(
        (batch, heads, ((length + 127) // 128) * 128, _padded_head_dim(head_dim)),
        dtype=q.dtype,
        device=q.device,
    )
    return output[:, :, :length, :head_dim]


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Apply SageAttention3 attention under the shared BF16 inference contract.

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
        "backend": "SageAttention3",
        "implementation": "sageattn3.api preprocessing/FP4 quantization + fp4attn_cuda.fwd",
        "kernel_selection": "SageAttention3 Blackwell FP4, per_block_mean=True",
        "input_dtype": "bfloat16",
        "output_dtype": "bfloat16",
        "attention_arithmetic": "FP4 Q/K/V with FP8 block scales and mean correction",
        "layout": "BHND",
        "causal": False,
        "dropout_p": 0.0,
        "softmax_scale": "explicit original_head_dim ** -0.5 passed to native FP4 kernel",
        "head_dimension_padding": {"48": 64, "66": 128, "72": 128},
        "sequence_padding": "upstream pads to 128, native kernel receives original key length",
        "key_input_mutation": "upstream mean subtraction operates on private padded/copied storage",
        "compile_integration": "inference-only torch.library.custom_op with fake implementation",
        "custom_op": "flumina_sage3::attention",
        "optional_package": "sageattn3",
        "package_version": package_version("sageattn3"),
        "upstream_api_reference": (
            "https://github.com/thu-ml/SageAttention/blob/"
            "d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/"
            "sageattention3_blackwell/sageattn3/api.py"
        ),
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
    if capability not in ((12, 0), (12, 1)):
        raise RuntimeError(
            f"The SageAttention3 native kernel supports SM120/SM121, got {capability}"
        )
    api = _load_api()
    result["kernel_module_file"] = api.__file__
    return result
