from __future__ import annotations

from importlib import metadata as package_metadata
from typing import Any

import torch


def package_version(name: str) -> str | None:
    """
    Read the installed version of an optional distribution.

    Args:
        name (str): Distribution name recognized by importlib.metadata.

    Returns:
        str | None: Installed version, or None when the distribution is absent.
    """
    try:
        return package_metadata.version(name)
    except package_metadata.PackageNotFoundError:
        return None


def validate_inputs(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout_p: float, is_causal: bool
) -> None:
    """
    Validate the shared BF16, noncausal inference contract without changing the inputs.

    Args:
        q (torch.Tensor): BF16 queries with shape (B, H, N, D).
        k (torch.Tensor): BF16 keys with the same shape and device as q.
        v (torch.Tensor): BF16 values with the same shape and device as q.
        dropout_p (float): Dropout probability; inference requires zero.
        is_causal (bool): Causal masking flag; inference requires False.

    Returns:
        None: Invalid inputs raise an exception.
    """
    if dropout_p != 0.0 or is_causal:
        raise ValueError(
            "These inference adapters require dropout_p=0 and noncausal attention"
        )
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("Flumina attention requires equal Q/K/V shapes (B, H, N, D)")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
        raise TypeError(
            "Attention Q/K/V must be BF16; run the FP32 model under BF16 autocast"
        )
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q/K/V must share one device")
    if q.shape[-1] <= 0 or q.shape[-1] > 128:
        raise ValueError(
            "The Flumina comparison supports attention head dimensions 1 through 128"
        )
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        raise RuntimeError(
            "The isolated attention adapters require no_grad or inference_mode"
        )


def validate_output(output: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """
    Verify that an attention kernel preserves the required output contract.

    Args:
        output (torch.Tensor): Kernel output to validate.
        query (torch.Tensor): Reference query defining the output shape and device.

    Returns:
        torch.Tensor: The original output tensor after validation.
    """
    if not isinstance(output, torch.Tensor):
        raise TypeError("Attention kernel did not return a Tensor")
    if output.shape != query.shape or output.device != query.device:
        raise RuntimeError(
            "Attention kernel changed the requested output shape or device"
        )
    if output.dtype != torch.bfloat16:
        raise RuntimeError(
            "Attention kernel must return BF16 without a precision conversion"
        )
    return output


def cuda_environment(device: str | torch.device) -> dict[str, Any]:
    """
    Inspect the CUDA environment and require native BF16 hardware support.

    Args:
        device (str | torch.device): CUDA device used to inspect hardware support.

    Returns:
        dict[str, Any]: Device identifier, GPU name, compute capability, and PyTorch/CUDA versions.
    """
    requested = torch.device(device)
    if requested.type != "cuda":
        raise RuntimeError("The attention benchmark requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the active environment")
    index = torch.cuda.current_device() if requested.index is None else requested.index
    properties = torch.cuda.get_device_properties(index)
    if properties.major < 8:
        raise RuntimeError(
            "The attention benchmark requires native BF16 CUDA support (SM80+)"
        )
    return {
        "device": f"cuda:{index}",
        "gpu_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
    }
