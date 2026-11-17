from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn


def _supports_cuda_bf16(device: torch.device) -> bool:
    """
    Check CUDA BF16 support for inference autocast.

    Args:
        device (torch.device): Inference device.

    Returns:
        bool: Whether CUDA is available and reports BF16 support.
    """

    return bool(
        device.type == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )


def _spatial_tuple(value: int | Sequence[int], *, name: str) -> tuple[int, int, int]:
    """
    Normalize a window size to three positive dimensions.

    Args:
        value (int | Sequence[int]): Isotropic size or sizes in (D, H, W) order.
        name (str): Parameter name used in validation errors.

    Returns:
        tuple[int, int, int]: Window dimensions in (D, H, W) order.
    """
    values = (
        (int(value),) * 3 if isinstance(value, int) else tuple(int(v) for v in value)
    )
    if len(values) != 3 or any(v <= 0 for v in values):
        raise ValueError(f"{name} must contain three positive values, got {values}")
    return values


def _window_starts(size: int, window: int, stride: int) -> tuple[int, ...]:
    """
    Place windows along one axis, including the final boundary.

    Args:
        size (int): Padded axis length, at least as large as the window.
        window (int): Window length.
        stride (int): Positive distance between window starts.

    Returns:
        tuple[int, ...]: Start indices ending at size - window.
    """
    starts = list(range(0, size - window + 1, stride))
    final = size - window
    if not starts or starts[-1] != final:
        starts.append(final)
    return tuple(starts)


@lru_cache(maxsize=32)
def _cpu_gaussian_importance(
    crop: tuple[int, int, int], sigma_scale: float
) -> torch.Tensor:
    """
    Build cached Gaussian weights for overlapping windows.

    Args:
        crop (tuple[int, int, int]): Window dimensions in (D, H, W) order.
        sigma_scale (float): Standard deviation relative to each axis length.

    Returns:
        torch.Tensor: Positive float32 CPU weights shaped (1, 1, D, H, W), normalized to a maximum of one.
    """
    axes = []
    for size in crop:
        coordinates = torch.arange(size, dtype=torch.float32)
        center = (size - 1) / 2.0
        sigma = max(float(size) * sigma_scale, 1e-6)
        axes.append(torch.exp(-0.5 * ((coordinates - center) / sigma) ** 2))
    importance = (
        axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    )
    return (importance / importance.max()).clamp_min(1e-7)[None, None]


def sliding_window_inference(
    model: nn.Module,
    inputs: torch.Tensor,
    num_classes: int,
    crop_size: int | Sequence[int],
    overlap: float = 0.5,
    *,
    blend_mode: str = "gaussian",
    sigma_scale: float = 0.125,
    return_logits: bool = False,
) -> torch.Tensor:
    """
    Blend overlapping 3D windows in float32 and restore the model mode.

    Args:
        model (nn.Module): Maps (B, C, d, h, w) windows to (B, K, d, h, w) logits.
        inputs (torch.Tensor): Floating volume batch shaped (B, C, D, H, W).
        num_classes (int): Number of output classes K, including background.
        crop_size (int | Sequence[int]): Isotropic size or window sizes (d, h, w).
        overlap (float): Adjacent window overlap fraction in [0, 1).
        blend_mode (str): Window weighting, "gaussian" or "constant".
        sigma_scale (float): Gaussian standard deviation relative to window size.
        return_logits (bool): Return blended logits instead of class indices.

    Returns:
        torch.Tensor: Float32 logits (B, K, D, H, W) when return_logits is True; otherwise int64 labels (B, D, H, W), on the
            input device.
    """

    if inputs.ndim != 5 or inputs.shape[0] <= 0 or inputs.shape[1] <= 0:
        raise ValueError(
            f"inputs must have nonempty shape [B,C,D,H,W], got {inputs.shape}"
        )
    if not inputs.is_floating_point():
        raise TypeError("inputs must be floating point")
    if int(num_classes) <= 1:
        raise ValueError("num_classes must be greater than one")
    if not 0.0 <= float(overlap) < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    if blend_mode not in {"constant", "gaussian"}:
        raise ValueError("blend_mode must be 'constant' or 'gaussian'")
    if sigma_scale <= 0:
        raise ValueError("sigma_scale must be positive")

    crop = _spatial_tuple(crop_size, name="crop_size")
    original = tuple(int(v) for v in inputs.shape[2:])
    padding = tuple(max(window - actual, 0) for window, actual in zip(crop, original))
    # F.pad uses W, H, D order; pad only the high ends for later cropping.
    if any(padding):
        inputs = F.pad(inputs, (0, padding[2], 0, padding[1], 0, padding[0]))
    spatial = tuple(int(v) for v in inputs.shape[2:])
    stride = tuple(max(1, int(window * (1.0 - float(overlap)))) for window in crop)
    starts = tuple(
        _window_starts(actual, window, step)
        for actual, window, step in zip(spatial, crop, stride)
    )

    batch = int(inputs.shape[0])
    prediction = torch.zeros(
        (batch, int(num_classes), *spatial), device=inputs.device, dtype=torch.float32
    )
    normalization = torch.zeros(
        (1, 1, *spatial), device=inputs.device, dtype=torch.float32
    )
    if blend_mode == "gaussian":
        weight = _cpu_gaussian_importance(crop, float(sigma_scale)).to(inputs.device)
    else:
        weight = torch.ones((1, 1, *crop), device=inputs.device, dtype=torch.float32)

    was_training = bool(model.training)
    use_cuda_bf16 = _supports_cuda_bf16(inputs.device)
    model.eval()
    try:
        with torch.inference_mode():
            for d in starts[0]:
                for h in starts[1]:
                    for w in starts[2]:
                        slices = (
                            slice(d, d + crop[0]),
                            slice(h, h + crop[1]),
                            slice(w, w + crop[2]),
                        )
                        window = inputs[(slice(None), slice(None), *slices)]
                        with (
                            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                            if use_cuda_bf16
                            else nullcontext()
                        ):
                            logits = model(window)
                        expected = (batch, int(num_classes), *crop)
                        if tuple(logits.shape) != expected:
                            raise ValueError(
                                f"Model must return main logits shaped {expected}, got {tuple(logits.shape)}"
                            )
                        if not torch.isfinite(logits).all():
                            raise FloatingPointError(
                                "Model returned non-finite inference logits"
                            )
                        prediction[(slice(None), slice(None), *slices)] += (
                            logits.float() * weight
                        )
                        normalization[(slice(None), slice(None), *slices)] += weight
                        del logits, window
        if normalization.amin().item() <= 0:
            raise RuntimeError("Sliding-window inference left voxels uncovered")
        # Normalize in place to avoid a second full-volume logit allocation.
        prediction.div_(normalization)
        del normalization, weight
        prediction = prediction[:, :, : original[0], : original[1], : original[2]]
        return prediction if return_logits else prediction.argmax(dim=1)
    finally:
        model.train(was_training)


__all__ = ["sliding_window_inference"]
