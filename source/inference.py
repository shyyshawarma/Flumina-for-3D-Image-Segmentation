from collections.abc import Callable
from itertools import combinations
from typing import cast

import torch


FLIP_AXES = tuple(axes for count in range(4) for axes in combinations((2, 3, 4), count))


def inference_with_augmentation(
    inputs: torch.Tensor,
    predictor: Callable[[torch.Tensor], torch.Tensor],
    inference_augument: bool = True,
) -> torch.Tensor:
    """
    Average aligned logits over all eight spatial flip combinations.

    Args:
        inputs (torch.Tensor): Volume batch shaped (B, C, D, H, W).
        predictor (Callable[[torch.Tensor], torch.Tensor]): Maps input volumes to floating logits shaped (B, K, D, H, W).
        inference_augument (bool): Enable flip averaging; False calls the predictor once and returns its output unchanged.

    Returns:
        torch.Tensor: Logits shaped (B, K, D, H, W). Flip averaging promotes float16/bfloat16 logits to float32 and preserves
            other floating dtypes.
    """
    if not inference_augument:
        return predictor(inputs)
    if inputs.ndim != 5:
        raise ValueError("Augmented inference requires inputs shaped (B, C, D, H, W)")

    total = None
    for axes in FLIP_AXES:
        view = inputs.flip(axes) if axes else inputs
        logits = predictor(view)
        if not torch.is_tensor(logits) or logits.ndim != 5:
            raise ValueError("The predictor must return logits shaped (B, K, D, H, W)")
        if not logits.is_floating_point():
            raise TypeError("The predictor must return floating point logits")
        if logits.shape[0] != inputs.shape[0] or logits.shape[2:] != inputs.shape[2:]:
            raise ValueError(
                "Predictor logits must share the input batch and spatial shape"
            )
        aligned = logits.flip(axes) if axes else logits
        if total is None:
            dtype = (
                torch.float32
                if logits.dtype in (torch.float16, torch.bfloat16)
                else logits.dtype
            )
            total = aligned.to(dtype=dtype).clone()
        elif logits.shape != total.shape:
            raise ValueError("Every augmented view must return the same logit shape")
        else:
            total.add_(aligned)
    return cast(torch.Tensor, total).div_(len(FLIP_AXES))


__all__ = ["FLIP_AXES", "inference_with_augmentation"]
