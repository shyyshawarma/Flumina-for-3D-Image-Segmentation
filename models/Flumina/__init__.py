from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from torch import nn

    Flumina: type[nn.Module]
    FluminaS: type[nn.Module]
    FluminaB: type[nn.Module]
    FluminaM: type[nn.Module]
    FluminaL: type[nn.Module]


_SIZES = {
    "s": "S",
    "small": "S",
    "b": "B",
    "base": "B",
    "m": "M",
    "medium": "M",
    "l": "L",
    "large": "L",
}


def _normalize_size(value: str) -> str:
    """
    Resolve a size selector to its canonical letter.

    Args:
        value (str): size letter, descriptive name, or Flumina-prefixed name

    Returns:
        str: one of S, B, M, or L
    """
    normalized = value.strip().lower().replace("-", "").replace("_", "")
    normalized = normalized.removeprefix("flumina")
    try:
        return _SIZES[normalized]
    except KeyError as error:
        raise ValueError(
            f"Unknown Flumina size {value!r}; expected S, B, M, or L"
        ) from error


def create_flumina(
    in_channels: int,
    num_classes: int,
    *,
    crop_size: int | Sequence[int] = (96, 96, 96),
    model_size: str | None = None,
    variant: str | None = None,
    use_checkpoint: bool = False,
    attention_post_norm: bool = True,
) -> nn.Module:
    """
    Construct the selected Flumina variant.

    Args:
        in_channels (int): number of input image channels
        num_classes (int): number of output segmentation channels
        crop_size (int | Sequence[int]): crop extent or spatial shape (D, H, W)
        model_size (str | None): size selector; defaults to Base when variant is also None
        variant (str | None): alternative size selector; must agree with model_size when both are provided
        use_checkpoint (bool): recompute tokenizer downsampling stages during backward
        attention_post_norm (bool): apply RMS normalization after attention and before its output projection

    Returns:
        nn.Module: Flumina model mapping images of shape (B, C_in, D, H, W) to raw logits of shape (B, num_classes, D, H, W)
    """
    size = None if model_size is None else _normalize_size(model_size)
    selected_variant = None if variant is None else _normalize_size(variant)
    if size is not None and selected_variant is not None and size != selected_variant:
        raise ValueError(
            "model_size and variant must select the same Flumina architecture"
        )
    selected = size or selected_variant or "B"
    module = import_module(f".Flumina-{selected}", __name__)
    return module.create_flumina(
        in_channels,
        num_classes,
        crop_size=crop_size,
        use_checkpoint=use_checkpoint,
        attention_post_norm=attention_post_norm,
    )


def __getattr__(name: str) -> type[nn.Module]:
    """
    Lazily load a Flumina model class.

    Args:
        name (str): Flumina, FluminaS, FluminaB, FluminaM, or FluminaL

    Returns:
        type[nn.Module]: requested Flumina variant class; Flumina selects Base
    """
    sizes = {
        "Flumina": "B",
        "FluminaS": "S",
        "FluminaB": "B",
        "FluminaM": "M",
        "FluminaL": "L",
    }
    if name not in sizes:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return import_module(f".Flumina-{sizes[name]}", __name__).Flumina


__all__ = ["Flumina", "FluminaS", "FluminaB", "FluminaM", "FluminaL", "create_flumina"]
