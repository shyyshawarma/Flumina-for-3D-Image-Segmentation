from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from torch import nn


@dataclass(frozen=True)
class VariantSpec:
    """
    Metadata for one Flumina size.

    Args:
        name (str): canonical size letter: s, b, m, or l
        description (str): display name for the model size
        aliases (tuple[str, ...]): additional names accepted for this size
    """

    name: str
    description: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelSpec:
    """
    Model family metadata and available sizes.

    Args:
        name (str): canonical model family name
        variants (tuple[VariantSpec, ...]): supported sizes and their aliases
        default_variant (str): size selected when no name is provided
    """

    name: str
    variants: tuple[VariantSpec, ...]
    default_variant: str = "b"

    def get_variant(self, name: str | None = None) -> VariantSpec:
        """
        Resolve a size name or alias.

        Args:
            name (str | None): case-insensitive size name; None selects the default

        Returns:
            VariantSpec: metadata for the selected model size
        """
        requested = (name or self.default_variant).strip().lower()
        for variant in self.variants:
            if requested in (variant.name, *variant.aliases):
                return variant
        raise ValueError(f"unknown Flumina variant {name!r}; expected s, b, m, or l")


_FLUMINA = ModelSpec(
    name="flumina",
    variants=(
        VariantSpec("s", "Flumina Small", ("small",)),
        VariantSpec("b", "Flumina Base", ("base",)),
        VariantSpec("m", "Flumina Medium", ("medium",)),
        VariantSpec("l", "Flumina Large", ("large",)),
    ),
)


@dataclass(frozen=True)
class ModelConfig:
    """
    Inputs for constructing a Flumina segmentation model.

    Args:
        name (str): model family name
        in_channels (int): number of input image channels
        out_channels (int): number of output segmentation channels
        crop_size (int | Sequence[int]): crop extent or spatial shape (D, H, W)
        variant (str | None): size letter or alias; None selects Base
        kwargs (Mapping[str, bool | str | None]): additional factory options,
            such as attention_post_norm or an equivalent variant selector
        gradient_checkpointing (bool): recompute tokenizer downsampling stages during backward
    """

    name: str
    in_channels: int
    out_channels: int
    crop_size: int | Sequence[int] = (96, 96, 96)
    variant: str | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    gradient_checkpointing: bool = False


def get_model_spec(name: str) -> ModelSpec:
    """
    Look up the Flumina family specification.

    Args:
        name (str): case-insensitive model family name

    Returns:
        ModelSpec: Flumina metadata with its four supported sizes
    """
    if name.strip().lower() != "flumina":
        raise ValueError(f"unknown model {name!r}; available models: flumina")
    return _FLUMINA


def available_models() -> tuple[str, ...]:
    """
    List the supported model families.

    Returns:
        tuple[str, ...]: family names; currently contains only flumina
    """
    return ("flumina",)


def get_model(name: str) -> Callable[..., nn.Module]:
    """
    Load the factory for a model family.

    Args:
        name (str): case-insensitive model family name

    Returns:
        Callable[..., nn.Module]: factory that constructs the selected Flumina size
    """
    get_model_spec(name)
    from models.Flumina import create_flumina

    return create_flumina


def _crop_shape(value: int | Sequence[int]) -> tuple[int, int, int]:
    """
    Validate a crop for the shared token grid and reflection padding.

    Args:
        value (int | Sequence[int]): crop extent or three spatial dimensions, each divisible by 8 and at least 16

    Returns:
        tuple[int, int, int]: validated spatial shape (D, H, W)
    """
    shape = (value,) * 3 if type(value) is int else tuple(cast(Sequence[int], value))
    if len(shape) != 3 or any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError("crop_size must contain three positive integers")
    if any(size % 8 for size in shape):
        raise ValueError(f"Flumina crop dimensions must be divisible by 8, got {shape}")
    if any(size < 16 for size in shape):
        raise ValueError(f"Flumina crop dimensions must be at least 16, got {shape}")
    return shape


def build_model(
    config: ModelConfig | str | None = None,
    *,
    name: str | None = None,
    in_channels: int | None = None,
    out_channels: int | None = None,
    crop_size: int | Sequence[int] = (96, 96, 96),
    variant: str | None = None,
    gradient_checkpointing: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """
    Build a Flumina model from a configuration or explicit arguments.

    Args:
        config (ModelConfig | str | None): complete configuration or positional family name;
            a configuration cannot be combined with overrides
        name (str | None): family name when config is omitted
        in_channels (int | None): number of input image channels when passing a family name
        out_channels (int | None): number of output segmentation channels when passing a family name
        crop_size (int | Sequence[int]): crop extent or spatial shape (D, H, W)
        variant (str | None): size letter or alias; None selects Base
        gradient_checkpointing (bool): recompute tokenizer downsampling stages during backward
        **kwargs (bool): additional factory options, such as attention_post_norm

    Returns:
        nn.Module: Flumina model mapping images of shape (B, C_in, D, H, W) to raw logits of shape (B, C_out, D, H, W)
    """
    if isinstance(config, ModelConfig):
        if (
            any(
                value is not None
                for value in (name, in_channels, out_channels, variant)
            )
            or kwargs
            or gradient_checkpointing
            or _crop_shape(crop_size) != (96, 96, 96)
        ):
            raise TypeError("a ModelConfig cannot be combined with explicit overrides")
        resolved = config
    else:
        if config is not None and name is not None:
            raise TypeError("pass the model name either positionally or by keyword")
        selected_name = config if config is not None else name
        if selected_name is None or in_channels is None or out_channels is None:
            raise TypeError("model name, in_channels, and out_channels are required")
        resolved = ModelConfig(
            selected_name,
            in_channels,
            out_channels,
            crop_size,
            variant,
            kwargs,
            gradient_checkpointing,
        )
    selected = get_model_spec(resolved.name).get_variant(resolved.variant)
    return get_model(resolved.name)(
        in_channels=resolved.in_channels,
        num_classes=resolved.out_channels,
        crop_size=_crop_shape(resolved.crop_size),
        model_size=selected.name,
        use_checkpoint=resolved.gradient_checkpointing,
        **resolved.kwargs,
    )
