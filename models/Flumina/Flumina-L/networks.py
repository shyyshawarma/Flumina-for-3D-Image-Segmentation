from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from .config import Config, MODEL_NAME, MODEL_SIZE, SpatialShape
from .decoder import SegmentationDecoder
from .encoder import Encoder, EncoderOutput


if TYPE_CHECKING:
    from typing import TypedDict

    class _ConfigOverrides(TypedDict, total=False):
        """
        Optional configuration fields accepted by the model constructor.

        Args:
            in_channels (int): number of input image channels
            num_segmentation_classes (int): number of output segmentation channels
            use_checkpoint (bool): recompute tokenizer downsampling stages during backward
            attention_post_norm (bool): normalize attention before its output projection
        """

        in_channels: int
        num_segmentation_classes: int
        use_checkpoint: bool
        attention_post_norm: bool


class Flumina(nn.Module):
    """3D segmentation model with four persistent residual streams."""

    def __init__(
        self,
        *,
        in_channels: int | None = None,
        num_classes: int | None = None,
        config: Config | None = None,
        use_checkpoint: bool | None = None,
        attention_post_norm: bool | None = None,
    ) -> None:
        """
        Args:
            in_channels (int | None): number of input image channels; defaults to four without a config
            num_classes (int | None): number of output segmentation channels; defaults to four without a config
            config (Config | None): complete model settings; explicit arguments must agree with them
            use_checkpoint (bool | None): recompute tokenizer downsampling stages during backward; None uses the config value
            attention_post_norm (bool | None): normalize attention before its output projection; None uses the config value
        """
        super().__init__()
        requested = {
            name: value
            for name, value in (
                ("in_channels", in_channels),
                ("num_segmentation_classes", num_classes),
                ("use_checkpoint", use_checkpoint),
                ("attention_post_norm", attention_post_norm),
            )
            if value is not None
        }
        if config is None:
            config = Config(**cast("_ConfigOverrides", requested))
        elif not isinstance(config, Config):
            raise TypeError("config must be a Config")
        elif any(value != getattr(config, name) for name, value in requested.items()):
            raise ValueError("Explicit constructor arguments conflict with config")
        self.config = config
        self.in_channels = config.in_channels
        self.num_classes = config.num_segmentation_classes
        self.num_segmentation_classes = config.num_segmentation_classes
        self.embed_dim = config.embed_dim
        self.model_name = MODEL_NAME
        self.model_size = MODEL_SIZE
        self.encoder = Encoder(config)
        self.decoder = SegmentationDecoder(
            config.embed_dim, config.num_segmentation_classes
        )
        self._initialize_decoder()

    def _initialize_decoder(self) -> None:
        """
        Initialize decoder convolutions while preserving transposed-convolution initialization.
        """
        for module in self.decoder.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> EncoderOutput:
        """
        Args:
            x (torch.Tensor): floating image tensor with shape (B, C_in, D, H, W), matching the configured channels and crop

        Returns:
            EncoderOutput: CNN pyramid and projected/final streams; each stream tensor has shape (B, 4, C, D/8, H/8, W/8),
                where C is embed_dim
        """
        return self.encoder(x)

    def decode(self, features: EncoderOutput | torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (EncoderOutput | torch.Tensor): encoder output or final streams with shape (B, 4, C, d, h, w);
                tensor inputs use the configured crop grids

        Returns:
            torch.Tensor: raw segmentation logits with shape (B, num_classes, 8*d, 8*h, 8*w)
        """
        match features:
            case EncoderOutput():
                streams = features.streams
                target_sizes = tuple(
                    tuple(feature.shape[2:]) for feature in features.pyramid
                )
            case torch.Tensor():
                streams = features
                target_sizes = tuple(
                    tuple(size // factor for size in self.config.crop_shape)
                    for factor in (1, 2, 4, 8)
                )
            case _:
                raise TypeError(
                    "features must be EncoderOutput or a Tensor of final states"
                )
        return self.decoder(streams, target_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): floating image tensor with shape (B, C_in, D, H, W), matching the configured channels and crop

        Returns:
            torch.Tensor: raw segmentation logits with shape (B, num_classes, D, H, W)
        """
        return self.decode(self.forward_features(x))


def create_flumina(
    in_channels: int,
    num_classes: int,
    *,
    crop_size: int | Sequence[int] = (96, 96, 96),
    use_checkpoint: bool = False,
    attention_post_norm: bool = True,
) -> Flumina:
    """
    Construct Flumina-L for the input channels and crop.

    Args:
        in_channels (int): number of input image channels
        num_classes (int): number of output segmentation channels
        crop_size (int | Sequence[int]): crop extent or spatial shape (D, H, W)
        use_checkpoint (bool): recompute tokenizer downsampling stages during backward
        attention_post_norm (bool): apply RMS normalization before the attention output projection

    Returns:
        Flumina: model mapping images of shape (B, C_in, D, H, W) to raw logits of shape (B, num_classes, D, H, W)
    """
    shape = (
        (crop_size,) * 3
        if type(crop_size) is int
        else tuple(cast(Sequence[int], crop_size))
    )
    config = Config(
        in_channels=in_channels,
        num_segmentation_classes=num_classes,
        crop_shape=cast(SpatialShape, shape),
        use_checkpoint=use_checkpoint,
        attention_post_norm=attention_post_norm,
    )
    return Flumina(config=config)
