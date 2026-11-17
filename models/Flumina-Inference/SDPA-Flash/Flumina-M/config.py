from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, cast


SpatialShape = tuple[int, int, int]
ChannelShape = tuple[int, int, int, int]

MODEL_SIZE: Final[str] = "Medium"
MODEL_NAME: Final[str] = "Flumina-M"
EMBED_DIM: Final[int] = 792
LAYERS: Final[int] = 12
HEADS: Final[int] = 12
HEAD_DIM: Final[int] = 66
HC_LOWRANK: Final[int] = EMBED_DIM // 8
HC_RMS_NORM_EPS: Final[float] = 1e-6
FEATURE_CHANNELS: Final[ChannelShape] = (32, 64, 128, 256)
DECODER_CHANNELS: Final[ChannelShape] = (32, 64, 128, 256)
POOL_FACTORS: Final[ChannelShape] = (8, 4, 2, 1)
PATCH_SIZE: Final[int] = 8
NUM_STREAMS: Final[int] = 4


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    """
    Attention, routing, and position settings for the Transformer.

    Args:
        embed_dim (int): channel dimension of each residual stream
        depth (int): number of Transformer blocks
        num_heads (int): number of attention heads
        drop_path_rate (float): upper bound on the stochastic depth probability across blocks
        mlp_dropout (float): dropout probability in the feed-forward branch
        attn_dropout (float): dropout probability for attention weights
        proj_dropout (float): dropout probability after the attention output projection
        layer_scale_init_value (float | None): initial value for both branch scales; None disables layer scaling
        rope_base (float): geometric frequency base for rotary embeddings
        use_learned_pos_embed (bool): add learned spatial position embeddings shared by all streams
        tokens_spatial_shape (SpatialShape): token grid dimensions (d, h, w)
        use_rotary_pos_embed (bool): apply axial rotary embeddings to spatial queries and keys
        num_register_tokens (int): number of non-spatial tokens prepended to each stream
        num_residual_streams (int): number of persistent residual streams
        hc_lowrank (int): bottleneck dimension of the dynamic read gates
        hc_rms_norm_eps (float): stability constant for grouped read-gate normalization
        attention_post_norm (bool): apply RMS normalization before the attention output projection
    """

    embed_dim: int = EMBED_DIM
    depth: int = LAYERS
    num_heads: int = HEADS
    drop_path_rate: float = 0.2
    mlp_dropout: float = 0.0
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    layer_scale_init_value: float | None = 0.1
    rope_base: float = 10000.0
    use_learned_pos_embed: bool = True
    tokens_spatial_shape: SpatialShape = (12, 12, 12)
    use_rotary_pos_embed: bool = True
    num_register_tokens: int = 0
    num_residual_streams: int = NUM_STREAMS
    hc_lowrank: int = HC_LOWRANK
    hc_rms_norm_eps: float = HC_RMS_NORM_EPS
    attention_post_norm: bool = True


@dataclass(frozen=True, slots=True)
class Config:
    """
    Immutable model and crop settings for Flumina-M.

    Args:
        in_channels (int): number of input image channels
        num_segmentation_classes (int): number of output segmentation channels
        crop_shape (SpatialShape): crop dimensions (D, H, W), each divisible by 8 and at least 16
        patch_embed_channels (ChannelShape): four CNN channel dimensions from finest to coarsest scale
        use_learned_pos_embed (bool): add learned spatial position embeddings
        use_rotary_pos_embed (bool): apply axial rotary embeddings to spatial queries and keys
        rope_base (float): geometric frequency base for rotary embeddings
        drop_path_rate (float): maximum stochastic depth probability across the Transformer blocks
        mlp_dropout (float): dropout probability in the feed-forward branch
        attn_dropout (float): dropout probability for attention weights
        proj_dropout (float): dropout probability after the attention output projection
        layer_scale_init_value (float | None): initial value for both branch scales; None disables layer scaling
        num_register_tokens (int): number of non-spatial tokens prepended to each stream
        hc_lowrank (int): bottleneck dimension of the dynamic read gates
        hc_rms_norm_eps (float): stability constant for grouped read-gate normalization
        attention_post_norm (bool): apply RMS normalization before the attention output projection
        use_checkpoint (bool): recompute tokenizer downsampling stages during backward
    """

    in_channels: int = 4
    num_segmentation_classes: int = 4
    crop_shape: SpatialShape = (96, 96, 96)
    patch_embed_channels: ChannelShape = FEATURE_CHANNELS
    use_learned_pos_embed: bool = True
    use_rotary_pos_embed: bool = True
    rope_base: float = 10000.0
    drop_path_rate: float = 0.2
    mlp_dropout: float = 0.0
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    layer_scale_init_value: float | None = 0.1
    num_register_tokens: int = 0
    hc_lowrank: int = HC_LOWRANK
    hc_rms_norm_eps: float = HC_RMS_NORM_EPS
    attention_post_norm: bool = True
    use_checkpoint: bool = False

    def __post_init__(self) -> None:
        """Validate dimensions, flags, and numerical settings."""
        for name in ("in_channels", "num_segmentation_classes", "hc_lowrank"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.num_register_tokens) is not int or self.num_register_tokens < 0:
            raise ValueError("num_register_tokens must be a non-negative integer")
        for name, length in (("crop_shape", 3), ("patch_embed_channels", 4)):
            values = tuple(getattr(self, name))
            if len(values) != length or any(
                type(v) is not int or v <= 0 for v in values
            ):
                raise ValueError(f"{name} must contain {length} positive integers")
            object.__setattr__(self, name, values)
        if any(size % PATCH_SIZE or size < 2 * PATCH_SIZE for size in self.crop_shape):
            raise ValueError(
                "crop_shape dimensions must be divisible by 8 and at least 16"
            )
        for name in (
            "use_learned_pos_embed",
            "use_rotary_pos_embed",
            "attention_post_norm",
            "use_checkpoint",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool")
        for name in ("rope_base", "hc_rms_norm_eps"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number")
        for name in ("drop_path_rate", "mlp_dropout", "attn_dropout", "proj_dropout"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value < 1
            ):
                raise ValueError(f"{name} must be a finite number in [0, 1)")
        value = self.layer_scale_init_value
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(
                "layer_scale_init_value must be None or finite and non-negative"
            )

    @property
    def embed_dim(self) -> int:
        """
        Returns:
            int: channel dimension of each residual stream
        """
        return EMBED_DIM

    @property
    def depth(self) -> int:
        """
        Returns:
            int: number of Transformer blocks
        """
        return LAYERS

    @property
    def num_heads(self) -> int:
        """
        Returns:
            int: number of attention heads
        """
        return HEADS

    @property
    def head_dim(self) -> int:
        """
        Returns:
            int: channel dimension of each attention head
        """
        return HEAD_DIM

    @property
    def patch_shape(self) -> SpatialShape:
        """
        Returns:
            SpatialShape: spatial reduction factors (8, 8, 8) for the shared token grid
        """
        return (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)

    @property
    def tokens_spatial_shape(self) -> SpatialShape:
        """
        Returns:
            SpatialShape: token grid dimensions (D/8, H/8, W/8) for the configured crop
        """
        return cast(SpatialShape, tuple(size // PATCH_SIZE for size in self.crop_shape))

    @property
    def feature_channels(self) -> ChannelShape:
        """
        Returns:
            ChannelShape: four CNN channel dimensions from finest to coarsest scale
        """
        return self.patch_embed_channels

    def transformer_config(self) -> TransformerConfig:
        """
        Collect the Transformer settings for this variant and crop.

        Returns:
            TransformerConfig: attention, routing, and position configuration
        """
        return TransformerConfig(
            embed_dim=EMBED_DIM,
            depth=LAYERS,
            num_heads=HEADS,
            drop_path_rate=float(self.drop_path_rate),
            mlp_dropout=float(self.mlp_dropout),
            attn_dropout=float(self.attn_dropout),
            proj_dropout=float(self.proj_dropout),
            layer_scale_init_value=(
                None
                if self.layer_scale_init_value is None
                else float(self.layer_scale_init_value)
            ),
            rope_base=float(self.rope_base),
            use_learned_pos_embed=self.use_learned_pos_embed,
            tokens_spatial_shape=self.tokens_spatial_shape,
            use_rotary_pos_embed=self.use_rotary_pos_embed,
            num_register_tokens=self.num_register_tokens,
            num_residual_streams=NUM_STREAMS,
            hc_lowrank=self.hc_lowrank,
            hc_rms_norm_eps=float(self.hc_rms_norm_eps),
            attention_post_norm=self.attention_post_norm,
        )
