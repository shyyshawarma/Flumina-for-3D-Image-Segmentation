from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from ._attention import attention
from .config import TransformerConfig
from .rope import RoPE3D, apply_rope


class DropPath(nn.Module):
    """Sample-wise stochastic depth for residual branch outputs."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        """
        Args:
            drop_prob (float): probability of dropping each sample during training
        """
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): branch output with shape (B, ...)

        Returns:
            torch.Tensor: tensor with shape (B, ...); surviving samples are rescaled during training
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            mask.div_(keep_prob)
        return x * mask


class ChannelRMSNorm(nn.Module):
    """Channel RMS normalization for tensors with channels in dimension one."""

    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        """
        Args:
            num_channels (int): number of channels to normalize
            eps (float): stability constant added to the mean square
        """
        super().__init__()
        self.num_channels = num_channels
        self.norm = nn.RMSNorm(num_channels, eps=eps, elementwise_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape (B, C, ...)

        Returns:
            torch.Tensor: normalized tensor with shape (B, C, ...)
        """
        return self.norm(x.movedim(1, -1)).movedim(-1, 1)


class RMSNorm(nn.Module):
    """Final-dimension normalization with a learnable offset scale."""

    def __init__(self, dim: int) -> None:
        """
        Args:
            dim (int): dimension of the input and output
        """
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape (..., dim)

        Returns:
            torch.Tensor: normalized tensor with shape (..., dim)
        """
        return F.normalize(x, dim=-1) * self.scale * (self.gamma + 1)


class GroupedRMSNorm(nn.Module):
    """Independent RMS normalization of residual streams with float32 accumulation."""

    def __init__(self, hidden_size: int, group_size: int, eps: float = 1e-6) -> None:
        """
        Args:
            hidden_size (int): total channel dimension of the concatenated streams
            group_size (int): channel dimension of one stream; must divide hidden_size
            eps (float): stability constant added to each group mean square
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.group_size = group_size
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): concatenated streams with shape (B, N, S*C),
                where C is group_size and S*C is hidden_size

        Returns:
            torch.Tensor: normalized streams with shape (B, N, S*C) and the input dtype
        """
        input_dtype = hidden_states.dtype
        grouped = hidden_states.float().unflatten(-1, (-1, self.group_size))
        variance = grouped.square().mean(dim=-1, keepdim=True)
        normalized = (grouped * torch.rsqrt(variance + self.eps)).flatten(-2)
        return (normalized * (1.0 + self.weight.float())).to(input_dtype)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Reset the learned offset so every channel has unit scale."""
        self.weight.zero_()


class SwiGLU(nn.Module):
    """Gated feed-forward layer with intermediate RMS normalization."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        """
        Args:
            dim (int): dimension of the input and output
            dropout (float): dropout probability before normalization and after the output projection
        """
        super().__init__()
        hidden_features = int(round((8.0 / 3.0) * dim))
        self.fc1_g = nn.Linear(dim, hidden_features, bias=True)
        self.fc1_x = nn.Linear(dim, hidden_features, bias=True)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        self.norm = nn.RMSNorm(hidden_features, eps=1e-5, elementwise_affine=True)
        self.fc2 = nn.Linear(hidden_features, dim, bias=True)
        self.drop2 = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize the gate projection near a constant positive activation."""
        nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tokens with shape (B, N, C)

        Returns:
            torch.Tensor: output tokens with shape (B, N, C)
        """
        x = self.act(self.fc1_g(x)) * self.fc1_x(x)
        return self.drop2(self.fc2(self.norm(self.drop1(x))))


class Attention(nn.Module):
    """Multi-head self-attention with optional axial 3D rotary embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        num_register_tokens: int = 0,
        attention_post_norm: bool = True,
    ) -> None:
        """
        Args:
            dim (int): dimension of the input and output
            num_heads (int): number of attention heads; must divide dim
            attn_dropout (float): dropout probability for attention weights
            proj_dropout (float): dropout probability after the output projection
            num_register_tokens (int): number of leading tokens excluded from rotary embedding
            attention_post_norm (bool): apply RMS normalization before the output projection
        """
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_register_tokens = num_register_tokens
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.post_attn_norm = (
            nn.RMSNorm(dim, eps=1e-5, elementwise_affine=True)
            if attention_post_norm
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, rope_embed: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tokens with shape (B, N, C), including any register prefix
            rope_embed (torch.Tensor | None): optional rotary table with shape (N, 2*head_dim),
                including any register prefix rows

        Returns:
            torch.Tensor: attention output with shape (B, N, C)
        """
        qkv = self.qkv(x).reshape(
            x.shape[0], x.shape[1], 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        if rope_embed is not None:
            q, k = apply_rope(q, k, rope_embed, self.num_register_tokens)
        attended = attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.dim)
        return self.proj_drop(self.proj(self.post_attn_norm(attended)))


@dataclass(frozen=True)
class RouterState:
    """
    Residual streams and write gates retained for a sublayer update.

    Args:
        residuals (torch.Tensor): packed streams with shape (B*S, N, C)
        beta (torch.Tensor): token-wise write gates with shape (B, N, S)
    """

    residuals: torch.Tensor
    beta: torch.Tensor


class StreamRouter(nn.Module):
    """Dynamic gated reads and writes over persistent residual streams."""

    def __init__(
        self,
        num_residual_streams: int,
        *,
        dim: int,
        layer_index: int = 0,
        lowrank_size: int | None = None,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        """
        Args:
            num_residual_streams (int): number S of persistent residual streams
            dim (int): channel dimension C of each stream
            layer_index (int): sublayer index selecting the initially favored write stream
            lowrank_size (int | None): read-gate bottleneck dimension; None uses dim//8
            rms_norm_eps (float): stability constant for grouped read-gate normalization
        """
        super().__init__()
        self.num_residual_streams = num_residual_streams
        self.dim = dim
        self.flat_hidden_size = dim * num_residual_streams
        self.lowrank_size = dim // 8 if lowrank_size is None else lowrank_size
        self.residual_transform = nn.Identity()
        self.hc_norm = GroupedRMSNorm(
            self.flat_hidden_size, group_size=dim, eps=rms_norm_eps
        )
        self.input_mix_weight_down = nn.Linear(
            self.flat_hidden_size, self.lowrank_size, bias=False
        )
        self.input_mix_weight_up = nn.Linear(
            self.lowrank_size, self.flat_hidden_size, bias=False
        )
        self.norm = RMSNorm(self.flat_hidden_size)
        initial_stream = layer_index % num_residual_streams
        self.register_buffer(
            "h_res",
            torch.eye(num_residual_streams, dtype=torch.float32),
            persistent=False,
        )
        beta = torch.full((num_residual_streams,), -1.0)
        beta[initial_stream] = 1.0
        self.static_beta = nn.Parameter(beta)
        self.dynamic_beta_fn = nn.Parameter(
            torch.zeros(self.flat_hidden_size, num_residual_streams)
        )
        self.h_post_scale = nn.Parameter(torch.ones(()) * 1e-2)
        self.dropout = nn.Dropout(0.0)
        self.init_read_weights()

    @torch.no_grad()
    def init_read_weights(self, init_std: float = 0.02) -> None:
        """
        Reset grouped normalization and initialize the read-gate projections.

        Args:
            init_std (float): standard deviation of the truncated normal projection weights
        """
        self.hc_norm.reset_parameters()
        nn.init.trunc_normal_(self.input_mix_weight_down.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.input_mix_weight_up.weight, mean=0.0, std=init_std)

    def mix(self, residuals: torch.Tensor) -> tuple[torch.Tensor, RouterState]:
        """
        Read the streams and retain the residuals and dynamic write gates.

        Args:
            residuals (torch.Tensor): packed residual streams with shape (B*S, N, C)

        Returns:
            tuple[torch.Tensor, RouterState]: sublayer input with shape (B, N, C), followed by packed residuals (B*S, N, C)
                and write gates (B, N, S) stored in RouterState
        """
        residuals = self.residual_transform(residuals)
        working = residuals.reshape(
            residuals.shape[0] // self.num_residual_streams,
            self.num_residual_streams,
            *residuals.shape[1:-1],
            self.dim,
        ).movedim(1, -2)
        flattened = working.flatten(start_dim=-2)
        read_normalized = self.hc_norm(flattened)
        projection_input = read_normalized.to(
            dtype=self.input_mix_weight_down.weight.dtype
        )
        gates = F.silu(
            self.input_mix_weight_down(projection_input) / self.num_residual_streams
        )
        gates = torch.sigmoid(self.input_mix_weight_up(gates))
        branch_input = (
            gates.unflatten(-1, (self.num_residual_streams, self.dim))
            * read_normalized.unflatten(-1, (self.num_residual_streams, self.dim))
        ).mean(dim=-2)
        write_normalized = self.norm(flattened)
        beta = (
            (write_normalized @ self.dynamic_beta_fn) * self.h_post_scale
            + self.static_beta
        ).sigmoid() * 2.0
        return branch_input, RouterState(residuals.to(dtype=branch_input.dtype), beta)

    def combine(self, branch_output: torch.Tensor, state: RouterState) -> torch.Tensor:
        """
        Add a gated sublayer output to each residual stream.

        Args:
            branch_output (torch.Tensor): sublayer output with shape (B, N, C)
            state (RouterState): packed residuals with shape (B*S, N, C) and write gates with shape (B, N, S)

        Returns:
            torch.Tensor: updated residual streams with shape (B*S, N, C)
        """
        output = torch.einsum("...c,...s->...sc", branch_output, state.beta)
        output = output.movedim(-2, 1).reshape(
            output.shape[0] * self.num_residual_streams, *output.shape[1:-2], self.dim
        )
        return self.dropout(torch.add(output, state.residuals))

    def forward(self, residuals: torch.Tensor) -> tuple[torch.Tensor, RouterState]:
        """
        Args:
            residuals (torch.Tensor): packed residual streams with shape (B*S, N, C)

        Returns:
            tuple[torch.Tensor, RouterState]: sublayer input with shape (B, N, C), followed by packed residuals (B*S, N, C)
                and write gates (B, N, S) stored in RouterState
        """
        return self.mix(residuals)


class Block(nn.Module):
    """Attention and SwiGLU branches with independent residual stream routing."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dropout: float,
        attn_dropout: float,
        proj_dropout: float,
        drop_path: float,
        layer_scale_init_value: float | None,
        num_register_tokens: int,
        num_residual_streams: int = 4,
        layer_index: int = 0,
        hc_lowrank: int | None = None,
        hc_rms_norm_eps: float = 1e-6,
        attention_post_norm: bool = True,
    ) -> None:
        """
        Args:
            dim (int): dimension of the input and output
            num_heads (int): number of attention heads
            mlp_dropout (float): dropout probability in the feed-forward branch
            attn_dropout (float): dropout probability for attention weights
            proj_dropout (float): dropout probability after the attention output projection
            drop_path (float): stochastic depth probability for both residual branches
            layer_scale_init_value (float | None): initial value for both branch scales; None disables layer scaling
            num_register_tokens (int): number of non-spatial tokens prepended to each stream
            num_residual_streams (int): number of persistent residual streams
            layer_index (int): block index used to initialize its two write gates
            hc_lowrank (int | None): read-gate bottleneck dimension; None uses dim//8
            hc_rms_norm_eps (float): stability constant for grouped read-gate normalization
            attention_post_norm (bool): apply RMS normalization before the attention output projection
        """
        super().__init__()
        self.ln1 = nn.RMSNorm(dim, eps=1e-5, elementwise_affine=True)
        self.attn = Attention(
            dim,
            num_heads,
            attn_dropout,
            proj_dropout,
            num_register_tokens,
            attention_post_norm,
        )
        self.drop_path1 = DropPath(drop_path)
        self.ln2 = nn.RMSNorm(dim, eps=1e-5, elementwise_affine=True)
        self.mlp = SwiGLU(dim, mlp_dropout)
        self.drop_path2 = DropPath(drop_path)
        self.gamma1: nn.Parameter | None = None
        self.gamma2: nn.Parameter | None = None
        if layer_scale_init_value is not None:
            self.gamma1 = nn.Parameter(torch.full((dim,), layer_scale_init_value))
            self.gamma2 = nn.Parameter(torch.full((dim,), layer_scale_init_value))
        self.attn_resid = StreamRouter(
            num_residual_streams,
            dim=dim,
            layer_index=2 * layer_index,
            lowrank_size=hc_lowrank,
            rms_norm_eps=hc_rms_norm_eps,
        )
        self.mlp_resid = StreamRouter(
            num_residual_streams,
            dim=dim,
            layer_index=2 * layer_index + 1,
            lowrank_size=hc_lowrank,
            rms_norm_eps=hc_rms_norm_eps,
        )

    def forward(
        self, x: torch.Tensor, rope_embed: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): packed streams with shape (B*S, N, C), where N includes any register prefix
            rope_embed (torch.Tensor | None): optional rotary table with shape (N, 2*head_dim),
                including any register prefix rows

        Returns:
            torch.Tensor: updated residual streams with shape (B*S, N, C)
        """
        attn_input, attn_state = self.attn_resid(x)
        attn_output = self.attn(self.ln1(attn_input), rope_embed=rope_embed)
        if self.gamma1 is not None:
            attn_output = attn_output * self.gamma1
        x = self.attn_resid.combine(self.drop_path1(attn_output), attn_state)
        mlp_input, mlp_state = self.mlp_resid(x)
        mlp_output = self.mlp(self.ln2(mlp_input))
        if self.gamma2 is not None:
            mlp_output = mlp_output * self.gamma2
        return self.mlp_resid.combine(self.drop_path2(mlp_output), mlp_state)


class Transformer(nn.Module):
    """
    Shared Transformer over residual streams initialized from multiple CNN scales.
    """

    def __init__(self, config: TransformerConfig) -> None:
        """
        Args:
            config (TransformerConfig): attention, routing, regularization, and token grid settings
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.depth = config.depth
        self.num_heads = config.num_heads
        self.tokens_spatial_shape = config.tokens_spatial_shape
        self.sequence_length = math.prod(config.tokens_spatial_shape)
        self.num_register_tokens = config.num_register_tokens
        self.num_residual_streams = config.num_residual_streams
        self.pos_embed: nn.Parameter | None = None
        if config.use_learned_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.sequence_length, config.embed_dim)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.register_tokens: nn.Parameter | None = None
        if config.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, config.num_register_tokens, config.embed_dim)
            )
            nn.init.normal_(self.register_tokens, std=1e-5)
        self.rope: RoPE3D | None = None
        if config.use_rotary_pos_embed:
            head_dim = config.embed_dim // config.num_heads
            self.rope = RoPE3D((2 * head_dim) // 3, config.rope_base)
        drop_paths = torch.linspace(
            0.0,
            float(config.drop_path_rate),
            config.depth,
            dtype=torch.float32,
            device="cpu",
        ).tolist()
        self.blocks = nn.ModuleList(
            Block(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_dropout=config.mlp_dropout,
                attn_dropout=config.attn_dropout,
                proj_dropout=config.proj_dropout,
                drop_path=drop_paths[index],
                layer_scale_init_value=config.layer_scale_init_value,
                num_register_tokens=config.num_register_tokens,
                num_residual_streams=config.num_residual_streams,
                layer_index=index,
                hc_lowrank=config.hc_lowrank,
                hc_rms_norm_eps=config.hc_rms_norm_eps,
                attention_post_norm=config.attention_post_norm,
            )
            for index in range(config.depth)
        )
        self.final_norm = nn.RMSNorm(
            config.embed_dim, eps=1e-5, elementwise_affine=True
        )
        self.initialize()

    def initialize(self) -> None:
        """Initialize weights and scale branch projections by layer depth."""
        nn.init.ones_(self.final_norm.weight)
        for block in cast(Iterable[Block], self.blocks):
            nn.init.ones_(block.ln1.weight)
            nn.init.ones_(block.ln2.weight)
            nn.init.trunc_normal_(block.attn.qkv.weight, std=0.02)
            nn.init.zeros_(block.attn.qkv.bias)
            nn.init.trunc_normal_(block.attn.proj.weight, std=0.02)
            nn.init.zeros_(block.attn.proj.bias)
            if isinstance(block.attn.post_attn_norm, nn.RMSNorm):
                nn.init.ones_(block.attn.post_attn_norm.weight)
            nn.init.ones_(block.mlp.norm.weight)
            nn.init.trunc_normal_(block.mlp.fc1_x.weight, std=0.02)
            nn.init.zeros_(block.mlp.fc1_x.bias)
            nn.init.trunc_normal_(block.mlp.fc2.weight, std=0.02)
            nn.init.zeros_(block.mlp.fc2.bias)
        with torch.no_grad():
            for layer_index, block in enumerate(
                cast(Iterable[Block], self.blocks), start=1
            ):
                scale = math.sqrt(2.0 * layer_index)
                block.attn.proj.weight.div_(scale)
                block.mlp.fc2.weight.div_(scale)

    def forward(self, projected_streams: torch.Tensor) -> torch.Tensor:
        """
        Args:
            projected_streams (torch.Tensor): initial streams with shape (B, S, C, d, h, w),
                matching the configured stream count, embedding dimension, and token grid

        Returns:
            torch.Tensor: normalized final streams with shape (B, S, C, d, h, w), with register tokens removed
        """
        batch_size = projected_streams.shape[0]
        tokens = projected_streams.permute(0, 1, 3, 4, 5, 2).reshape(
            batch_size * self.num_residual_streams, self.sequence_length, self.embed_dim
        )
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed
        if self.register_tokens is not None:
            prefix = self.register_tokens.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat((prefix, tokens), dim=1)
        rope_embed = None
        if self.rope is not None:
            rope_embed = self.rope.get_embed(
                self.tokens_spatial_shape, device=tokens.device, dtype=tokens.dtype
            )
            rope_embed = self.rope.prepend_register_tokens(
                rope_embed,
                self.num_register_tokens,
                device=tokens.device,
                dtype=tokens.dtype,
            )
        for block in self.blocks:
            tokens = block(tokens, rope_embed=rope_embed)
        tokens = self.final_norm(tokens)
        first_spatial_token = self.num_register_tokens
        tokens = tokens[:, first_spatial_token:, :]
        return (
            tokens.reshape(
                batch_size,
                self.num_residual_streams,
                *self.tokens_spatial_shape,
                self.embed_dim,
            )
            .permute(0, 1, 5, 2, 3, 4)
            .contiguous()
        )
