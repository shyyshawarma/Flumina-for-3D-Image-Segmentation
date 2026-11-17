from typing import Protocol

import torch
from torch import nn


class OptimizerConfig(Protocol):
    """Read-only AdamW settings supplied by training configurations."""

    @property
    def base_lr(self) -> float:
        """
        Returns:
            float: Base learning rate.
        """
        ...

    @property
    def weight_decay(self) -> float:
        """
        Returns:
            float: Weight-decay coefficient.
        """
        ...

    @property
    def adam_beta1(self) -> float:
        """
        Returns:
            float: First-moment decay coefficient.
        """
        ...

    @property
    def adam_beta2(self) -> float:
        """
        Returns:
            float: Second-moment decay coefficient.
        """
        ...

    @property
    def optimizer_eps(self) -> float:
        """
        Returns:
            float: AdamW denominator epsilon.
        """
        ...


def setup_optimizer(args: OptimizerConfig, model: nn.Module) -> torch.optim.AdamW:
    """
    Build AdamW with the configured training hyperparameters.

    Args:
        args (OptimizerConfig): Learning rate, weight decay, betas, and epsilon.
        model (nn.Module): Model whose parameters will be optimized.

    Returns:
        torch.optim.AdamW: Optimizer over all model parameters.
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.optimizer_eps,
        amsgrad=False,
    )
