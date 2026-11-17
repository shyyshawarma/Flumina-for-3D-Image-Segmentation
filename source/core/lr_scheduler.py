import math
from collections.abc import Mapping
from typing import Any, Protocol

from torch.optim import Optimizer


class SchedulerConfig(Protocol):
    """Read-only cosine schedule settings supplied by training configurations."""

    @property
    def num_epochs(self) -> int:
        """
        Returns:
            int: Number of training epochs.
        """
        ...

    @property
    def iter_per_epoch(self) -> int:
        """
        Returns:
            int: Scheduled optimizer updates per epoch.
        """
        ...

    @property
    def warmup_ratio(self) -> float:
        """
        Returns:
            float: Fraction of updates reserved for warmup.
        """
        ...

    @property
    def warmup_lr(self) -> float:
        """
        Returns:
            float: Initial learning rate during warmup.
        """
        ...

    @property
    def min_lr(self) -> float:
        """
        Returns:
            float: Final learning rate after cosine decay.
        """
        ...


class CosineLRScheduler:
    """
    Linear warmup and cosine decay indexed by accepted optimizer updates.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        total_updates: int,
        warmup_updates: int = 0,
        warmup_lr: float = 0.0,
        min_lr: float = 0.0,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Optimizer whose group learning rates are updated.
            total_updates (int): Total scheduled optimizer updates.
            warmup_updates (int): Number of linear warmup updates.
            warmup_lr (float): Initial rate of the highest-rate parameter group.
            min_lr (float): Final rate of the highest-rate parameter group.
        """
        if type(total_updates) is not int or type(warmup_updates) is not int:
            raise ValueError("Update counts must be integers")
        if not 0 <= warmup_updates < total_updates:
            raise ValueError("Expected 0 <= warmup_updates < total_updates")
        self.optimizer = optimizer
        self.total_updates = total_updates
        self.warmup_updates = warmup_updates
        self.warmup_lr = float(warmup_lr)
        self.min_lr = float(min_lr)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        maximum = max(self.base_lrs)
        if maximum <= 0 or any(not math.isfinite(lr) or lr < 0 for lr in self.base_lrs):
            raise ValueError("Learning rates must be finite with a positive maximum")
        if any(not 0 <= lr <= maximum for lr in (self.warmup_lr, self.min_lr)):
            raise ValueError("Warmup and minimum rates must be in [0, base_lr]")
        self.warmup_lrs = [self.warmup_lr * lr / maximum for lr in self.base_lrs]
        self.min_lrs = [self.min_lr * lr / maximum for lr in self.base_lrs]
        self.last_update = 0
        self._apply_update(0)

    def _learning_rates_at(self, num_updates: int) -> list[float]:
        """
        Compute group learning rates at an absolute update count.

        Args:
            num_updates (int): Number of completed optimizer updates.

        Returns:
            list[float]: Learning rates in parameter-group order.
        """
        if self.warmup_updates and num_updates <= self.warmup_updates:
            progress = num_updates / self.warmup_updates
            return [
                start + (end - start) * progress
                for start, end in zip(self.warmup_lrs, self.base_lrs)
            ]
        completed = max(0, num_updates - self.warmup_updates)
        progress = min(1.0, completed / (self.total_updates - self.warmup_updates))
        multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            minimum + (base - minimum) * multiplier
            for base, minimum in zip(self.base_lrs, self.min_lrs)
        ]

    def _apply_update(self, num_updates: int) -> None:
        """
        Apply the scheduled learning rates to optimizer groups.

        Args:
            num_updates (int): Number of completed optimizer updates.
        """
        for group, lr in zip(
            self.optimizer.param_groups, self._learning_rates_at(num_updates)
        ):
            group["lr"] = lr

    def step_update(self, num_updates: int) -> None:
        """
        Install learning rates for the next optimizer update.

        Args:
            num_updates (int): Absolute accepted-update count, nondecreasing and no larger than total_updates.
        """
        if (
            type(num_updates) is not int
            or not self.last_update <= num_updates <= self.total_updates
        ):
            raise ValueError("Update count must advance within the schedule")
        self.last_update = num_updates
        self._apply_update(self.last_update)

    def get_last_lr(self) -> list[float]:
        """
        Returns:
            list[float]: Current learning rates in parameter-group order.
        """
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        """
        Serialize the schedule configuration and progress.

        Returns:
            dict[str, int | float | str | list[float] | dict]: Schedule settings, base learning rates,
                and last accepted-update count.
        """
        return {
            "schedule_name": "cosine",
            "total_updates": self.total_updates,
            "warmup_updates": self.warmup_updates,
            "warmup_lr": self.warmup_lr,
            "min_lr": self.min_lr,
            "base_lrs": list(self.base_lrs),
            "last_update": self.last_update,
            "extra_state": {},
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """
        Restore progress after checking the schedule configuration.

        Args:
            state_dict (Mapping[str, int | float | str | list[float] | dict]): State returned by this scheduler's
                state_dict method.
        """
        expected = self.state_dict()
        changed = [
            key
            for key, value in expected.items()
            if key != "last_update" and state_dict.get(key) != value
        ]
        if changed:
            raise ValueError(f"Incompatible scheduler state: {changed}")
        update = state_dict.get("last_update")
        if type(update) is not int or not 0 <= update <= self.total_updates:
            raise ValueError("Saved update count is outside the schedule")
        self.last_update = update
        self._apply_update(update)


def setup_scheduler(args: SchedulerConfig, optimizer: Optimizer) -> CosineLRScheduler:
    """
    Build the cosine schedule for the epoch and update budget.

    Args:
        args (SchedulerConfig): Epoch, warmup, and learning-rate settings.
        optimizer (Optimizer): Optimizer whose learning rates follow the schedule.

    Returns:
        CosineLRScheduler: Scheduler initialized at update zero.
    """
    total = args.num_epochs * args.iter_per_epoch
    return CosineLRScheduler(
        optimizer,
        total_updates=total,
        warmup_updates=int(total * args.warmup_ratio),
        warmup_lr=args.warmup_lr,
        min_lr=args.min_lr,
    )
