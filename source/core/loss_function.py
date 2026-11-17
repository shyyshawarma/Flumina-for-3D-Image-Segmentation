import torch
import torch.nn.functional as F
from torch import nn


class DiceLoss(nn.Module):
    """
    Soft multiclass Dice averaged across classes after batch aggregation.
    """

    def __init__(self, include_background: bool = False, smooth: float = 1e-5) -> None:
        """
        Args:
            include_background (bool): Include class zero in the class average.
            smooth (float): Constant added to the Dice numerator and denominator.
        """
        super().__init__()
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Class logits shaped (B, K, D, H, W).
            target (torch.Tensor): One-hot labels shaped (B, K, D, H, W).

        Returns:
            torch.Tensor: Scalar float32 Dice loss.
        """
        probabilities = torch.softmax(logits.float(), dim=1)
        target = target.float()
        axes = (0, 2, 3, 4)
        intersection = (probabilities * target).sum(axes)
        denominator = probabilities.sum(axes) + target.sum(axes)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        if not self.include_background and dice.numel() > 1:
            dice = dice[1:]
        return 1.0 - dice.mean()


class CrossEntropyLoss(nn.Module):
    """
    Multiclass cross entropy with channel-first one-hot targets.
    """

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Class logits shaped (B, K, D, H, W).
            target (torch.Tensor): One-hot labels shaped (B, K, D, H, W).

        Returns:
            torch.Tensor: Scalar cross entropy averaged over all voxels.
        """
        return F.cross_entropy(logits, target.argmax(dim=1))


class SegmentationLoss(nn.Module):
    """
    Sum of foreground Dice loss and multiclass cross entropy.
    """

    def __init__(self) -> None:
        """
        Initialize foreground Dice and multiclass cross-entropy criteria.
        """
        super().__init__()
        self.dice = DiceLoss(include_background=False)
        self.cross_entropy = CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Class logits shaped (B, K, D, H, W).
            target (torch.Tensor): One-hot labels shaped (B, K, D, H, W).

        Returns:
            torch.Tensor: Scalar sum of Dice loss and cross entropy.
        """
        return self.dice(logits, target) + self.cross_entropy(logits, target)
