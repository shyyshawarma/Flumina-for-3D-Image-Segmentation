from __future__ import annotations

import os
import random

import numpy as np
import torch


def setup_seed(random_seed: int, *, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy, and PyTorch random streams.

    Args:
        random_seed (int): Nonnegative process seed.
        deterministic (bool): Enable deterministic PyTorch algorithms and disable cuDNN benchmarking and TF32 when True.
    """

    seed = int(random_seed)
    if seed < 0:
        raise ValueError("random_seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.set_float32_matmul_precision("highest" if deterministic else "high")
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.allow_tf32 = not deterministic
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = not deterministic


__all__ = ["setup_seed"]
