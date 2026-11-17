from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

MODALITY_ORDER = ("FLAIR", "T1ce", "T1", "T2")
NUM_MODALITIES = len(MODALITY_ORDER)


def resolve_split_file(
    root: str | Path,
    *,
    role: str,
) -> Path:
    """
    Resolve a dataset's fixed train, validation, or test case list.

    Args:
        root (str | Path): Processed dataset directory.
        role (str): Split role: train, val, or test.

    Returns:
        Path: Existing case-list path under root.

    Raises:
        ValueError: If the split role is invalid.
        FileNotFoundError: If the case-list file is missing.
    """

    if role not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split role: {role}")
    path = Path(root) / f"{role}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    return path


def read_split_names(file_path: str | Path) -> list[str]:
    """
    Read and sort a validated case list.

    Args:
        file_path (str | Path): UTF-8 text file with one case identifier per line.

    Returns:
        list[str]: Sorted identifiers after blank lines are removed.

    Raises:
        FileNotFoundError: If file_path does not identify a file.
        ValueError: If the list is empty, repeats an identifier, or contains a path component.
    """

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")

    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError(f"Split file is empty: {path}")
    if any(Path(name).name != name or name in {".", ".."} for name in names):
        raise ValueError(f"Split file contains an invalid case identifier: {path}")

    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(
            f"Split file contains duplicate case identifiers: {duplicates[:5]}"
        )
    return sorted(names)


def ensure_disjoint_splits(**splits: Sequence[str]) -> None:
    """
    Reject case overlap between named splits.

    Args:
        **splits (Sequence[str]): Case identifiers for each keyword-named split.

    Raises:
        ValueError: If two splits share a case identifier.
    """

    split_names = list(splits)
    for index, left_name in enumerate(split_names):
        left = set(splits[left_name])
        following_index = index + 1
        for right_name in split_names[following_index:]:
            overlap = sorted(left.intersection(splits[right_name]))
            if overlap:
                raise ValueError(
                    f"Dataset splits {left_name!r} and {right_name!r} overlap: {overlap[:5]}"
                )


def channel_last_to_first(
    volume: np.ndarray,
    *,
    expected_channels: int | None = None,
) -> np.ndarray:
    """
    Move image channels from the last axis to the first.

    Args:
        volume (np.ndarray): Finite floating image with shape (X, Y, Z, C).
        expected_channels (int | None): Required channel count, or None to skip this check.

    Returns:
        np.ndarray: Contiguous float32 image with shape (C, X, Y, Z).

    Raises:
        ValueError: If the rank, channel count, or finite-value check fails.
        TypeError: If the image dtype is not floating point.
    """

    array = np.asarray(volume)
    if array.ndim != 4:
        raise ValueError(f"Expected a 4D XYZC volume, got shape {array.shape}")
    if expected_channels is not None and array.shape[-1] != int(expected_channels):
        raise ValueError(
            f"Expected {int(expected_channels)} image channels, got shape {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"Expected a floating-point volume, got dtype {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("Volume contains NaN or infinite values")
    return np.ascontiguousarray(np.moveaxis(array, -1, 0), dtype=np.float32)


def xyzc_to_cxyz(volume: np.ndarray) -> np.ndarray:
    """
    Convert a four-modality BraTS image to channel-first layout.

    Args:
        volume (np.ndarray): Floating image with shape (X, Y, Z, 4), ordered FLAIR, T1ce, T1, T2.

    Returns:
        np.ndarray: Contiguous float32 image with shape (4, X, Y, Z), preserving modality order.
    """

    return channel_last_to_first(volume, expected_channels=NUM_MODALITIES)


def validate_case_arrays(
    volume: np.ndarray,
    label: np.ndarray,
    *,
    num_classes: int,
    expected_channels: int | None = None,
) -> None:
    """
    Check aligned image and label arrays before tensor conversion.

    This check validates shapes and values; the caller handles dtype conversion.

    Args:
        volume (np.ndarray): Image with shape (X, Y, Z, C) and at least one channel.
        label (np.ndarray): Label map with shape (X, Y, Z); values must lie in [0, num_classes).
        num_classes (int): Exclusive upper bound on label values.
        expected_channels (int | None): Required channel count, or None to accept any positive count.

    Raises:
        ValueError: If shapes disagree, values are nonfinite, or labels fall outside the class range.
    """

    if volume.ndim != 4 or (
        expected_channels is not None and volume.shape[-1] != int(expected_channels)
    ):
        expected = (
            "any positive channel count"
            if expected_channels is None
            else str(expected_channels)
        )
        raise ValueError(
            f"Expected XYZC volume with {expected} channels, got {volume.shape}"
        )
    if volume.shape[-1] <= 0:
        raise ValueError(f"Volume must have at least one channel, got {volume.shape}")
    if label.shape != volume.shape[:3]:
        raise ValueError(
            f"Volume/label shape mismatch: {volume.shape} vs {label.shape}"
        )
    if not np.isfinite(volume).all():
        raise ValueError("Volume contains NaN or infinite values")
    if not np.isfinite(label).all():
        raise ValueError("Label contains NaN or infinite values")
    invalid = (label < 0) | (label >= int(num_classes))
    if np.any(invalid):
        values = np.unique(label[invalid]).tolist()
        raise ValueError(
            f"Invalid label values {values}; expected [0, {num_classes - 1}]"
        )
