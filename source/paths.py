from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

_DATA_PATH_ENVIRONMENT_VARIABLES = ("DATA_PATH", "FLUMINA_DATA_ROOT")


def configured_data_path(
    explicit: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """
    Select the configured data path and expand its user directory.

    Args:
        explicit (str | Path | None): Data parent or direct processed dataset path.
        environ (Mapping[str, str] | None): Environment values; None uses os.environ.

    Returns:
        Path: First configured path from explicit, DATA_PATH, FLUMINA_DATA_ROOT, or the repository's data directory,
            in that order.
    """

    if explicit is not None and str(explicit):
        return Path(explicit).expanduser()

    environment = os.environ if environ is None else environ
    for variable in _DATA_PATH_ENVIRONMENT_VARIABLES:
        value = environment.get(variable)
        if value:
            return Path(value).expanduser()
    return REPOSITORY_ROOT / "data"


def _validated_dataset_directory_name(dataset_name: str) -> str:
    """
    Require a dataset name containing one path component.

    Args:
        dataset_name (str): Nonempty directory name other than "." or "..".

    Returns:
        str: Validated dataset directory name.
    """
    if (
        not isinstance(dataset_name, str)
        or not dataset_name
        or Path(dataset_name).name != dataset_name
        or dataset_name in {".", ".."}
    ):
        raise ValueError(
            f"dataset_name must be a path-safe directory name: {dataset_name!r}"
        )
    return dataset_name


def _looks_like_processed_dataset(path: Path) -> bool:
    """
    Recognize a dataset directory by its case-list files.

    Args:
        path (Path): Candidate processed dataset directory.

    Returns:
        bool: Whether labeled.txt and a train, val, or test list exist.
    """

    return (path / "labeled.txt").is_file() and any(
        (path / f"{role}.txt").is_file() for role in ("train", "val", "test")
    )


def resolve_dataset_root(
    data_path: str | Path,
    dataset_name: str,
    *,
    direct_dataset_name: str | None = None,
) -> Path:
    """
    Resolve a dataset under a shared parent or directly identified root.

    Args:
        data_path (str | Path): Shared data parent or processed dataset directory.
        dataset_name (str): Canonical dataset directory name.
        direct_dataset_name (str | None): Dataset identity for a direct root whose directory name differs from
            the canonical name.

    Returns:
        Path: Resolved dataset path; the selected path may not yet exist.
    """

    canonical_name = _validated_dataset_directory_name(dataset_name)
    if direct_dataset_name is not None:
        direct_dataset_name = _validated_dataset_directory_name(direct_dataset_name)

    base = Path(data_path).expanduser()
    if (base.exists() and not base.is_dir()) or (
        base.is_symlink() and not base.exists()
    ):
        raise NotADirectoryError(f"Configured data path is not a directory: {base}")

    nested = base / canonical_name
    direct_layout = _looks_like_processed_dataset(base)
    nested_layout = _looks_like_processed_dataset(nested)
    if direct_layout and nested_layout:
        raise ValueError(
            "Configured data path is ambiguous: both the path itself and its "
            f"{canonical_name!r} child look like processed datasets: {base}"
        )

    if direct_layout:
        if base.name == canonical_name:
            return base
        if direct_dataset_name is None:
            raise ValueError(
                "An arbitrary-name direct dataset root requires its configured "
                f"dataset identity: {base}"
            )
        if direct_dataset_name != canonical_name:
            raise ValueError(
                f"Direct data path {base} is configured for "
                f"{direct_dataset_name!r}, not {canonical_name!r}; cross-dataset "
                "evaluation requires a shared data parent"
            )
        return base

    if nested_layout or nested.is_dir():
        return nested
    if base.name == canonical_name:
        return base
    return nested


__all__ = [
    "REPOSITORY_ROOT",
    "configured_data_path",
    "resolve_dataset_root",
]
