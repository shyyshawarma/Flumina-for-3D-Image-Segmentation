from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence, cast

import torch


CONFIG_ENV = "FLUMINA_FA4_TUNING_FILE"
_trial_override: dict[str, Any] | None = None


def shape_key(shape: Sequence[int], dtype: str = "bfloat16") -> str:
    """
    Normalize a positive BF16 attention shape into a tuning lookup key.

    Args:
        shape (Sequence[int]): Query dimensions in (B, H, N, D) order.
        dtype (str): BF16 dtype name, optionally prefixed with torch.

    Returns:
        str: BF16 shape key with dimensions separated by x.
    """
    dimensions = tuple(int(value) for value in shape)
    if len(dimensions) != 4 or min(dimensions) <= 0:
        raise ValueError("FA4 tuning requires a positive BHND shape")
    if dtype.removeprefix("torch.") != "bfloat16":
        raise ValueError("The FA4 tuning experiment requires BF16")
    return "bfloat16:" + "x".join(map(str, dimensions))


def validate_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """
    Validate a supported FA4 configuration and normalize its tile dimensions.

    Args:
        options (Mapping[str, Any]): Public-default or private-forward kernel options.

    Returns:
        dict[str, Any]: Copied options with nonempty tile dimensions normalized to a list.
    """
    normalized = dict(options)
    kind = normalized.get("implementation")
    if kind == "public_default":
        if normalized != {"implementation": "public_default", "input_layout": "native"}:
            raise ValueError("Invalid public default FA4 configuration")
        return normalized
    if kind != "private_fwd" or normalized.get("input_layout") != "bnhd_contiguous":
        raise ValueError("Unknown tuned FA4 implementation/layout")
    if set(normalized) != {"implementation", "input_layout", "tile_mn", "num_splits"}:
        raise ValueError("Unexpected FA4 tuning options")
    tile = normalized["tile_mn"]
    if tile is not None:
        if (
            not isinstance(tile, (list, tuple))
            or len(tile) != 2
            or any(type(value) is not int or value <= 0 for value in tile)
        ):
            raise ValueError("tile_mn must contain two positive integers")
        if tile[0] % 64 or tile[1] % 16:
            raise ValueError("SM120 FA4 tiles require M divisible by 64 and N by 16")
        normalized["tile_mn"] = list(tile)
    if normalized["num_splits"] != 1:
        raise ValueError("SM120 forward only supports num_splits=1")
    return normalized


def configuration_id(options: Mapping[str, Any]) -> str:
    """
    Hash a normalized FA4 configuration for stable trial identification.

    Args:
        options (Mapping[str, Any]): Kernel options accepted by validate_options.

    Returns:
        str: First 16 hexadecimal characters of the normalized configuration hash.
    """
    return sha256(
        json.dumps(validate_options(options), sort_keys=True).encode()
    ).hexdigest()[:16]


def candidate_options(head_dim: int) -> list[dict[str, Any]]:
    """
    Enumerate effective SM120 tuning options; upstream fixes threads, stages, and splits.

    Args:
        head_dim (int): Unpadded attention head dimension from 1 through 128.

    Returns:
        list[dict[str, Any]]: Public-default and private-forward configurations within shared-memory bounds.
    """
    if not 1 <= head_dim <= 128:
        raise ValueError(
            "The current FA4 tuning adapter supports head dimensions up to 128"
        )
    padded = ((head_dim + 7) // 8) * 8
    # The kernel allocates head widths rounded to 16 despite accepting multiples of 8; use this for SMEM bounds.
    allocated = ((padded + 15) // 16) * 16
    candidates: list[dict[str, Any]] = [
        {"implementation": "public_default", "input_layout": "native"},
        {
            "implementation": "private_fwd",
            "input_layout": "bnhd_contiguous",
            "tile_mn": None,
            "num_splits": 1,
        },
    ]
    for tile_m in (64, 128, 192, 256):
        for tile_n in (32, 64, 96, 128, 192, 256):
            # Leave room in the 99 KiB SMEM for alignment/barriers; the compiler rejects any unsupported tiles.
            required = 2 * allocated * (tile_m + 2 * tile_n)
            if required > 96 * 1024:
                continue
            candidates.append(
                {
                    "implementation": "private_fwd",
                    "input_layout": "bnhd_contiguous",
                    "tile_mn": [tile_m, tile_n],
                    "num_splits": 1,
                }
            )
    return candidates


@lru_cache(maxsize=4)
def environment_fingerprint(device: str) -> dict[str, Any]:
    """
    Identify the hardware, installed dependencies, adapter sources, and architecture settings.

    Args:
        device (str): CUDA device used for the tuning experiment.

    Returns:
        dict[str, Any]: Hardware details, package versions, source hashes, and relevant environment variables.
    """
    requested = torch.device(device)
    properties = torch.cuda.get_device_properties(requested)
    interface = import_module("flash_attn.cute.interface")
    package_root = Path(cast(str, interface.__file__)).parent
    source_files = {
        "fa4_interface": Path(cast(str, interface.__file__)),
        "fa4_fwd": package_root / "flash_fwd.py",
        "fa4_fwd_sm120": package_root / "flash_fwd_sm120.py",
        "flumina_adapter": Path(__file__).with_name("_attention.py"),
        "flumina_tuning": Path(__file__),
    }
    return {
        "gpu_name": properties.name,
        "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
        "compute_capability": [properties.major, properties.minor],
        "multiprocessor_count": properties.multi_processor_count,
        "total_memory": properties.total_memory,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "packages": {
            name: metadata.version(name)
            for name in (
                "flash-attn-4",
                "nvidia-cutlass-dsl",
                "apache-tvm-ffi",
                "quack-kernels",
            )
        },
        "source_sha256": {
            name: sha256(path.read_bytes()).hexdigest()
            for name, path in source_files.items()
        },
        "architecture_environment": {
            name: os.environ.get(name)
            for name in (
                "CUTE_DSL_ARCH",
                "FLASH_ATTENTION_ARCH",
                "FLASH_ATTENTION_NUM_SMS",
                "CUTE_DSL_PTXAS_PATH",
            )
        },
    }


@lru_cache(maxsize=1)
def _load_configuration(
    path: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
    """
    Load completed tuning choices backed by successful comparative measurements.

    Args:
        path (str): Path to the JSON tuning result.

    Returns:
        tuple[dict[str, Any], dict[str, dict[str, Any]], str]: Parsed document, per-shape options, and file SHA-256.
    """
    raw = Path(path).read_bytes()
    document = json.loads(raw)
    if document.get("status") != "complete":
        raise RuntimeError("FA4 tuning file must be a complete tuning result")
    choices = {}
    for row in document["shapes"]:
        key = shape_key(row["shape"], row.get("dtype", "bfloat16"))
        if key in choices or row.get("status") != "complete":
            raise RuntimeError(f"Duplicate or incomplete FA4 tuning shape: {key}")
        options = validate_options(row["best_options"])
        matching = [
            trial
            for trial in row["trials"]
            if trial.get("status") == "passed" and trial.get("options") == options
        ]
        if not matching or len(row["trials"]) < 2:
            raise RuntimeError(f"FA4 choice lacks completed comparative trials: {key}")
        choices[key] = options
    if not choices:
        raise RuntimeError("FA4 tuning configuration is empty")
    return document, choices, sha256(raw).hexdigest()


@lru_cache(maxsize=4)
def _resolved_configuration_path(path: str) -> str:
    # Cache path resolution during preflight to avoid filesystem syscalls in every compiled attention operation.
    """
    Resolve and cache the immutable tuning file path.

    Args:
        path (str): Configured tuning file path.

    Returns:
        str: Absolute path with symbolic links resolved.
    """
    return str(Path(path).resolve())


@lru_cache(maxsize=4)
def _verified_configuration(path: str, device: str) -> dict[str, dict[str, Any]]:
    """
    Require the measured environment and source fingerprint before using tuning choices.

    Args:
        path (str): Resolved tuning result path.
        device (str): CUDA device on which the choices will run.

    Returns:
        dict[str, dict[str, Any]]: Verified options indexed by BF16 attention shape.
    """
    document, choices, _ = _load_configuration(path)
    actual = environment_fingerprint(device)
    if document["environment_fingerprint"] != actual:
        differences = [
            key
            for key in actual
            if document["environment_fingerprint"].get(key) != actual[key]
        ]
        raise RuntimeError(
            "FA4 tuning environment/source mismatch: " + ", ".join(differences)
        )
    return choices


def selected_options(shape: Sequence[int], device: str) -> dict[str, Any] | None:
    """
    Select trial options or a measured configuration for the requested attention shape.

    Args:
        shape (Sequence[int]): Query dimensions in (B, H, N, D) order.
        device (str): CUDA device on which attention will run.

    Returns:
        dict[str, Any] | None: Trial or measured options, or None when tuning is disabled.
    """
    if _trial_override is not None:
        return _trial_override
    path = os.environ.get(CONFIG_ENV)
    if not path:
        return None
    choices = _verified_configuration(_resolved_configuration_path(path), device)
    key = shape_key(shape)
    if key not in choices:
        raise RuntimeError(
            f"FA4 tuning has no measured configuration for {key}; run the tuner first"
        )
    return choices[key]


def set_trial_override(options: Mapping[str, Any] | None) -> None:
    """
    Change opaque-operation options between tuning trials without changing model graphs.

    Args:
        options (Mapping[str, Any] | None): Trial configuration, or None to clear the process-local override.

    Returns:
        None: Updates the process-local override; an active tuning file raises an error.
    """
    global _trial_override
    if os.environ.get(CONFIG_ENV):
        raise RuntimeError(
            f"Unset {CONFIG_ENV} before searching new FA4 configurations"
        )
    _trial_override = None if options is None else validate_options(options)


def tuning_metadata(device: str | None = None) -> dict[str, Any]:
    """
    Describe optional measured tuning choices and their recorded environment.

    Args:
        device (str | None): CUDA device to verify, or None to read metadata without environment validation.

    Returns:
        dict[str, Any]: Tuning policy and, when enabled, the file path, hash, choices, and environment fingerprint.
    """
    path = os.environ.get(CONFIG_ENV)
    if not path:
        return {"tuning_enabled": False, "tuning_policy": "upstream public default"}
    resolved = _resolved_configuration_path(path)
    document, choices, digest = _load_configuration(resolved)
    if device is not None:
        _verified_configuration(resolved, device)
    return {
        "tuning_enabled": True,
        "tuning_file": resolved,
        "tuning_file_sha256": digest,
        "tuning_policy": "strict measured per-BHND-shape choice; no missing-shape fallback",
        "tuning_choices": choices,
        "tuning_environment_fingerprint": document["environment_fingerprint"],
    }
