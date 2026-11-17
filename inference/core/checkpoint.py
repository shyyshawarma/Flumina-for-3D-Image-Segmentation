from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


MODELS = ("Flumina-S", "Flumina-B", "Flumina-M", "Flumina-L")


@dataclass(frozen=True)
class LoadedCheckpoint:
    """
    Checkpoint state tensors on CPU with their saved metadata.

    Args:
        state_dict (dict[str, torch.Tensor]): CPU parameters and buffers keyed by unwrapped state name.
        metadata (dict[str, Any]): Saved model, dataset, seed, and epoch fields.
    """

    state_dict: dict[str, torch.Tensor]
    metadata: dict[str, Any]


def model_variant(name: str) -> str:
    """
    Normalize a formal Flumina model name to its size suffix.

    Args:
        name (str): Model name or size suffix, matched without case sensitivity.

    Returns:
        str: One of s, b, m, or l.
    """
    variant = name.lower().replace("flumina-", "").replace("flumina_", "")
    if variant not in {"s", "b", "m", "l"}:
        raise ValueError(f"Expected one of {', '.join(MODELS)}")
    return variant


def normalize_state_dict(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """
    Strip distributed and compiled wrapper prefixes from model state names.

    Args:
        state (Mapping[str, torch.Tensor]): Nonempty saved model state mapping.

    Returns:
        dict[str, torch.Tensor]: Unwrapped names with tensor values and shapes preserved.
    """
    if not state:
        raise ValueError("Checkpoint state dictionary is empty")
    normalized = {}
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise TypeError("Checkpoint weights must contain string/tensor pairs")
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        if key in normalized:
            raise ValueError(f"Wrapper removal duplicates checkpoint key {key!r}")
        normalized[key] = value
    return normalized


def read_checkpoint(path: str | Path) -> LoadedCheckpoint:
    """
    Load checkpoint state tensors on CPU using the restricted unpickler.

    Args:
        path (str | Path): File containing model_state_dict, state_dict, model, or a bare tensor mapping.

    Returns:
        LoadedCheckpoint: Normalized CPU state tensors and saved metadata, including the top-level epoch when it is available
            and absent from metadata.
    """
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError as error:
        if "mmap can only be used" not in str(error):
            raise
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint payload must be a mapping")
    if "model_state_dict" in payload:
        state = payload["model_state_dict"]
    elif "state_dict" in payload:
        state = payload["state_dict"]
    elif (
        isinstance(payload.get("model"), Mapping)
        and payload["model"]
        and all(torch.is_tensor(value) for value in payload["model"].values())
    ):
        state = payload["model"]
    elif payload and all(torch.is_tensor(value) for value in payload.values()):
        state = payload
    else:
        raise ValueError("Checkpoint has no unambiguous tensor state dictionary")
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint weights must be a mapping")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("Checkpoint metadata must be a mapping")
    metadata = dict(metadata)
    if "epoch" in payload and not torch.is_tensor(payload["epoch"]):
        metadata.setdefault("epoch", payload["epoch"])
    state = normalize_state_dict(state)
    return LoadedCheckpoint(state, metadata)


def load_model(
    name: str,
    config: Mapping[str, Any],
    checkpoint: LoadedCheckpoint,
    device: str = "cpu",
):
    """
    Construct a formal Flumina model and load its complete model state.

    Args:
        name (str): Formal model name or size suffix.
        config (Mapping[str, Any]): Dataset, channel, class, crop, seed, and model settings.
        checkpoint (LoadedCheckpoint): Normalized CPU state tensors and saved metadata.
        device (str): Device for the loaded model.

    Returns:
        torch.nn.Module: Evaluation model mapping inputs of shape (B, C, D, H, W) to raw class logits of shape (B, K, D, H, W).
    """
    from models import build_model

    variant = model_variant(name)
    kwargs = dict(config.get("model_kwargs") or {})
    if set(kwargs) - {"attention_post_norm"}:
        raise ValueError("Formal model_kwargs only supports attention_post_norm")
    kwargs.setdefault("attention_post_norm", config.get("attention_post_norm", True))
    model = build_model(
        name="flumina",
        variant=variant,
        in_channels=int(config["in_channels"]),
        out_channels=int(config["num_classes"]),
        crop_size=config["crop_size"],
        gradient_checkpointing=False,
        **kwargs,
    )
    model.load_state_dict(checkpoint.state_dict, strict=True)
    return model.to(device).eval()
