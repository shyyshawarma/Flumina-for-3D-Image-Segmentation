from collections.abc import Mapping

import torch
from torch import nn


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Remove nested DDP and torch.compile wrappers.

    Args:
        model (nn.Module): Model with optional runtime wrappers.

    Returns:
        nn.Module: Innermost reachable module; wrapper cycles stop traversal.
    """
    current = model
    visited: set[int] = set()

    while id(current) not in visited:
        visited.add(id(current))

        if isinstance(wrapped := getattr(current, "module", None), nn.Module):
            current = wrapped
        elif isinstance(wrapped := getattr(current, "_orig_mod", None), nn.Module):
            current = wrapped
        else:
            break

    return current


def clone_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Detach and clone every state tensor on the CPU.

    Args:
        state_dict (Mapping[str, torch.Tensor]): Named model state tensors.

    Returns:
        dict[str, torch.Tensor]: Independent CPU copies preserving keys, tensor shapes, and dtypes.
    """
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def strip_runtime_prefixes(name: str) -> str:
    """
    Remove repeated DDP and torch.compile prefixes from a state key.

    Args:
        name (str): State-dict key with optional runtime prefixes.

    Returns:
        str: Key with leading "module." and "_orig_mod." prefixes removed.
    """
    stripped = name
    prefixes = ("module.", "_orig_mod.")

    while stripped.startswith(prefixes):
        for prefix in prefixes:
            stripped = stripped.removeprefix(prefix)

    return stripped


def load_model_state_dict(
    model: nn.Module, state_dict: Mapping[str, torch.Tensor], *, strict: bool = True
) -> object:
    """
    Load model weights after removing runtime wrapper prefixes.

    Args:
        model (nn.Module): Destination model with optional runtime wrappers.
        state_dict (Mapping[str, torch.Tensor]): Source tensors with matching destination shapes and compatible dtypes.
        strict (bool): Require an exact match between source and model keys.

    Returns:
        torch.nn.modules.module._IncompatibleKeys: Named tuple containing missing_keys and unexpected_keys
            from nn.Module.load_state_dict.
    """
    base_model = unwrap_model(model)
    normalized_state_dict: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        normalized_name = strip_runtime_prefixes(name)
        if normalized_name in normalized_state_dict:
            raise ValueError(
                "Runtime-prefix normalization produced a duplicate load key: "
                f"{normalized_name}"
            )
        normalized_state_dict[normalized_name] = tensor
    return base_model.load_state_dict(normalized_state_dict, strict=strict)
