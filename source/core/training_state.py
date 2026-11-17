from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn

from source.utils.runtime import clone_state_dict, unwrap_model


def _swap_named_tensors(
    named_tensors: Iterator[tuple[str, torch.Tensor]],
    shadow_tensors: dict[str, torch.Tensor],
) -> None:
    """
    Exchange live and EMA values while preserving tensor storage.

    Args:
        named_tensors (Iterator[tuple[str, torch.Tensor]]): Live named tensors with shapes matching their EMA counterparts.
        shadow_tensors (dict[str, torch.Tensor]): EMA tensors with the same names and shapes;
            each side keeps its device and dtype.
    """

    for name, tensor in named_tensors:
        shadow = shadow_tensors[name]
        live_value = tensor.detach().clone()
        tensor.copy_(shadow.to(device=tensor.device, dtype=tensor.dtype))
        shadow.copy_(live_value.to(device=shadow.device, dtype=shadow.dtype))


class StatefulEMA:
    """
    Exponential moving average of model parameters with copied buffers.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        """
        Args:
            model (nn.Module): Model used to initialize shadow parameters and buffers.
            decay (float): Exponential decay strictly between zero and one.
        """
        if not 0.0 < float(decay) < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}.")

        self.decay = float(decay)
        self.num_updates = 0
        base_model = unwrap_model(model)
        self.shadow_params = {
            name: parameter.detach().clone()
            for name, parameter in base_model.named_parameters()
        }
        self.shadow_buffers = {
            name: buffer.detach().clone() for name, buffer in base_model.named_buffers()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Average parameters and copy buffers from the live model.

        Args:
            model (nn.Module): Model with parameter and buffer names and shapes matching the EMA state. Values are converted to
                shadow dtypes/devices.
        """

        base_model = unwrap_model(model)
        current_params = dict(base_model.named_parameters())
        current_buffers = dict(base_model.named_buffers())

        one_minus_decay = 1.0 - self.decay
        for name, parameter in current_params.items():
            shadow = self.shadow_params[name]
            shadow.mul_(self.decay).add_(
                parameter.detach().to(device=shadow.device, dtype=shadow.dtype),
                alpha=one_minus_decay,
            )
        for name, buffer in current_buffers.items():
            shadow = self.shadow_buffers[name]
            shadow.copy_(buffer.detach().to(device=shadow.device, dtype=shadow.dtype))
        self.num_updates += 1

    @contextmanager
    def use_ema_weights(self, model: nn.Module) -> Iterator[None]:
        """
        Temporarily replace live model values with EMA values.

        Args:
            model (nn.Module): Model whose named tensors match the EMA state.

        Yields:
            None: Control with EMA weights installed; live values are restored when the context exits, including on exceptions.
        """

        base_model = unwrap_model(model)
        current_params = dict(base_model.named_parameters())
        current_buffers = dict(base_model.named_buffers())
        with torch.no_grad():
            _swap_named_tensors(iter(current_params.items()), self.shadow_params)
            _swap_named_tensors(iter(current_buffers.items()), self.shadow_buffers)
        try:
            yield
        finally:
            with torch.no_grad():
                _swap_named_tensors(iter(current_params.items()), self.shadow_params)
                _swap_named_tensors(iter(current_buffers.items()), self.shadow_buffers)

    def state_dict(self) -> dict[str, Any]:
        """
        Serialize EMA progress and independent CPU tensor copies.

        Returns:
            dict[str, float | int | dict[str, torch.Tensor]]: Decay, update count, and named parameter/buffer copies preserving
                source shapes and dtypes.
        """

        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": clone_state_dict(self.shadow_params),
            "shadow_buffers": clone_state_dict(self.shadow_buffers),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """
        Restore EMA settings and tensors after checking names and shapes.

        Args:
            state_dict (Mapping[str, float | int | Mapping[str, torch.Tensor]]): Saved decay, update count,
                parameters, and buffers.
        """

        decay = float(state_dict["decay"])
        if not 0.0 < decay < 1.0:
            raise ValueError(f"Checkpoint EMA decay must be in (0, 1), got {decay}.")

        shadow_params = state_dict.get("shadow_params")
        shadow_buffers = state_dict.get("shadow_buffers")
        if not isinstance(shadow_params, Mapping) or not isinstance(
            shadow_buffers, Mapping
        ):
            raise TypeError(
                "EMA checkpoint must contain parameter and buffer mappings."
            )

        self._load_tensor_mapping("parameter", self.shadow_params, shadow_params)
        self._load_tensor_mapping("buffer", self.shadow_buffers, shadow_buffers)
        self.decay = decay
        self.num_updates = int(state_dict.get("num_updates", 0))
        if self.num_updates < 0:
            raise ValueError("EMA num_updates cannot be negative.")

    @staticmethod
    def _validate_names_and_shapes(
        kind: str,
        expected: Mapping[str, torch.Tensor],
        actual: Mapping[str, torch.Tensor],
    ) -> None:
        """
        Check that two named tensor inventories have identical shapes.

        Args:
            kind (str): Inventory name used in errors.
            expected (Mapping[str, torch.Tensor]): Required names and tensor shapes.
            actual (Mapping[str, torch.Tensor]): Candidate names and tensor shapes.
        """

        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        if missing or extra:
            raise ValueError(
                f"EMA {kind} inventory changed; missing={missing}, extra={extra}."
            )
        mismatched = {
            name: (tuple(expected[name].shape), tuple(actual[name].shape))
            for name in expected
            if expected[name].shape != actual[name].shape
        }
        if mismatched:
            raise ValueError(f"EMA {kind} shapes changed: {mismatched}.")

    @classmethod
    def _load_tensor_mapping(
        cls,
        kind: str,
        destination: dict[str, torch.Tensor],
        source: Mapping[str, Any],
    ) -> None:
        """
        Copy checkpoint tensors into existing storage after validation.

        Args:
            kind (str): Inventory name used in errors.
            destination (dict[str, torch.Tensor]): Tensors whose storage, dtype, and device are retained.
            source (Mapping[str, torch.Tensor]): Saved tensors with matching names and shapes, converted to the destination
                dtype and device.
        """

        tensor_source = {
            str(name): value
            for name, value in source.items()
            if isinstance(value, torch.Tensor)
        }
        if len(tensor_source) != len(source):
            raise TypeError(f"EMA {kind} state contains non-tensor values.")
        cls._validate_names_and_shapes(kind, destination, tensor_source)
        for name, value in tensor_source.items():
            destination[name].copy_(
                value.detach().to(
                    device=destination[name].device, dtype=destination[name].dtype
                )
            )
