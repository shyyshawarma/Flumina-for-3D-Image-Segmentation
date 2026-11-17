import argparse
import json
import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from source.dataset.runtime_contracts import get_dataset_contract
from source.paths import REPOSITORY_ROOT, configured_data_path


_TRAINING_CROPS = {
    "BraTS-2025-GLI": (128, 128, 128),
    "AMOS-2022": (64, 160, 192),
    "MAMA-MIA": (64, 160, 192),
    "KiTS-2023": (128, 128, 128),
    "LiTS": (128, 128, 128),
}


def _crop_size(values: Sequence[int]) -> tuple[int, int, int]:
    """
    Normalize one or three crop dimensions.

    Args:
        values (Sequence[int]): One isotropic size or three positive dimensions.

    Returns:
        tuple[int, int, int]: Crop dimensions in (D, H, W) order.
    """
    crop = tuple(int(value) for value in values)
    if len(crop) == 1:
        crop *= 3
    if len(crop) != 3 or any(value <= 0 for value in crop):
        raise ValueError("crop_size must contain one or three positive integers")
    return crop


def _boolean(value: str) -> bool:
    """
    Parse a command-line boolean value.

    Args:
        value (str): Case-insensitive true/false, yes/no, or 1/0.

    Returns:
        bool: Parsed boolean value.
    """
    match value.lower():
        case "true" | "1" | "yes":
            return True
        case "false" | "0" | "no":
            return False
        case _:
            raise argparse.ArgumentTypeError("Expected True or False")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """
    Training and final-evaluation settings for a segmentation run.

    Args:
        run_id (str): Run name used as the final output-directory component.
        model_name (str): Registered segmentation model name.
        model_variant (str): Model size variant: s, b, m, or l.
        dataset_name (str): Training dataset name.
        data_path (str): Shared data parent or processed dataset root.
        log_root (str): Parent directory for run artifacts.
        in_channels (int): Number of image channels C.
        num_classes (int): Number of label classes K, including background.
        crop_size (tuple[int, int, int]): Training and inference window (D, H, W).
        batch_size (int): Number of training samples per batch.
        num_workers (int): Number of data-loader workers.
        persistent_workers (bool): Keep loader workers between iterations.
        prefetch_factor (int): Number of batches prefetched per worker.
        base_lr (float): Peak AdamW learning rate.
        weight_decay (float): AdamW weight-decay coefficient.
        optimizer_eps (float): AdamW denominator epsilon.
        adam_beta1 (float): First-moment decay coefficient.
        adam_beta2 (float): Second-moment decay coefficient.
        num_epochs (int): Total number of training epochs.
        iter_per_epoch (int): Attempted optimizer updates per epoch.
        warmup_ratio (float): Fraction of scheduled updates used for warmup.
        warmup_lr (float): Learning rate at the start of warmup.
        min_lr (float): Learning rate at the end of cosine decay.
        grad_clip (float | None): Maximum gradient norm; None disables clipping.
        overlap (float): Sliding-window overlap fraction.
        top_k (int): Number of checkpoints retained by validation score.
        evaluation_top_k (int | None): Retained checkpoints to evaluate; None uses all top_k checkpoints.
        validation_start_epoch (int): First validation epoch, counted from one.
        validation_interval (int): Epochs between validation traversals.
        checkpoint_interval (int): Epochs between recovery checkpoints.
        seed (int): Process and data-loader random seed.
        deterministic (bool): Request deterministic PyTorch execution.
        mixed_precision (str): Autocast mode: no, fp16, or bf16.
        compile (bool): Compile the model with torch.compile.
        progress (bool): Display training and validation progress bars.
        use_ema (bool): Use EMA weights for validation and inference snapshots.
        ema_decay (float): Exponential decay for EMA parameters.
        gradient_checkpointing (bool): Enable model activation checkpointing.
        device (str): Training device or "auto" for CUDA/CPU selection.
        resume (str | None): Path to this run's last_state.pth checkpoint.
        allow_nonexact_resume (bool): Allow supported worker-setting and CUDA topology changes when restoring training state.
        inference_augument (bool): Average spatial flip views in final evaluation.
    """

    run_id: str
    model_name: str = "flumina"
    model_variant: str = "b"
    dataset_name: str = "BraTS-2025-GLI"
    data_path: str = field(default_factory=lambda: str(configured_data_path()))
    log_root: str = "runs"
    in_channels: int = 4
    num_classes: int = 5
    crop_size: tuple[int, int, int] = (128, 128, 128)
    batch_size: int = 2
    num_workers: int = 4
    persistent_workers: bool = True
    prefetch_factor: int = 2
    base_lr: float = 3e-4
    weight_decay: float = 5e-2
    optimizer_eps: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    num_epochs: int = 1000
    iter_per_epoch: int = 250
    warmup_ratio: float = 0.05
    warmup_lr: float = 1e-5
    min_lr: float = 1e-6
    grad_clip: float | None = 1.0
    overlap: float = 0.5
    top_k: int = 10
    evaluation_top_k: int | None = None
    validation_start_epoch: int = 400
    validation_interval: int = 10
    checkpoint_interval: int = 200
    seed: int = 42
    deterministic: bool = False
    mixed_precision: str = "bf16"
    compile: bool = True
    progress: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999
    gradient_checkpointing: bool = False
    device: str = "auto"
    resume: str | None = None
    allow_nonexact_resume: bool = False
    inference_augument: bool = True

    @property
    def run_dir(self) -> Path:
        """
        Returns:
            Path: Artifact directory grouped by dataset, model, variant, and run.
        """
        return (
            Path(self.log_root)
            / self.dataset_name
            / self.model_name
            / self.model_variant
            / self.run_id
        )

    @property
    def total_updates(self) -> int:
        """
        Returns:
            int: Scheduled update budget, num_epochs multiplied by iter_per_epoch.
        """
        return self.num_epochs * self.iter_per_epoch

    @property
    def num_classes_without_background(self) -> int:
        """
        Returns:
            int: Number of foreground classes, excluding class zero.
        """
        return self.num_classes - 1

    @property
    def resolved_evaluation_top_k(self) -> int:
        """
        Returns:
            int: Evaluation checkpoint count, defaulting to top_k.
        """
        return self.top_k if self.evaluation_top_k is None else self.evaluation_top_k

    def as_dict(self) -> dict[str, Any]:
        """
        Convert configuration fields to JSON-compatible values.

        Returns:
            dict[str, str | int | float | bool | list[int] | None]: Configuration with crop dimensions as a list and paths
                relative to the repository.
        """
        result = asdict(self)
        result["crop_size"] = list(self.crop_size)
        for name in ("data_path", "log_root", "resume"):
            if result[name] is not None:
                result[name] = os.path.relpath(
                    Path(result[name]).expanduser().resolve(), REPOSITORY_ROOT
                )
        return result

    @classmethod
    def from_args(cls, argv: Sequence[str] | None = None) -> "TrainingConfig":
        """
        Read JSON defaults, apply command-line overrides, and validate settings.

        Args:
            argv (Sequence[str] | None): Arguments to parse; None uses sys.argv.

        Returns:
            TrainingConfig: Validated settings with dataset channels, classes, data path, and crop dimensions resolved.
        """
        preliminary = argparse.ArgumentParser(add_help=False)
        preliminary.add_argument("--config")
        known, _ = preliminary.parse_known_args(argv)
        parser = build_training_parser()
        if known.config:
            values = json.loads(Path(known.config).read_text(encoding="utf-8"))
            if not isinstance(values, dict):
                raise ValueError("Training config must be a JSON object")
            # Saved run configurations include these retired default settings.
            for name, default in (("split_type", "Normal"), ("train_ratio", None)):
                if values.get(name) == default:
                    values.pop(name, None)
            unknown = set(values) - {item.name for item in fields(cls)}
            if unknown:
                raise ValueError(
                    f"Unknown training configuration fields: {sorted(unknown)}"
                )
            parser.set_defaults(**values)
        values = vars(parser.parse_args(argv))
        values.pop("config")
        if not values["run_id"]:
            parser.error("--run-id is required unless provided by --config")
        contract = get_dataset_contract(values["dataset_name"])
        if not contract.trainable:
            raise ValueError(f"{contract.dataset_name} is evaluation-only")
        values["dataset_name"] = contract.dataset_name
        values["data_path"] = str(configured_data_path(values["data_path"]))
        crop = values["crop_size"]
        values["crop_size"] = (
            _TRAINING_CROPS[contract.dataset_name] if crop is None else _crop_size(crop)
        )
        for name, expected in (
            ("in_channels", contract.in_channels),
            ("num_classes", contract.num_classes),
        ):
            supplied = values[name]
            if supplied is not None and supplied != expected:
                raise ValueError(
                    f"{name}={supplied} conflicts with {contract.dataset_name} ({expected})"
                )
            values[name] = expected
        return cls(**values)._validate()

    def _validate(self) -> "TrainingConfig":
        """
        Check configuration values and cross-field constraints.

        Returns:
            TrainingConfig: This configuration after validation succeeds.
        """
        if (
            not self.run_id.strip()
            or self.run_id in {".", ".."}
            or Path(self.run_id).name != self.run_id
        ):
            raise ValueError("run_id must be a nonempty path-safe name")
        if self.model_name != "flumina" or self.model_variant not in {
            "s",
            "b",
            "m",
            "l",
        }:
            raise ValueError("Select Flumina variant s, b, m, or l")
        if len(self.crop_size) != 3 or any(
            type(value) is not int or value <= 0 for value in self.crop_size
        ):
            raise ValueError("crop_size must contain three positive integers")
        for name in (
            "batch_size",
            "num_epochs",
            "iter_per_epoch",
            "prefetch_factor",
            "top_k",
            "validation_start_epoch",
            "validation_interval",
            "checkpoint_interval",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_workers < 0 or self.seed < 0:
            raise ValueError("num_workers and seed must be nonnegative")
        for name in ("base_lr", "optimizer_eps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        for name in ("adam_beta1", "adam_beta2", "warmup_ratio", "overlap"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        for name in ("warmup_lr", "min_lr"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= self.base_lr:
                raise ValueError(f"{name} must be in [0, base_lr]")
        if self.grad_clip is not None and (
            not math.isfinite(self.grad_clip) or self.grad_clip <= 0
        ):
            raise ValueError("grad_clip must be finite and positive or None")
        if not 0 < self.ema_decay < 1:
            raise ValueError("ema_decay must be in (0, 1)")
        if not 0 < self.resolved_evaluation_top_k <= self.top_k:
            raise ValueError("evaluation_top_k must be in [1, top_k]")
        if self.allow_nonexact_resume and self.resume is None:
            raise ValueError("--allow-nonexact-resume requires --resume")
        if self.device not in {"auto", "cpu", "cuda"} and not self.device.startswith(
            "cuda:"
        ):
            raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")
        return self


def build_training_parser() -> argparse.ArgumentParser:
    """
    Build the training command-line parser.

    Returns:
        argparse.ArgumentParser: Parser with model, dataset, optimization, checkpoint, and inference options.
    """
    parser = argparse.ArgumentParser(description="Train a Flumina segmentation model")
    parser.add_argument(
        "--config", help="JSON recipe; explicit CLI arguments take precedence"
    )
    parser.add_argument("--run-id", "--run-name", dest="run_id")
    parser.add_argument(
        "--model-name",
        "--model",
        choices=("flumina",),
        default="flumina",
        dest="model_name",
    )
    parser.add_argument("--model-variant", choices=("s", "b", "m", "l"), default="b")
    parser.add_argument(
        "--dataset-name", "--dataset", default="BraTS-2025-GLI", dest="dataset_name"
    )
    parser.add_argument("--data-path")
    parser.add_argument("--log-root", default="runs")
    parser.add_argument("--in-channels", type=int)
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--crop-size", type=int, nargs="+")
    for name in (
        "batch_size",
        "num_workers",
        "prefetch_factor",
        "num_epochs",
        "iter_per_epoch",
        "top_k",
        "validation_start_epoch",
        "validation_interval",
        "checkpoint_interval",
        "seed",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=int,
            default=TrainingConfig.__dataclass_fields__[name].default,
        )
    parser.add_argument("--evaluation-top-k", type=int)
    for name in (
        "base_lr",
        "weight_decay",
        "optimizer_eps",
        "adam_beta1",
        "adam_beta2",
        "warmup_ratio",
        "warmup_lr",
        "min_lr",
        "grad_clip",
        "overlap",
        "ema_decay",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=float,
            default=TrainingConfig.__dataclass_fields__[name].default,
        )
    for name in (
        "persistent_workers",
        "deterministic",
        "compile",
        "progress",
        "use_ema",
        "gradient_checkpointing",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            action=argparse.BooleanOptionalAction,
            default=TrainingConfig.__dataclass_fields__[name].default,
        )
    parser.add_argument(
        "--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume")
    parser.add_argument("--allow-nonexact-resume", action="store_true")
    parser.add_argument(
        "--inference_augument",
        "--inference-augument",
        type=_boolean,
        nargs="?",
        const=True,
        default=True,
        dest="inference_augument",
        help="Average eight flipped views during final test evaluation",
    )
    parser.add_argument(
        "--no-inference_augument",
        "--no-inference-augument",
        action="store_false",
        dest="inference_augument",
    )
    return parser


def get_args(argv: Sequence[str] | None = None) -> TrainingConfig:
    """
    Parse and validate training configuration.

    Args:
        argv (Sequence[str] | None): Arguments to parse; None uses sys.argv.

    Returns:
        TrainingConfig: Resolved training configuration.
    """
    return TrainingConfig.from_args(argv)
