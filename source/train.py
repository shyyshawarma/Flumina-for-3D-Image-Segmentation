from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from source.config import TrainingConfig
from source.core import SegmentationLoss
from source.core.training_state import StatefulEMA
from source.csv_outputs import initialize_validation_history
from source.core.lr_scheduler import CosineLRScheduler, setup_scheduler
from source.core.optimizer import setup_optimizer
from source.utils.runtime import load_model_state_dict, unwrap_model
from source.paths import REPOSITORY_ROOT, resolve_dataset_root
from source.validate import ValidationResult, validate_model


@dataclass(frozen=True, slots=True)
class EpochStats:
    """
    Update counts and scalar summaries for one training epoch.

    Args:
        epoch (int): One-based epoch index.
        attempted_updates (int): Number of batches attempted in the epoch.
        successful_updates (int): Number of accepted optimizer updates.
        nonfinite_skips (int): Updates skipped for nonfinite loss or gradients.
        mean_loss (float): Mean accepted-update loss, or NaN if none succeeded.
        learning_rate (float): Highest parameter-group rate at the end of the epoch.
    """

    epoch: int
    attempted_updates: int
    successful_updates: int
    nonfinite_skips: int
    mean_loss: float
    learning_rate: float


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    """
    Inference checkpoint ranked by its validation score.

    Args:
        epoch (int): One-based checkpoint epoch.
        path (Path): Inference-weights path.
        validation_score (float): Score used to rank the checkpoint.
    """

    epoch: int
    path: Path
    validation_score: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """
    Run artifacts and completed training progress.

    Args:
        run_dir (Path): Run artifact directory.
        last_checkpoint (Path): Final inference-weights path.
        top_checkpoints (tuple[CheckpointRecord, ...]): Retained checkpoints in validation-rank order.
        completed_epochs (int): Total completed epochs.
        successful_updates (int): Total accepted optimizer updates.
    """

    run_dir: Path
    last_checkpoint: Path
    top_checkpoints: tuple[CheckpointRecord, ...]
    completed_epochs: int
    successful_updates: int


def _atomic_json(path: Path, payload: object) -> None:
    """
    Write a complete JSON document before replacing the destination.

    Args:
        path (Path): Output JSON path.
        payload (dict[str, Any]): JSON-serializable fields with finite numbers.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch_save(path: Path, payload: object) -> None:
    """
    Serialize a checkpoint before atomically replacing its destination.

    Args:
        path (Path): Output checkpoint path.
        payload (dict[str, Any]): Model weights or complete training state.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_inference_weights_atomic(
    path: Path,
    model: nn.Module,
    *,
    metadata: Mapping[str, Any],
) -> None:
    """
    Save unwrapped model weights as independent CPU tensors.

    Args:
        path (Path): Output inference-checkpoint path.
        model (nn.Module): Model whose current weights are saved.
        metadata (Mapping[str, Any]): Checkpoint identity and descriptive fields.
    """

    base_model = unwrap_model(model)
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in base_model.state_dict().items()
    }
    _atomic_torch_save(
        path,
        {
            "model_state_dict": state,
            "metadata": dict(metadata),
        },
    )


def _ema_inference_metadata(ema: StatefulEMA | None) -> dict[str, Any]:
    """
    Describe the EMA state used for an inference snapshot.

    Args:
        ema (StatefulEMA | None): Moving-average state, or None for live weights.

    Returns:
        dict[str, str | float | int]: EMA variant, decay, and update count, or an empty mapping when EMA is disabled.
    """

    if ema is None:
        return {}
    return {
        "weights_variant": "ema",
        "ema_decay": float(ema.decay),
        "ema_num_updates": int(ema.num_updates),
    }


def fixed_iteration_batches(loader: Iterable[object], count: int) -> Iterator[object]:
    """
    Yield a fixed batch count, restarting a finite loader as needed.

    Args:
        loader (Iterable[tuple | list | Mapping[str, torch.Tensor]]): Nonempty training-batch iterable that can be restarted
            when exhausted.
        count (int): Positive number of batches to yield.

    Yields:
        tuple | list | Mapping[str, torch.Tensor]: Next training batch unchanged.
    """

    if isinstance(count, bool) or int(count) != count or int(count) <= 0:
        raise ValueError("count must be a positive integer")
    iterator = iter(loader)
    for _ in range(int(count)):
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
            try:
                yield next(iterator)
            except StopIteration as exc:
                raise ValueError("Training loader is empty") from exc


def _unpack_training_batch(batch: object) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract image and target tensors from a training batch.

    Args:
        batch (tuple | list | Mapping[str, torch.Tensor]): Sequence beginning with image/target tensors, or mapping using
            image/volume and target/label.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Images (B, C, D, H, W) and targets (B, D, H, W) or (B, K, D, H, W), in that order.
    """
    if isinstance(batch, Mapping):
        return batch.get("image", batch.get("volume")), batch.get(
            "target", batch.get("label")
        )
    sequence = cast(Sequence[torch.Tensor], batch)
    return sequence[0], sequence[1]


def _one_hot_target(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert categorical labels to one-hot floating targets.

    Args:
        target (torch.Tensor): Labels (B, D, H, W) or one-hot targets (B, K, D, H, W).
        num_classes (int): Number of one-hot channels K.

    Returns:
        torch.Tensor: Float32 one-hot targets shaped (B, K, D, H, W).
    """
    if target.ndim == 5:
        return target.float()
    return F.one_hot(target.long(), num_classes=num_classes).movedim(-1, 1).float()


def _autocast_context(device: torch.device, mixed_precision: str):
    """
    Select BF16, CUDA FP16, or an inactive autocast context.

    Args:
        device (torch.device): Device used for training.
        mixed_precision (str): Autocast mode: no, fp16, or bf16.

    Returns:
        torch.autocast | nullcontext: BF16 context, CUDA-only FP16 context, or a no-op context for the remaining combinations.
    """

    match mixed_precision:
        case "bf16":
            return torch.autocast(device.type, dtype=torch.bfloat16)
        case "fp16" if device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.float16)
        case _:
            return nullcontext()


def _gradients_are_finite(model: nn.Module) -> bool:
    """
    Check all materialized parameter gradients.

    Args:
        model (nn.Module): Model whose parameter gradients are inspected.

    Returns:
        bool: Whether every existing gradient contains only finite values.
    """

    return all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def train_one_epoch(
    model: nn.Module,
    train_loader: Iterable[object],
    optimizer: Optimizer,
    scheduler: CosineLRScheduler,
    loss_function: nn.Module,
    *,
    epoch: int,
    iter_per_epoch: int,
    device: torch.device | str,
    successful_updates: int = 0,
    grad_clip: float | None = None,
    mixed_precision: str = "no",
    scaler: torch.amp.GradScaler | None = None,
    ema: StatefulEMA | None = None,
    on_update: Callable[[int, float], None] | None = None,
    show_progress: bool = False,
    total_epochs: int | None = None,
) -> EpochStats:
    """
    Run a fixed number of attempted optimizer updates.

    Args:
        model (nn.Module): Maps images (B, C, D, H, W) to logits (B, K, D, H, W).
        train_loader (Iterable[tuple | list | Mapping[str, torch.Tensor]]): Batches with images and categorical (B, D, H, W) or
            one-hot (B, K, D, H, W) labels.
        optimizer (Optimizer): Optimizer for the model parameters.
        scheduler (CosineLRScheduler): Schedule advanced after accepted updates.
        loss_function (nn.Module): Objective returning a scalar tensor from logits and one-hot targets shaped (B, K, D, H, W).
        epoch (int): One-based epoch index.
        iter_per_epoch (int): Number of attempted updates.
        device (torch.device | str): Training device.
        successful_updates (int): Accepted-update count before this epoch.
        grad_clip (float | None): Maximum gradient norm; None disables clipping.
        mixed_precision (str): Autocast mode: no, fp16, or bf16.
        scaler (torch.amp.GradScaler | None): Optional FP16 gradient scaler.
        ema (StatefulEMA | None): Moving average updated after accepted steps.
        on_update (Callable[[int, float], None] | None): Callback receiving the absolute accepted-update count and scalar loss.
        show_progress (bool): Display a training progress bar.
        total_epochs (int | None): Optional total shown beside the current epoch.

    Returns:
        EpochStats: Attempt, success, and skip counts, mean accepted loss, and the ending learning rate for this epoch.
    """

    resolved_device = torch.device(device)

    model.train()
    attempted = 0
    successful = 0
    nonfinite = 0
    loss_sum = 0.0
    description = (
        f"Train {epoch}/{int(total_epochs)}"
        if total_epochs is not None
        else f"Train epoch {epoch}"
    )
    batches = fixed_iteration_batches(train_loader, iter_per_epoch)
    progress_bar = (
        tqdm(
            batches,
            total=int(iter_per_epoch),
            desc=description,
            unit="iter",
            dynamic_ncols=True,
            leave=True,
        )
        if show_progress
        else None
    )
    iterator: Iterable[object] = progress_bar if progress_bar is not None else batches

    def refresh_progress(last_loss: float | None) -> None:
        """
        Refresh the progress bar with loss, rate, and update counts.

        Args:
            last_loss (float | None): Latest accepted loss; None marks a skipped step.
        """
        if progress_bar is None:
            return
        learning_rate = max(float(group["lr"]) for group in optimizer.param_groups)
        postfix = {
            "loss": "nonfinite" if last_loss is None else f"{last_loss:.5g}",
            "mean": f"{loss_sum / successful:.5g}" if successful else "n/a",
            "lr": f"{learning_rate:.3g}",
            "ok": successful,
            "skip": nonfinite,
        }
        progress_bar.set_postfix(postfix, refresh=False)

    try:
        for batch in iterator:
            attempted += 1
            images, targets = _unpack_training_batch(batch)
            images = images.to(resolved_device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(resolved_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with _autocast_context(resolved_device, mixed_precision):
                logits = model(images)
                normalized_target = _one_hot_target(targets, int(logits.shape[1]))
                loss = loss_function(logits, normalized_target)
            if not torch.isfinite(loss.detach()):
                nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                refresh_progress(None)
                continue

            use_scaler = scaler is not None and scaler.is_enabled()
            active_scaler = cast(torch.amp.GradScaler, scaler)
            if use_scaler:
                active_scaler.scale(loss).backward()
                active_scaler.unscale_(optimizer)
            else:
                loss.backward()

            step_succeeded = _gradients_are_finite(model)
            if step_succeeded:
                previous_scale = (
                    float(active_scaler.get_scale()) if use_scaler else None
                )
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                if use_scaler:
                    active_scaler.step(optimizer)
                    active_scaler.update()
                    step_succeeded = float(active_scaler.get_scale()) >= cast(
                        float, previous_scale
                    )
                else:
                    optimizer.step()
            elif use_scaler:
                # unscale_ recorded the overflow; update the scale after skipping.
                active_scaler.update()

            if not step_succeeded:
                nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                refresh_progress(None)
                continue
            if ema is not None:
                ema.update(model)
            successful += 1
            absolute_update = successful_updates + successful
            scheduler.step_update(absolute_update)
            scalar_loss = float(loss.detach().float().cpu())
            loss_sum += scalar_loss
            refresh_progress(scalar_loss)
            if on_update is not None:
                on_update(absolute_update, scalar_loss)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    mean_loss = loss_sum / successful if successful else float("nan")
    learning_rate = max(float(group["lr"]) for group in optimizer.param_groups)

    return EpochStats(
        epoch=int(epoch),
        attempted_updates=attempted,
        successful_updates=successful,
        nonfinite_skips=nonfinite,
        mean_loss=mean_loss,
        learning_rate=learning_rate,
    )


class TopKCheckpointManager:
    """
    Validation-ranked inference checkpoints with atomic inventory updates.
    """

    def __init__(self, directory: str | Path, k: int) -> None:
        """
        Args:
            directory (str | Path): Directory owned by this checkpoint inventory.
            k (int): Positive number of highest-scoring checkpoints to retain.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        self.directory = Path(directory)
        if self.directory.is_symlink():
            raise ValueError("Checkpoint directory cannot be a symlink")
        self.directory.mkdir(parents=True, exist_ok=True)
        self._canonical_directory = self.directory.resolve(strict=True)
        self.k = int(k)
        self._records: list[CheckpointRecord] = []

    @property
    def records(self) -> tuple[CheckpointRecord, ...]:
        """
        Returns:
            tuple[CheckpointRecord, ...]: Retained records sorted by descending validation score, then ascending epoch.
        """
        return tuple(
            sorted(
                self._records,
                key=lambda record: (-record.validation_score, record.epoch),
            )
        )

    def _validated_directory(self) -> Path:
        """
        Verify that the managed directory still has its original identity.

        Returns:
            Path: Canonical managed directory after checking for path changes.
        """
        resolved = self.directory.resolve(strict=True)
        if (
            self.directory.is_symlink()
            or resolved != self._canonical_directory
            or not resolved.is_dir()
        ):
            raise ValueError("Checkpoint directory identity changed")
        return resolved

    def _validated_member_path(self, path: Path) -> Path:
        """
        Require a regular checkpoint file inside the managed directory.

        Args:
            path (Path): Full checkpoint path or filename within the managed directory.

        Returns:
            Path: Checkpoint path expressed under the managed directory.
        """
        candidate = Path(path)
        if candidate.parent == Path("."):
            candidate = self.directory / candidate
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate.resolve().parent != self._validated_directory()
        ):
            raise ValueError(
                f"Checkpoint must be a regular file in {self.directory}: {candidate}"
            )
        return self.directory / candidate.name

    @staticmethod
    def _owned_checkpoint_epoch(name: str) -> int | None:
        """
        Recognize the manager's canonical checkpoint filenames.

        Args:
            name (str): Candidate filename.

        Returns:
            int | None: Positive epoch encoded by epoch_NNNN.pth, or None for filenames outside the manager's naming convention.
        """
        if not name.startswith("epoch_") or not name.endswith(".pth"):
            return None
        encoded = name[6:-4]
        if not encoded.isascii() or not encoded.isdigit():
            return None
        epoch = int(encoded)
        return epoch if epoch > 0 and name == f"epoch_{epoch:04d}.pth" else None

    def consider(
        self,
        model: nn.Module,
        *,
        epoch: int,
        validation_score: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> CheckpointRecord | None:
        """
        Stage weights when the candidate enters the retained checkpoint set.

        Args:
            model (nn.Module): Model supplying the candidate inference weights.
            epoch (int): Positive epoch not already in the retained set.
            validation_score (float): Finite ranking score; higher is better.
            metadata (Mapping[str, Any] | None): Optional inference metadata.

        Returns:
            CheckpointRecord | None: Staged candidate, or None when its score does not improve a full retained set.
        """
        score = float(validation_score)
        if epoch <= 0 or not math.isfinite(score):
            raise ValueError("Checkpoint epoch must be positive and score finite")
        if any(record.epoch == epoch for record in self._records):
            raise ValueError(f"Epoch {epoch} was already considered")
        if len(self._records) >= self.k and score <= self.records[-1].validation_score:
            return None
        self._validated_directory()
        path = self.directory / f"epoch_{epoch:04d}.pth"
        _save_inference_weights_atomic(
            path,
            model,
            metadata={
                **dict(metadata or {}),
                "epoch": epoch,
                "validation_score": score,
            },
        )
        candidate = CheckpointRecord(epoch, path, score)
        self._records = sorted(
            (*self._records, candidate),
            key=lambda record: (-record.validation_score, record.epoch),
        )[: self.k]
        return candidate

    def commit_inventory(self) -> None:
        """
        Publish the inventory and remove stale owned weight files.

        Commit resumable training state first so it references only retained files.
        """
        directory = self._validated_directory()
        records = self.records
        _atomic_json(
            self.directory / "top_k.json",
            {
                "top_k": self.k,
                "checkpoints": [
                    {
                        "rank": rank,
                        "epoch": record.epoch,
                        "checkpoint": record.path.name,
                        "validation_score": record.validation_score,
                    }
                    for rank, record in enumerate(records, 1)
                ],
            },
        )
        retained = {record.path.name for record in records}
        for candidate in self.directory.iterdir():
            if (
                candidate.name in retained
                or self._owned_checkpoint_epoch(candidate.name) is None
            ):
                continue
            if (
                not candidate.is_symlink()
                and candidate.is_file()
                and candidate.resolve().parent == directory
            ):
                candidate.unlink()

    def restore(self, records: Iterable[CheckpointRecord]) -> None:
        """
        Restore a saved inventory after checking checkpoint paths and epochs.

        Args:
            records (Iterable[CheckpointRecord]): Retained records with unique epochs and regular checkpoint files inside the
                managed directory.
        """
        restored = list(records)
        if len(restored) > self.k or len({record.epoch for record in restored}) != len(
            restored
        ):
            raise ValueError("Invalid retained checkpoint count or duplicate epochs")
        for record in restored:
            path = self._validated_member_path(record.path)
            if (
                record.epoch <= 0
                or not math.isfinite(record.validation_score)
                or path.name != f"epoch_{record.epoch:04d}.pth"
            ):
                raise ValueError(f"Invalid retained checkpoint: {path}")
        self._records = [
            CheckpointRecord(
                record.epoch, self.directory / record.path.name, record.validation_score
            )
            for record in restored
        ]


_RESUME_IDENTITY_FIELDS = (
    "run_id",
    "model_name",
    "model_variant",
    "dataset_name",
    "data_path",
    "in_channels",
    "num_classes",
    "crop_size",
    "batch_size",
    "base_lr",
    "weight_decay",
    "optimizer_eps",
    "adam_beta1",
    "adam_beta2",
    "num_epochs",
    "iter_per_epoch",
    "warmup_ratio",
    "warmup_lr",
    "min_lr",
    "grad_clip",
    "overlap",
    "top_k",
    "validation_interval",
    "seed",
    "deterministic",
    "mixed_precision",
    "gradient_checkpointing",
    "num_workers",
    "persistent_workers",
    "prefetch_factor",
    "validation_start_epoch",
    "checkpoint_interval",
    "compile",
    "use_ema",
    "ema_decay",
)

_NONEXACT_MUTABLE_IDENTITY_FIELDS = frozenset(
    {
        "num_workers",
        "persistent_workers",
        "prefetch_factor",
        "checkpoint_interval",
    }
)


def _normalized_identity_value(
    name: str,
    value: Any,
    *,
    dataset_name: str | None = None,
) -> Any:
    """
    Normalize crop dimensions and dataset paths for resume comparison.

    Args:
        name (str): Training-configuration field name.
        value (str | Path | int | float | bool | tuple[int, int, int] | None): Field value before normalization.
        dataset_name (str | None): Dataset identity used to resolve data_path.

    Returns:
        str | int | float | bool | list[int] | None: Crop dimensions as a list, repository-relative dataset path, or the
            original scalar field value.
    """
    match name:
        case "crop_size":
            return [int(item) for item in value]
        case "data_path":
            if dataset_name is None:
                return os.path.relpath(
                    Path(str(value)).resolve(strict=False), REPOSITORY_ROOT
                )
            dataset_root = resolve_dataset_root(
                value,
                dataset_name,
                direct_dataset_name=dataset_name,
            )
            return os.path.relpath(dataset_root.resolve(strict=False), REPOSITORY_ROOT)
        case _:
            return value


def _training_identity(args: TrainingConfig) -> dict[str, Any]:
    """
    Collect normalized settings used to validate training continuation.

    Args:
        args (TrainingConfig): Requested training configuration.

    Returns:
        dict[str, str | int | float | bool | list[int] | None]: Resume identity, including the optimizer, schedule, and
            repository-relative run path.
    """
    identity = {
        name: _normalized_identity_value(
            name, getattr(args, name), dataset_name=args.dataset_name
        )
        for name in _RESUME_IDENTITY_FIELDS
    }
    identity.update(optimizer="adamw", lr_scheduler="cosine")
    identity["run_dir"] = os.path.relpath(
        Path(args.run_dir).resolve(strict=False), REPOSITORY_ROOT
    )
    return identity


def _validate_training_identity(state: Mapping[str, Any], args: TrainingConfig) -> None:
    """
    Check saved settings against the requested resume configuration.

    Args:
        state (Mapping[str, Any]): Training checkpoint containing an identity mapping.
        args (TrainingConfig): Requested configuration and resume policy.
    """
    stored = state.get("identity")
    if not isinstance(stored, Mapping):
        raise ValueError("Training state is missing its configuration identity")
    stored = dict(stored)
    # Retired selectors must describe the fixed membership used for continuation.
    for name, default in (("split_type", "Normal"), ("train_ratio", None)):
        if stored.get(name, default) != default:
            raise ValueError(f"Resume state uses unsupported {name}: {stored[name]!r}")
    if "data_path" in stored and isinstance(stored.get("dataset_name"), str):
        stored["data_path"] = _normalized_identity_value(
            "data_path",
            REPOSITORY_ROOT / stored["data_path"],
            dataset_name=stored["dataset_name"],
        )
    if "run_dir" in stored:
        stored["run_dir"] = os.path.relpath(
            REPOSITORY_ROOT / stored["run_dir"], REPOSITORY_ROOT
        )
    expected = _training_identity(args)
    changed = [
        name
        for name, value in expected.items()
        if stored.get(name) != value
        and not (
            args.allow_nonexact_resume and name in _NONEXACT_MUTABLE_IDENTITY_FIELDS
        )
    ]
    if changed:
        raise ValueError(
            f"Resume configuration fields changed or missing: {sorted(changed)}"
        )


def _load_state_payload(path: Path, args: TrainingConfig) -> Mapping[str, Any]:
    """
    Load training state and validate its identity and resume contract.

    Args:
        path (Path): Trusted full training-state checkpoint.
        args (TrainingConfig): Requested configuration and resume policy.

    Returns:
        Mapping[str, Any]: Supported training-state payload after validation.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise ValueError(f"Unsupported training state: {path}")
    _validate_training_identity(state, args)
    contract = state.get("resume_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Training state is missing its resume contract")
    if contract.get("granularity") != "configured_checkpoint_boundary":
        raise ValueError("Training state has an unsupported resume contract")
    interval = contract.get("checkpoint_interval")
    if type(interval) is not int or interval <= 0:
        raise ValueError("Training state has an invalid checkpoint interval")
    if interval != args.checkpoint_interval and not args.allow_nonexact_resume:
        raise ValueError("Checkpoint interval changed on exact resume")
    if (
        contract.get("exact_resume_supported") is not True
        and not args.allow_nonexact_resume
    ):
        raise ValueError(
            f"This state cannot be resumed exactly: {contract.get('reason', '')}"
        )
    return state


def _validated_resume_path(args: TrainingConfig) -> Path:
    """
    Require the resume file to be this run's regular last_state.pth.

    Args:
        args (TrainingConfig): Configuration with a non-None resume path.

    Returns:
        Path: Validated last_state.pth path inside the run's checkpoints directory.
    """
    run_dir = Path(args.run_dir)
    directory = run_dir / "checkpoints"
    expected = directory / "last_state.pth"
    supplied = Path(cast(str, args.resume))
    if any(path.is_symlink() for path in (run_dir, directory, supplied, expected)):
        raise ValueError("Resume paths cannot be symlinks")
    if (
        supplied.resolve(strict=True) != expected.resolve(strict=True)
        or not supplied.is_file()
    ):
        raise ValueError(f"Resume must use this run's state: {expected}")
    return expected


def prepare_run_directory(args: TrainingConfig) -> Path:
    """
    Create an empty run directory or validate the requested resumed run.

    Args:
        args (TrainingConfig): Run path, configuration identity, and resume settings.

    Returns:
        Path: Prepared run directory.
    """

    run_dir = Path(args.run_dir)
    if args.resume is None:
        if run_dir.is_symlink():
            raise FileExistsError(
                f"Refusing to use a symlink as a fresh run: {run_dir}"
            )
        if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
            raise FileExistsError(
                f"Refusing to mix a fresh run with existing artifacts: {run_dir}"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    state_path = _validated_resume_path(args)
    _load_state_payload(state_path, args)
    return run_dir


def build_training_components(
    model: nn.Module,
    args: TrainingConfig,
) -> tuple[Optimizer, CosineLRScheduler, SegmentationLoss]:
    """
    Build the configured optimizer, schedule, and segmentation loss.

    Args:
        model (nn.Module): Model whose parameters will be optimized.
        args (TrainingConfig): AdamW and cosine learning-rate settings.

    Returns:
        tuple[torch.optim.AdamW, CosineLRScheduler, SegmentationLoss]: Optimizer, scheduler, and Dice-plus-cross-entropy
            objective, in that order.
    """
    optimizer = setup_optimizer(args, model)
    return optimizer, setup_scheduler(args, optimizer), SegmentationLoss()


def _loader_worker_settings(loader: Iterable[object]) -> tuple[int, bool]:
    """
    Read worker settings from a training or validation loader.

    Args:
        loader (Iterable[tuple | list | Mapping[str, torch.Tensor]]): Batch iterable with optional num_workers and
            persistent_workers attributes.

    Returns:
        tuple[int, bool]: Worker count and persistence flag, defaulting to (0, False).
    """
    return int(getattr(loader, "num_workers", 0)), bool(
        getattr(loader, "persistent_workers", False)
    )


def _release_persistent_loader_workers(loader: Iterable[object]) -> int:
    """
    Release persistent training workers and their prefetched batches.

    Args:
        loader (Iterable[tuple | list | Mapping[str, torch.Tensor]]): Training loader whose cached iterator can be recreated on
            the next traversal.

    Returns:
        int: Number of released workers, or zero when no active pool exists.
    """
    num_workers, persistent = _loader_worker_settings(loader)
    iterator = getattr(loader, "_iterator", None)
    if not num_workers or not persistent or iterator is None:
        return 0
    iterator._shutdown_workers()
    cast(DataLoader[object], loader)._iterator = None
    del iterator
    gc.collect()
    return num_workers


def _loader_generators(
    loader: Iterable[object],
) -> tuple[tuple[str, torch.Generator], ...]:
    """
    Collect distinct random generators from a loader and its samplers.

    Args:
        loader (Iterable[tuple | list | Mapping[str, torch.Tensor]]): Training or validation batch iterable with optional
            loader/sampler generators.

    Returns:
        tuple[tuple[str, torch.Generator], ...]: Generator names and instances in loader, sampler, then batch-sampler order,
            without duplicates.
    """

    sampler = getattr(loader, "sampler", None)
    batch_sampler = getattr(loader, "batch_sampler", None)
    candidates = (
        ("loader.generator", getattr(loader, "generator", None)),
        ("loader.sampler.generator", getattr(sampler, "generator", None)),
        (
            "loader.batch_sampler.sampler.generator",
            getattr(getattr(batch_sampler, "sampler", None), "generator", None),
        ),
    )
    generators: list[tuple[str, torch.Generator]] = []
    seen: set[int] = set()
    for name, generator in candidates:
        if not isinstance(generator, torch.Generator) or id(generator) in seen:
            continue
        seen.add(id(generator))
        generators.append((name, generator))
    return tuple(generators)


def _loader_random_state(loader: Iterable[object]) -> dict[str, Any]:
    """
    Capture loader worker settings and generator states.

    Args:
        loader (Iterable[tuple | list | Mapping[str, torch.Tensor]]): Training or validation batch iterable.

    Returns:
        dict[str, int | bool | list[dict[str, str | torch.Tensor]]]: Worker settings and named CPU byte-state tensors for each
            distinct generator.
    """

    num_workers, persistent_workers = _loader_worker_settings(loader)
    return {
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
        "generators": [
            {
                "name": name,
                "state": generator.get_state().detach().cpu().clone(),
            }
            for name, generator in _loader_generators(loader)
        ],
    }


def _exact_resume_blockers(
    args: TrainingConfig,
    train_loader: Iterable[object],
    val_loader: Iterable[object],
) -> tuple[str, ...]:
    """
    Identify loader settings whose random state cannot be restored exactly.

    Args:
        args (TrainingConfig): Configured data-loader worker settings.
        train_loader (Iterable[tuple | list | Mapping[str, torch.Tensor]]): Training loader whose worker-local augmentation
            state is not saved.
        val_loader (Iterable[tuple | list]): Deterministic validation loader.

    Returns:
        tuple[str, ...]: Reasons exact continuation is unavailable; empty if supported.
    """

    blockers: list[str] = []
    configured_workers = int(args.num_workers)
    if configured_workers > 0:
        blockers.append(f"configuration uses num_workers={configured_workers}")
    # Deterministic validation does not require worker-local RNG restoration.
    _loader_worker_settings(val_loader)
    num_workers, persistent_workers = _loader_worker_settings(train_loader)
    if num_workers > 0:
        suffix = " with persistent workers" if persistent_workers else ""
        blockers.append(f"training loader uses {num_workers} workers{suffix}")
    return tuple(blockers)


def _resume_contract(
    args: TrainingConfig,
    train_loader: Iterable[object],
    val_loader: Iterable[object],
) -> dict[str, Any]:
    """
    Describe the checkpoint boundary and supported resume guarantees.

    Args:
        args (TrainingConfig): Checkpoint interval and worker settings.
        train_loader (Iterable[tuple | list | Mapping[str, torch.Tensor]]): Training batch iterable.
        val_loader (Iterable[tuple | list]): Validation batch iterable.

    Returns:
        dict[str, str | int | bool | None]: Checkpoint boundary, interval, exact-resume status, and optional blocker reason.
    """

    blockers = _exact_resume_blockers(args, train_loader, val_loader)
    reason = None
    if blockers:
        reason = (
            "; ".join(blockers)
            + ". Worker-local Python/NumPy augmentation RNG is not serializable; "
            "start the run with --num-workers 0 to enable exact resume"
        )
    return {
        "granularity": "configured_checkpoint_boundary",
        "checkpoint_interval": int(args.checkpoint_interval),
        "exact_resume_supported": not blockers,
        "reason": reason,
    }


def _restore_loader_random_state(
    loader: Iterable[object],
    payload: Mapping[str, Any],
    *,
    role: str,
    allow_worker_change: bool = False,
) -> None:
    """
    Restore named loader/sampler generators and check worker settings.

    Args:
        loader (Iterable[tuple | list | Mapping[str, torch.Tensor]]): Destination training or validation loader.
        payload (Mapping[str, int | bool | list[dict[str, str | torch.Tensor]]]): Saved worker settings and
            generator byte-state tensors.
        role (str): Loader name used in compatibility errors.
        allow_worker_change (bool): Permit different worker settings on resume.
    """
    if not isinstance(payload, Mapping):
        raise ValueError(f"Missing {role} loader RNG state")
    saved_settings = (payload["num_workers"], payload["persistent_workers"])
    if not allow_worker_change and saved_settings != _loader_worker_settings(loader):
        raise ValueError(f"{role} loader worker settings changed on exact resume")
    generators = dict(_loader_generators(loader))
    entries = payload["generators"]
    saved = {entry["name"]: entry["state"] for entry in entries}
    if len(saved) != len(entries) or set(saved) != set(generators):
        raise ValueError(f"{role} loader generator inventory changed")
    for name, generator in generators.items():
        generator.set_state(saved[name].detach().cpu())


def _restore_cuda_random_state(
    payload: object, *, allow_device_change: bool = False
) -> None:
    """
    Restore CUDA random states by logical device index.

    Args:
        payload (list[torch.Tensor] | tuple[torch.Tensor, ...] | None): Saved CPU byte-state tensors, one per CUDA device, or no
            CUDA state.
        allow_device_change (bool): Restore overlapping device indices when the saved and current device counts differ.
    """
    if payload is None or not torch.cuda.is_available():
        return
    if not isinstance(payload, (list, tuple)):
        raise ValueError("Invalid CUDA RNG inventory")
    count = torch.cuda.device_count()
    if len(payload) != count and not allow_device_change:
        raise ValueError(
            "CUDA device count changed; use --allow-nonexact-resume to restore overlapping devices"
        )
    for device, state in enumerate(payload[:count]):
        torch.cuda.set_rng_state(state.detach().cpu(), device=device)


def _training_state(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: CosineLRScheduler,
    scaler: torch.amp.GradScaler | None,
    args: TrainingConfig,
    *,
    ema: StatefulEMA | None,
    epoch: int,
    successful_updates: int,
    top_checkpoints: Iterable[CheckpointRecord],
    train_loader: Iterable[object],
    val_loader: Iterable[object],
    pending_validation_epoch: int | None = None,
) -> dict[str, Any]:
    """
    Assemble the model, optimizer, loader, and random state for resuming.

    Args:
        model (nn.Module): Live model supplying unwrapped training weights.
        optimizer (Optimizer): Optimizer supplying parameter-group and update state.
        scheduler (CosineLRScheduler): Learning-rate schedule and progress.
        scaler (torch.amp.GradScaler | None): Optional FP16 gradient-scaler state.
        args (TrainingConfig): Configuration identity and resume settings.
        ema (StatefulEMA | None): Optional moving-average state.
        epoch (int): Last completed training epoch.
        successful_updates (int): Total accepted optimizer updates.
        top_checkpoints (Iterable[CheckpointRecord]): Current retained checkpoints.
        train_loader (Iterable[tuple | list | Mapping[str, torch.Tensor]]): Training loader supplying generator state.
        val_loader (Iterable[tuple | list]): Validation loader with generator state.
        pending_validation_epoch (int | None): Completed training epoch whose validation must be replayed after resume, or None.

    Returns:
        dict[str, Any]: Training-state payload with checkpoint inventory, resume contract, and process/loader random states.
    """
    return {
        "identity": _training_identity(args),
        "resume_contract": _resume_contract(args, train_loader, val_loader),
        "loader_random_state": {
            "training": _loader_random_state(train_loader),
            "validation": _loader_random_state(val_loader),
        },
        "epoch": int(epoch),
        "pending_validation_epoch": (
            None if pending_validation_epoch is None else int(pending_validation_epoch)
        ),
        "successful_updates": int(successful_updates),
        "model_state_dict": unwrap_model(model).state_dict(),
        "ema_state_dict": None if ema is None else ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": None if scaler is None else scaler.state_dict(),
        "top_checkpoints": [
            {
                "epoch": record.epoch,
                "filename": record.path.name,
                "validation_score": record.validation_score,
            }
            for record in top_checkpoints
        ],
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
        "cuda_random_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _is_validation_epoch(
    epoch: int,
    *,
    start_epoch: int,
    interval: int,
) -> bool:
    """
    Check membership in the configured validation schedule.

    Args:
        epoch (int): Epoch to check.
        start_epoch (int): First validation epoch.
        interval (int): Positive spacing between validation epochs.

    Returns:
        bool: Whether epoch equals start_epoch plus a nonnegative interval multiple.
    """

    return epoch >= start_epoch and (epoch - start_epoch) % interval == 0


def _validation_epoch_count(
    *,
    num_epochs: int,
    start_epoch: int,
    interval: int,
) -> int:
    """
    Count validation epochs within the training schedule.

    Args:
        num_epochs (int): Final training epoch, inclusive.
        start_epoch (int): First validation epoch.
        interval (int): Positive spacing between validation epochs.

    Returns:
        int: Number of scheduled validation epochs, or zero when none are reached.
    """

    if num_epochs < start_epoch:
        return 0
    return 1 + (num_epochs - start_epoch) // interval


def _load_training_state(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: CosineLRScheduler,
    scaler: torch.amp.GradScaler | None,
    ema: StatefulEMA | None,
    manager: TopKCheckpointManager,
    args: TrainingConfig,
    train_loader: Iterable[object],
    val_loader: Iterable[object],
) -> tuple[int, int, int | None]:
    """
    Restore checkpoint state and the random streams needed for continuation.

    Args:
        path (str | Path): Full training-state checkpoint path.
        model (nn.Module): Destination model for live training weights.
        optimizer (Optimizer): Optimizer whose update state is restored.
        scheduler (CosineLRScheduler): Scheduler whose progress is restored.
        scaler (torch.amp.GradScaler | None): Optional destination gradient scaler.
        ema (StatefulEMA | None): Optional destination EMA state.
        manager (TopKCheckpointManager): Manager receiving retained checkpoints.
        args (TrainingConfig): Requested training configuration and resume policy.
        train_loader (Iterable[tuple | list | Mapping[str, torch.Tensor]]): Training loader receiving saved generator states.
        val_loader (Iterable[tuple | list]): Validation loader to restore.

    Returns:
        tuple[int, int, int | None]: Completed epoch, accepted-update count, and pending validation epoch, in that order.
    """
    state_path = Path(path)
    state = _load_state_payload(state_path, args)
    blockers = _exact_resume_blockers(args, train_loader, val_loader)
    if blockers and not args.allow_nonexact_resume:
        raise ValueError(
            "Exact epoch-boundary resume requires --num-workers 0; "
            + "; ".join(blockers)
        )

    epoch = state.get("epoch")
    successful_updates = state.get("successful_updates")
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or not 0 <= epoch <= args.num_epochs
    ):
        raise ValueError("Training-state epoch is outside the configured schedule")
    if (
        isinstance(successful_updates, bool)
        or not isinstance(successful_updates, int)
        or not 0 <= successful_updates <= epoch * args.iter_per_epoch
    ):
        raise ValueError("Training-state successful_updates is inconsistent with epoch")

    if "pending_validation_epoch" not in state:
        raise ValueError("Training state is missing pending-validation state")
    pending_validation_epoch = state.get("pending_validation_epoch")
    saved_checkpoint_interval = int(state["resume_contract"]["checkpoint_interval"])
    if pending_validation_epoch is not None and (
        isinstance(pending_validation_epoch, bool)
        or not isinstance(pending_validation_epoch, int)
        or pending_validation_epoch != epoch
        or not _is_validation_epoch(
            pending_validation_epoch,
            start_epoch=int(args.validation_start_epoch),
            interval=int(args.validation_interval),
        )
        or not (
            pending_validation_epoch % saved_checkpoint_interval == 0
            or pending_validation_epoch == int(args.num_epochs)
        )
    ):
        raise ValueError("Training state has an invalid pending validation epoch")

    restored: list[CheckpointRecord] = []
    for item in state.get("top_checkpoints", ()):
        if "filename" not in item:
            raise ValueError("Top-k checkpoint inventory entry has no filename")
        filename = str(item["filename"])
        relative = Path(filename)
        if (
            relative.is_absolute()
            or relative.name != filename
            or filename in {"", ".", ".."}
        ):
            raise ValueError(f"Invalid top-k checkpoint filename: {filename!r}")
        restored.append(
            CheckpointRecord(
                epoch=int(item["epoch"]),
                path=manager.directory / filename,
                validation_score=float(item["validation_score"]),
            )
        )
    if any(
        record.epoch > epoch
        or record.epoch == pending_validation_epoch
        or not _is_validation_epoch(
            record.epoch,
            start_epoch=args.validation_start_epoch,
            interval=args.validation_interval,
        )
        for record in restored
    ):
        raise ValueError(
            "Retained checkpoint epochs conflict with the validation schedule"
        )
    manager.restore(restored)

    load_model_state_dict(model, state["model_state_dict"], strict=True)
    ema_state = state.get("ema_state_dict")
    if ema is not None:
        ema.load_state_dict(cast(Mapping[str, Any], ema_state))
        if ema.num_updates != successful_updates:
            raise ValueError(
                "EMA progress does not match training-state successful_updates"
            )
    elif ema_state is not None:
        raise ValueError("Training state contains EMA state but EMA is disabled")
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    if scheduler.last_update != successful_updates:
        raise ValueError(
            "Scheduler progress does not match training-state successful_updates"
        )
    if scaler is not None and state.get("scaler_state_dict") is not None:
        scaler.load_state_dict(state["scaler_state_dict"])
    random.setstate(state["random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.random.set_rng_state(state["torch_random_state"])
    _restore_cuda_random_state(
        state.get("cuda_random_state"),
        allow_device_change=args.allow_nonexact_resume,
    )
    loader_random_state = state["loader_random_state"]
    _restore_loader_random_state(
        train_loader,
        loader_random_state.get("training"),
        role="training",
        allow_worker_change=args.allow_nonexact_resume,
    )
    _restore_loader_random_state(
        val_loader,
        loader_random_state.get("validation"),
        role="validation",
        allow_worker_change=args.allow_nonexact_resume,
    )
    manager.commit_inventory()
    return epoch, successful_updates, pending_validation_epoch


def train_model(
    model: nn.Module,
    args: TrainingConfig,
    train_loader: Iterable[object],
    val_loader: Iterable[object],
    *,
    device: torch.device | str,
    optimizer: Optimizer | None = None,
    scheduler: CosineLRScheduler | None = None,
    loss_function: nn.Module | None = None,
    writer: Any | None = None,
    logger: Any | None = None,
    validation_fn: Callable[..., ValidationResult] = validate_model,
    _run_directory_prepared: bool = False,
) -> TrainingResult:
    """
    Train, validate on schedule, and retain the highest-scoring checkpoints.

    Args:
        model (nn.Module): Segmentation model mapping (B, C, D, H, W) images to (B, K, D, H, W) class logits.
        args (TrainingConfig): Training, checkpoint, and validation settings.
        train_loader (Iterable[tuple | list | Mapping[str, torch.Tensor]]): Reusable iterable of image and categorical/one-hot
            target batches.
        val_loader (Iterable[tuple | list]): Validation cases with image, target, identifier, and physical-space metadata.
        device (torch.device | str): Device used for training and validation.
        optimizer (Optimizer | None): Custom optimizer; None builds configured AdamW.
        scheduler (CosineLRScheduler | None): Custom schedule; None builds cosine decay.
        loss_function (nn.Module | None): Scalar objective; None uses Dice plus cross entropy on logits and one-hot targets
            shaped (B, K, D, H, W).
        writer (SummaryWriter | None): Optional TensorBoard metric writer.
        logger (Logger | None): Optional text and structured run logger.
        validation_fn (Callable[..., ValidationResult]): Validation callback accepting the validate_model arguments and
            returning a selection score.
        _run_directory_prepared (bool): Whether the caller already claimed or validated the run directory
            through prepare_run_directory.

    Returns:
        TrainingResult: Run paths, retained checkpoints, and completed progress.
    """

    resolved_device = torch.device(device)
    validation_epochs = _validation_epoch_count(
        num_epochs=int(args.num_epochs),
        start_epoch=int(args.validation_start_epoch),
        interval=int(args.validation_interval),
    )
    if validation_epochs < int(args.top_k):
        raise ValueError(
            f"top_k={args.top_k} requires at least {args.top_k} validation epochs; "
            f"this schedule provides {validation_epochs}"
        )
    run_dir = Path(args.run_dir)
    if _run_directory_prepared:
        if not run_dir.is_dir() or run_dir.is_symlink():
            raise ValueError(f"Prepared run directory is invalid: {run_dir}")
        if args.resume is not None:
            state_path = _validated_resume_path(args)
            _load_state_payload(state_path, args)
    else:
        prepare_run_directory(args)

    model.to(resolved_device)
    ema = StatefulEMA(model, decay=float(args.ema_decay)) if args.use_ema else None
    if logger is not None and ema is not None:
        logger.write(f"EMA enabled: decay={ema.decay:.6f}.\n")
    if optimizer is None:
        optimizer = setup_optimizer(args, model)
    if scheduler is None:
        scheduler = setup_scheduler(args, optimizer)
    if loss_function is None:
        loss_function = SegmentationLoss()

    checkpoints_dir = run_dir / "checkpoints"
    top_dir = checkpoints_dir / "top_k"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    if args.resume is None:
        _atomic_json(run_dir / "config.json", args.as_dict())
    manager = TopKCheckpointManager(top_dir, int(args.top_k))

    scaler: torch.amp.GradScaler | None = None
    if resolved_device.type == "cuda" and args.mixed_precision == "fp16":
        scaler = torch.amp.GradScaler("cuda")
    start_epoch = 0
    successful_updates = 0
    pending_validation_epoch: int | None = None
    if args.resume is not None:
        start_epoch, successful_updates, pending_validation_epoch = (
            _load_training_state(
                args.resume,
                model,
                optimizer,
                scheduler,
                scaler,
                ema,
                manager,
                args,
                train_loader,
                val_loader,
            )
        )
        if args.allow_nonexact_resume and logger is not None:
            logger.write(
                "WARNING: non-exact resume enabled; model, EMA, optimizer, "
                "scheduler, and compatible process RNG state were restored, but "
                "data-loader worker or CUDA device-topology RNG continuity is not "
                "guaranteed.\n"
            )
            if callable(getattr(logger, "flush", None)):
                logger.flush()

    last_checkpoint = checkpoints_dir / "last.pth"
    state_path = checkpoints_dir / "last_state.pth"
    validation_history_path = initialize_validation_history(
        run_dir / "valid.csv",
        dataset_name=args.dataset_name,
    )

    def inference_metadata(epoch: int) -> dict[str, Any]:
        """
        Collect model identity and EMA fields for an inference checkpoint.

        Args:
            epoch (int): Epoch associated with the saved weights.

        Returns:
            dict[str, str | int | float]: Epoch, dataset/model identity, channel counts, and optional EMA metadata.
        """
        return {
            "epoch": int(epoch),
            "dataset": args.dataset_name,
            "model": args.model_name,
            "variant": args.model_variant,
            "in_channels": args.in_channels,
            "num_classes": args.num_classes,
            **_ema_inference_metadata(ema),
        }

    def commit_checkpoint(
        epoch: int,
        *,
        pending_validation: int | None,
        write_inference: bool,
    ) -> None:
        """
        Commit recovery state before publishing the retained checkpoint inventory.

        Args:
            epoch (int): Last completed training epoch.
            pending_validation (int | None): Validation epoch to replay on resume, or None when validation is complete or not
                scheduled.
            write_inference (bool): Also replace the last inference-weight snapshot.
        """
        if logger is not None:
            if pending_validation is not None:
                message = (
                    f"Saving pre-validation recovery checkpoint at epoch {epoch}...\n"
                )
            elif write_inference:
                message = (
                    f"Saving inference and resumable checkpoints at epoch {epoch}...\n"
                )
            else:
                message = f"Refreshing resumable checkpoint after validation at epoch {epoch}...\n"
            logger.write(message)
            if callable(getattr(logger, "flush", None)):
                logger.flush()

        checkpoint_started = time.perf_counter()
        if write_inference:
            with ema.use_ema_weights(model) if ema is not None else nullcontext():
                _save_inference_weights_atomic(
                    last_checkpoint,
                    model,
                    metadata=inference_metadata(epoch),
                )
        _atomic_torch_save(
            state_path,
            _training_state(
                model,
                optimizer,
                scheduler,
                scaler,
                args,
                ema=ema,
                epoch=epoch,
                successful_updates=successful_updates,
                top_checkpoints=manager.records,
                train_loader=train_loader,
                val_loader=val_loader,
                pending_validation_epoch=pending_validation,
            ),
        )
        manager.commit_inventory()
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        if logger is not None:
            action = "refreshed" if not write_inference else "committed"
            logger.write(
                f"Checkpoint epoch {epoch} {action} in {checkpoint_seconds:.2f}s.\n"
            )
            if callable(getattr(logger, "log_metrics", None)):
                logger.log_metrics(
                    {
                        "type": "timing",
                        "phase": (
                            "checkpoint_state_refresh"
                            if not write_inference
                            else "checkpoint"
                        ),
                        "epoch": epoch,
                        "duration_seconds": checkpoint_seconds,
                    }
                )

    def prepare_validation_memory(epoch: int) -> None:
        # The next training traversal recreates the released loader iterator.
        """
        Release gradients and persistent training workers before validation.

        Args:
            epoch (int): Validation epoch used in progress messages.
        """
        optimizer.zero_grad(set_to_none=True)
        released_workers = _release_persistent_loader_workers(train_loader)
        if logger is not None and released_workers:
            logger.write(
                f"Released {released_workers} persistent training data-loader "
                f"workers before validation epoch {epoch}.\n"
            )
            if callable(getattr(logger, "flush", None)):
                logger.flush()

    def perform_validation(epoch: int) -> None:
        """
        Evaluate current inference weights and stage any qualifying checkpoint.

        Args:
            epoch (int): Epoch used for validation records and checkpoint ranking.
        """
        if logger is not None:
            logger.write(
                f"Starting validation at epoch {epoch} "
                f"(every {args.validation_interval} epochs from "
                f"epoch {args.validation_start_epoch})...\n"
            )
            if callable(getattr(logger, "flush", None)):
                logger.flush()
        validation_started = time.perf_counter()
        with ema.use_ema_weights(model) if ema is not None else nullcontext():
            validation = validation_fn(
                model,
                args,
                val_loader,
                epoch,
                writer,
                logger,
                validation_history_path,
                device=resolved_device,
            )
        validation_seconds = time.perf_counter() - validation_started
        if logger is not None:
            logger.write(
                f"Validation epoch {epoch} completed in {validation_seconds:.2f}s.\n"
            )
            if callable(getattr(logger, "log_metrics", None)):
                logger.log_metrics(
                    {
                        "type": "timing",
                        "phase": "validation",
                        "epoch": epoch,
                        "duration_seconds": validation_seconds,
                    }
                )

        top_k_started = time.perf_counter()
        with ema.use_ema_weights(model) if ema is not None else nullcontext():
            candidate = manager.consider(
                model,
                epoch=epoch,
                validation_score=float(validation.selection_score),
                metadata=inference_metadata(epoch),
            )
        top_k_seconds = time.perf_counter() - top_k_started
        if logger is not None and candidate is not None:
            logger.write(
                f"Top-k inference weights for epoch {epoch} saved in "
                f"{top_k_seconds:.2f}s.\n"
            )
            if callable(getattr(logger, "log_metrics", None)):
                logger.log_metrics(
                    {
                        "type": "timing",
                        "phase": "top_k_checkpoint",
                        "epoch": epoch,
                        "duration_seconds": top_k_seconds,
                    }
                )

    if pending_validation_epoch is not None:
        if logger is not None:
            logger.write(
                f"Replaying pending validation at epoch "
                f"{pending_validation_epoch} before training resumes.\n"
            )
            if callable(getattr(logger, "flush", None)):
                logger.flush()
        prepare_validation_memory(pending_validation_epoch)
        perform_validation(pending_validation_epoch)
        commit_checkpoint(
            pending_validation_epoch,
            pending_validation=None,
            write_inference=False,
        )

    for epoch in range(start_epoch + 1, int(args.num_epochs) + 1):
        stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_function,
            epoch=epoch,
            iter_per_epoch=int(args.iter_per_epoch),
            device=resolved_device,
            successful_updates=successful_updates,
            grad_clip=args.grad_clip,
            mixed_precision=args.mixed_precision,
            scaler=scaler,
            ema=ema,
            show_progress=bool(args.progress),
            total_epochs=int(args.num_epochs),
        )
        if stats.successful_updates == 0:
            raise FloatingPointError(
                f"Epoch {epoch} completed with zero successful optimizer updates "
                f"(attempted={stats.attempted_updates}, "
                f"nonfinite={stats.nonfinite_skips})"
            )
        successful_updates += stats.successful_updates
        if logger is not None:
            loss_message = (
                f"Epoch {epoch}/{args.num_epochs}: loss={stats.mean_loss:.6f}, "
                f"updates={stats.successful_updates}/{stats.attempted_updates}, "
                f"lr={stats.learning_rate:.8g}"
            )
            logger.write(loss_message + "\n")
            if callable(getattr(logger, "log_metrics", None)):
                logger.log_metrics(
                    {
                        "type": "training",
                        "epoch": epoch,
                        "loss": stats.mean_loss,
                        "attempted_updates": stats.attempted_updates,
                        "successful_updates": stats.successful_updates,
                        "nonfinite_skips": stats.nonfinite_skips,
                        "learning_rate": stats.learning_rate,
                    }
                )
        if writer is not None:
            writer.add_scalar("training/loss", stats.mean_loss, epoch)
            writer.add_scalar("training/learning_rate", stats.learning_rate, epoch)

        validation_due = _is_validation_epoch(
            epoch,
            start_epoch=int(args.validation_start_epoch),
            interval=int(args.validation_interval),
        )
        checkpoint_due = epoch % int(args.checkpoint_interval) == 0 or epoch == int(
            args.num_epochs
        )

        if validation_due:
            prepare_validation_memory(epoch)

        # Save before validation so resume can replay it without retraining the epoch.
        if checkpoint_due:
            commit_checkpoint(
                epoch,
                pending_validation=epoch if validation_due else None,
                write_inference=True,
            )

        if validation_due:
            perform_validation(epoch)
            if checkpoint_due:
                # Save updated loader RNG and rankings; inference weights are unchanged.
                commit_checkpoint(
                    epoch,
                    pending_validation=None,
                    write_inference=False,
                )

    if writer is not None and callable(getattr(writer, "flush", None)):
        writer.flush()
    return TrainingResult(
        run_dir=run_dir,
        last_checkpoint=last_checkpoint,
        top_checkpoints=manager.records,
        completed_epochs=int(args.num_epochs),
        successful_updates=successful_updates,
    )


__all__ = [
    "CheckpointRecord",
    "EpochStats",
    "TopKCheckpointManager",
    "TrainingResult",
    "build_training_components",
    "fixed_iteration_batches",
    "prepare_run_directory",
    "train_model",
    "train_one_epoch",
]
