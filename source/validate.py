from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import nn
from tqdm import tqdm

from source.core import sliding_window_inference
from source.dataset import validate_evaluation_metadata
from source.evaluation import MetricAccumulator, MetricRecord, evaluate_case, get_task
from source.utils.runtime import unwrap_model


class ValidationConfig(Protocol):
    """Required read-only dataset and window settings for case evaluation."""

    @property
    def dataset_name(self) -> str:
        """
        Returns:
            str: Registered dataset identifier.
        """
        ...

    @property
    def num_classes(self) -> int:
        """
        Returns:
            int: Number of label classes, including background.
        """
        ...

    @property
    def crop_size(self) -> int | Sequence[int]:
        """
        Returns:
            int | Sequence[int]: Isotropic window size or dimensions in (D, H, W) order.
        """
        ...


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """
    Per-case metrics, aggregates, and score from one evaluation traversal.

    Args:
        dataset_name (str): Evaluated dataset name.
        epoch (int): Epoch associated with the evaluated weights.
        selection_score (float): Score used to rank checkpoints.
        num_cases (int): Number of evaluated cases.
        records (tuple[MetricRecord, ...]): Per-case metric observations.
        aggregates (tuple[MetricRecord, ...]): Aggregate metric observations.
    """

    dataset_name: str
    epoch: int
    selection_score: float
    num_cases: int
    records: tuple[MetricRecord, ...]
    aggregates: tuple[MetricRecord, ...]

    def __float__(self) -> float:
        """
        Returns:
            float: Checkpoint-selection score.
        """
        return self.selection_score

    @property
    def composite_score(self) -> float:
        """
        Returns:
            float: Checkpoint-selection score reported as "Average" in CSV output.
        """

        return self.selection_score


def _model_device(model: nn.Module) -> torch.device:
    """
    Find the device of the first model parameter or buffer.

    Args:
        model (nn.Module): Model whose device is needed.

    Returns:
        torch.device: First tensor's device, or CPU for a model without tensors.
    """

    tensor = next(iter(model.parameters()), None)
    if tensor is None:
        tensor = next(iter(model.buffers()), None)
    return torch.device("cpu") if tensor is None else tensor.device


def _case_metadata(value: object, index: int, batch_size: int) -> dict[str, Any]:
    """
    Extract one case's metadata from a collated batch.

    Args:
        value (Mapping[str, Any] | Sequence[Mapping[str, Any] | None] | None): Batched metadata mapping, per-case mappings, or
            absent metadata.
        index (int): Case index within the batch.
        batch_size (int): Number of cases in the batch.

    Returns:
        dict[str, Any]: Per-case metadata with collated tensors converted to scalars or lists and spacing assembled in
            spatial-axis order.
    """

    if value is None:
        return {}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Mapping)):
        if len(value) != batch_size:
            raise ValueError(
                f"Expected {batch_size} metadata entries, got {len(value)}"
            )
        item = value[index]
        if item is None:
            return {}
        if not isinstance(item, Mapping):
            raise TypeError("Per-case metadata entries must be mappings")
        return dict(item)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if torch.is_tensor(item) and item.ndim > 0 and item.shape[0] == batch_size:
                selected = item[index]
                item = selected.item() if selected.numel() == 1 else selected.tolist()
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                if (
                    str(key) == "spacing"
                    and len(item) == 3
                    and all(
                        torch.is_tensor(component)
                        and component.ndim > 0
                        and component.shape[0] == batch_size
                        for component in item
                    )
                ):
                    item = tuple(float(component[index]) for component in item)
                elif len(item) == batch_size:
                    item = item[index]
            result[str(key)] = item
        return result
    raise TypeError("Metadata must be a mapping, a sequence of mappings, or None")


def _normalize_case_target(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert a case target to a categorical class map.

    Args:
        target (torch.Tensor): One-hot (K, D, H, W) or categorical (D, H, W) labels.
        num_classes (int): Expected number of one-hot channels K.

    Returns:
        torch.Tensor: Detached int64 CPU class map shaped (D, H, W).
    """

    match target.ndim:
        case 4:
            if target.shape[0] != num_classes:
                raise ValueError(
                    "A rank-four target must be channel-first one-hot data with "
                    f"{num_classes} channels, got {tuple(target.shape)}"
                )
            target = target.argmax(dim=0)
        case 3:
            pass
        case _:
            raise ValueError(
                f"Target must have shape [K,D,H,W] or [D,H,W], got {target.shape}"
            )
    return target.detach().to(device="cpu", dtype=torch.int64)


def _validation_case_total(loader: Iterable[object]) -> int | None:
    """
    Read the case count from a loader's dataset when available.

    Args:
        loader (Iterable[tuple | list]): Evaluation loader or batch iterable.

    Returns:
        int | None: Nonnegative dataset length, or None if unavailable.
    """

    dataset = getattr(loader, "dataset", None)
    try:
        return len(cast(Sized, dataset))
    except (TypeError, AttributeError):
        return None


def _iter_cases(
    batch: object,
) -> Iterable[tuple[torch.Tensor, torch.Tensor, str, dict[str, Any]]]:
    """
    Unpack stacked batches or lists of variable-size cases.

    Args:
        batch (tuple | list): Images, targets, names, and optional metadata, either batched together or stored as individual
            sample tuples.

    Yields:
        tuple[torch.Tensor, torch.Tensor, str, dict[str, Any]]: Image (C, D, H, W), categorical (D, H, W) or one-hot
            (K, D, H, W) target, case identifier, and metadata, in that order.
    """

    # Variable-size cases remain separate instead of being stacked.
    if isinstance(batch, list) and batch and isinstance(batch[0], (tuple, list)):
        for sample in batch:
            if len(sample) not in {3, 4}:
                raise ValueError(
                    "A validation sample must contain image, target, name[, metadata]"
                )
            image, target, name = sample[:3]
            metadata = sample[3] if len(sample) == 4 else None
            if not torch.is_tensor(image) or not torch.is_tensor(target):
                raise TypeError("Validation images and targets must be tensors")
            yield image, target, str(name), _case_metadata(metadata, 0, 1)
        return

    if not isinstance(batch, (tuple, list)) or len(batch) not in {3, 4}:
        raise TypeError(
            "Validation batches must be (images, targets, names[, metadata]) or "
            "a list of individual samples"
        )
    images, targets, names = batch[:3]
    metadata = batch[3] if len(batch) == 4 else None
    if not torch.is_tensor(images) or not torch.is_tensor(targets):
        raise TypeError("Validation images and targets must be tensors")
    if images.ndim != 5 or targets.ndim not in {4, 5}:
        raise ValueError("Stacked validation tensors must include a batch dimension")
    batch_size = int(images.shape[0])
    if int(targets.shape[0]) != batch_size:
        raise ValueError("Validation image and target batch sizes differ")
    identifiers = (
        (str(names),) if isinstance(names, str) else tuple(str(name) for name in names)
    )
    if len(identifiers) != batch_size:
        raise ValueError(
            f"Expected {batch_size} case identifiers, got {len(identifiers)}"
        )
    for index, identifier in enumerate(identifiers):
        yield (
            images[index],
            targets[index],
            identifier,
            _case_metadata(metadata, index, batch_size),
        )


def _write_validation_csv(path: str | Path, result: ValidationResult) -> None:
    """
    Write validation aggregates using the shared checkpoint metric schema.

    Args:
        path (str | Path): Validation-history CSV destination.
        result (ValidationResult): Evaluation result containing aggregate records.
    """

    # Import CSV serialization only when validation output is requested.
    from source.csv_outputs import (
        CheckpointIdentity,
        TopKResult,
        write_validation_history,
    )

    identity = CheckpointIdentity(
        epoch=result.epoch,
        checkpoint=f"validation_epoch_{result.epoch:04d}",
    )
    task = get_task(result.dataset_name)
    rows = (
        TopKResult(
            rank=1,
            identity=identity,
            validation_score=result.composite_score,
            dataset=record.dataset,
            protocol_id=record.protocol_id or task.protocol_id,
            protocol_status=record.protocol_status or task.protocol_status,
            selection_proxy_id=task.selection_proxy_id,
            subset="validation",
            evaluation_space=record.evaluation_space,
            metric_scope=record.metric_scope or task.metric_scope,
            lesion_matching_method=(
                record.lesion_matching_method or task.lesion_matching_method
            ),
            lesion_matching_threshold=record.lesion_matching_threshold,
            metric=record.metric,
            region=record.region,
            tolerance_mm=record.tolerance_mm,
            unit=record.unit,
            aggregation=record.aggregation,
            true_positive=record.true_positive,
            false_positive=record.false_positive,
            false_negative=record.false_negative,
            value=record.value,
        )
        for record in result.aggregates
        if record.applicable and torch.isfinite(torch.tensor(record.value)).item()
    )
    write_validation_history(path, rows)


def validate_model(
    model: nn.Module,
    args: ValidationConfig,
    val_loader: Iterable[object],
    epoch: int = 0,
    writer: Any | None = None,
    logger: Any | None = None,
    validation_results_csv: str | Path | None = None,
    *,
    device: torch.device | str | None = None,
    predictor: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> ValidationResult:
    """
    Evaluate all cases and compute the checkpoint-selection score.

    Args:
        model (nn.Module): Segmentation model; runtime wrappers are removed for sliding-window inference and its training mode
            is restored.
        args (ValidationConfig): Dataset, label, sliding-window, and optional progress settings.
        val_loader (Iterable[tuple | list]): Batched or individual cases with images (C, D, H, W), labels, identifiers, and
            physical-space metadata.
        epoch (int): Epoch associated with the evaluated weights.
        writer (SummaryWriter | None): Optional TensorBoard metric writer.
        logger (Logger | None): Optional run logger.
        validation_results_csv (str | Path | None): Optional validation-history CSV.
        device (torch.device | str | None): Inference device; None uses model device.
        predictor (Callable[[torch.Tensor], torch.Tensor] | None): Optional predictor from images (B, C, D, H, W) to labels
            (B, D, H, W). None uses unaugmented sliding-window inference.

    Returns:
        ValidationResult: Per-case metrics, aggregate metrics, and selection score.
    """

    inference_model = unwrap_model(model)
    resolved_device = (
        _model_device(inference_model) if device is None else torch.device(device)
    )
    dataset_name = str(args.dataset_name)
    num_classes = int(args.num_classes)
    crop_size = args.crop_size
    overlap = float(getattr(args, "overlap", 0.5))
    accumulator = MetricAccumulator()
    num_cases = 0

    was_training = bool(model.training)
    progress_bar = None
    try:
        model.eval()
        if bool(getattr(args, "progress", False)):
            progress_bar = tqdm(
                total=_validation_case_total(val_loader),
                desc=f"Validate epoch {epoch}",
                unit="case",
                dynamic_ncols=True,
                leave=True,
            )
            progress_bar.set_postfix({"stage": "loading"}, refresh=True)
        for batch in val_loader:
            for image, target, case_id, metadata in _iter_cases(batch):
                if image.ndim != 4:
                    raise ValueError(
                        f"Validation image for {case_id!r} must be [C,D,H,W], got {image.shape}"
                    )
                validate_evaluation_metadata(
                    dataset_name,
                    (case_id,),
                    {case_id: metadata},
                )
                # Physical-space metrics require the validated voxel spacing.
                spacing = metadata.pop("spacing")
                if progress_bar is not None:
                    progress_bar.set_postfix(
                        {"case": case_id, "stage": "inference"},
                        refresh=True,
                    )
                image_batch = image.unsqueeze(0).to(
                    resolved_device, dtype=torch.float32, non_blocking=True
                )
                if predictor is None:
                    prediction = sliding_window_inference(
                        inference_model,
                        image_batch,
                        num_classes=num_classes,
                        crop_size=crop_size,
                        overlap=overlap,
                    )[0].cpu()
                else:
                    prediction = predictor(image_batch)[0].cpu()
                del image_batch
                target_classes = _normalize_case_target(target, num_classes)
                if progress_bar is not None:
                    progress_bar.set_postfix(
                        {"case": case_id, "stage": "metrics"},
                        refresh=True,
                    )
                records = evaluate_case(
                    dataset_name,
                    prediction,
                    target_classes,
                    case_id=case_id,
                    spacing=spacing,
                    metadata=metadata,
                )
                accumulator.update(records)
                num_cases += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_postfix({"stage": "loading"}, refresh=True)
    finally:
        if progress_bar is not None:
            progress_bar.close()
        model.train(was_training)

    if num_cases == 0:
        raise ValueError("Validation loader produced no cases")
    result = ValidationResult(
        dataset_name=dataset_name,
        epoch=int(epoch),
        selection_score=accumulator.selection_score(dataset_name),
        num_cases=num_cases,
        records=accumulator.records,
        aggregates=accumulator.aggregate(),
    )

    if writer is not None:
        writer.add_scalar(
            "validation/composite_score", result.composite_score, int(epoch)
        )
        for record in result.aggregates:
            if record.applicable:
                writer.add_scalar(
                    f"validation/{record.metric}/{record.region}",
                    record.value,
                    int(epoch),
                )
    if logger is not None:
        logger.write(
            f"Validation epoch {epoch}: {num_cases} cases, "
            f"composite_score={result.composite_score:.6f}\n"
        )
        if callable(getattr(logger, "log_metrics", None)):
            logger.log_metrics(
                {
                    "type": "validation",
                    "epoch": int(epoch),
                    "dataset": dataset_name,
                    "num_cases": num_cases,
                    "composite_score": result.composite_score,
                }
            )
        if callable(getattr(logger, "flush", None)):
            logger.flush()
    if validation_results_csv is not None:
        _write_validation_csv(validation_results_csv, result)
    return result


__all__ = ["ValidationConfig", "ValidationResult", "validate_model"]
