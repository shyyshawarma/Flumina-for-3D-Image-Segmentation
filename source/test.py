from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from source.csv_outputs import (
    CSVOutputPaths,
    CaseMetricRow,
    CheckpointIdentity,
    TopKResult,
    write_topk_csvs,
)
from source.evaluation import get_task
from source.core.inference import sliding_window_inference
from source.inference import inference_with_augmentation
from source.utils.runtime import load_model_state_dict, unwrap_model
from source.validate import ValidationConfig, ValidationResult, validate_model


@dataclass(frozen=True, slots=True)
class EvaluatedCheckpoint:
    """
    Validation-ranked checkpoint and its held-out evaluation results.

    Args:
        rank (int): One-based rank by validation score.
        epoch (int): Epoch of the checkpoint.
        checkpoint (Path): Inference-weights path.
        validation_score (float): Score used to select the checkpoint.
        result (ValidationResult): Held-out metrics for this checkpoint.
    """

    rank: int
    epoch: int
    checkpoint: Path
    validation_score: float
    result: ValidationResult


@dataclass(frozen=True, slots=True)
class TopKEvaluation:
    """
    Evaluated checkpoints and their exported CSV paths.

    Args:
        checkpoints (tuple[EvaluatedCheckpoint, ...]): Checkpoints in rank order.
        csv_paths (CSVOutputPaths): Aggregate and per-case CSV output paths.
    """

    checkpoints: tuple[EvaluatedCheckpoint, ...]
    csv_paths: CSVOutputPaths


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """
    Deserialized inference weights and metadata available for reuse.

    Args:
        path (Path): Source checkpoint path.
        state_dict (Mapping[str, torch.Tensor]): Named model weights on the CPU.
        metadata (Mapping[str, Any]): Checkpoint identity and descriptive fields.
    """

    path: Path
    state_dict: Mapping[str, torch.Tensor]
    metadata: Mapping[str, Any]


def read_checkpoint(path: str | Path) -> LoadedCheckpoint:
    """
    Read a weights artifact or raw tensor mapping with weights-only loading.

    Args:
        path (str | Path): Checkpoint file to load on the CPU.

    Returns:
        LoadedCheckpoint: Validated tensor mapping, metadata, and source path.
    """

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint payload must be a mapping: {checkpoint}")
    if "model_state_dict" in payload:
        state = payload["model_state_dict"]
        metadata = payload.get("metadata", {})
    elif "state_dict" in payload:
        state = payload["state_dict"]
        metadata = payload.get("metadata", {})
    elif payload and all(torch.is_tensor(value) for value in payload.values()):
        state = payload
        metadata = {}
    else:
        raise ValueError(
            f"Checkpoint contains no unambiguous model state dictionary: {checkpoint}"
        )
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and torch.is_tensor(value)
        for name, value in state.items()
    ):
        raise TypeError(f"Invalid model state dictionary in {checkpoint}")
    if not isinstance(metadata, Mapping):
        raise TypeError(f"Checkpoint metadata must be a mapping: {checkpoint}")
    return LoadedCheckpoint(
        path=checkpoint,
        state_dict=state,  # type: ignore[arg-type]
        metadata=metadata,
    )


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    loaded_checkpoint: LoadedCheckpoint | None = None,
) -> Mapping[str, Any]:
    """
    Load model weights with exact key matching and return their metadata.

    Args:
        model (nn.Module): Destination segmentation model.
        checkpoint_path (str | Path): Inference checkpoint path.
        loaded_checkpoint (LoadedCheckpoint | None): Previously loaded weights from the same path; None reads the file.

    Returns:
        Mapping[str, Any]: Metadata stored with the checkpoint, or an empty mapping.
    """

    path = Path(checkpoint_path)
    loaded = read_checkpoint(path) if loaded_checkpoint is None else loaded_checkpoint
    if loaded.path.resolve() != path.resolve():
        raise ValueError(
            "Preloaded checkpoint does not match the requested path: "
            f"{loaded.path} != {path}"
        )
    state, metadata = loaded.state_dict, loaded.metadata
    load_model_state_dict(model, state, strict=True)
    return metadata


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    args: ValidationConfig,
    test_loader: Iterable[object],
    *,
    device: torch.device | str,
    epoch: int | None = None,
    logger: Any | None = None,
    loaded_checkpoint: LoadedCheckpoint | None = None,
) -> ValidationResult:
    """
    Load one checkpoint and evaluate one traversal of held-out cases.

    Args:
        checkpoint_path (str | Path): Inference checkpoint path.
        model (nn.Module): Segmentation model receiving the checkpoint weights.
        args (ValidationConfig): Dataset and sliding-window settings, with optional model identity and flip settings.
        test_loader (Iterable[tuple | list]): Cases containing images, targets, identifiers, and evaluation metadata.
        device (torch.device | str): Inference device.
        epoch (int | None): Reported epoch; None uses checkpoint metadata or zero.
        logger (Logger | None): Optional run logger.
        loaded_checkpoint (LoadedCheckpoint | None): Optional weights already read from checkpoint_path.

    Returns:
        ValidationResult: Per-case metrics, aggregates, and selection score.
    """

    metadata = load_checkpoint(
        model,
        checkpoint_path,
        loaded_checkpoint=loaded_checkpoint,
    )
    resolved_epoch = int(metadata.get("epoch", 0) if epoch is None else epoch)
    prediction_kwargs = {}
    if getattr(args, "inference_augument", True):
        inference_model = unwrap_model(model)

        def predict_logits(images: torch.Tensor) -> torch.Tensor:
            """
            Args:
                images (torch.Tensor): Volume batch shaped (B, C, D, H, W).

            Returns:
                torch.Tensor: Blended float32 logits shaped (B, K, D, H, W).
            """
            return sliding_window_inference(
                inference_model,
                images,
                num_classes=args.num_classes,
                crop_size=args.crop_size,
                overlap=getattr(args, "overlap", 0.5),
                return_logits=True,
            )

        def predict_labels(images: torch.Tensor) -> torch.Tensor:
            """
            Args:
                images (torch.Tensor): Volume batch shaped (B, C, D, H, W).

            Returns:
                torch.Tensor: Int64 labels (B, D, H, W) from averaged flip logits.
            """
            return inference_with_augmentation(images, predict_logits, True).argmax(
                dim=1
            )

        prediction_kwargs["predictor"] = predict_labels
    return validate_model(
        model,
        args,
        test_loader,
        resolved_epoch,
        None,
        logger,
        None,
        device=device,
        **prediction_kwargs,
    )


def _checkpoint_fields(
    record: object,
) -> tuple[int, Path, float, LoadedCheckpoint | None]:
    """
    Read the epoch, path, score, and optional preloaded weights of a record.

    Args:
        record (CheckpointRecord | EvaluatedCheckpoint): Checkpoint record; equivalent records may expose path/checkpoint and
            score/validation_score.

    Returns:
        tuple[int, Path, float, LoadedCheckpoint | None]: Epoch, checkpoint path, validation score, and optional preloaded
            weights, in that order.
    """

    epoch = getattr(record, "epoch", None)
    path = getattr(record, "path", getattr(record, "checkpoint", None))
    score = getattr(record, "validation_score", getattr(record, "score", None))
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"Invalid checkpoint epoch: {epoch!r}")
    if path is None:
        raise ValueError("Checkpoint record has no path")
    validation_score = float(cast(float, score))
    if not math.isfinite(validation_score):
        raise ValueError("Checkpoint validation score must be finite")
    loaded = getattr(record, "loaded_checkpoint", None)
    if loaded is not None and not isinstance(loaded, LoadedCheckpoint):
        raise TypeError("loaded_checkpoint must be a LoadedCheckpoint")
    return epoch, Path(path), validation_score, loaded


def evaluate_checkpoints(
    checkpoint_records: Iterable[object],
    model: nn.Module,
    args: ValidationConfig,
    test_loader: Iterable[object],
    *,
    output_dir: str | Path,
    top_k: int,
    device: torch.device | str,
    subset: str = "test",
    logger: Any | None = None,
    results_path: str | Path | None = None,
    cases_path: str | Path | None = None,
) -> TopKEvaluation:
    """
    Evaluate selected checkpoints and atomically export their metric CSVs.

    Args:
        checkpoint_records (Iterable[CheckpointRecord | EvaluatedCheckpoint]): Selected records, sorted here by descending score
            then ascending epoch.
        model (nn.Module): Segmentation model reused for each checkpoint.
        args (ValidationConfig): Dataset evaluation settings and optional model identity.
        test_loader (Iterable[tuple | list]): Reusable iterable of evaluation batches.
        output_dir (str | Path): Parent directory for default CSV outputs.
        top_k (int): Required number of selected checkpoints.
        device (torch.device | str): Inference device.
        subset (str): Cohort label stored in metric rows.
        logger (Logger | None): Optional run logger.
        results_path (str | Path | None): Aggregate CSV path; None uses topk_results.csv.
        cases_path (str | Path | None): Per-case CSV path; None uses topk_cases.csv.

    Returns:
        TopKEvaluation: Results in validation-rank order and the written CSV paths.
    """

    materialized = tuple(_checkpoint_fields(record) for record in checkpoint_records)
    if len(materialized) != int(top_k):
        raise ValueError(
            f"Final evaluation requires exactly top_k={top_k} checkpoints, "
            f"got {len(materialized)}"
        )
    ranked = sorted(materialized, key=lambda item: (-item[2], item[0]))
    if len({epoch for epoch, _, _, _ in ranked}) != len(ranked):
        raise ValueError("Top-k checkpoints contain duplicate epochs")

    aggregate_rows: list[TopKResult] = []
    case_rows: list[CaseMetricRow] = []
    evaluated: list[EvaluatedCheckpoint] = []
    for rank, (epoch, checkpoint, validation_score, loaded_checkpoint) in enumerate(
        ranked, start=1
    ):
        result = evaluate_checkpoint(
            checkpoint,
            model,
            args,
            test_loader,
            device=device,
            epoch=epoch,
            logger=logger,
            loaded_checkpoint=loaded_checkpoint,
        )
        identity = CheckpointIdentity.from_path(epoch=epoch, path=checkpoint)
        task = get_task(result.dataset_name)
        for record in result.aggregates:
            if not record.applicable or not math.isfinite(record.value):
                continue
            aggregate_rows.append(
                TopKResult(
                    rank=rank,
                    identity=identity,
                    validation_score=validation_score,
                    dataset=record.dataset,
                    protocol_id=record.protocol_id or task.protocol_id,
                    protocol_status=record.protocol_status or task.protocol_status,
                    selection_proxy_id=task.selection_proxy_id,
                    subset=subset,
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
            )
        for record in result.records:
            if not record.applicable or not math.isfinite(record.value):
                continue
            case_rows.append(
                CaseMetricRow(
                    rank=rank,
                    identity=identity,
                    validation_score=validation_score,
                    dataset=record.dataset,
                    protocol_id=record.protocol_id or task.protocol_id,
                    protocol_status=record.protocol_status or task.protocol_status,
                    selection_proxy_id=task.selection_proxy_id,
                    subset=subset,
                    case_id=record.case_id,
                    patient_id=record.patient_id or "",
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
                    true_positive=record.true_positive,
                    false_positive=record.false_positive,
                    false_negative=record.false_negative,
                    value=record.value,
                )
            )
        evaluated.append(
            EvaluatedCheckpoint(
                rank=rank,
                epoch=epoch,
                checkpoint=checkpoint,
                validation_score=validation_score,
                result=result,
            )
        )

    csv_paths = write_topk_csvs(
        output_dir,
        aggregate_rows,
        case_rows,
        top_k=int(top_k),
        results_path=results_path,
        cases_path=cases_path,
    )
    return TopKEvaluation(tuple(evaluated), csv_paths)


__all__ = [
    "EvaluatedCheckpoint",
    "LoadedCheckpoint",
    "TopKEvaluation",
    "evaluate_checkpoint",
    "evaluate_checkpoints",
    "load_checkpoint",
    "read_checkpoint",
]
