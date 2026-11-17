from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from os import PathLike
from pathlib import PurePosixPath, PureWindowsPath


def dice_summary_columns(regions: Sequence[str]) -> tuple[str, ...]:
    """
    Build compact Dice columns in registered region order.

    Args:
        regions (Sequence[str]): Ordered names of the task's evaluation regions.

    Returns:
        tuple[str, ...]: Epoch, the supplied region names, then Average.
    """
    return ("Epoch", *regions, "Average")


DICE_SUMMARY_COLUMNS = dice_summary_columns(("WT", "TC", "ET"))

_EVALUATION_SPACE = "processed_crop"
_COMPLETE_INTEGRITY = "complete"
_CROPPED_LABEL_LOSS = "cropped_nonzero_label_loss"

RESULT_COLUMNS = (
    "rank",
    "epoch",
    "checkpoint",
    "validation_score",
    "dataset",
    "protocol_id",
    "protocol_status",
    "selection_proxy_id",
    "subset",
    "evaluation_space",
    "data_integrity_status",
    "dropped_nonzero_label_voxels",
    "metric_scope",
    "lesion_matching_method",
    "lesion_matching_threshold",
    "metric",
    "region",
    "tolerance_mm",
    "unit",
    "aggregation",
    "true_positive",
    "false_positive",
    "false_negative",
    "value",
)

CASE_COLUMNS = (
    "rank",
    "epoch",
    "checkpoint",
    "validation_score",
    "dataset",
    "protocol_id",
    "protocol_status",
    "selection_proxy_id",
    "subset",
    "evaluation_space",
    "data_integrity_status",
    "dropped_nonzero_label_voxels",
    "case_id",
    "patient_id",
    "metric_scope",
    "lesion_matching_method",
    "lesion_matching_threshold",
    "metric",
    "region",
    "tolerance_mm",
    "unit",
    "true_positive",
    "false_positive",
    "false_negative",
    "value",
)


def _validate_checkpoint(value: str) -> None:
    """
    Require a filename without path components or control characters.

    Args:
        value (str): Serialized checkpoint filename.
    """
    if not value or value in {".", ".."}:
        raise ValueError("checkpoint must be a nonempty file identifier")
    if PurePosixPath(value).name != value or PureWindowsPath(value).name != value:
        raise ValueError("checkpoint must be a file identifier without path components")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("checkpoint must not contain control characters")


def _validate_data_integrity(
    *,
    evaluation_space: str,
    data_integrity_status: str,
    dropped_nonzero_label_voxels: int,
) -> None:
    """
    Check that crop provenance agrees with the dropped-label count.

    Args:
        evaluation_space (str): Evaluation domain; must be processed_crop.
        data_integrity_status (str): complete for zero dropped voxels, otherwise cropped_nonzero_label_loss.
        dropped_nonzero_label_voxels (int): Nonnegative foreground label count removed by cropping; booleans are rejected.
    """
    if evaluation_space != _EVALUATION_SPACE:
        raise ValueError(f"evaluation_space must be {_EVALUATION_SPACE!r}")
    if (
        not isinstance(dropped_nonzero_label_voxels, Integral)
        or isinstance(dropped_nonzero_label_voxels, bool)
        or dropped_nonzero_label_voxels < 0
    ):
        raise ValueError("dropped_nonzero_label_voxels must be a nonnegative integer")
    expected = (
        _COMPLETE_INTEGRITY
        if dropped_nonzero_label_voxels == 0
        else _CROPPED_LABEL_LOSS
    )
    if data_integrity_status != expected:
        raise ValueError("data_integrity_status does not match the dropped label count")


def _validate_metric_row(row: TopKResult | CaseMetricRow) -> None:
    """
    Reject nonfinite scores and inconsistent crop provenance.

    Args:
        row (TopKResult | CaseMetricRow): Metric row to validate.
    """
    if not math.isfinite(row.validation_score) or not math.isfinite(row.value):
        raise ValueError("CSV scores must be finite")
    _validate_data_integrity(
        evaluation_space=row.evaluation_space,
        data_integrity_status=row.data_integrity_status,
        dropped_nonzero_label_voxels=row.dropped_nonzero_label_voxels,
    )


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    """
    Portable epoch and filename identifying a selected checkpoint.

    Args:
        epoch (int): Nonnegative training epoch; booleans are rejected.
        checkpoint (str): Filename without directory components or controls.
    """

    epoch: int
    checkpoint: str

    def __post_init__(self) -> None:
        """Validate the epoch and portable checkpoint filename."""
        if (
            not isinstance(self.epoch, Integral)
            or isinstance(self.epoch, bool)
            or self.epoch < 0
        ):
            raise ValueError("epoch must be a nonnegative integer")
        _validate_checkpoint(self.checkpoint)

    @classmethod
    def from_path(
        cls, *, epoch: int, path: str | PathLike[str]
    ) -> "CheckpointIdentity":
        """
        Extract a checkpoint filename from a path.

        Args:
            epoch (int): Epoch represented by the checkpoint.
            path (str | os.PathLike[str]): Checkpoint path using POSIX or Windows separators.

        Returns:
            CheckpointIdentity: Validated epoch and filename without its directory.
        """
        checkpoint = os.fspath(path).replace("\\", "/").rsplit("/", 1)[-1]
        return cls(epoch=epoch, checkpoint=checkpoint)


@dataclass(frozen=True, slots=True)
class TopKResult:
    """
    One aggregate metric for a ranked validation checkpoint.

    Args:
        rank (int): Checkpoint rank, starting at one.
        identity (CheckpointIdentity): Checkpoint epoch and portable filename.
        validation_score (float): Finite score used to rank the checkpoint.
        dataset (str): Canonical dataset name.
        protocol_id (str): Evaluation protocol identifier.
        protocol_status (str): Relationship to the source evaluation protocol.
        selection_proxy_id (str): Checkpoint selection policy identifier.
        subset (str): Evaluated data split, such as validation or test.
        metric_scope (str): Unit of evaluation, such as subject.
        lesion_matching_method (str): Lesion matching policy, or not_applicable.
        lesion_matching_threshold (float | None): Threshold used for lesion matches.
        metric (str): Metric identifier.
        region (str): Evaluated binary region name.
        tolerance_mm (float | None): Surface Dice tolerance in millimetres, if used.
        unit (str): Unit of the metric value.
        aggregation (str): Reduction policy, such as case_macro.
        true_positive (int | None): Aggregated matched lesion count, when measured.
        false_positive (int | None): Aggregated unmatched prediction count, if used.
        false_negative (int | None): Aggregated missed reference count, if used.
        value (float): Finite aggregate metric value.
        evaluation_space (str): Evaluation domain; must be processed_crop.
        data_integrity_status (str): complete or cropped_nonzero_label_loss, consistent with the dropped voxel count.
        dropped_nonzero_label_voxels (int): Nonnegative foreground voxel count removed by cropping.
    """

    rank: int
    identity: CheckpointIdentity
    validation_score: float
    dataset: str
    protocol_id: str
    protocol_status: str
    selection_proxy_id: str
    subset: str
    metric_scope: str
    lesion_matching_method: str
    lesion_matching_threshold: float | None
    metric: str
    region: str
    tolerance_mm: float | None
    unit: str
    aggregation: str
    true_positive: int | None
    false_positive: int | None
    false_negative: int | None
    value: float
    evaluation_space: str = _EVALUATION_SPACE
    data_integrity_status: str = _COMPLETE_INTEGRITY
    dropped_nonzero_label_voxels: int = 0

    def __post_init__(self) -> None:
        """Validate finite scores and crop provenance before serialization."""
        _validate_metric_row(self)


@dataclass(frozen=True, slots=True)
class CaseMetricRow:
    """
    One case metric for a ranked validation checkpoint.

    Args:
        rank (int): Checkpoint rank, starting at one.
        identity (CheckpointIdentity): Checkpoint epoch and portable filename.
        validation_score (float): Finite score used to rank the checkpoint.
        dataset (str): Canonical dataset name.
        protocol_id (str): Evaluation protocol identifier.
        protocol_status (str): Relationship to the source evaluation protocol.
        selection_proxy_id (str): Checkpoint selection policy identifier.
        subset (str): Evaluated data split, such as validation or test.
        case_id (str): Case identifier.
        patient_id (str): Serialized patient identifier.
        metric_scope (str): Unit of evaluation, such as subject.
        lesion_matching_method (str): Lesion matching policy, or not_applicable.
        lesion_matching_threshold (float | None): Threshold used for lesion matches.
        metric (str): Metric identifier.
        region (str): Evaluated binary region name.
        tolerance_mm (float | None): Surface Dice tolerance in millimetres, if used.
        unit (str): Unit of the metric value.
        true_positive (int | None): Matched lesion count, when measured.
        false_positive (int | None): Unmatched predicted lesion count, when measured.
        false_negative (int | None): Missed reference lesion count, when measured.
        value (float): Finite metric value for this case.
        evaluation_space (str): Evaluation domain; must be processed_crop.
        data_integrity_status (str): complete or cropped_nonzero_label_loss, consistent with the dropped voxel count.
        dropped_nonzero_label_voxels (int): Nonnegative foreground voxel count removed by cropping.
    """

    rank: int
    identity: CheckpointIdentity
    validation_score: float
    dataset: str
    protocol_id: str
    protocol_status: str
    selection_proxy_id: str
    subset: str
    case_id: str
    patient_id: str
    metric_scope: str
    lesion_matching_method: str
    lesion_matching_threshold: float | None
    metric: str
    region: str
    tolerance_mm: float | None
    unit: str
    true_positive: int | None
    false_positive: int | None
    false_negative: int | None
    value: float
    evaluation_space: str = _EVALUATION_SPACE
    data_integrity_status: str = _COMPLETE_INTEGRITY
    dropped_nonzero_label_voxels: int = 0

    def __post_init__(self) -> None:
        """Validate finite scores and crop provenance before serialization."""
        _validate_metric_row(self)


__all__ = [
    "CASE_COLUMNS",
    "DICE_SUMMARY_COLUMNS",
    "RESULT_COLUMNS",
    "CaseMetricRow",
    "CheckpointIdentity",
    "TopKResult",
    "dice_summary_columns",
]
