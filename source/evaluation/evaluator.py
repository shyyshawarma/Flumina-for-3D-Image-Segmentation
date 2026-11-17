from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, cast

import numpy as np

from source.evaluation.metrics import dice_score
from source.evaluation.registry import get_task


@dataclass(frozen=True, slots=True)
class MetricRecord:
    """
    One segmentation metric with case and protocol metadata.

    Args:
        dataset (str): Canonical dataset name.
        case_id (str): Case identifier, or __aggregate__ for a summary.
        region (str): Name of the evaluated binary region.
        metric (str): Metric identifier.
        value (float): Metric value, possibly NaN for an inapplicable summary.
        applicable (bool): Whether the value is eligible for aggregation.
        gt_voxels (int): Foreground reference voxel count.
        pred_voxels (int): Foreground prediction voxel count.
        aggregation (str): Reduction label, such as case or case_macro.
        patient_id (str | None): Patient identifier when available.
        protocol_id (str): Evaluation protocol identifier.
        protocol_status (str): Relationship to the referenced evaluation protocol.
        metric_scope (str): Unit of evaluation, such as subject.
        lesion_matching_method (str): Lesion matching policy, when applicable.
        lesion_matching_threshold (float | None): Matching threshold, when used.
        tolerance_mm (float | None): Surface distance tolerance in millimetres.
        unit (str): Unit of the metric value.
        true_positive (int | None): Matched lesion count, when measured.
        false_positive (int | None): Unmatched predicted lesion count, when measured.
        false_negative (int | None): Missed reference lesion count, when measured.
        evaluation_space (str): Spatial domain on which the metric was computed.
        data_integrity_status (str): Whether cropping removed foreground labels.
        dropped_nonzero_label_voxels (int): Foreground label voxels lost to cropping.
    """

    dataset: str
    case_id: str
    region: str
    metric: str
    value: float
    applicable: bool
    gt_voxels: int
    pred_voxels: int
    aggregation: str = "case"
    patient_id: str | None = None
    protocol_id: str = ""
    protocol_status: str = ""
    metric_scope: str = ""
    lesion_matching_method: str = ""
    lesion_matching_threshold: float | None = None
    tolerance_mm: float | None = None
    unit: str = "unitless"
    true_positive: int | None = None
    false_positive: int | None = None
    false_negative: int | None = None
    evaluation_space: str = "processed_crop"
    data_integrity_status: str = "complete"
    dropped_nonzero_label_voxels: int = 0

    def to_dict(self) -> dict[str, Any]:
        """
        Convert all record fields to a dictionary.

        Returns:
            dict[str, Any]: Field names and their stored values.
        """
        return asdict(self)


def _class_map(value: Any) -> np.ndarray:
    """
    Convert categorical labels to a CPU NumPy array.

    Args:
        value (numpy.ndarray | torch.Tensor): Integer class labels of any shape.

    Returns:
        numpy.ndarray: Int64 labels with the input shape. Tensor inputs are detached and moved to CPU before conversion.
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("Class maps must contain categorical integer labels")
    return array.astype(np.int64, copy=False)


def _spacing(values: Sequence[float]) -> tuple[float, float, float]:
    """
    Validate three voxel spacings in array-axis order.

    Args:
        values (Sequence[float]): Voxel spacing (s0, s1, s2) in millimetres.

    Returns:
        tuple[float, float, float]: Three finite, positive spacings.
    """
    spacing = tuple(float(value) for value in values)
    if len(spacing) != 3 or any(
        not np.isfinite(value) or value <= 0 for value in spacing
    ):
        raise ValueError("Spacing must contain three finite positive values")
    return spacing


def _integrity(metadata: Mapping[str, Any]) -> tuple[str, int]:
    """
    Validate the paired cropped-label integrity fields.

    Args:
        metadata (Mapping[str, Any]): Case metadata with optional label_support.dropped_nonzero_label_voxels and
            label_support.integrity_status entries; both must be present together.

    Returns:
        tuple[str, int]: Integrity status and dropped foreground voxel count. Missing fields default to complete and zero.
    """
    count_key = "label_support.dropped_nonzero_label_voxels"
    status_key = "label_support.integrity_status"
    if count_key not in metadata and status_key not in metadata:
        return "complete", 0
    if count_key not in metadata or status_key not in metadata:
        raise ValueError(
            "Evaluation metadata must provide label loss count and integrity status"
        )
    count = int(metadata[count_key])
    expected = "cropped_nonzero_label_loss" if count > 0 else "complete"
    if count < 0 or metadata[status_key] != expected:
        raise ValueError("Label loss count and integrity status disagree")
    return expected, count


def evaluate_case(
    dataset_name: str,
    prediction: Any,
    target: Any,
    *,
    case_id: str = "case",
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    metadata: Mapping[str, Any] | None = None,
) -> tuple[MetricRecord, ...]:
    """
    Evaluate whole-volume Dice for each registered region.

    Args:
        dataset_name (str): Supported dataset name.
        prediction (numpy.ndarray | torch.Tensor): Integer class map with shape (S0, S1, S2) and labels in [0, num_classes).
        target (numpy.ndarray | torch.Tensor): Integer reference labels on the same processed grid as prediction.
        case_id (str): Case identifier used in metric records.
        spacing (Sequence[float]): Three finite, positive voxel spacings (s0, s1, s2), in millimetres and in array-axis order.
        metadata (Mapping[str, Any] | None): Optional patient_id and paired label_support integrity fields.

    Returns:
        tuple[MetricRecord, ...]: One applicable Dice record per registered region. Two empty masks score 1; one empty mask
            scores 0. Every region, including BraTS resection cavity, is scored for every case.
    """
    task = get_task(dataset_name)
    pred, ref = _class_map(prediction), _class_map(target)
    if pred.shape != ref.shape or pred.ndim != 3:
        raise ValueError("Prediction and target must have the same 3D shape")
    if any(np.any((array < 0) | (array >= task.num_classes)) for array in (pred, ref)):
        raise ValueError(f"Class maps must contain labels 0..{task.num_classes - 1}")
    _spacing(spacing)
    metadata = metadata or {}
    integrity, dropped = _integrity(metadata)
    patient = metadata.get("patient_id")
    records = []
    for region in task.regions:
        pred_mask = np.isin(pred, region.labels)
        target_mask = np.isin(ref, region.labels)
        records.append(
            MetricRecord(
                dataset=task.dataset_name,
                case_id=case_id,
                region=region.name,
                metric="dice",
                value=dice_score(pred_mask, target_mask),
                applicable=True,
                gt_voxels=int(np.count_nonzero(target_mask)),
                pred_voxels=int(np.count_nonzero(pred_mask)),
                patient_id=None if patient is None else str(patient),
                protocol_id=task.protocol_id,
                protocol_status=task.protocol_status,
                metric_scope=task.metric_scope,
                lesion_matching_method=task.lesion_matching_method,
                data_integrity_status=integrity,
                dropped_nonzero_label_voxels=dropped,
            )
        )
    return tuple(records)


def evaluate_batch(
    dataset_name: str,
    predictions: Any,
    targets: Any,
    *,
    case_ids: Sequence[str] | None = None,
    spacings: Sequence[float] | Sequence[Sequence[float]] = (1.0, 1.0, 1.0),
    metadata: Sequence[Mapping[str, Any] | None] | Mapping[str, Any] | None = None,
) -> tuple[MetricRecord, ...]:
    """
    Evaluate Dice for a nonempty batch of categorical maps.

    Args:
        dataset_name (str): Supported dataset name.
        predictions (numpy.ndarray | torch.Tensor): Integer class maps with shape (B, S0, S1, S2), where B is positive.
        targets (numpy.ndarray | torch.Tensor): Reference labels with the same shape and grid as predictions.
        case_ids (Sequence[str] | None): B case identifiers, or None to generate names of the form case-00000.
        spacings (Sequence[float] | Sequence[Sequence[float]]): One shared spacing triple or B triples, in millimetres and in
            spatial array-axis order.
        metadata (Sequence[Mapping[str, Any] | None] | Mapping[str, Any] | None): B metadata entries, one shared mapping, or
            None.

    Returns:
        tuple[MetricRecord, ...]: Case records in batch order, then registered region order, with B times the number of regions
            entries.
    """
    pred, ref = _class_map(predictions), _class_map(targets)
    if pred.shape != ref.shape or pred.ndim != 4 or pred.shape[0] == 0:
        raise ValueError("Batch maps must have the same nonempty (B, D, H, W) shape")
    batch = pred.shape[0]
    identifiers = (
        tuple(case_ids)
        if case_ids is not None
        else tuple(f"case-{i:05d}" for i in range(batch))
    )
    raw_spacing = tuple(spacings)
    common = len(raw_spacing) == 3 and all(np.isscalar(value) for value in raw_spacing)
    spacing_items = (
        (_spacing(cast(Sequence[float], raw_spacing)),) * batch
        if common
        else tuple(_spacing(cast(Sequence[float], value)) for value in raw_spacing)
    )
    metadata_items = (
        (metadata,) * batch
        if metadata is None or isinstance(metadata, Mapping)
        else tuple(metadata)
    )
    if any(
        len(values) != batch for values in (identifiers, spacing_items, metadata_items)
    ):
        raise ValueError("Case IDs, spacings, and metadata must match batch size")
    return tuple(
        record
        for index in range(batch)
        for record in evaluate_case(
            dataset_name,
            pred[index],
            ref[index],
            case_id=identifiers[index],
            spacing=spacing_items[index],
            metadata=metadata_items[index],
        )
    )


def validation_selection_score(
    records: Iterable[MetricRecord], dataset_name: str | None = None
) -> float:
    """
    Average applicable finite case scores across the selection regions.

    Args:
        records (Iterable[MetricRecord]): Case metric records to filter and average.
        dataset_name (str | None): Dataset to select, or None to infer it when the records identify exactly one dataset.

    Returns:
        float: Equal-weight mean of matching case Dice records.
    """
    rows = tuple(records)
    if dataset_name is None:
        datasets = {row.dataset for row in rows}
        if len(datasets) != 1:
            raise ValueError(
                "Specify a dataset when records do not identify exactly one"
            )
        dataset_name = next(iter(datasets))
    task = get_task(dataset_name)
    values = [
        row.value
        for row in rows
        if row.dataset == task.dataset_name
        and row.aggregation == "case"
        and row.metric == task.selection_metric
        and row.region in task.selection_regions
        and row.applicable
        and np.isfinite(row.value)
    ]
    if not values:
        raise ValueError("No applicable finite validation scores")
    return float(np.mean(values, dtype=np.float64))


_GROUP_FIELDS = (
    "dataset",
    "protocol_id",
    "protocol_status",
    "metric_scope",
    "evaluation_space",
    "lesion_matching_method",
    "lesion_matching_threshold",
    "region",
    "metric",
    "tolerance_mm",
    "unit",
)


class MetricAccumulator:
    """Collect case metrics and average applicable finite values by group."""

    def __init__(self) -> None:
        """Initialize an empty sequence of case records."""
        self._records: list[MetricRecord] = []

    @property
    def records(self) -> tuple[MetricRecord, ...]:
        """
        Return the accumulated records in insertion order.

        Returns:
            tuple[MetricRecord, ...]: Immutable snapshot of the stored records.
        """
        return tuple(self._records)

    def update(self, records: MetricRecord | Iterable[MetricRecord]) -> None:
        """
        Append one or more metric records.

        Args:
            records (MetricRecord | Iterable[MetricRecord]): Records to append in iteration order.
        """
        self._records.extend(
            (records,) if isinstance(records, MetricRecord) else records
        )

    def aggregate(self) -> tuple[MetricRecord, ...]:
        """
        Average case records that share region, metric, and protocol metadata.

        Returns:
            tuple[MetricRecord, ...]: One case_macro record per group in first-seen order. Values and counts use applicable
                finite records. Groups without an eligible value are inapplicable with NaN values; integrity metadata includes
                all cases in the group.
        """
        if not self._records:
            raise ValueError("Cannot aggregate zero case records")
        groups = defaultdict(list)
        for row in self._records:
            if row.aggregation != "case":
                raise ValueError("Only case records can be accumulated")
            groups[tuple(getattr(row, field) for field in _GROUP_FIELDS)].append(row)
        summaries = []
        for group in groups.values():
            integrity_by_case = {}
            for row in group:
                integrity = (
                    row.data_integrity_status,
                    row.dropped_nonzero_label_voxels,
                )
                if integrity_by_case.setdefault(row.case_id, integrity) != integrity:
                    raise ValueError(
                        "Case integrity metadata differs between metric records"
                    )
            dropped = sum(count for _, count in integrity_by_case.values())
            included = [
                row for row in group if row.applicable and np.isfinite(row.value)
            ]
            counts = {
                name: sum(getattr(row, name) or 0 for row in included)
                if any(getattr(row, name) is not None for row in included)
                else None
                for name in ("true_positive", "false_positive", "false_negative")
            }
            summaries.append(
                replace(
                    group[0],
                    case_id="__aggregate__",
                    patient_id=None,
                    value=float(np.mean([row.value for row in included]))
                    if included
                    else float("nan"),
                    applicable=bool(included),
                    gt_voxels=sum(row.gt_voxels for row in included),
                    pred_voxels=sum(row.pred_voxels for row in included),
                    aggregation="case_macro",
                    data_integrity_status="cropped_nonzero_label_loss"
                    if dropped
                    else "complete",
                    dropped_nonzero_label_voxels=dropped,
                    **counts,
                )
            )
        return tuple(summaries)

    def selection_score(self, dataset_name: str | None = None) -> float:
        """
        Compute the validation selection score from stored case records.

        Args:
            dataset_name (str | None): Dataset to select, or None to infer the single dataset represented by the stored records.

        Returns:
            float: Equal-weight mean of applicable finite selection-region Dice.
        """
        return validation_selection_score(self._records, dataset_name)


__all__ = [
    "MetricAccumulator",
    "MetricRecord",
    "evaluate_batch",
    "evaluate_case",
    "validation_selection_score",
]
