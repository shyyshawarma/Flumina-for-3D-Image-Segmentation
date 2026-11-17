from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from source.evaluation.metrics import case_diagonal_mm
from source.evaluation.registry import get_task
from source.evaluation.surface_distance import (
    compute_robust_hausdorff,
    compute_surface_distances,
)


METRIC_NAMES = ("DSC", "HD95", "IoU", "ASD")
METRIC_PROTOCOL_ID = "unibench.case_macro_surfel_hd95_iou_pooled_asd"
MRI_UNAVAILABLE_REGIONS = frozenset({"bladder", "prostate/uterus"})


def _dataset_identity(dataset_name: str) -> tuple[str, str]:
    """
    Resolve evaluation and parent label-schema identities.

    Args:
        dataset_name (str): Registered dataset alias or AMOS modality view.

    Returns:
        tuple[str, str]: Canonical evaluation name followed by the physical dataset name.
    """

    if not isinstance(dataset_name, str):
        raise ValueError("Dataset name must be a nonempty string")
    token = "".join(
        character for character in dataset_name.casefold() if character.isalnum()
    )
    match token:
        case "amos2022ct":
            return "AMOS-2022-CT", "AMOS-2022"
        case "amos2022mri":
            return "AMOS-2022-MRI", "AMOS-2022"
    canonical = get_task(dataset_name).dataset_name
    return canonical, canonical


def get_regions(dataset_name: str) -> dict[str, tuple[int, ...]]:
    """
    Read the registered regions in their CSV column order.

    Args:
        dataset_name (str): Dataset alias or AMOS modality view.

    Returns:
        dict[str, tuple[int, ...]]: Ordered region names mapped to their categorical labels.
    """

    _, parent = _dataset_identity(dataset_name)
    return {region.name: region.labels for region in get_task(parent).regions}


def _spacing_tuple(spacing: Sequence[float]) -> tuple[float, float, float]:
    """
    Normalize three physical voxel spacings.

    Args:
        spacing (Sequence[float]): Positive, finite spacing in array axis order, in mm.

    Returns:
        tuple[float, float, float]: Validated spacing along the three spatial axes.
    """
    try:
        values = tuple(float(value) for value in spacing)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Spacing must contain three finite positive values in mm"
        ) from exc
    if len(values) != 3 or any(
        not np.isfinite(value) or value <= 0.0 for value in values
    ):
        raise ValueError(
            f"Spacing must contain three finite positive values in mm, got {values}"
        )
    return values  # type: ignore[return-value]


def _binary_pair(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate aligned binary masks and expose boolean arrays.

    Args:
        prediction (np.ndarray): Nonempty 3D boolean or numeric 0/1 mask.
        target (np.ndarray): Reference mask with the same shape and binary encoding.

    Returns:
        tuple[np.ndarray, np.ndarray]: Boolean prediction and target, each of shape (D, H, W).
    """
    arrays = tuple(np.asarray(value) for value in (prediction, target))
    if arrays[0].shape != arrays[1].shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {arrays[0].shape} vs {arrays[1].shape}"
        )
    if arrays[0].ndim != 3 or 0 in arrays[0].shape:
        raise ValueError("Metrics require nonzero-sized 3D spatial masks")
    for array in arrays:
        if array.dtype.kind not in "biuf" or not np.all((array == 0) | (array == 1)):
            raise ValueError(
                "Binary metrics require bool or 0/1 masks, without NaN/ignore labels"
            )
    return arrays[0].astype(bool, copy=False), arrays[1].astype(bool, copy=False)


def _case_maps(
    prediction: np.ndarray, target: np.ndarray, parent_dataset: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate aligned categorical maps against a dataset label schema.

    Args:
        prediction (np.ndarray): Integer class map of shape (D, H, W).
        target (np.ndarray): Reference class map on the same nonempty grid.
        parent_dataset (str): Physical dataset defining the allowed class indices.

    Returns:
        tuple[np.ndarray, np.ndarray]: Prediction and target of shape (D, H, W) with their integer dtypes preserved and all
            labels in range.
    """
    arrays = tuple(np.asarray(value) for value in (prediction, target))
    if arrays[0].shape != arrays[1].shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {arrays[0].shape} vs {arrays[1].shape}"
        )
    if arrays[0].ndim != 3 or 0 in arrays[0].shape:
        raise ValueError("Evaluation expects nonzero-sized 3D categorical class maps")
    count = get_task(parent_dataset).num_classes
    for name, array in zip(("prediction", "target"), arrays):
        if array.dtype.kind not in "biu":
            raise TypeError(
                f"{name} must contain categorical integer labels, got {array.dtype}"
            )
        invalid = (array < 0) | (array >= count)
        if np.any(invalid):
            raise ValueError(
                f"{name} contains invalid labels {np.unique(array[invalid]).tolist()}; "
                f"{parent_dataset} expects 0..{count - 1}, with no voxel ignore label"
            )
    return arrays[0], arrays[1]


def _mask_status(pred_count: int, target_count: int) -> str:
    """
    Describe whether either region mask has no foreground voxels.

    Args:
        pred_count (int): Number of predicted foreground voxels.
        target_count (int): Number of reference foreground voxels.

    Returns:
        str: both_empty, prediction_empty, target_empty, or ok.
    """
    match pred_count, target_count:
        case 0, 0:
            return "both_empty"
        case 0, _:
            return "prediction_empty"
        case _, 0:
            return "target_empty"
        case _:
            return "ok"


def binary_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    spacing: Sequence[float],
) -> dict[str, float]:
    """
    Compute overlap and physical surface distances for aligned binary masks.

    Args:
        prediction (np.ndarray): Binary prediction on the complete processed volume with shape (S0, S1, S2).
        target (np.ndarray): Binary reference on the same orthogonal grid.
        spacing (Sequence[float]): Physical voxel spacing in array axis order, in mm.

    Returns:
        dict[str, float]: DSC and IoU fractions, the maximum directional area-weighted HD95, and pooled symmetric ASD in mm.
            Both-empty masks have overlap one and distance zero; one-sided empty masks use the full physical volume diagonal.
    """

    pred, ref = _binary_pair(prediction, target)
    physical_spacing = _spacing_tuple(spacing)
    pred_count = int(np.count_nonzero(pred))
    target_count = int(np.count_nonzero(ref))
    status = _mask_status(pred_count, target_count)
    if status == "both_empty":
        return {"DSC": 1.0, "HD95": 0.0, "IoU": 1.0, "ASD": 0.0}
    if status != "ok":
        penalty = case_diagonal_mm(pred.shape, physical_spacing)
        return {"DSC": 0.0, "HD95": penalty, "IoU": 0.0, "ASD": penalty}

    intersection = int(np.count_nonzero(pred & ref))
    union = pred_count + target_count - intersection
    distances = compute_surface_distances(ref, pred, physical_spacing)
    area_gt = distances["surfel_areas_gt"]
    area_pred = distances["surfel_areas_pred"]
    total_area = float(np.sum(area_gt) + np.sum(area_pred))
    numerator = float(
        np.sum(distances["distances_gt_to_pred"] * area_gt)
        + np.sum(distances["distances_pred_to_gt"] * area_pred)
    )
    return {
        "DSC": float(2.0 * intersection / (pred_count + target_count)),
        "HD95": float(compute_robust_hausdorff(distances, 95.0)),
        "IoU": float(intersection / union),
        "ASD": float(numerator / total_area),
    }


@dataclass(frozen=True, slots=True)
class MetricRecord:
    """
    A region metric with case identity, foreground counts, and annotation status.

    Args:
        dataset (str): Canonical evaluation dataset, including any AMOS modality suffix.
        case_id (str): Case identifier used for aggregation.
        region (str): Registered region name.
        metric (str): Benchmark metric name: DSC, HD95, IoU, or ASD.
        value (float): Metric value; unannotated regions use NaN.
        status (str): Mask emptiness or not_annotated status.
        included (bool): Whether the record contributes to aggregate metrics.
        pred_voxels (int): Number of predicted region voxels.
        gt_voxels (int): Number of reference region voxels.
        unit (str): ratio for overlap metrics or mm for distances.
        empty_penalty_mm (float): Computed physical diagonal of this case's complete processed volume, in mm.
        protocol_id (str): Identifier of the metric definition and aggregation rules.
    """

    dataset: str
    case_id: str
    region: str
    metric: str
    value: float
    status: str
    included: bool
    pred_voxels: int
    gt_voxels: int
    unit: str
    empty_penalty_mm: float
    protocol_id: str = METRIC_PROTOCOL_ID

    def to_dict(self) -> dict[str, Any]:
        """
        Convert every metric and audit field to a serializable mapping.

        Returns:
            dict[str, str | float | bool | int]: Field names and their scalar values.
        """
        return asdict(self)


def evaluate_case_records(
    prediction: np.ndarray,
    target: np.ndarray,
    dataset_name: str,
    *,
    spacing: Sequence[float],
    case_id: str = "case",
) -> tuple[MetricRecord, ...]:
    """
    Score registered regions while retaining emptiness and annotation status.

    Args:
        prediction (np.ndarray): Integer class map of the complete processed volume with shape (S0, S1, S2).
        target (np.ndarray): Categorical reference on the same processed grid.
        dataset_name (str): Dataset alias or AMOS modality view.
        spacing (Sequence[float]): Physical voxel spacing in array axis order, in mm.
        case_id (str): Nonempty identifier stored in every metric record.

    Returns:
        tuple[MetricRecord, ...]: Four metric records per registered region. AMOS MRI bladder and prostate/uterus entries carry
            NaN and are excluded.
    """

    dataset, parent = _dataset_identity(dataset_name)
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("case_id must be a nonempty string")
    physical_spacing = _spacing_tuple(spacing)
    pred, ref = _case_maps(prediction, target, parent)
    penalty = case_diagonal_mm(pred.shape, physical_spacing)
    is_mri = dataset == "AMOS-2022-MRI"
    if is_mri and np.any((ref == 14) | (ref == 15)):
        raise ValueError(
            "AMOS-2022-MRI reference unexpectedly contains unannotated label 14 or 15"
        )

    records: list[MetricRecord] = []
    for region, labels in get_regions(dataset).items():
        pred_mask = np.isin(pred, labels)
        target_mask = np.isin(ref, labels)
        pred_count = int(np.count_nonzero(pred_mask))
        target_count = int(np.count_nonzero(target_mask))
        included = not (is_mri and region in MRI_UNAVAILABLE_REGIONS)
        if included:
            values = binary_metrics(
                pred_mask,
                target_mask,
                spacing=physical_spacing,
            )
            status = _mask_status(pred_count, target_count)
        else:
            values = {metric: float("nan") for metric in METRIC_NAMES}
            status = "not_annotated"
        records.extend(
            MetricRecord(
                dataset=dataset,
                case_id=case_id,
                region=region,
                metric=metric,
                value=values[metric],
                status=status,
                included=included,
                pred_voxels=pred_count,
                gt_voxels=target_count,
                unit="mm" if metric in {"HD95", "ASD"} else "ratio",
                empty_penalty_mm=penalty,
            )
            for metric in METRIC_NAMES
        )
    return tuple(records)


def evaluate_case(
    prediction: np.ndarray,
    target: np.ndarray,
    dataset_name: str,
    *,
    spacing: Sequence[float],
    case_id: str = "case",
) -> dict[str, dict[str, float]]:
    """
    Return benchmark region scores grouped by metric.

    Args:
        prediction (np.ndarray): Integer class map of the complete processed volume with shape (S0, S1, S2).
        target (np.ndarray): Categorical reference on the same processed grid.
        dataset_name (str): Dataset alias or AMOS modality view.
        spacing (Sequence[float]): Physical voxel spacing in array axis order, in mm.
        case_id (str): Case identifier passed to record evaluation.

    Returns:
        dict[str, dict[str, float]]: Scores indexed by metric and region; overlap uses fractions, distances use mm, and
            unavailable annotations use NaN.
    """

    result: dict[str, dict[str, float]] = {metric: {} for metric in METRIC_NAMES}
    for record in evaluate_case_records(
        prediction,
        target,
        dataset_name,
        spacing=spacing,
        case_id=case_id,
    ):
        result[record.metric][record.region] = record.value
    return result


def _validated_value(metric: str, value: float) -> float:
    """
    Validate an included score before computing an aggregate.

    Args:
        metric (str): Metric name; DSC and IoU are limited to the unit interval.
        value (float): Included score, which must be nonnegative and not NaN.

    Returns:
        float: Validated score, retaining positive infinity for distance metrics.
    """
    score = float(value)
    if np.isnan(score):
        raise ValueError(
            f"Included {metric} value is NaN; exclude only explicit not-annotated rows"
        )
    if score < 0 or (metric in {"DSC", "IoU"} and score > 1.0):
        raise ValueError(f"Invalid {metric} value {score}")
    return score


def aggregate_records(records: Iterable[MetricRecord]) -> dict[str, dict[str, float]]:
    """
    Average region scores over equally weighted cases with complete coverage.

    Duplicate or missing case/metric/region rows are rejected. Case weights do not depend on foreground volume or surface area.

    Args:
        records (Iterable[MetricRecord]): Complete records from one dataset and metric protocol; case diagonals may differ.

    Returns:
        dict[str, dict[str, float]]: Case means indexed by metric and region. Unannotated regions remain NaN, and included
            infinities propagate.
    """

    rows = tuple(records)
    if not rows:
        raise ValueError("Cannot aggregate zero cases")
    if len({row.dataset for row in rows}) != 1:
        raise ValueError("Aggregate one dataset at a time")
    if len({row.protocol_id for row in rows}) != 1:
        raise ValueError("Cannot mix metric protocols")
    dataset = rows[0].dataset
    expected_pairs = {
        (metric, region) for metric in METRIC_NAMES for region in get_regions(dataset)
    }
    by_case: dict[str, set[tuple[str, str]]] = defaultdict(set)
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        pair = (row.metric, row.region)
        if pair not in expected_pairs:
            raise ValueError(f"Unknown metric/region pair {pair}")
        if pair in by_case[row.case_id]:
            raise ValueError(f"Duplicate case/metric/region: {row.case_id!r}/{pair}")
        by_case[row.case_id].add(pair)
        expected_inclusion = not (
            dataset == "AMOS-2022-MRI" and row.region in MRI_UNAVAILABLE_REGIONS
        )
        if row.included != expected_inclusion:
            raise ValueError(
                f"Unexpected annotation applicability for {dataset}/{row.region}"
            )
        if row.included:
            groups[pair].append(_validated_value(row.metric, row.value))
        elif row.status != "not_annotated" or not np.isnan(row.value):
            raise ValueError(
                "Excluded rows must explicitly be not_annotated with a NaN value"
            )
    for case_id, pairs in by_case.items():
        if pairs != expected_pairs:
            raise ValueError(f"Incomplete metric/region coverage for case {case_id!r}")
    return {
        metric: {
            region: float(np.mean(groups[(metric, region)], dtype=np.float64))
            if groups[(metric, region)]
            else float("nan")
            for region in get_regions(dataset)
        }
        for metric in METRIC_NAMES
    }


def aggregate_cases(
    cases: Iterable[Mapping[str, Mapping[str, float]]],
    *,
    nan_policy: Literal["raise", "omit"] = "raise",
) -> dict[str, dict[str, float]]:
    """
    Average equally weighted case dictionaries under an explicit NaN policy.

    Args:
        cases (Iterable[Mapping[str, Mapping[str, float]]]): Case scores with all four metrics and identical ordered region
            schemas.
        nan_policy (Literal["raise", "omit"]): Reject NaN by default; omit is for known unavailable annotations such as AMOS MRI
            regions.

    Returns:
        dict[str, dict[str, float]]: Case means indexed by metric and region. Infinities propagate; a region with only omitted
            values remains NaN.
    """

    if nan_policy not in {"raise", "omit"}:
        raise ValueError("nan_policy must be 'raise' or 'omit'")
    items = tuple(cases)
    if not items:
        raise ValueError("Cannot aggregate zero cases")
    schema = {metric: tuple(regions) for metric, regions in items[0].items()}
    if set(schema) != set(METRIC_NAMES) or any(
        not regions for regions in schema.values()
    ):
        raise ValueError(
            "Each case must contain all four metrics and nonempty region columns"
        )
    for case in items:
        if {metric: tuple(regions) for metric, regions in case.items()} != schema:
            raise ValueError("Cases must have identical ordered metric/region schemas")
    result: dict[str, dict[str, float]] = {metric: {} for metric in schema}
    for metric, regions in schema.items():
        for region in regions:
            values = [float(case[metric][region]) for case in items]
            if nan_policy == "omit":
                values = [value for value in values if not np.isnan(value)]
            validated = [_validated_value(metric, value) for value in values]
            result[metric][region] = (
                float(np.mean(validated, dtype=np.float64))
                if validated
                else float("nan")
            )
    return result


def metric_protocol() -> dict[str, Any]:
    """
    Describe the benchmark formulas and empty-region rules for run metadata.

    Returns:
        dict[str, Any]: Serializable metric, unit, aggregation, and annotation rules.
    """

    return {
        "id": METRIC_PROTOCOL_ID,
        "scope": "whole_volume_per_case",
        "aggregation": "mean_over_cases_per_region_then_mean_over_included_regions",
        "DSC": "2 * intersection / (prediction_voxels + target_voxels)",
        "IoU": "intersection / union; ratio in core, percentage in CSV",
        "HD95": "max of directional area-weighted surfel distance 95th percentiles",
        "ASD": "sum of area * distance over both directions / total area of both surfaces",
        "distance_unit": "mm; spacing follows array axis order",
        "both_empty": "DSC=IoU=1; HD95=ASD=0; included",
        "one_sided_empty": "DSC=IoU=0; HD95=ASD=empty_penalty_mm; included",
        "empty_penalty_rule": "case-diagonal",
        "empty_penalty_formula": "sqrt(sum((shape[k] * spacing[k]) ** 2 for k in range(len(shape))))",
        "empty_penalty_note": "shape and spacing describe the complete processed volume in stored array axis order",
        "unannotated_regions": "AMOS-2022-MRI bladder and prostate/uterus only; NaN excluded, CSV '-'",
        "voxel_ignore_labels": "unsupported; invalid labels and NaN inputs raise",
        "infinite_distance": "included in case and region means; never silently filtered",
        "surface_backend": "BraTS-2023-Metrics 43c905242b2eecf421d4ab2da7af8ece9777d322, vendored through repository",
    }
