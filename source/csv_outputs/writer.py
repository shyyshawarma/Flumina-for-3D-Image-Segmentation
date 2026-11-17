from __future__ import annotations

import csv
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from source.csv_outputs.schema import (
    CASE_COLUMNS,
    CaseMetricRow,
    TopKResult,
    dice_summary_columns,
)
from source.evaluation.registry import get_task


@dataclass(frozen=True, slots=True)
class CSVOutputPaths:
    """
    Paths of the aggregate and per-case checkpoint CSV files.

    Args:
        results (Path): Compact aggregate Dice CSV path.
        cases (Path): Detailed per-case metric CSV path.
    """

    results: Path
    cases: Path


def _validate_top_k(top_k: int) -> None:
    """
    Require a positive integer checkpoint count.

    Args:
        top_k (int): Number of selected checkpoints; booleans are rejected.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k!r}")


def _float_text(value: float) -> str:
    """
    Format a metric value with round-trip precision.

    Args:
        value (float): Numeric value to serialize.

    Returns:
        str: Value formatted with 17 significant digits.
    """
    return format(float(value), ".17g")


def _display_float_text(value: float) -> str:
    """
    Format a value for compact epoch comparison tables.

    Args:
        value (float): Numeric value to display.

    Returns:
        str: Value formatted with six digits after the decimal point.
    """

    return format(float(value), ".6f")


def _optional_float_text(value: float | None) -> str:
    """
    Serialize an optional floating-point metric field.

    Args:
        value (float | None): Numeric value, or None for an absent field.

    Returns:
        str: Value with 17 significant digits, or an empty string for None.
    """
    return "" if value is None else _float_text(value)


def _optional_integer_text(value: int | None) -> str:
    """
    Serialize an optional integer metric field.

    Args:
        value (int | None): Count, or None for an absent field.

    Returns:
        str: Integer text, or an empty string for None.
    """
    return "" if value is None else str(value)


def _checkpoint_signature(
    identity,
    validation_score: float,
    dataset: str,
    protocol_id: str,
    protocol_status: str,
    selection_proxy_id: str,
    subset: str,
    lesion_matching_method: str,
    lesion_matching_threshold: float | None,
) -> tuple[object, ...]:
    """
    Collect the checkpoint metadata that must agree within a rank.

    Args:
        identity (CheckpointIdentity): Checkpoint epoch and portable filename.
        validation_score (float): Score used to rank the checkpoint.
        dataset (str): Canonical dataset name.
        protocol_id (str): Evaluation protocol identifier.
        protocol_status (str): Relationship to the source evaluation protocol.
        selection_proxy_id (str): Checkpoint selection policy identifier.
        subset (str): Evaluated data split.
        lesion_matching_method (str): Lesion matching policy.
        lesion_matching_threshold (float | None): Threshold used for lesion matches.

    Returns:
        tuple[int, str, float, str, str, str, str, str, str, float | None]: Epoch, filename, score, dataset, protocol ID,
            protocol status, selection policy, subset, matching method, and matching threshold.
    """
    return (
        identity.epoch,
        identity.checkpoint,
        float(validation_score),
        dataset,
        protocol_id,
        protocol_status,
        selection_proxy_id,
        subset,
        lesion_matching_method,
        lesion_matching_threshold,
    )


def _validate_rank_set(ranks: set[int], *, top_k: int, rows_name: str) -> None:
    """
    Require every rank from one through top_k.

    Args:
        ranks (set[int]): Ranks represented by the input rows.
        top_k (int): Required number of consecutive checkpoint ranks.
        rows_name (str): Row category included in validation errors.
    """
    expected = set(range(1, top_k + 1))
    if ranks != expected:
        raise ValueError(
            f"{rows_name} ranks must be exactly 1..{top_k}; received {sorted(ranks)}"
        )


def _validate_results(
    rows: Sequence[TopKResult], *, top_k: int
) -> dict[int, tuple[object, ...]]:
    """
    Validate aggregate row keys and checkpoint metadata across ranks.

    Args:
        rows (Sequence[TopKResult]): Nonempty aggregate metric rows.
        top_k (int): Required checkpoint ranks, from one through top_k.

    Returns:
        dict[int, tuple[object, ...]]: Checkpoint signature for each rank.
    """
    _validate_top_k(top_k)
    if not rows:
        raise ValueError("results must contain at least one row")

    signatures: dict[int, tuple[object, ...]] = {}
    keys: set[tuple[object, ...]] = set()
    for row in rows:
        if not isinstance(row, TopKResult):
            raise TypeError("results must contain only TopKResult instances")
        signature = _checkpoint_signature(
            row.identity,
            row.validation_score,
            row.dataset,
            row.protocol_id,
            row.protocol_status,
            row.selection_proxy_id,
            row.subset,
            row.lesion_matching_method,
            row.lesion_matching_threshold,
        )
        previous = signatures.setdefault(row.rank, signature)
        if previous != signature:
            raise ValueError(f"inconsistent checkpoint metadata for rank {row.rank}")
        key = (
            row.rank,
            row.dataset,
            row.subset,
            row.metric,
            row.region,
            row.tolerance_mm,
            row.aggregation,
        )
        if key in keys:
            raise ValueError(f"duplicate aggregate metric key: {key!r}")
        keys.add(key)

    _validate_rank_set(set(signatures), top_k=top_k, rows_name="result")
    return signatures


def _validate_cases(
    rows: Sequence[CaseMetricRow],
    *,
    top_k: int,
    expected_signatures: Mapping[int, tuple[object, ...]] | None = None,
) -> dict[int, tuple[object, ...]]:
    """
    Validate case row keys and checkpoint metadata across ranks.

    Args:
        rows (Sequence[CaseMetricRow]): Per-case metric rows; may be empty.
        top_k (int): Required checkpoint ranks when rows are present.
        expected_signatures (Mapping[int, tuple[object, ...]] | None): Aggregate checkpoint signatures to match, or None for
            independent validation.

    Returns:
        dict[int, tuple[object, ...]]: Checkpoint signature for each rank, or an empty dictionary when rows are empty.
    """
    _validate_top_k(top_k)
    if not rows:
        return {}

    signatures: dict[int, tuple[object, ...]] = {}
    keys: set[tuple[object, ...]] = set()
    for row in rows:
        if not isinstance(row, CaseMetricRow):
            raise TypeError("case_metrics must contain only CaseMetricRow instances")
        signature = _checkpoint_signature(
            row.identity,
            row.validation_score,
            row.dataset,
            row.protocol_id,
            row.protocol_status,
            row.selection_proxy_id,
            row.subset,
            row.lesion_matching_method,
            row.lesion_matching_threshold,
        )
        previous = signatures.setdefault(row.rank, signature)
        if previous != signature:
            raise ValueError(f"inconsistent checkpoint metadata for rank {row.rank}")
        if (
            expected_signatures is not None
            and expected_signatures.get(row.rank) != signature
        ):
            raise ValueError(
                f"case metric metadata does not match aggregate result for rank {row.rank}"
            )
        key = (
            row.rank,
            row.dataset,
            row.subset,
            row.case_id,
            row.patient_id,
            row.metric,
            row.region,
            row.tolerance_mm,
        )
        if key in keys:
            raise ValueError(f"duplicate per-case metric key: {key!r}")
        keys.add(key)

    _validate_rank_set(set(signatures), top_k=top_k, rows_name="case metric")
    return signatures


def _case_record(row: CaseMetricRow) -> dict[str, object]:
    """
    Convert a case metric to the serialized CSV fields.

    Args:
        row (CaseMetricRow): Validated case metric row.

    Returns:
        dict[str, str | int]: CSV fields with formatted numeric scores and empty strings for absent optional values.
    """
    return {
        "rank": row.rank,
        "epoch": row.identity.epoch,
        "checkpoint": row.identity.checkpoint,
        "validation_score": _float_text(row.validation_score),
        "dataset": row.dataset,
        "protocol_id": row.protocol_id,
        "protocol_status": row.protocol_status,
        "selection_proxy_id": row.selection_proxy_id,
        "subset": row.subset,
        "evaluation_space": row.evaluation_space,
        "data_integrity_status": row.data_integrity_status,
        "dropped_nonzero_label_voxels": row.dropped_nonzero_label_voxels,
        "case_id": row.case_id,
        "patient_id": row.patient_id,
        "metric_scope": row.metric_scope,
        "lesion_matching_method": row.lesion_matching_method,
        "lesion_matching_threshold": _optional_float_text(
            row.lesion_matching_threshold
        ),
        "metric": row.metric,
        "region": row.region,
        "tolerance_mm": _optional_float_text(row.tolerance_mm),
        "unit": row.unit,
        "true_positive": _optional_integer_text(row.true_positive),
        "false_positive": _optional_integer_text(row.false_positive),
        "false_negative": _optional_integer_text(row.false_negative),
        "value": _float_text(row.value),
    }


def _atomic_write_csv(
    path: str | Path,
    *,
    columns: Sequence[str],
    records: Iterable[Mapping[str, object]],
) -> Path:
    """
    Write a CSV beside its destination, then atomically replace the file.

    Args:
        path (str | Path): Destination CSV path.
        columns (Sequence[str]): Ordered header names.
        records (Iterable[Mapping[str, object]]): Row mappings keyed by the header.

    Returns:
        Path: Destination of the completed UTF-8 CSV, including its header.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(
                handle,
                fieldnames=columns,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(records)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output


def _dice_summary_spec(
    rows: Sequence[TopKResult],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """
    Resolve the registered region columns for one dataset.

    Args:
        rows (Sequence[TopKResult]): Aggregate rows identifying exactly one dataset.

    Returns:
        tuple[str, tuple[str, ...], tuple[str, ...]]: Canonical dataset name, ordered selection regions, and the full CSV
            header.
    """

    datasets = {row.dataset for row in rows}
    if len(datasets) != 1:
        raise ValueError(
            "one Dice summary CSV must contain exactly one dataset, got "
            f"{sorted(datasets)}"
        )
    dataset = next(iter(datasets))
    task = get_task(dataset)
    regions = task.selection_regions
    return task.dataset_name, regions, dice_summary_columns(regions)


def _dice_summary_records(
    rows: Sequence[TopKResult],
) -> tuple[dict[str, str | int], ...]:
    """
    Pivot regional case-macro Dice into one compact row per epoch.

    Args:
        rows (Sequence[TopKResult]): Subject-level case-macro Dice rows, with one row per registered selection region for each
            epoch.

    Returns:
        tuple[dict[str, str | int], ...]: Records sorted by epoch, containing Epoch, each region, and their arithmetic Average.
            Scores use six decimal places; validation averages must match the checkpoint selection score.
    """

    dataset, regions, _ = _dice_summary_spec(rows)

    grouped: dict[int, list[TopKResult]] = {}
    for row in rows:
        grouped.setdefault(row.identity.epoch, []).append(row)

    records: list[dict[str, str | int]] = []
    for epoch in sorted(grouped):
        epoch_rows = grouped[epoch]
        if len(epoch_rows) != len(regions):
            raise ValueError(
                f"{dataset} epoch {epoch} must contain exactly one Dice row "
                f"for each of {regions}, got {len(epoch_rows)} rows"
            )
        if any(
            row.metric != "dice"
            or row.aggregation != "case_macro"
            or row.metric_scope != "subject"
            for row in epoch_rows
        ):
            raise ValueError(
                f"{dataset} summaries require subject-level case-macro Dice"
            )
        values = {row.region: float(row.value) for row in epoch_rows}
        if set(values) != set(regions):
            raise ValueError(
                f"{dataset} epoch {epoch} regions must be {regions}, "
                f"got {tuple(sorted(values))}"
            )
        average = float(sum(values[region] for region in regions) / len(regions))
        if {row.subset for row in epoch_rows} == {"validation"} and not math.isclose(
            average,
            float(epoch_rows[0].validation_score),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                f"validation Average for epoch {epoch} must equal composite_score: "
                f"{average} != {epoch_rows[0].validation_score}"
            )
        records.append(
            {
                "Epoch": epoch,
                **{region: _display_float_text(values[region]) for region in regions},
                "Average": _display_float_text(average),
            }
        )
    return tuple(records)


def _read_dice_summary(
    path: Path,
    *,
    dataset: str,
    regions: Sequence[str],
) -> tuple[dict[str, str], ...]:
    """
    Read a compact Dice history and validate its columns and values.

    Args:
        path (Path): Existing UTF-8 Dice history CSV.
        dataset (str): Dataset name used in validation errors.
        regions (Sequence[str]): Required region columns in their registered order.

    Returns:
        tuple[dict[str, str], ...]: Records in file order, with unique nonnegative epochs, finite scores, and Average consistent
            with the regional mean.
    """

    columns = dice_summary_columns(regions)

    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != columns:
                raise ValueError(
                    f"{dataset} Dice history header must exactly match "
                    f"{','.join(columns)}; legacy metric histories must be "
                    "regenerated from checkpoints"
                )
            records: list[dict[str, str]] = []
            seen_epochs: set[int] = set()
            for line_number, record in enumerate(reader, start=2):
                if None in record or any(value is None for value in record.values()):
                    raise ValueError(
                        f"malformed Dice history row at line {line_number}"
                    )
                try:
                    epoch = int(record["Epoch"])
                    values = [float(record[region]) for region in regions]
                    average = float(record["Average"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid Dice history row at line {line_number}"
                    ) from exc
                if epoch < 0 or epoch in seen_epochs:
                    raise ValueError(
                        f"Dice history contains an invalid or duplicate epoch {epoch}"
                    )
                if any(not math.isfinite(value) for value in (*values, average)):
                    raise ValueError(
                        f"Dice history contains a non-finite value at line {line_number}"
                    )
                if not math.isclose(
                    average,
                    sum(values) / len(regions),
                    rel_tol=0.0,
                    abs_tol=1.0e-6,
                ):
                    raise ValueError(
                        f"Dice history Average is inconsistent at line {line_number}"
                    )
                seen_epochs.add(epoch)
                records.append(dict(record))  # type: ignore[arg-type]
    except UnicodeDecodeError as exc:
        raise ValueError("Dice history must be UTF-8 CSV") from exc
    return tuple(records)


def _write_dice_validation_history(
    path: str | Path,
    rows: Sequence[TopKResult],
) -> Path:
    """
    Append one epoch while rejecting conflicting or out-of-order rewrites.

    Args:
        path (str | Path): Validation history CSV path.
        rows (Sequence[TopKResult]): Region summaries identifying one epoch.

    Returns:
        Path: History path. An identical serialized epoch is accepted without rewriting the file; a new epoch must exceed every
            stored epoch.
    """

    summaries = _dice_summary_records(rows)
    dataset, regions, columns = _dice_summary_spec(rows)
    if len(summaries) != 1:
        raise ValueError("one validation call must produce exactly one epoch summary")
    incoming = summaries[0]
    incoming_epoch = int(incoming["Epoch"])
    output = Path(path)
    existing = (
        _read_dice_summary(output, dataset=dataset, regions=regions)
        if output.exists()
        else ()
    )
    by_epoch = {int(record["Epoch"]): record for record in existing}
    previous = by_epoch.get(incoming_epoch)
    normalized_incoming = {key: str(value) for key, value in incoming.items()}
    if previous == normalized_incoming:
        return output
    if previous is not None:
        raise ValueError(
            f"conflicting validation history rewrite for epoch {incoming_epoch}"
        )
    if by_epoch and incoming_epoch <= max(by_epoch):
        raise ValueError(
            "validation history epochs must append monotonically; "
            f"got epoch {incoming_epoch} after epoch {max(by_epoch)}"
        )
    records: tuple[Mapping[str, object], ...] = (*existing, incoming)
    return _atomic_write_csv(
        output,
        columns=columns,
        records=records,
    )


def write_validation_history(
    path: str | Path,
    rows: Iterable[TopKResult],
) -> Path:
    """
    Append one validation epoch to the compact Dice history.

    Args:
        path (str | Path): Output CSV path.
        rows (Iterable[TopKResult]): Registered region summaries for one epoch, each with rank one.

    Returns:
        Path: History path. Identical serialized epochs leave the file unchanged; conflicting or out-of-order epochs are
            rejected.
    """

    incoming = tuple(rows)
    _validate_results(incoming, top_k=1)
    return _write_dice_validation_history(path, incoming)


def initialize_validation_history(
    path: str | Path,
    *,
    dataset_name: str | None = None,
    resume_epoch: int | None = None,
) -> Path:
    """
    Create a Dice history or rewind it to a saved training epoch.

    Args:
        path (str | Path): Validation CSV path.
        dataset_name (str | None): Dataset determining the region columns; a supported name is required and None raises
            ValueError.
        resume_epoch (int | None): Last committed epoch. Later rows are removed when supplied; None retains all existing rows.

    Returns:
        Path: Validated history path, with the registered header if newly created.
    """

    output = Path(path)
    if dataset_name is None:
        raise ValueError("dataset_name is required to initialize validation history")
    if resume_epoch is not None and (
        isinstance(resume_epoch, bool)
        or not isinstance(resume_epoch, int)
        or resume_epoch < 0
    ):
        raise ValueError("resume_epoch must be a non-negative integer or None")
    task = get_task(dataset_name)
    columns = dice_summary_columns(task.selection_regions)
    if not output.exists():
        return _atomic_write_csv(output, columns=columns, records=())
    existing = _read_dice_summary(
        output,
        dataset=task.dataset_name,
        regions=task.selection_regions,
    )
    if resume_epoch is None:
        return output
    retained = tuple(
        record for record in existing if int(record["Epoch"]) <= resume_epoch
    )
    if len(retained) == len(existing):
        return output
    return _atomic_write_csv(output, columns=columns, records=retained)


def write_topk_results(
    path: str | Path,
    rows: Iterable[TopKResult],
    *,
    top_k: int,
) -> Path:
    """
    Atomically write compact Dice summaries in ascending epoch order.

    Args:
        path (str | Path): Aggregate CSV destination.
        rows (Iterable[TopKResult]): Registered region summaries for the selected checkpoints, with one subject-level case-macro
            Dice row per region.
        top_k (int): Number of checkpoint ranks, which must span one through top_k.

    Returns:
        Path: Written aggregate CSV path.
    """

    materialized = tuple(rows)
    _validate_results(materialized, top_k=top_k)
    _, _, columns = _dice_summary_spec(materialized)
    return _atomic_write_csv(
        path,
        columns=columns,
        records=_dice_summary_records(materialized),
    )


def write_topk_cases(
    path: str | Path,
    rows: Iterable[CaseMetricRow],
    *,
    top_k: int,
) -> Path:
    """
    Atomically write case metrics sorted by rank and case metadata.

    Args:
        path (str | Path): Per-case CSV destination.
        rows (Iterable[CaseMetricRow]): Case metric rows for all checkpoint ranks, or an empty iterable to write only the
            header.
        top_k (int): Number of required checkpoint ranks when rows are present.

    Returns:
        Path: Written per-case CSV path.
    """

    materialized = tuple(rows)
    _validate_cases(materialized, top_k=top_k)
    ordered = sorted(
        materialized,
        key=lambda row: (
            row.rank,
            row.dataset,
            row.subset,
            row.case_id,
            row.patient_id,
            row.metric,
            row.region,
            row.tolerance_mm if row.tolerance_mm is not None else -1.0,
        ),
    )
    return _atomic_write_csv(
        path,
        columns=CASE_COLUMNS,
        records=(_case_record(row) for row in ordered),
    )


def write_topk_csvs(
    output_dir: str | Path,
    results: Iterable[TopKResult],
    case_metrics: Iterable[CaseMetricRow] = (),
    *,
    top_k: int,
    results_path: str | Path | None = None,
    cases_path: str | Path | None = None,
) -> CSVOutputPaths:
    """
    Write aggregate Dice and detailed case metrics for ranked checkpoints.

    Args:
        output_dir (str | Path): Directory used for default output filenames.
        results (Iterable[TopKResult]): Region summaries for every checkpoint rank.
        case_metrics (Iterable[CaseMetricRow]): Optional case rows whose checkpoint metadata must match the aggregate rows.
        top_k (int): Number of selected checkpoint ranks.
        results_path (str | Path | None): Aggregate CSV destination, or None for topk_results.csv under output_dir.
        cases_path (str | Path | None): Per-case CSV destination, or None for topk_cases.csv under output_dir.

    Returns:
        CSVOutputPaths: Aggregate and per-case CSV paths. Aggregate rows are ordered by epoch; case rows by rank and case
            metadata. Empty case input still produces the standard header.
    """

    _validate_top_k(top_k)
    result_rows = tuple(results)
    case_rows = tuple(case_metrics)
    signatures = _validate_results(result_rows, top_k=top_k)
    _validate_cases(
        case_rows,
        top_k=top_k,
        expected_signatures=signatures,
    )

    directory = Path(output_dir)
    result_output = (
        Path(results_path)
        if results_path is not None
        else directory / "topk_results.csv"
    )
    cases_output = (
        Path(cases_path) if cases_path is not None else directory / "topk_cases.csv"
    )
    if result_output.resolve() == cases_output.resolve():
        raise ValueError("results_path and cases_path must refer to different files")

    ordered_results = sorted(
        result_rows,
        key=lambda row: (
            row.rank,
            row.dataset,
            row.subset,
            row.metric,
            row.region,
            row.tolerance_mm if row.tolerance_mm is not None else -1.0,
            row.aggregation,
        ),
    )
    ordered_cases = sorted(
        case_rows,
        key=lambda row: (
            row.rank,
            row.dataset,
            row.subset,
            row.case_id,
            row.patient_id,
            row.metric,
            row.region,
            row.tolerance_mm if row.tolerance_mm is not None else -1.0,
        ),
    )
    _, _, columns = _dice_summary_spec(ordered_results)
    _atomic_write_csv(
        result_output,
        columns=columns,
        records=_dice_summary_records(ordered_results),
    )
    _atomic_write_csv(
        cases_output,
        columns=CASE_COLUMNS,
        records=(_case_record(row) for row in ordered_cases),
    )
    return CSVOutputPaths(results=result_output, cases=cases_output)


__all__ = [
    "CSVOutputPaths",
    "write_topk_cases",
    "write_topk_csvs",
    "write_topk_results",
    "initialize_validation_history",
    "write_validation_history",
]
