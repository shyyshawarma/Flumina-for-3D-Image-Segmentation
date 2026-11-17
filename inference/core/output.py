import csv
import io
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4


def atomic_text(path: Path, content: str) -> None:
    """
    Write text through a sibling temporary file and atomically replace the destination.

    Args:
        path (Path): Destination file; missing parent directories are created.
        content (str): Complete UTF-8 text to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def portable_values(value: Any) -> Any:
    """
    Convert metadata paths and nonfinite numbers to portable JSON values.

    Args:
        value (Mapping | list | tuple | Path | str | int | float | bool | None): Nested metadata containing JSON scalars,
            containers, and file paths.

    Returns:
        dict | list | str | int | float | bool | None: Converted metadata with basename-only paths, list containers, and text
            for NaN or infinities.
    """
    match value:
        case float() if not math.isfinite(value):
            if math.isnan(value):
                return "nan"
            return "inf" if value > 0 else "-inf"
        case Path():
            return value.name
        case str() if value.startswith("/") or ":\\" in value:
            return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        case Mapping():
            return {str(key): portable_values(item) for key, item in value.items()}
        case tuple() | list():
            return [portable_values(item) for item in value]
        case _:
            return value


def write_json(path: Path, value: Any) -> None:
    """
    Atomically save indented JSON after converting metadata to portable values.

    Args:
        path (Path): Destination JSON file.
        value (Mapping | list | tuple | Path | str | int | float | bool | None): Input metadata tree accepted by
            portable_values.
    """
    atomic_text(
        path, json.dumps(portable_values(value), indent=2, allow_nan=False) + "\n"
    )


def render_csv(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """
    Render a header and rows as CSV with newline-terminated records.

    Args:
        header (Sequence[str]): Ordered column names.
        rows (Sequence[Sequence[str | int | float | bool | None]]): Values in column order.

    Returns:
        str: CSV text containing the header followed by all supplied rows.
    """
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue()


def metric_cell(value: float) -> str:
    """
    Format a metric value for a summary CSV cell.

    Args:
        value (float): Metric value, including NaN or infinity.

    Returns:
        str: Six-decimal finite value, a dash for NaN, or signed infinity text.
    """
    if math.isnan(value):
        return "-"
    return "inf" if value == math.inf else f"{value:.6f}"


def summary_row(
    epoch: int, values: Mapping[str, float], scale: float = 1.0
) -> list[str]:
    """
    Format region values and their unweighted mean for one epoch.

    Args:
        epoch (int): Checkpoint epoch written in the first column.
        values (Mapping[str, float]): Ordered region scores with at least one non-NaN value.
        scale (float): Multiplier applied to every score and the mean.

    Returns:
        list[str]: Epoch, region cells, and the mean over non-NaN regions; included infinities are retained.
    """
    included = [
        float(value) for value in values.values() if not math.isnan(float(value))
    ]
    average = sum(included) / len(included)
    return [
        str(epoch),
        *(metric_cell(value * scale) for value in values.values()),
        metric_cell(average * scale),
    ]
