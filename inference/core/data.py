from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from source.dataset.contracts import resolve_split_file
from source.dataset.runtime_contracts import get_dataset_contract


def physical_dataset(dataset: str) -> str:
    """
    Resolve the dataset identity used by the stored data and label contract.

    Args:
        dataset (str): Dataset alias or AMOS modality view.

    Returns:
        str: Canonical storage dataset name; AMOS CT and MRI share AMOS-2022.
    """
    token = "".join(c for c in dataset.casefold() if c.isalnum())
    if token in {"amos2022ct", "amos2022mri"}:
        return "AMOS-2022"
    return get_dataset_contract(dataset).dataset_name


def canonical_dataset(dataset: str) -> str:
    """
    Normalize a dataset name while preserving an AMOS modality selection.

    Args:
        dataset (str): Dataset alias or AMOS modality view.

    Returns:
        str: Canonical evaluation name, preserving any explicit CT or MRI suffix.
    """
    token = "".join(c for c in dataset.casefold() if c.isalnum())
    match token:
        case "amos2022ct":
            return "AMOS-2022-CT"
        case "amos2022mri":
            return "AMOS-2022-MRI"
        case _:
            return physical_dataset(dataset)


def sha256_file(path: Path) -> str:
    """
    Hash a file in bounded memory.

    Args:
        path (Path): File to read as bytes.

    Returns:
        str: Hexadecimal SHA-256 digest of the full file.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """
    Read a UTF-8 JSON document and require an object at its root.

    Args:
        path (Path): JSON file to load.

    Returns:
        dict[str, Any]: Decoded object with its stored fields preserved.
    """
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_case_list(path: Path) -> tuple[str, ...]:
    """
    Read unique case identifiers from nonblank lines.

    Args:
        path (Path): Text file containing one plain case identifier per line.

    Returns:
        tuple[str, ...]: Nonempty, duplicate-free identifiers in file order.
    """
    names = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not names or len(names) != len(set(names)):
        raise ValueError(f"Case list must be nonempty and duplicate-free: {path}")
    if any(Path(name).name != name or name in {".", ".."} for name in names):
        raise ValueError(f"Invalid case identifier in {path}")
    return names


def _dataset_identity(payload: Mapping[str, Any]) -> str:
    """
    Resolve one physical dataset from a processed manifest descriptor.

    Args:
        payload (Mapping[str, Any]): Manifest with a dataset name or identity mapping.

    Returns:
        str: Canonical dataset shared by all populated identity fields.
    """
    match payload.get("dataset"):
        case str() as identity:
            return physical_dataset(identity)
        case Mapping() as identity:
            names = [
                identity[key]
                for key in ("key", "physical_output", "name")
                if identity.get(key)
            ]
            resolved = {physical_dataset(str(name)) for name in names}
            if len(resolved) == 1:
                return resolved.pop()
    raise ValueError(
        "Processed manifest must unambiguously identify its physical dataset"
    )


def resolve_data_root(data_path: str | Path, dataset: str) -> Path:
    """
    Locate a processed dataset directory and verify its manifest identity.

    Args:
        data_path (str | Path): Dataset directory or shared parent directory.
        dataset (str): Requested dataset or AMOS modality view.

    Returns:
        Path: Resolved dataset directory containing the matching manifest.json.
    """
    name = physical_dataset(dataset)
    supplied = Path(data_path).expanduser()
    candidate = supplied if (supplied / "manifest.json").is_file() else supplied / name
    manifest = candidate / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Processed dataset manifest not found: {manifest}")
    actual = _dataset_identity(read_json(manifest))
    if actual != name:
        raise ValueError(
            f"Dataset mismatch: requested {name}, manifest {manifest} declares {actual}"
        )
    return candidate.resolve()


def _relative_file(root: Path, value: object, *, kind: str) -> Path:
    """
    Resolve a manifest file path within the processed dataset root.

    Args:
        root (Path): Resolved dataset directory.
        value (str): Nonempty relative file path without parent traversal.
        kind (str): File role used in validation errors.

    Returns:
        Path: Existing resolved file contained by root.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"Manifest {kind} path is missing")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"Manifest {kind} path must remain inside dataset root: {value}"
        )
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise FileNotFoundError(
            f"Manifest {kind} file is missing or escapes dataset root: {root / path}"
        )
    return resolved


def _spacing(value: object, *, case_id: str) -> tuple[float, float, float]:
    """
    Validate a manifest spacing triple in stored spatial axis order.

    Args:
        value (list[float | int] | tuple[float | int, ...]): Three positive, finite mm values.
        case_id (str): Case identifier used in validation errors.

    Returns:
        tuple[float, float, float]: Physical voxel spacing in array axis order.
    """
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(
            isinstance(x, bool)
            or not isinstance(x, (float, int))
            or not math.isfinite(x)
            or x <= 0
            for x in value
        )
    ):
        raise ValueError(
            f"{case_id}: geometry.canonical_spacing must give three positive finite mm values"
        )
    return cast(tuple[float, float, float], tuple(float(x) for x in value))


@dataclass(frozen=True)
class Case:
    """
    Files and geometry for one processed image and label pair.

    Args:
        case_id (str): Unique case identifier.
        volume_path (Path): Stored image array file.
        label_path (Path): Stored categorical label array file.
        backend (str): Array format: npy, b2nd, or nnunet_b2nd.
        spacing (tuple[float, float, float]): Voxel spacing in stored spatial axis order, in mm.
        metadata (dict[str, Any]): Case identity, modality, and label integrity fields.
        storage (dict[str, Any]): Manifest output descriptor with native and canonical layouts.
    """

    case_id: str
    volume_path: Path
    label_path: Path
    backend: str
    spacing: tuple[float, float, float]
    metadata: dict[str, Any]
    storage: dict[str, Any]


@dataclass(frozen=True)
class EvaluationData:
    """
    Selected evaluation cases with their membership and manifest provenance.

    Args:
        dataset (str): Canonical evaluation dataset, including any AMOS modality suffix.
        root (Path): Resolved processed dataset directory.
        membership_path (Path): Case list used to select the evaluation cohort.
        cases (tuple[Case, ...]): Validated labeled cases in evaluation order.
        provenance (dict[str, Any]): Manifest hashes, split hashes, selection, and storage metadata.
    """

    dataset: str
    root: Path
    membership_path: Path
    cases: tuple[Case, ...]
    provenance: dict[str, Any]


def _validate_release_list(
    payload: Mapping[str, Any], path: Path, names: Sequence[str]
) -> None:
    """
    Verify declared split counts and hashes when release metadata matches the file.

    Args:
        payload (Mapping[str, Any]): Processed manifest with optional release_lists entries.
        path (Path): Membership file whose release metadata is checked.
        names (Sequence[str]): Case identifiers already read from the file.
    """
    releases = payload.get("release_lists", {})
    item = releases.get(path.stem) if isinstance(releases, Mapping) else None
    if not isinstance(item, Mapping) or item.get("path") != path.name:
        return
    if item.get("count") is not None and int(item["count"]) != len(names):
        raise ValueError(f"Case count differs from manifest release_lists: {path}")
    if item.get("sha256") and sha256_file(path) != item["sha256"]:
        raise ValueError(
            f"Split content differs from manifest release_lists SHA-256: {path}"
        )


def prepare_data(
    dataset: str,
    config: Mapping[str, Any],
    data_path: str | Path,
    *,
    membership_file: str | Path | None = None,
) -> EvaluationData:
    """
    Resolve labeled test cases and validate their manifest and split separation.

    Args:
        dataset (str): Dataset name or AMOS modality view.
        config (Mapping[str, Any]): Saved dataset, channel, and class settings.
        data_path (str | Path): Processed dataset directory or shared parent.
        membership_file (str | Path | None): Explicit case list overriding the saved membership_file or default test.txt.

    Returns:
        EvaluationData: Selected case descriptors with validated spacing, label contracts, and split provenance. Image and label
            arrays are opened by load_case.
    """
    dataset = canonical_dataset(dataset)
    physical = physical_dataset(dataset)
    contract = get_dataset_contract(physical)
    if (
        physical_dataset(str(config.get("dataset_name", config.get("dataset", ""))))
        != physical
    ):
        raise ValueError(
            "Training configuration dataset differs from requested benchmark dataset"
        )
    for key, expected in (
        ("in_channels", contract.in_channels),
        ("num_classes", contract.num_classes),
    ):
        if key not in config or int(config[key]) != expected:
            raise ValueError(
                f"Training config {key} must be {expected} for {physical}, got {config.get(key)!r}"
            )
    root = resolve_data_root(data_path, dataset)
    manifest_path = root / "manifest.json"
    payload = read_json(manifest_path)
    descriptor = payload["dataset"]
    if isinstance(descriptor, Mapping):
        channels = descriptor.get("channel_order")
        if channels is None and isinstance(descriptor.get("modalities"), list):
            channels = [m.get("name") for m in descriptor["modalities"]]
        if channels is not None and tuple(channels) != contract.channel_names:
            raise ValueError(
                f"Channel order differs from trained contract: {channels} versus {contract.channel_names}"
            )
        labels = descriptor.get("label_names", descriptor.get("labels"))
        if labels is not None:
            ids = (
                sorted(int(k) for k in labels)
                if isinstance(labels, Mapping)
                else sorted(int(x[0]) for x in labels)
            )
            if ids != list(range(contract.num_classes)):
                raise ValueError(f"Manifest label set differs from {physical}: {ids}")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("Processed manifest must contain cases[]")
    records: dict[str, dict[str, Any]] = {}
    for record in raw_cases:
        if not isinstance(record, dict):
            raise ValueError("Each manifest case must be an object")
        name = record.get("case_id")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in {".", ".."}
            or name in records
        ):
            raise ValueError(f"Invalid or repeated manifest case_id: {name!r}")
        records[name] = record

    selected_file = membership_file or config.get("membership_file")
    if selected_file:
        membership_path = Path(selected_file)
        if not membership_path.is_absolute():
            membership_path = root / membership_path
    else:
        membership_path = resolve_split_file(root, role="test")
    membership_path = membership_path.resolve()
    requested = read_case_list(membership_path)
    labeled_path = root / "labeled.txt"
    labeled_names = read_case_list(labeled_path)
    _validate_release_list(payload, labeled_path, labeled_names)
    _validate_release_list(payload, membership_path, requested)
    labeled = set(labeled_names)
    selected = tuple(name for name in requested if name in labeled)
    excluded = tuple(name for name in requested if name not in labeled)
    if not selected:
        raise ValueError(f"Test list has no labeled cases: {membership_path}")
    if any(name not in records for name in selected):
        raise ValueError("Selected labeled case is absent from processed manifest")

    train_path = resolve_split_file(root, role="train")
    val_path = resolve_split_file(root, role="val")
    train_names, val_names = read_case_list(train_path), read_case_list(val_path)
    _validate_release_list(payload, train_path, train_names)
    _validate_release_list(payload, val_path, val_names)
    cohorts = {
        "train": set(train_names) & labeled,
        "val": set(val_names) & labeled,
        "test": set(selected),
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = cohorts[left] & cohorts[right]
        if overlap:
            raise ValueError(f"{left}/{right} case overlap: {sorted(overlap)[:5]}")
    if physical == "BraTS-2025-GLI":
        patients: dict[str, set[str]] = {}
        for role, names in cohorts.items():
            ids = [records.get(name, {}).get("patient_id") for name in names]
            if any(not isinstance(pid, str) or not pid.strip() for pid in ids):
                raise ValueError(
                    f"Missing patient_id prevents patient separation verification for {role}"
                )
            patients[role] = set(cast(list[str], ids))
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            if patients[left] & patients[right]:
                raise ValueError(f"{left}/{right} patient overlap")

    modality = (
        "CT" if dataset.endswith("-CT") else "MRI" if dataset.endswith("-MRI") else None
    )
    if physical == "AMOS-2022" and any(
        records[name].get("modality") not in {"CT", "MRI"} for name in selected
    ):
        raise ValueError(
            "AMOS evaluation requires explicit CT/MRI modality for every selected case"
        )
    if modality:
        selected = tuple(
            name for name in selected if records[name]["modality"] == modality
        )
        if not selected:
            raise ValueError(f"No {modality} cases in selected AMOS cohort")

    cases: list[Case] = []
    for name in selected:
        record = records[name]
        geometry = record.get("geometry", {})
        if not isinstance(geometry, Mapping):
            raise ValueError(f"{name}: geometry must be an object")
        spacing = _spacing(geometry.get("canonical_spacing"), case_id=name)
        support = record.get("label_support", {})
        dropped = (
            support.get("dropped_nonzero_label_voxels")
            if isinstance(support, Mapping)
            else None
        )
        if isinstance(dropped, bool) or not isinstance(dropped, int) or dropped < 0:
            raise ValueError(
                f"{name}: missing/invalid label_support.dropped_nonzero_label_voxels"
            )
        if record.get("label_status", record.get("status")) != "labeled":
            raise ValueError(
                f"{name}: labeled.txt conflicts with manifest label_status"
            )
        if physical == "BraTS-2025-GLI" and record.get("treatment_status") not in {
            "preoperative",
            "postoperative",
        }:
            raise ValueError(f"{name}: missing or invalid treatment_status")
        output = record.get("output")
        if not isinstance(output, dict):
            raise ValueError(f"{name}: manifest output object missing")
        backend = record.get("storage_backend")
        if backend not in {"npy", "b2nd", "nnunet_b2nd"}:
            raise ValueError(
                f"{name}: unsupported explicit storage_backend {backend!r}"
            )
        if backend == "b2nd":
            transform = output.get("transform", {})
            if not str(transform.get("volume_axis", "")).startswith(
                "identity"
            ) or not str(transform.get("label_axis", "")).startswith("identity"):
                raise ValueError(
                    f"{name}: canonical B2ND data must declare identity spatial transforms"
                )
        if modality == "MRI":
            counts = output.get("canonical", {}).get("label", {}).get("counts")
            if not isinstance(counts, dict) or any(
                counts.get(str(label), 0) != 0 for label in (14, 15)
            ):
                raise ValueError(
                    f"{name}: MRI unsupported labels 14/15 are present or label counts missing"
                )
        metadata = {
            key: record.get(key)
            for key in (
                "patient_id",
                "treatment_status",
                "modality",
                "cohort",
                "source_role",
            )
        }
        metadata.update(
            {
                "evaluation_space": "processed_crop",
                "array_spatial_order": geometry.get(
                    "array_spatial_order",
                    "XYZ" if physical.startswith("BraTS-") else "DHW",
                ),
                "dropped_nonzero_label_voxels": dropped,
                "data_integrity_status": "complete"
                if dropped == 0
                else "cropped_nonzero_label_loss",
            }
        )
        cases.append(
            Case(
                name,
                _relative_file(
                    root,
                    output.get("volume"),
                    kind="volume",
                ),
                _relative_file(
                    root,
                    output.get("label"),
                    kind="label",
                ),
                backend,
                spacing,
                metadata,
                output,
            )
        )

    provenance = {
        "physical_dataset": physical,
        "dataset_root": root.name,
        "processed_manifest": manifest_path.name,
        "processed_manifest_sha256": sha256_file(manifest_path),
        "membership_file": membership_path.name,
        "membership_sha256": sha256_file(membership_path),
        "train_membership_sha256": sha256_file(train_path),
        "val_membership_sha256": sha256_file(val_path),
        "requested_case_count": len(requested),
        "excluded_unlabeled_cases": list(excluded),
        "modality_filter": modality,
        "case_ids": list(selected),
        "case_count": len(cases),
        "evaluation_space": "processed_crop",
        "preprocessing": "stored normalization and geometry; dtype conversion and channel-last to channel-first only",
        "spacing_source": "cases[].geometry.canonical_spacing in stored array axis order",
        "label_support_masking": "none; preserves source evaluation on canonical labels including mapped background",
        "cropped_nonzero_label_voxels": sum(
            case.metadata["dropped_nonzero_label_voxels"] for case in cases
        ),
        "storage_backends": sorted({case.backend for case in cases}),
    }
    return EvaluationData(dataset, root, membership_path, tuple(cases), provenance)


def _read_array(path: Path, backend: str):
    """
    Open a stored array while preserving its native dtype and axis order.

    Args:
        path (Path): NPY or B2ND array file.
        backend (str): Storage format: npy, b2nd, or nnunet_b2nd.

    Returns:
        np.ndarray: Array with its stored shape; NPY files use a read-only memory map.
    """
    import numpy as np

    if backend == "npy":
        return np.load(path, allow_pickle=False, mmap_mode="r")
    import blosc2

    # Blosc2 4.1.2 annotates all keyword values as dict, but documents mmap_mode as str.
    open_array = cast(Callable[..., blosc2.NDArray], blosc2.open)
    handle = open_array(str(path), mode="r", mmap_mode="r", dparams={"nthreads": 1})
    try:
        return np.asarray(handle[tuple(slice(None) for _ in handle.shape)])
    finally:
        close = getattr(handle, "close", None)
        if callable(close):
            close()


def load_case(case: Case, *, in_channels: int, num_classes: int):
    """
    Load processed arrays on their stored spatial grid with channel-first images.

    Canonical NPY and B2ND images use (D, H, W, C). nnU-Net stores images as (C, D, H, W) and labels as (1, D, H, W), with -1
    mapped to background. Spatial axis order and image normalization are preserved.

    Args:
        case (Case): Processed case descriptor with storage layout and geometry.
        in_channels (int): Expected image channel count C.
        num_classes (int): Number of categorical classes, including background.

    Returns:
        tuple[np.ndarray, np.ndarray]: Contiguous float32 image of shape (C, D, H, W) and uint8 target of shape (D, H, W), both
            on CPU.
    """
    import numpy as np

    image = _read_array(case.volume_path, case.backend)
    target = _read_array(case.label_path, case.backend)
    native = case.storage.get("native", {})
    for key, array in (("volume", image), ("label", target)):
        descriptor = native.get(key, {}) if isinstance(native, Mapping) else {}
        if (
            descriptor.get("shape") is not None
            and tuple(descriptor["shape"]) != array.shape
        ):
            raise ValueError(
                f"{case.case_id}: decoded {key} shape disagrees with manifest"
            )
        if (
            descriptor.get("dtype") is not None
            and np.dtype(descriptor["dtype"]) != array.dtype
        ):
            raise ValueError(
                f"{case.case_id}: decoded {key} dtype disagrees with manifest"
            )
    if not np.issubdtype(image.dtype, np.floating) or not np.issubdtype(
        target.dtype, np.integer
    ):
        raise ValueError(
            f"{case.case_id}: expected floating image and integer categorical labels"
        )
    if case.backend == "nnunet_b2nd":
        if (
            image.ndim != 4
            or image.shape[0] != in_channels
            or target.ndim != 4
            or target.shape[0] != 1
        ):
            raise ValueError(f"{case.case_id}: nnU-Net data must be CDHW / 1DHW")
        image = np.moveaxis(image, 0, -1)
        target = target[0]
        if np.any(target < -1):
            raise ValueError(f"{case.case_id}: invalid nnU-Net labels below -1")
        target = np.where(target == -1, 0, target)
    if (
        image.ndim != 4
        or target.ndim != 3
        or image.shape[:-1] != target.shape
        or image.shape[-1] != in_channels
    ):
        raise ValueError(
            f"{case.case_id}: image/label shape mismatch {image.shape} / {target.shape}"
        )
    if (
        not np.all(np.isfinite(image))
        or min(target.shape) <= 0
        or np.any(target < 0)
        or np.any(target >= num_classes)
    ):
        raise ValueError(
            f"{case.case_id}: nonfinite image, empty spatial axis or invalid class label"
        )
    canonical = case.storage.get("canonical", {})
    for key, array in (("volume", image), ("label", target)):
        descriptor = canonical.get(key, {}) if isinstance(canonical, Mapping) else {}
        if (
            descriptor.get("shape") is not None
            and tuple(descriptor["shape"]) != array.shape
        ):
            raise ValueError(
                f"{case.case_id}: canonical {key} shape disagrees with manifest"
            )
    image = np.ascontiguousarray(np.moveaxis(image, -1, 0), dtype=np.float32)
    # Moving a single channel can preserve a read-only NPY memory map. Copy it so tensors passed to the model have writable
    # backing storage.
    if not image.flags.writeable:
        image = image.copy()
    return image, np.ascontiguousarray(target, dtype=np.uint8)
