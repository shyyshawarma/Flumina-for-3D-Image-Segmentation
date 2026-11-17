from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import cast

import blosc2
import nibabel as nib
import numpy as np
import yaml


DATASET = "BraTS-2025-GLI"
CHANNELS = ("t2f", "t1c", "t1n", "t2w")
LABEL_VALUES = (0, 1, 2, 3, 4)


def load_config(path: str | Path) -> dict:
    """
    Load and validate preprocessing and storage settings.

    Args:
        path (str | Path): YAML configuration file; project_root is relative to its directory.

    Returns:
        dict: Settings with absolute Path values for raw_root, output_root, and split_root.
    """
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    if config["dataset"] != DATASET:
        raise ValueError(f"Expected dataset {DATASET}")
    base = (path.parent / config.get("project_root", "../..")).resolve()
    for key in ("raw_root", "output_root", "split_root"):
        config[key] = (base / config[key]).resolve()
    if config["processing"]["channel_order"] != list(CHANNELS):
        raise ValueError("The frozen channel order is FLAIR, T1ce, T1, T2")
    if config["processing"]["label_values"] != list(LABEL_VALUES):
        raise ValueError("The frozen categorical label vocabulary changed")
    if config["processing"]["minimum_size"] < 1:
        raise ValueError("minimum_size must be positive")
    if config.get("workers", 1) < 1:
        raise ValueError("workers must be positive")
    storage = config["storage"]
    if (
        storage["backend"],
        storage["codec"],
        storage["support_bitorder"],
        storage["foreground_dtype"],
    ) != ("b2nd", "LZ4HC", "little", "uint32"):
        raise ValueError("The frozen B2ND and sidecar storage contract changed")
    if (
        storage["chunk_spatial"] < storage["block_spatial"]
        or storage["block_spatial"] < 1
    ):
        raise ValueError("Storage chunks must contain positive blocks")
    return config


def safe_id(value: str) -> str:
    """
    Validate a case identifier for use as one path component.

    Args:
        value (str): Candidate case identifier.

    Returns:
        str: Validated case identifier, unchanged.
    """
    if not value or value in (".", "..") or any(c in value for c in "/\\\x00"):
        raise ValueError(f"Unsafe case ID: {value!r}")
    return value


def read_inventory(config: dict) -> tuple[list[dict], dict[str, list[str]]]:
    """
    Resolve source scans and validate the frozen split assignments.

    Args:
        config (dict): Resolved raw_root and split_root, source directory mapping, case_table_sha256, and split_sha256
            checksums.

    Returns:
        tuple[list[dict], dict[str, list[str]]]: Case records in CSV order with modality paths, optional segmentation paths, and
            membership, followed by ordered case lists keyed by split.
    """
    root = config["split_root"]
    if (
        hashlib.sha256((root / "cases.csv").read_bytes()).hexdigest()
        != config["case_table_sha256"]
    ):
        raise ValueError("Frozen source case mapping changed")
    lists = {}
    for role, expected in config["split_sha256"].items():
        payload = (root / f"{role}.txt").read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Frozen {role} list changed")
        names = [safe_id(line) for line in payload.decode().splitlines()]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate IDs in {role}")
        lists[role] = names
    primary = [set(lists[role]) for role in ("train", "val", "test")]
    if any(primary[i] & primary[j] for i in range(3) for j in range(i)):
        raise ValueError("Train, validation, and test overlap")
    labeled, unlabeled = set(lists["labeled"]), set(lists["unlabeled"])
    if labeled & unlabeled or set.union(*primary) != labeled | unlabeled:
        raise ValueError("Split membership disagrees with labeled cases")
    with (root / "cases.csv").open(newline="") as handle:
        cases = list(csv.DictReader(handle))
    ids = [safe_id(row["case_id"]) for row in cases]
    if len(ids) != len(set(ids)) or set(ids) != labeled | unlabeled:
        raise ValueError("Frozen case mapping disagrees with the split lists")
    membership = {
        name: role for role in ("train", "val", "test") for name in lists[role]
    }
    patient_roles, component_roles = {}, {}
    for case in cases:
        role = membership[case["case_id"]]
        for key, assignments in (
            ("patient_id", patient_roles),
            ("duplicate_component", component_roles),
        ):
            identifier = case[key]
            if identifier and assignments.setdefault(identifier, role) != role:
                raise ValueError(f"{key} crosses train/validation/test boundaries")
    if (
        set(lists["test_preoperative"]) & set(lists["test_postoperative"])
        or set(lists["test_preoperative"]) | set(lists["test_postoperative"])
        != primary[2]
    ):
        raise ValueError("Pre/postoperative test views must partition test")
    seen_source_files = set()
    for case in cases:
        case_id = case["case_id"]
        original = safe_id(case["raw_case_id"])
        source = config["raw_root"] / config["sources"][case["source_key"]]
        case_dir = source / original
        if not case_dir.is_dir():
            matches = list(source.glob(f"**/{original}"))
            matches = [path for path in matches if path.is_dir()]
            if len(matches) != 1:
                raise FileNotFoundError(f"Expected one source directory for {case_id}")
            case_dir = matches[0]
        files = {token: case_dir / f"{original}-{token}.nii.gz" for token in CHANNELS}
        files["seg"] = (
            case_dir / f"{original}-seg.nii.gz" if case_id in labeled else None
        )
        for path in files.values():
            if path is not None and not path.is_file():
                raise FileNotFoundError(path)
        if files["t2f"] in seen_source_files:
            raise ValueError(f"Repeated source scan for {case_id}")
        seen_source_files.add(files["t2f"])
        case["files"] = files
        case["subset"] = membership.get(case_id, "unlabeled")
        case["status"] = "labeled" if case_id in labeled else "unlabeled"
        case["label_status"] = case["status"]
        case["source_role"] = case["source_key"]
    return cases, lists


def load_canonical(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    """
    Read one NIfTI image in canonical RAS orientation.

    Args:
        path (Path): MRI channel or categorical segmentation NIfTI file.

    Returns:
        tuple[np.ndarray, nib.Nifti1Image]: Float32 data with shape (X, Y, Z), followed by the reoriented image containing its
            affine and voxel spacing.
    """
    image = cast(nib.Nifti1Image, nib.load(path))
    affine = cast(np.ndarray, image.affine)
    if len(image.shape) != 3 or not np.isfinite(affine).all():
        raise ValueError(f"Expected a finite 3-D NIfTI: {path.name}")
    if abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError(f"Singular affine: {path.name}")
    image = nib.as_closest_canonical(image)
    if nib.aff2axcodes(image.affine) != ("R", "A", "S"):
        raise ValueError(f"Canonical orientation is not RAS: {path.name}")
    data = image.get_fdata(dtype=np.float32, caching="unchanged")
    if not np.isfinite(data).all():
        raise ValueError(f"Nonfinite image: {path.name}")
    return data, image


def crop_bounds(volume: np.ndarray, minimum: int) -> tuple[tuple, tuple, dict]:
    """
    Compute mmFormer bounds using foreground maxima as exclusive stops.

    Crop support uses the raw maximum across channels. Short axes are expanded within the source grid, then padded if needed to
    reach minimum.

    Args:
        volume (np.ndarray): Raw float32 MRI with shape (4, X, Y, Z), in FLAIR, T1ce, T1, T2 order.
        minimum (int): Minimum output extent for each spatial axis after padding.

    Returns:
        tuple[tuple[slice, ...], tuple[tuple[int, int], ...], dict]: XYZ crop slices, per-axis (before, after) zero padding, and
            crop/repair geometry.
    """
    coordinates = np.where(np.amax(volume, axis=0) != 0)
    if not coordinates[0].size:
        raise ValueError("MRI crop support is empty")
    bounds, padding, foreground, legacy, repaired, lengths = [], [], [], [], [], []
    for axis, positions in enumerate(coordinates):
        low, high = int(positions.min()), int(positions.max())
        foreground.append([low, high])
        if high - low < minimum:
            extra = int((minimum - (high - low)) / 2)
            high += extra + 1
            low -= extra
        if low < 0:
            high -= low
            low = 0
        legacy.append([low, high])
        size = volume.shape[axis + 1]
        low = max(0, min(low, size))
        high = max(low, min(high, size))
        lengths.append(high - low)
        needs_repair = high - low < minimum
        repaired.append(needs_repair)
        before = after = 0
        if needs_repair:
            missing = max(0, min(minimum, size) - (high - low))
            low -= missing // 2
            high += missing - missing // 2
            if low < 0:
                high = min(size, high - low)
                low = 0
            if high > size:
                low = max(0, low - (high - size))
                high = size
            missing = max(0, minimum - (high - low))
            before, after = missing // 2, missing - missing // 2
        bounds.append(slice(low, high))
        padding.append((before, after))
    geometry = {
        "crop_support": "np.amax(canonical_raw_cxyz, axis=0) != 0",
        "crop_mode": "mmformer",
        "mri_foreground_bounds_inclusive": foreground,
        "mmformer_legacy_bounds_max_as_exclusive": legacy,
        "mmformer_legacy_realized_shape": lengths,
        "crop_bounds": [[s.start, s.stop] for s in bounds],
        "padding": [list(pair) for pair in padding],
        "corrected_axes": repaired,
    }
    return tuple(bounds), tuple(padding), geometry


def normalize(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Z-normalize each modality on shared nonzero MRI support.

    Each channel uses every voxel where any modality is nonzero; values outside that support remain zero. Statistics are
    computed before cropping.

    Args:
        volume (np.ndarray): Uncropped float32 MRI with shape (4, X, Y, Z), in FLAIR, T1ce, T1, T2 order.

    Returns:
        tuple[np.ndarray, np.ndarray, list[dict]]: Float32 normalized MRI with the input shape, bool support with shape
            (X, Y, Z), and per-channel statistics in the input modality order.
    """
    support = np.any(volume != 0, axis=0)
    output = np.zeros_like(volume, dtype=np.float32)
    statistics = []
    for index, channel in enumerate(volume):
        own = channel[channel != 0].astype(np.float64, copy=False)
        if not own.size or own.std() < 1e-6:
            raise ValueError(f"Degenerate MRI channel {CHANNELS[index]}")
        values = channel[support].astype(np.float64, copy=False)
        mean, std = float(values.mean()), float(values.std())
        if not np.isfinite([mean, std]).all() or std < 1e-6:
            raise ValueError("Invalid shared-support normalization statistics")
        output[index, support] = ((values - mean) / std).astype(np.float32)
        statistics.append(
            {
                "channel": CHANNELS[index],
                "raw_mean": mean,
                "raw_std": std,
                "raw_min": float(values.min()),
                "raw_max": float(values.max()),
                "raw_nonzero_voxels": int(own.size),
                "raw_nonzero_std": float(own.std()),
                "normalized_min": float(output[index, support].min()),
                "normalized_max": float(output[index, support].max()),
                "normalized_mean": float(output[index, support].mean(dtype=np.float64)),
                "normalized_std": float(output[index, support].std(dtype=np.float64)),
            }
        )
    return output, support, statistics


def preprocess_case(case: dict, config: dict) -> tuple[dict, dict]:
    """
    Normalize the full scan, then crop and pad the storage arrays.

    Args:
        case (dict): Source case metadata and files mapping each CHANNELS token to a NIfTI Path; files['seg'] is a label Path or
            None for an unlabeled case.
        config (dict): Preprocessing settings with processing.minimum_size for the minimum spatial crop extent.

    Returns:
        tuple[dict, dict]: Named arrays and a case record with geometry and statistics. Arrays contain float32 volume
            (X, Y, Z, 4) in FLAIR, T1ce, T1, T2 order and 1D uint8 support packed in C order with little bit order. Labeled
            cases also include uint8 labels (X, Y, Z) and sorted uint32 C-order foreground indices (F,), where F is the number
            of nonzero labels.
    """
    images, arrays = [], []
    for token in CHANNELS:
        array, image = load_canonical(case["files"][token])
        arrays.append(array)
        images.append(image)
    reference = images[0]
    label = None
    if case["files"]["seg"] is not None:
        label, image = load_canonical(case["files"]["seg"])
        images.append(image)
        if not set(np.unique(label)) <= set(LABEL_VALUES):
            raise ValueError(f"Invalid categorical labels: {case['case_id']}")
        label = label.astype(np.uint8)
    for image in images[1:]:
        if image.shape != reference.shape or not np.allclose(
            image.affine, reference.affine, rtol=1e-5, atol=1e-4
        ):
            raise ValueError(f"Image/label geometry mismatch: {case['case_id']}")
    raw = np.stack(arrays, axis=0)
    crop, padding, geometry = crop_bounds(raw, config["processing"]["minimum_size"])
    normalized, support, statistics = normalize(raw)
    volume = np.ascontiguousarray(
        np.pad(
            np.moveaxis(normalized[(slice(None), *crop)], 0, -1), (*padding, (0, 0))
        ),
        dtype=np.float32,
    )
    support = np.pad(support[crop], padding)
    result = {
        "volume": volume,
        "support": np.packbits(support.reshape(-1), bitorder="little"),
    }
    outside = dropped = 0
    if label is not None:
        outside = int(np.count_nonzero((label != 0) & ~np.any(raw != 0, axis=0)))
        dropped = int(np.count_nonzero(label) - np.count_nonzero(label[crop]))
        result["label"] = np.ascontiguousarray(
            np.pad(label[crop], padding), dtype=np.uint8
        )
        if result["label"].size > np.iinfo(np.uint32).max:
            raise ValueError("Foreground indexing exceeds uint32 capacity")
        result["foreground_index"] = np.flatnonzero(
            result["label"].reshape(-1) != 0
        ).astype(np.uint32)
    affine = reference.affine.copy()
    offset = np.array([s.start - padding[axis][0] for axis, s in enumerate(crop)])
    affine[:3, 3] = reference.affine[:3, :3] @ offset + reference.affine[:3, 3]
    geometry.update(
        {
            "canonical_shape": list(reference.shape),
            "canonical_orientation": ["R", "A", "S"],
            "canonical_spacing": [float(x) for x in reference.header.get_zooms()[:3]],
            "canonical_affine": reference.affine.tolist(),
            "output_affine": affine.tolist(),
            "output_shape": list(volume.shape[:3]),
        }
    )
    record = {key: value for key, value in case.items() if key != "files"}
    record.update(
        {
            "geometry": geometry,
            "normalization": statistics,
            "label_support": {
                "outside_shared_mri_foreground_voxels": outside,
                "dropped_nonzero_label_voxels": dropped,
                "valid_support_voxels": int(np.count_nonzero(support)),
            },
        }
    )
    return result, record


def array_digest(array: np.ndarray) -> str:
    """
    Hash the C-order array payload with SHA-256.

    Args:
        array (np.ndarray): Numeric array of any shape and dtype.

    Returns:
        str: Hexadecimal digest of the raw array bytes, excluding metadata.
    """
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def write_case(case: dict, config: dict) -> dict:
    """
    Write one B2ND case and verify every stored array.

    Args:
        case (dict): Source case record with resolved file paths and case metadata.
        config (dict): Resolved output_root, processing settings, and storage settings for preprocessing, compression, tiling,
            and NumPy sidecars.

    Returns:
        dict: Case record with verified output paths, array hashes, and storage metadata.
    """
    arrays, record = preprocess_case(case, config)
    case_id, output = case["case_id"], config["output_root"]
    layout = {
        "volume": ("vol", "vol", "b2nd"),
        "label": ("seg", "seg", "b2nd"),
        "support": ("support", "support", "npy"),
        "foreground_index": ("sampling", "foreground", "npy"),
    }
    saved = {
        "volume": None,
        "label": None,
        "support": None,
        "foreground_index": None,
        "canonical": {},
        "native": {},
        "transform": {
            "volume_axis": "identity XYZC",
            "label_axis": "identity XYZ",
            "label_values": "identity",
            "support": "identity stored packed uint8 little-bit-order",
            "foreground_index": "numpy.flatnonzero(label.reshape(-1,C-order)!=0) cast to uint32",
        },
    }
    for kind, array in arrays.items():
        folder, suffix, extension = layout[kind]
        path = output / folder / f"{case_id}_{suffix}.{extension}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        meta = {
            "path": path.relative_to(output).as_posix(),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": array_digest(array),
        }
        if extension == "b2nd":
            storage = config["storage"]
            chunks = tuple(min(storage["chunk_spatial"], n) for n in array.shape[:3])
            blocks = tuple(min(storage["block_spatial"], n) for n in array.shape[:3])
            if array.ndim == 4:
                chunks, blocks = (*chunks, array.shape[-1]), (*blocks, array.shape[-1])
            blosc2.asarray(
                array,
                urlpath=str(path),
                mode="w",
                chunks=chunks,
                blocks=blocks,
                cparams={
                    "codec": blosc2.Codec.LZ4HC,
                    "clevel": storage["clevel"],
                    "nthreads": 1,
                },
            )
            store = cast(
                blosc2.NDArray,
                blosc2.open(str(path), mode="r", dparams={"nthreads": 1}),
            )
            decoded = np.asarray(store[:])
            meta.update(
                {
                    "chunks": list(store.chunks),
                    "blocks": list(store.blocks),
                    "codec": "LZ4HC",
                    "clevel": storage["clevel"],
                }
            )
            del store
        else:
            np.save(path, array, allow_pickle=False)
            decoded = np.load(path, allow_pickle=False)
        if (
            decoded.dtype != array.dtype
            or decoded.shape != array.shape
            or array_digest(decoded) != meta["sha256"]
        ):
            raise ValueError(f"Storage roundtrip changed {case_id}: {kind}")
        saved[kind] = meta.pop("path")
        if extension == "b2nd":
            saved["native"][kind] = {
                key: meta[key] for key in ("shape", "dtype", "chunks", "blocks")
            }
        saved["canonical"][kind] = {
            key: meta[key] for key in ("shape", "dtype", "sha256")
        }
        match kind:
            case "label":
                values, counts = np.unique(array, return_counts=True)
                saved["canonical"][kind].update(
                    {
                        "values": values.tolist(),
                        "counts": {
                            str(int(value)): int(count)
                            for value, count in zip(values, counts)
                        },
                    }
                )
            case "support":
                saved["canonical"][kind].update(
                    {
                        "encoding": "numpy uint8 packbits, C-order, little bit order",
                        "valid_support_voxels": record["label_support"][
                            "valid_support_voxels"
                        ],
                    }
                )
            case "foreground_index":
                saved["canonical"][kind]["encoding"] = (
                    "C-order flat uint32 indices where canonical label != 0, strictly increasing"
                )
    record["output"] = saved
    record["storage_backend"] = "b2nd"
    return record


def run(config: dict, selected: list[str] | None = None, dry_run: bool = False) -> dict:
    """
    Build all cases or a selected subset from the frozen inventory.

    Args:
        config (dict): Resolved source, split, processing, and storage settings; workers controls the number of preprocessing
            processes.
        selected (list[str] | None): Case IDs for a partial build; None or an empty list selects all cases.
        dry_run (bool): Validate the source inventory and split lists without preprocessing or writing files.

    Returns:
        dict: Dataset name, selected case count, partial-build flag, and split counts.
    """
    cases, lists = read_inventory(config)
    if selected:
        requested = set(selected)
        if not requested <= {case["case_id"] for case in cases}:
            raise ValueError("Selected case is absent from the frozen inventory")
        cases = [case for case in cases if case["case_id"] in requested]
        lists = {
            role: [name for name in names if name in requested]
            for role, names in lists.items()
        }
    summary = {
        "dataset": DATASET,
        "cases": len(cases),
        "partial": bool(selected),
        "splits": {role: len(names) for role, names in lists.items()},
    }
    if dry_run:
        return summary
    if blosc2.__version__ != config["storage"]["blosc2_version"]:
        raise RuntimeError(
            f"Storage requires blosc2 {config['storage']['blosc2_version']}"
        )
    output, raw = config["output_root"], config["raw_root"]
    if output == raw or output in raw.parents or raw in output.parents:
        raise ValueError("Raw and output directories must be separate")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    workers = config.get("workers", 1)
    records = []
    if workers == 1:
        for index, case in enumerate(cases, 1):
            records.append(write_case(case, config))
            print(f"[{index}/{len(cases)}] {case['case_id']}", flush=True)
    else:
        from itertools import repeat

        with ProcessPoolExecutor(max_workers=workers) as pool:
            for index, record in enumerate(
                pool.map(write_case, cases, repeat(config)), 1
            ):
                records.append(record)
                print(f"[{index}/{len(cases)}] {record['case_id']}", flush=True)
    for role, names in lists.items():
        (output / f"{role}.txt").write_text("".join(f"{name}\n" for name in names))
    manifest = {
        **summary,
        "dataset": {
            "key": "brats2025_gli",
            "name": DATASET,
            "channel_order": ["FLAIR", "T1ce", "T1", "T2"],
            "labels": {
                str(i): label
                for i, label in enumerate(["background", "NETC", "SNFH", "ET", "RC"])
            },
        },
        "volume_axes": "XYZC",
        "label_axes": "XYZ",
        "processing": config["processing"],
        "storage": config["storage"],
        "cases": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "name_mapping.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "processed_name": row["case_id"],
                        "original_name": row["raw_case_id"],
                        "patient_id": row["patient_id"],
                        "treatment_status": row["treatment_status"],
                        "source_role": row["source_role"],
                        "spacing": row["geometry"]["canonical_spacing"],
                        "subset": row["subset"],
                    }
                    for row in records
                ]
            },
            indent=2,
        )
        + "\n"
    )
    return summary


def main() -> None:
    """
    Parse command-line settings and run preprocessing or an inventory check.

    Returns:
        None: Prints the selected case and split counts as JSON.
    """
    parser = argparse.ArgumentParser(
        description=f"Rebuild {DATASET} from official NIfTI"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--case", action="append", help="Build only this case; may be repeated"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate inventory without writing"
    )
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config), args.case, args.dry_run), indent=2))


if __name__ == "__main__":
    main()
