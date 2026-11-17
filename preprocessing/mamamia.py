from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import cast

import blosc2
import nibabel as nib
import numpy as np
import yaml
from numpy.typing import NDArray
from scipy.ndimage import binary_fill_holes, map_coordinates
from skimage.transform import resize


DATASET = "MAMA-MIA"


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
    settings = config["processing"]
    if (
        settings["selected_phase"] != "0001"
        or settings["expert_parent_id"] != "syn64871175"
    ):
        raise ValueError(
            "The benchmark requires the first post-contrast image and expert mask"
        )
    if settings["label_values"] != [0, 1] or settings["order_z"] != 0:
        raise ValueError(
            "The frozen binary label and separate-axis interpolation contract changed"
        )
    spacing = np.asarray(settings["target_spacing_dhw"], dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Target spacing must contain three finite positive values")
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


def file_digest(path: Path, algorithm: str = "md5") -> str:
    """
    Hash a file in fixed-size byte chunks.

    Args:
        path (Path): File whose contents are hashed.
        algorithm (str): Hashlib algorithm name; defaults to md5 for source verification.

    Returns:
        str: Hexadecimal digest of the file contents.
    """
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_inventory(config: dict) -> tuple[list[dict], dict[str, list[str]]]:
    """
    Select each first post-contrast image and the expert mask listed in the manifest.

    Args:
        config (dict): Resolved raw_root and split_root, source_manifest filename, split_sha256 checksums, and processing
            phase/expert-parent settings.

    Returns:
        tuple[list[dict], dict[str, list[str]]]: Cases in frozen labeled order with image/label paths, source MD5 values, and
            membership, followed by ordered case lists keyed by split.
    """
    lists = {}
    for role, digest in config["split_sha256"].items():
        payload = (config["split_root"] / f"{role}.txt").read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"Frozen {role} list changed")
        names = [safe_id(line) for line in payload.decode().splitlines()]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate case in {role}")
        lists[role] = names
    primary = [set(lists[role]) for role in ("train", "val", "test")]
    if any(primary[i] & primary[j] for i in range(3) for j in range(i)):
        raise ValueError("Train, validation, and test overlap")
    if lists["unlabeled"] or set.union(*primary) != set(lists["labeled"]):
        raise ValueError("Benchmark splits must partition all labeled cases")
    membership = {
        name: role for role in ("train", "val", "test") for name in lists[role]
    }
    rows_by_name = {}
    manifest = config["raw_root"] / config["source_manifest"]
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"name", "parentId", "path", "dataFileSizeBytes", "dataFileMD5Hex"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError("The source manifest lacks mask-selection fields")
        for row in reader:
            rows_by_name.setdefault(row["name"], []).append(row)
    cases = []
    used_paths = set()
    for case_id in lists["labeled"]:
        image_name = f"{case_id}_{config['processing']['selected_phase']}.nii.gz"
        image_rows = rows_by_name.get(image_name, [])
        expert_rows = [
            row
            for row in rows_by_name.get(f"{case_id}.nii.gz", [])
            if row["parentId"] == config["processing"]["expert_parent_id"]
        ]
        if len(image_rows) != 1 or len(expert_rows) != 1:
            raise ValueError(
                f"Expected one first post-contrast image and one expert mask for {case_id}"
            )
        files, checksums = {}, {}
        for kind, row in (("image", image_rows[0]), ("label", expert_rows[0])):
            name = safe_id(Path(row["path"]).name)
            path = config["raw_root"] / name
            if not path.is_file() or path.stat().st_size != int(
                row["dataFileSizeBytes"]
            ):
                raise ValueError(f"Missing or incomplete selected source file: {name}")
            if row.get("error"):
                raise ValueError(f"Source manifest records a download error for {name}")
            if path in used_paths:
                raise ValueError(f"Source file reused across cases: {name}")
            used_paths.add(path)
            files[kind] = path
            checksums[kind] = row["dataFileMD5Hex"]
        cases.append(
            {
                "case_id": case_id,
                "raw_case_id": case_id,
                "patient_id": case_id,
                "cohort": case_id.split("_", 1)[0],
                "subset": membership[case_id],
                "status": "labeled",
                "label_status": "labeled",
                "files": files,
                "source_md5": checksums,
            }
        )
    return cases, lists


def load_nifti(path: Path) -> nib.Nifti1Image:
    """
    Load a NIfTI-1 image through its file map.

    Args:
        path (Path): Source NIfTI-1 image or expert-mask file.

    Returns:
        nib.Nifti1Image: Loaded image with its source header and affine.
    """
    file_map = nib.Nifti1Image.make_file_map()
    file_map["image"].filename = str(path)
    return nib.Nifti1Image.from_file_map(file_map)


def header_diagnostic(image: nib.Nifti1Image) -> dict:
    """
    Compare the qform and sform stored in a NIfTI header.

    Args:
        image (nib.Nifti1Image): Source image with header transforms to inspect.

    Returns:
        dict[str, int | float | None]: Transform codes and their maximum absolute matrix difference, or None for the difference
            if either transform is absent.
    """
    qform, qcode = image.header.get_qform(coded=True)
    sform, scode = image.header.get_sform(coded=True)
    difference = (
        None if qform is None or sform is None else float(np.max(np.abs(qform - sform)))
    )
    return {
        "qform_code": int(qcode),
        "sform_code": int(scode),
        "qform_sform_max_abs_difference": difference,
    }


def load_case(case: dict) -> tuple[np.ndarray, np.ndarray, tuple, dict]:
    """
    Validate and reorient a selected image and expert mask.

    Args:
        case (dict): Case identifier and files mapping image and label to NIfTI Paths.

    Returns:
        tuple[np.ndarray, np.ndarray, tuple[float, float, float], dict]: Float32 image and int8 binary labels with shape
            (1, D, H, W), DHW spacing in millimeters, and source/reoriented geometry with header diagnostics.
    """
    image_original, label_original = (
        load_nifti(case["files"][key]) for key in ("image", "label")
    )
    for image in (image_original, label_original):
        affine = cast(np.ndarray, image.affine)
        if (
            len(image.shape) != 3
            or not np.isfinite(affine).all()
            or abs(np.linalg.det(affine[:3, :3])) <= 1e-12
        ):
            raise ValueError(f"Invalid 3-D source geometry: {case['case_id']}")
    spacing = tuple(float(x) for x in image_original.header.get_zooms()[:3])
    original_affine = cast(np.ndarray, image_original.affine)
    expert_affine = cast(np.ndarray, label_original.affine)
    if image_original.shape != label_original.shape or not np.allclose(
        original_affine, expert_affine, rtol=0, atol=1e-5
    ):
        raise ValueError(f"Image/expert geometry mismatch: {case['case_id']}")
    if not np.allclose(
        spacing, label_original.header.get_zooms()[:3], rtol=0, atol=1e-6
    ):
        raise ValueError(f"Image/expert spacing mismatch: {case['case_id']}")
    if not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError("Invalid source spacing")
    image = image_original.as_reoriented(
        cast(Sequence[Sequence[int]], nib.orientations.io_orientation(original_affine))
    )
    label = label_original.as_reoriented(
        cast(Sequence[Sequence[int]], nib.orientations.io_orientation(expert_affine))
    )
    image_affine = cast(np.ndarray, image.affine)
    label_affine = cast(np.ndarray, label.affine)
    if image.shape != label.shape or not np.allclose(
        image_affine, label_affine, rtol=0, atol=1e-5
    ):
        raise ValueError(
            f"Reoriented image/expert geometry mismatch: {case['case_id']}"
        )
    image_xyz = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    label_xyz = np.asarray(label.get_fdata(dtype=np.float32), dtype=np.float32)
    if not np.isfinite(image_xyz).all() or not np.isfinite(label_xyz).all():
        raise ValueError("Nonfinite NIfTI payload")
    if list(np.unique(label_xyz)) != [0, 1]:
        raise ValueError("Expert masks must contain exactly background 0 and tumor 1")
    canonical_spacing = tuple(float(x) for x in image.header.get_zooms()[:3])
    if not np.allclose(
        canonical_spacing, label.header.get_zooms()[:3], rtol=0, atol=1e-6
    ):
        raise ValueError("Reoriented image/expert spacing mismatch")
    data = np.ascontiguousarray(image_xyz.transpose(2, 1, 0)[None], dtype=np.float32)
    segmentation = np.ascontiguousarray(
        label_xyz.astype(np.int8).transpose(2, 1, 0)[None]
    )
    geometry = {
        "source_shape_xyz": list(image_original.shape),
        "source_spacing_xyz": list(spacing),
        "source_selected_affine": original_affine.tolist(),
        "image_expert_selected_affine_max_abs_difference": float(
            np.max(np.abs(original_affine - expert_affine))
        ),
        "reoriented_shape_xyz": list(image.shape),
        "reoriented_spacing_xyz": list(canonical_spacing),
        "reoriented_affine": image_affine.tolist(),
        "image_header": header_diagnostic(image_original),
        "expert_header": header_diagnostic(label_original),
    }
    return data, segmentation, tuple(reversed(canonical_spacing)), geometry


def resize_label(label: np.ndarray, shape: tuple, order: int) -> np.ndarray:
    """
    Resize categorical labels using nearest neighbors or category masks.

    Nonzero orders threshold each mask at >= 0.5; larger categories win overlaps.

    Args:
        label (np.ndarray): Signed labels in a (D, H, W) volume or a 2D plane; synthetic label -1 marks ignored background.
        shape (tuple[int, ...]): Target dimensions in the input axis order.
        order (int): Interpolation order; zero samples labels directly, while other orders interpolate one mask per category.

    Returns:
        np.ndarray: Resized labels with the requested shape and input dtype.
    """
    if order == 0:
        return cast(
            np.ndarray,
            resize(
                label.astype(float),
                shape,
                order,
                mode="edge",
                clip=True,
                anti_aliasing=False,
            ),
        ).astype(label.dtype)
    output = np.zeros(shape, dtype=label.dtype)
    for value in np.unique(label):
        mask = resize(
            (label == value).astype(float),
            shape,
            order,
            mode="edge",
            clip=True,
            anti_aliasing=False,
        )
        output[mask >= 0.5] = value
    return output


def separate_axis(spacing: tuple, target: tuple, threshold: float) -> int | None:
    """
    Choose a unique coarse axis for separate resampling.

    Args:
        spacing (tuple[float, float, float]): Source spacing in millimeters, in DHW order.
        target (tuple[float, float, float]): Target spacing in millimeters, in DHW order.
        threshold (float): Maximum-to-minimum spacing ratio above which a grid is considered anisotropic.

    Returns:
        int | None: DHW axis index with the unique maximum spacing in the first anisotropic grid, checking source before target;
            None if none qualifies or that grid has tied maxima.
    """
    for grid in (spacing, target):
        if max(grid) / min(grid) > threshold:
            axes = np.where(max(grid) / np.asarray(grid) == 1)[0]
            return int(axes[0]) if len(axes) == 1 else None
    return None


def resample(
    data: np.ndarray, shape: tuple, axis: int | None, order: int, is_seg: bool
) -> np.ndarray:
    """
    Resize in float64 and optionally sample a coarse axis separately.

    Args:
        data (np.ndarray): Float32 image or int8 labels with shape (C, D, H, W).
        shape (tuple[int, int, int]): Target spatial shape in DHW order.
        axis (int | None): Spatial axis for nearest-neighbor sampling after in-plane resizing; None resizes all three dimensions
            together.
        order (int): Interpolation order for full-volume or in-plane resizing.
        is_seg (bool): Use categorical label interpolation instead of image interpolation.

    Returns:
        np.ndarray: Resampled array with shape (C, *shape) and the input dtype.
    """
    if tuple(data.shape[1:]) == tuple(shape):
        return data
    output = np.zeros((data.shape[0], *shape), dtype=data.dtype)
    data = data.astype(float, copy=False)
    for channel in range(data.shape[0]):
        if axis is None:
            output[channel] = (
                resize_label(data[channel], shape, order)
                if is_seg
                else resize(
                    data[channel], shape, order, mode="edge", anti_aliasing=False
                )
            )
            continue
        shape_2d = tuple(size for index, size in enumerate(shape) if index != axis)
        intermediate_shape = list(shape)
        intermediate_shape[axis] = data.shape[axis + 1]
        intermediate = np.zeros(intermediate_shape)
        for index in range(data.shape[axis + 1]):
            slicer = tuple(
                index if dimension == axis else slice(None) for dimension in range(3)
            )
            plane = data[channel][slicer]
            intermediate[slicer] = (
                resize_label(plane, shape_2d, order)
                if is_seg
                else resize(plane, shape_2d, order, mode="edge", anti_aliasing=False)
            )
        if data.shape[axis + 1] != shape[axis]:
            coordinates = np.mgrid[: shape[0], : shape[1], : shape[2]].astype(float)
            for dimension in range(3):
                coordinates[dimension] = (
                    intermediate.shape[dimension]
                    / shape[dimension]
                    * (coordinates[dimension] + 0.5)
                    - 0.5
                )
            output[channel] = map_coordinates(
                intermediate, coordinates, order=0, mode="nearest"
            )
        else:
            output[channel] = intermediate
    return output


def preprocess_case(case: dict, config: dict) -> tuple[dict, dict]:
    """
    Crop, Z-normalize, and resample the selected image and expert mask.

    Normalization uses every voxel in the cropped image, including background.

    Args:
        case (dict): Source record with image/label Paths, source_md5 values, and case metadata from read_inventory.
        config (dict): Processing settings for spacing, interpolation, and normalization, with optional verify_source_md5 to
            check source contents.

    Returns:
        tuple[dict, dict]: Named arrays and a case record with geometry and raw crop statistics. Arrays contain float32 volume
            (D, H, W, 1), uint8 labels (D, H, W), 1D uint8 support packed in C order with little bit order, and sorted uint32
            C-order foreground indices (F,). Negative resampled labels become background and are excluded from support; F counts
            nonzero labels.
    """
    if config.get("verify_source_md5", True):
        for kind, path in case["files"].items():
            if file_digest(path) != case["source_md5"][kind]:
                raise ValueError(f"Source checksum mismatch: {case['case_id']} {kind}")
    data, signed_seg, spacing, geometry = load_case(case)
    shape_before = list(data.shape[1:])
    support = cast(NDArray[np.bool_], binary_fill_holes(np.any(data != 0, axis=0)))
    coordinates = np.where(support)
    if not coordinates[0].size:
        raise ValueError("Empty image nonzero support")
    bbox = [[int(values.min()), int(values.max()) + 1] for values in coordinates]
    crop = tuple(slice(low, high) for low, high in bbox)
    positive_before = int(np.count_nonzero(signed_seg == 1))
    data = np.ascontiguousarray(data[(slice(None), *crop)], dtype=np.float32)
    signed_seg = np.ascontiguousarray(signed_seg[(slice(None), *crop)], dtype=np.int8)
    signed_seg[(signed_seg == 0) & ~support[crop][None]] = -1
    if not positive_before or np.count_nonzero(signed_seg == 1) != positive_before:
        raise ValueError("Nonzero crop discarded expert tumor labels")
    settings = config["processing"]
    target = tuple(settings["target_spacing_dhw"])
    shape = tuple(
        int(round(old / new * size))
        for old, new, size in zip(spacing, target, data.shape[1:])
    )
    if min(shape) <= 0:
        raise ValueError("Resampling produced a nonpositive extent")
    mean, std = float(data[0].mean()), float(data[0].std())
    if not np.isfinite([mean, std]).all() or std <= 0:
        raise ValueError("The complete cropped image cannot be Z-normalized")
    statistics = {
        "method": "ZScoreNormalization",
        "use_mask_for_norm": False,
        "mean": mean,
        "std": std,
        "min": float(data.min()),
        "max": float(data.max()),
    }
    data[0] -= mean
    data[0] /= max(std, settings["normalization_epsilon"])
    axis = separate_axis(spacing, target, settings["anisotropy_threshold"])
    shape_cropped = list(data.shape[1:])
    data = resample(data, shape, axis, settings["image_order"], False)
    signed_seg = resample(signed_seg, shape, axis, settings["segmentation_order"], True)
    if not np.isfinite(data).all() or not set(np.unique(signed_seg)) <= {-1, 0, 1}:
        raise ValueError("Resampling changed image finiteness or label vocabulary")
    support = np.ascontiguousarray(signed_seg[0] >= 0)
    label = np.ascontiguousarray(np.where(support, signed_seg[0], 0), dtype=np.uint8)
    if list(np.unique(label)) != [0, 1] or label.size > np.iinfo(np.uint32).max:
        raise ValueError("Final binary labels or uint32 sampling capacity are invalid")
    arrays = {
        "volume": np.ascontiguousarray(np.moveaxis(data, 0, -1), dtype=np.float32),
        "label": label,
        "support": np.packbits(support.reshape(-1), bitorder="little"),
        "foreground_index": np.flatnonzero(label.reshape(-1) != 0).astype(np.uint32),
    }
    geometry.update(
        {
            "internal_spacing_dhw": list(spacing),
            "shape_before_cropping_dhw": shape_before,
            "bbox_used_for_cropping": bbox,
            "shape_after_cropping_before_resampling_dhw": shape_cropped,
            "resampled_shape_dhw": list(shape),
            "target_spacing_dhw_mm": list(target),
            "canonical_spacing": list(target),
            "output_shape": list(shape),
            "separate_z": axis is not None,
            "lowres_axis_dhw": axis,
        }
    )
    record = {
        key: value for key, value in case.items() if key not in ("files", "source_md5")
    }
    record.update(
        {
            "geometry": geometry,
            "normalization": statistics,
            "label_support": {
                "dropped_nonzero_label_voxels": 0,
                "valid_support_voxels": int(np.count_nonzero(support)),
            },
        }
    )
    return arrays, record


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
            "volume_axis": "identity DHWC",
            "label_axis": "identity DHW",
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
            "key": "mama_mia",
            "name": DATASET,
            "channel_order": ["first_post_contrast_T1"],
            "labels": {"0": "background", "1": "tumor"},
        },
        "volume_axes": "DHWC",
        "label_axes": "DHW",
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
