from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import blosc2
import nibabel as nib
import numpy as np
import yaml
from numpy.typing import NDArray
from scipy.ndimage import binary_fill_holes, map_coordinates
from skimage.transform import resize

DATASET = "KiTS-2023"
CONFIG_NAME = "kits2023"
CASE_PATTERN = "case_[0-9]{5}"
NUM_CLASSES = 4
EXPECTED_COUNTS = {"train": 342, "val": 49, "test": 98, "labeled": 489, "unlabeled": 0}
TARGET_SPACING = (1.0, 0.78125, 0.78125)
READER = "nibabel_reorient"
NORMALIZATION = "CTNormalization"
INTENSITY = {
    "mean": 103.13636779785156,
    "std": 73.34317779541016,
    "percentile_00_5": -58.0,
    "percentile_99_5": 302.0,
}
CHUNKS = (16, 64, 64)
BLOCKS = (16, 16, 16)
NPY_VERSION = (2, 0)
LABEL_NAMES = ("background", "kidney", "tumor", "cyst")


def load_config(path: Path) -> dict[str, Any]:
    """
    Load dataset settings and resolve paths against project_root.

    Args:
        path (Path): YAML configuration file; project_root is relative to its directory.

    Returns:
        dict[str, Any]: Validated settings with absolute Path values for project_root, raw_root, output_root, split_root, and
            reference_manifest.
    """
    path = path.expanduser().resolve(strict=True)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("dataset") != DATASET:
        raise ValueError(f"Configuration must select {DATASET}")
    config = dict(value)
    root = (path.parent / config.get("project_root", "../..")).resolve()
    config["project_root"] = root
    for key in ("raw_root", "output_root", "split_root", "reference_manifest"):
        config[key] = (root / Path(config[key]).expanduser()).resolve()
    expected = {
        "target_spacing": list(TARGET_SPACING),
        "reader": READER,
        "normalization": NORMALIZATION,
        "use_mask_for_norm": False,
        "image_order": 3,
        "label_order": 1,
        "separate_axis_order": 0,
        "anisotropy_threshold": 3,
        "chunks": list(CHUNKS),
        "blocks": list(BLOCKS),
        "compression": "LZ4HC",
        "compression_level": 8,
    }
    for key, setting in expected.items():
        if config.get(key) != setting:
            raise ValueError(f"{key} must retain the published setting {setting!r}")
    if config.get("intensity_properties") != INTENSITY:
        raise ValueError("intensity_properties must retain the published values")
    for key in ("image_pattern", "label_pattern"):
        pattern = config[key]
        if not isinstance(pattern, str) or not any(
            field in pattern for field in ("{case_id}", "{case_number}")
        ):
            raise ValueError(f"{key} must contain {{case_id}} or {{case_number}}")
        candidate = Path(pattern.format(case_id="example", case_number=0))
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"{key} must be a safe relative pattern")
    return config


def read_splits(config: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """
    Validate frozen split membership and list order.

    Args:
        config (dict[str, Any]): Settings containing split_root and reference_manifest as resolved Path objects.

    Returns:
        tuple[dict[str, list[str]], dict[str, Any]]: Ordered case lists keyed by split, followed by the reference manifest with
            case records and list hashes.
    """
    reference = json.loads(config["reference_manifest"].read_text(encoding="utf-8"))
    if reference.get("dataset") != DATASET:
        raise ValueError("Reference manifest belongs to a different dataset")
    splits = {}
    for role, count in EXPECTED_COUNTS.items():
        payload = (config["split_root"] / f"{role}.txt").read_bytes()
        names = payload.decode("utf-8").splitlines()
        declaration = reference["lists"][f"{role}.txt"]
        if len(names) != count or len(set(names)) != len(names):
            raise ValueError(f"Invalid {role} split count or duplicate case")
        if any((re.fullmatch(CASE_PATTERN, name) is None for name in names)):
            raise ValueError(f"Unsafe or noncanonical case in {role}")
        if hashlib.sha256(payload).hexdigest() != declaration["sha256"]:
            raise ValueError(f"Frozen {role} membership or order changed")
        splits[role] = names
    grouped = [name for role in ("train", "val", "test") for name in splits[role]]
    if len(set(grouped)) != len(grouped) or set(grouped) != set(splits["labeled"]):
        raise ValueError("Splits overlap or do not cover all labeled cases")
    cases = {item["case_id"]: item for item in reference["cases"]}
    if len(cases) != len(reference["cases"]) or set(cases) != set(grouped):
        raise ValueError("Reference case inventory differs from the frozen lists")
    for relative, declaration in reference["lists"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Reference list paths must be safe and relative")
        payload = (config["split_root"] / path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != declaration["sha256"]:
            raise ValueError(f"Frozen list changed: {relative}")
    return splits, reference


def resolve_case_paths(config: dict[str, Any], case_id: str) -> tuple[Path, Path]:
    """
    Find the unique raw image and label pair for a case.

    Args:
        config (dict[str, Any]): Raw root and image/label patterns, with optional case_paths mapping case IDs to relative image
            and label overrides.
        case_id (str): Case identifier from the frozen inventory.

    Returns:
        tuple[Path, Path]: Existing image and label paths, in that order.
    """
    root = config["raw_root"]
    override = config.get("case_paths", {}).get(case_id)
    if override is not None:
        patterns = [(override["image"], override["label"])]
    else:
        patterns = [(config["image_pattern"], config["label_pattern"])]
        patterns.extend(
            [
                ("{case_id}/imaging.nii.gz", "{case_id}/segmentation.nii.gz"),
                (
                    "dataset/{case_id}/imaging.nii.gz",
                    "dataset/{case_id}/segmentation.nii.gz",
                ),
            ]
        )
    pairs = []
    for image_pattern, label_pattern in patterns:
        relative = [
            Path(
                pattern.format(
                    case_id=case_id, case_number=int(case_id.rsplit("_", 1)[1])
                )
            )
            for pattern in (image_pattern, label_pattern)
        ]
        if any(path.is_absolute() or ".." in path.parts for path in relative):
            raise ValueError(f"{case_id}: raw paths must be safe and relative")
        pair = tuple(root / path for path in relative)
        if all(path.is_file() for path in pair) and pair not in pairs:
            pairs.append(pair)
    match pairs:
        case []:
            raise FileNotFoundError(f"{case_id}: required raw image or label is absent")
        case [pair]:
            return pair
        case _:
            raise ValueError(
                f"{case_id}: multiple raw pairs match; select one through case_paths"
            )


def read_case(
    image_path: Path, label_path: Path
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Read and reorient a matching NIfTI image and label pair.

    Args:
        image_path (Path): Raw image NIfTI file.
        label_path (Path): Categorical label NIfTI file with matching geometry.

    Returns:
        tuple[np.ndarray, np.ndarray, dict[str, Any]]: Float32 image and labels with shape (D, H, W), followed by DHW spacing
            and original/reoriented affines.
    """
    image = cast(nib.Nifti1Image, nib.load(str(image_path)))
    label = cast(nib.Nifti1Image, nib.load(str(label_path)))
    if len(image.shape) != 3 or image.shape != label.shape:
        raise ValueError("Raw image and label must have equal three-dimensional shapes")
    affine_tolerance = 1e-05
    image_affine = cast(np.ndarray, image.affine)
    label_affine = cast(np.ndarray, label.affine)
    if (
        not np.isfinite(image_affine).all()
        or not np.isfinite(label_affine).all()
        or (not np.allclose(image_affine, label_affine, rtol=0, atol=affine_tolerance))
        or (
            not np.allclose(
                image.header.get_zooms(), label.header.get_zooms(), rtol=0, atol=1e-06
            )
        )
        or (
            not np.array_equal(
                nib.orientations.io_orientation(image_affine),
                nib.orientations.io_orientation(label_affine),
            )
        )
    ):
        raise ValueError("Raw image and label geometry differ")
    canonical = image.as_reoriented(
        cast(Sequence[Sequence[int]], nib.orientations.io_orientation(image_affine))
    )
    canonical_label = label.as_reoriented(
        cast(Sequence[Sequence[int]], nib.orientations.io_orientation(label_affine))
    )
    data = np.asarray(canonical.get_fdata().transpose(2, 1, 0), dtype=np.float32)
    segmentation = np.asarray(
        canonical_label.get_fdata().transpose(2, 1, 0), dtype=np.float32
    )
    spacing = [float(item) for item in canonical.header.get_zooms()[::-1]]
    geometry = {
        "spacing": spacing,
        "original_affine": image_affine.tolist(),
        "reoriented_affine": cast(np.ndarray, canonical.affine).tolist(),
    }
    if not np.isfinite(data).all() or not np.isfinite(segmentation).all():
        raise ValueError("Raw arrays contain nonfinite values")
    if not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError("Raw spacing must contain three positive finite values")
    if (
        segmentation.min() < 0
        or segmentation.max() >= NUM_CLASSES
        or (not np.equal(segmentation, np.rint(segmentation)).all())
    ):
        raise ValueError(f"Raw labels must be integers in 0..{NUM_CLASSES - 1}")
    return (data, segmentation, geometry)


def resize_labels(labels: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """
    Resize category masks with linear interpolation.

    Masks use a >= 0.5 threshold; larger categories win ties.

    Args:
        labels (np.ndarray): Signed labels in a (D, H, W) volume or a 2D plane; synthetic label -1 marks ignored background.
        shape (tuple[int, ...]): Target dimensions in the input axis order.

    Returns:
        np.ndarray: Resized labels with the requested shape and input dtype.
    """
    output = np.zeros(shape, dtype=labels.dtype)
    for category in np.unique(labels):
        weight = resize(
            (labels == category).astype(float),
            shape,
            order=1,
            mode="edge",
            clip=True,
            anti_aliasing=False,
        )
        output[weight >= 0.5] = category
    return output


def resample(
    array: np.ndarray, shape: tuple[int, int, int], spacing: list[float], is_label: bool
) -> np.ndarray:
    """
    Resample images and labels with nnU-Net interpolation.

    Source and target grids are checked in that order. A separate axis is used only when the first grid with a spacing ratio
    above three has one coarse axis; that axis uses nearest-neighbor sampling after in-plane resizing.

    Args:
        array (np.ndarray): Float32 image or signed labels with shape (D, H, W).
        shape (tuple[int, int, int]): Target shape in (D, H, W) order.
        spacing (list[float]): Source voxel spacing in millimeters, in (D, H, W) order.
        is_label (bool): Use linear category-mask interpolation for labels; otherwise use cubic image interpolation.

    Returns:
        np.ndarray: Resampled DHW array with the requested shape and input dtype.
    """
    if tuple(array.shape) == tuple(shape):
        return array
    axis = None
    for candidate_spacing in (spacing, TARGET_SPACING):
        values = np.asarray(candidate_spacing)
        if values.max() / values.min() > 3:
            candidates = np.flatnonzero(values.max() / values == 1)
            if len(candidates) == 1:
                axis = int(candidates[0])
            break
    source = array.astype(float, copy=False)

    def interpolate(values: np.ndarray, destination: tuple[int, ...]) -> np.ndarray:
        """
        Apply the selected image or label interpolation.

        Args:
            values (np.ndarray): Float64 image or labels in a 3D volume or 2D plane.
            destination (tuple[int, ...]): Target dimensions in the input axis order.

        Returns:
            np.ndarray: Float64 resized values with the requested shape.
        """
        if is_label:
            return resize_labels(values, destination)
        return cast(
            np.ndarray,
            resize(values, destination, order=3, mode="edge", anti_aliasing=False),
        )

    if axis is None:
        return interpolate(source, shape).astype(array.dtype)
    plane_shape = tuple((shape[index] for index in range(3) if index != axis))
    intermediate_shape = list(shape)
    intermediate_shape[axis] = array.shape[axis]
    intermediate = np.zeros(intermediate_shape, dtype=np.float64)
    for index in range(array.shape[axis]):
        selection_list = cast(list[int | slice], [slice(None)] * 3)
        selection_list[axis] = index
        selection = tuple(selection_list)
        intermediate[selection] = interpolate(source[selection], plane_shape)
    if array.shape[axis] == shape[axis]:
        return intermediate.astype(array.dtype)
    output = np.empty(shape, dtype=array.dtype)
    for start in range(0, shape[0], 16):
        stop = min(start + 16, shape[0])
        coordinates = np.mgrid[start:stop, : shape[1], : shape[2]].astype(float)
        for dimension in range(3):
            coordinates[dimension] = (
                float(intermediate.shape[dimension])
                / shape[dimension]
                * (coordinates[dimension] + 0.5)
                - 0.5
            )
        output[start:stop] = map_coordinates(
            intermediate, coordinates, order=0, mode="nearest"
        )
    return output


def preprocess(
    data: np.ndarray, segmentation: np.ndarray, geometry: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Crop, normalize, and resample one CT case.

    Clip the complete cropped image and normalize it with the fixed CT statistics.

    Args:
        data (np.ndarray): Raw float32 image with shape (D, H, W).
        segmentation (np.ndarray): Float32 categorical labels with the same shape.
        geometry (dict[str, Any]): Reader metadata containing source spacing in DHW order and millimeters.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]: Float32 image with shape (D', H', W', 1), uint8 labels with
            shape (D', H', W'), 1D uint8 support packed in C order with little bit order, and updated geometry. Negative
            resampled labels become background and are excluded from support.
    """
    data = data.astype(np.float32, copy=True)
    segmentation = segmentation.copy()
    support = cast(NDArray[np.bool_], binary_fill_holes(data != 0))
    bbox = []
    for axis in range(3):
        indices = np.flatnonzero(
            np.any(support, axis=tuple((i for i in range(3) if i != axis)))
        )
        bbox.append(
            [int(indices[0]), int(indices[-1]) + 1]
            if len(indices)
            else [0, data.shape[axis]]
        )
    crop = tuple((slice(start, stop) for start, stop in bbox))
    properties = dict(geometry)
    properties["shape_before_cropping"] = list(data.shape)
    data = data[crop]
    segmentation = segmentation[crop]
    segmentation[(segmentation == 0) & ~support[crop]] = -1
    properties["bbox_used_for_cropping"] = bbox
    properties["shape_after_cropping_and_before_resampling"] = list(data.shape)
    shape = cast(
        tuple[int, int, int],
        tuple(
            (
                int(round(old / new * size))
                for old, new, size in zip(
                    geometry["spacing"], TARGET_SPACING, data.shape
                )
            )
        ),
    )
    if min(shape) < 1:
        raise ValueError("Target spacing would produce an empty axis")
    np.clip(data, INTENSITY["percentile_00_5"], INTENSITY["percentile_99_5"], out=data)
    data -= INTENSITY["mean"]
    data /= max(INTENSITY["std"], 1e-08)
    volume = np.ascontiguousarray(
        resample(data, shape, geometry["spacing"], False)[..., None]
    )
    native_label = resample(segmentation, shape, geometry["spacing"], True).astype(
        np.int16
    )
    label = np.ascontiguousarray(np.maximum(native_label, 0), dtype=np.uint8)
    packed = np.packbits((native_label >= 0).reshape(-1), bitorder="little")
    properties.update(
        requested_spacing_rounded=[1.0, 0.78, 0.78],
        canonical_spacing=list(TARGET_SPACING),
        output_shape=list(shape),
        array_spatial_order="DHW (nnU-Net ZYX order retained)",
    )
    return (volume, label, packed, properties)


def payload_hash(array: np.ndarray) -> str:
    """
    Hash the C-order array payload with SHA-256.

    Args:
        array (np.ndarray): Numeric array of any shape and dtype.

    Returns:
        str: Hexadecimal digest of the raw array bytes, excluding metadata.
    """
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def array_statistics(volume: np.ndarray) -> dict[str, float]:
    """
    Accumulate whole-volume statistics in depth slabs.

    Args:
        volume (np.ndarray): Float32 image with shape (D, H, W, 1).

    Returns:
        dict[str, float]: Mean, population standard deviation, minimum, and maximum; means and variances are accumulated in
            float64.
    """
    count, mean, m2 = (0, 0.0, 0.0)
    for start in range(0, len(volume), 16):
        slab_slice = slice(start, start + 16)
        slab = volume[slab_slice]
        slab_mean, variance = (
            float(np.mean(slab, dtype=np.float64)),
            float(np.var(slab, dtype=np.float64)),
        )
        total, delta = (count + slab.size, slab_mean - mean)
        m2 += variance * slab.size + delta * delta * count * slab.size / total
        mean += delta * slab.size / total
        count = total
    return {
        "mean": mean,
        "std": math.sqrt(m2 / count),
        "min": float(volume.min()),
        "max": float(volume.max()),
    }


def case_record(
    case_id: str,
    volume: np.ndarray,
    label: np.ndarray,
    support: np.ndarray,
    geometry: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    """
    Record output geometry, class counts, and array hashes.

    Args:
        case_id (str): Case identifier from the frozen inventory.
        volume (np.ndarray): Processed float32 image with shape (D, H, W, 1).
        label (np.ndarray): Processed uint8 labels with shape (D, H, W).
        support (np.ndarray): 1D uint8 valid-label mask packed in C order with little bit order.
        geometry (dict[str, Any]): Source geometry, crop bounds, and resampling metadata.

    Returns:
        tuple[dict[str, Any], np.ndarray]: Serializable case record and sorted uint32 foreground indices with shape (F,), where
            F is the number of nonzero labels and indices refer to the C-order flattened label array.
    """
    if label.size > np.iinfo(np.uint32).max:
        raise ValueError(
            "Published foreground-index format requires at most uint32 voxels"
        )
    foreground = np.flatnonzero(label.reshape(-1) != 0).astype(np.uint32)
    arrays = {
        "volume": volume,
        "label": label,
        "support": support,
        "foreground_index": foreground,
    }
    canonical = {
        key: {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": payload_hash(array),
        }
        for key, array in arrays.items()
    }
    counts = np.bincount(label.reshape(-1), minlength=NUM_CLASSES)
    canonical["label"].update(
        values=np.flatnonzero(counts).tolist(),
        counts={str(i): int(value) for i, value in enumerate(counts) if value},
    )
    valid = int(
        np.unpackbits(support, bitorder="little", count=label.size).sum(dtype=np.int64)
    )
    canonical["support"].update(
        encoding="numpy uint8 packbits, C-order, little bit order",
        valid_support_voxels=valid,
    )
    canonical["foreground_index"]["encoding"] = (
        "C-order flat uint32 indices where canonical label != 0, strictly increasing"
    )
    record = {
        "case_id": case_id,
        "status": "labeled",
        "label_status": "labeled",
        "storage_backend": "b2nd",
        "modality": "CT",
        "geometry": geometry,
        "normalization": {
            "method": NORMALIZATION,
            "use_mask_for_norm": False,
            **array_statistics(volume),
        },
        "label_support": {
            "nnunet_ignore_voxels_mapped_to_background": label.size - valid,
            "valid_support_voxels": valid,
            "valid_support_fraction": valid / label.size,
            "dropped_nonzero_label_voxels": 0,
        },
        "output": {
            "volume": f"vol/{case_id}_vol.b2nd",
            "label": f"seg/{case_id}_seg.b2nd",
            "support": f"support/{case_id}_support.npy",
            "foreground_index": f"sampling/{case_id}_foreground.npy",
            "canonical": canonical,
            "properties": None,
            "transform": {
                "volume_axis": "identity DHWC",
                "label_axis": "identity DHW",
                "label_values": f"identity 0..{NUM_CLASSES - 1}",
                "support": "identity stored packed uint8 little-bit-order",
                "foreground_index": "identity stored exact sorted uint32 C-order flat indices",
            },
        },
    }
    return (record, foreground)


def verify_reference(record: dict[str, Any], reference: dict[str, Any]) -> None:
    """
    Compare the rebuilt case with its frozen reference.

    Args:
        record (dict[str, Any]): Computed geometry, logical array hashes, class counts, support counts, and normalization
            statistics.
        reference (dict[str, Any]): Frozen case record containing the expected values.

    Returns:
        None: Completes if the case matches the reference.

    Raises:
        ValueError: A logical array or metadata field differs from the reference.
    """
    case_id = record["case_id"]
    for name in ("volume", "label", "support", "foreground_index"):
        for field in ("shape", "dtype", "sha256"):
            if (
                record["output"]["canonical"][name][field]
                != reference["canonical"][name][field]
            ):
                raise ValueError(f"{case_id}: reference {name}.{field} mismatch")
    for field in reference["geometry"]:
        if record["geometry"][field] != reference["geometry"][field]:
            raise ValueError(f"{case_id}: reference geometry.{field} mismatch")
    if (
        record["output"]["canonical"]["label"]["counts"]
        != reference["canonical"]["label"]["counts"]
    ):
        raise ValueError(f"{case_id}: reference class counts mismatch")
    if record["label_support"] != reference["label_support"]:
        raise ValueError(f"{case_id}: reference support statistics mismatch")
    for field in ("mean", "std", "min", "max"):
        if not np.isclose(
            record["normalization"][field],
            reference["normalization"][field],
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(f"{case_id}: reference normalization.{field} mismatch")


def write_case(
    root: Path,
    record: dict[str, Any],
    volume: np.ndarray,
    label: np.ndarray,
    support: np.ndarray,
    foreground: np.ndarray,
) -> None:
    """
    Store B2ND images and labels with NumPy support and foreground arrays.

    Args:
        root (Path): Output dataset directory.
        record (dict[str, Any]): Case metadata with relative output paths; output.native is updated with the B2ND shapes,
            dtypes, chunks, and blocks.
        volume (np.ndarray): Float32 image with shape (D, H, W, 1).
        label (np.ndarray): Uint8 labels with shape (D, H, W).
        support (np.ndarray): 1D uint8 valid-label mask packed in C order with little bit order.
        foreground (np.ndarray): Sorted uint32 C-order indices with shape (F,), where F is the number of nonzero labels.

    Returns:
        None: Writes the arrays, verifies B2ND values, and updates record in place.
    """
    if blosc2.__version__ != "4.1.2":
        raise RuntimeError("Storage reproduction requires blosc2==4.1.2")
    blosc2.set_nthreads(1)
    record["output"]["native"] = {}
    for name, array in (("volume", volume), ("label", label)):
        path = root / record["output"][name]
        path.parent.mkdir(parents=True, exist_ok=True)
        chunks = tuple((min(size, tile) for size, tile in zip(array.shape[:3], CHUNKS)))
        blocks = tuple((min(size, tile) for size, tile in zip(array.shape[:3], BLOCKS)))
        if array.ndim == 4:
            chunks += (1,)
            blocks += (1,)
        store = blosc2.empty(
            array.shape,
            dtype=array.dtype,
            urlpath=str(path),
            mode="w",
            chunks=chunks,
            blocks=blocks,
            cparams={"codec": blosc2.Codec.LZ4HC, "clevel": 8, "nthreads": 1},
        )
        for start in range(0, len(array), CHUNKS[0]):
            chunk_slice = slice(start, start + CHUNKS[0])
            store[chunk_slice] = array[chunk_slice]
        record["output"]["native"][name] = {
            "shape": list(store.shape),
            "dtype": str(store.dtype),
            "chunks": list(store.chunks),
            "blocks": list(store.blocks),
        }
        del store
        saved = cast(
            blosc2.NDArray, blosc2.open(str(path), mode="r", dparams={"nthreads": 1})
        )
        for start in range(0, len(array), CHUNKS[0]):
            chunk_slice = slice(start, start + CHUNKS[0])
            if not np.array_equal(
                cast(np.ndarray, saved[chunk_slice]), array[chunk_slice]
            ):
                raise ValueError(f"{record['case_id']}: B2ND round trip changed {name}")
        del saved
    for name, array in (("support", support), ("foreground_index", foreground)):
        path = root / record["output"][name]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.lib.format.write_array(
                handle, array, version=NPY_VERSION, allow_pickle=False
            )


def run(
    config: dict[str, Any], cases: list[str] | None = None, audit_only: bool = False
) -> dict[str, Any]:
    """
    Build the selected cases or audit their frozen inventory and raw paths.

    Args:
        config (dict[str, Any]): Resolved settings from load_config, including input, output, split, and reference paths, raw
            patterns, and optional verify_reference.
        cases (list[str] | None): Frozen case IDs to build; None selects all labeled cases in the stored order.
        audit_only (bool): Check split membership and source files without writing.

    Returns:
        dict[str, Any]: Dataset manifest after a build, or an audit summary containing the dataset name, checked raw pair count,
            and zero writes.
    """
    splits, reference = read_splits(config)
    expected = {case["case_id"]: case for case in reference["cases"]}
    selected = (
        splits["labeled"]
        if cases is None
        else [case for case in splits["labeled"] if case in cases]
    )
    if not selected or (cases is not None and set(selected) != set(cases)):
        raise ValueError(
            "Requested cases must be a nonempty subset of the frozen labeled inventory"
        )
    raw, output = (config["raw_root"], config["output_root"])
    if raw == output or raw in output.parents or output in raw.parents:
        raise ValueError("Raw and output directories must be separate")
    pairs = {}
    for case in selected:
        pairs[case] = resolve_case_paths(config, case)
    if audit_only:
        return {"dataset": DATASET, "checked_raw_pairs": len(pairs), "writes": 0}
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(
            "Output must be absent or empty; existing datasets are never overwritten"
        )
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for case in selected:
        data, segmentation, geometry = read_case(*pairs[case])
        volume, label, support, geometry = preprocess(data, segmentation, geometry)
        del data, segmentation
        record, foreground = case_record(case, volume, label, support, geometry)
        record["split"] = expected[case]["split"]
        record["source"] = {
            "raw_image": pairs[case][0].relative_to(raw).as_posix(),
            "raw_label": pairs[case][1].relative_to(raw).as_posix(),
        }
        if config.get("verify_reference", True):
            verify_reference(record, expected[case])
        write_case(output, record, volume, label, support, foreground)
        if (
            config.get("verify_reference", True)
            and record["output"]["native"] != expected[case]["native"]
        ):
            raise ValueError(f"{case}: native B2ND tiling differs from reference")
        records.append(record)
        del volume, label, support, foreground
        print(f"{case}: saved", flush=True)
    selected_set = set(selected)
    for relative in reference["lists"]:
        source = config["split_root"] / relative
        payload = source.read_bytes()
        if selected_set != set(splits["labeled"]):
            names = [
                name for name in payload.decode().splitlines() if name in selected_set
            ]
            payload = "".join(f"{name}\n" for name in names).encode()
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    manifest = {
        "dataset": {
            "key": CONFIG_NAME,
            "name": DATASET,
            "channel_order": ["contrast_enhanced_CT"],
            "labels": [[index, name] for index, name in enumerate(LABEL_NAMES)],
            "volume_layout": "DHWC",
            "segmentation_layout": "DHW",
        },
        "storage": {"backend": "b2nd"},
        "preprocessing": {
            "reader": READER,
            "spacing_exact": list(TARGET_SPACING),
            "normalization": NORMALIZATION,
            "use_mask_for_norm": False,
            "intensity_properties": INTENSITY,
            "image_order": 3,
            "label_order": 1,
            "separate_axis_order": 0,
            "anisotropy_threshold": 3,
        },
        "complete": len(records) == EXPECTED_COUNTS["labeled"],
        "reference_verified": config.get("verify_reference", True),
        "cases": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    """
    Parse command-line settings and run preprocessing or an input audit.

    Returns:
        None: Prints the build or audit summary as JSON.
    """
    parser = argparse.ArgumentParser(
        description=f"Rebuild {DATASET} from raw NIfTI using its frozen configuration"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "preprocessing"
        / f"{CONFIG_NAME}.yaml",
    )
    parser.add_argument(
        "--case",
        action="append",
        help="Build one frozen case; repeat for a small subset",
    )
    parser.add_argument(
        "--output", type=Path, help="Override output_root; relative to project_root"
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Check raw paths and frozen splits without writing",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output is not None:
        config["output_root"] = (
            config["project_root"] / args.output.expanduser()
        ).resolve()
    result = run(config, args.case, args.audit_only)
    print(
        json.dumps(
            {
                "dataset": DATASET,
                "case_count": len(result.get("cases", [])),
                "complete": result.get("complete", False),
                "audit": result if args.audit_only else None,
            }
        )
    )


if __name__ == "__main__":
    main()
