from __future__ import annotations

import math
import os
import pickle
import random
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import numpy as np

if TYPE_CHECKING:
    from blosc2 import NDArray

StorageBackend = Literal["npy", "nnunet_b2nd", "b2nd"]

NPY_BACKEND: StorageBackend = "npy"
NNUNET_B2ND_BACKEND: StorageBackend = "nnunet_b2nd"
B2ND_BACKEND: StorageBackend = "b2nd"


class _ReadOnlyB2NDOpener(Protocol):
    """Blosc2's documented keyword contract for opening local B2ND arrays."""

    def __call__(
        self,
        urlpath: str,
        *,
        mode: Literal["r"],
        mmap_mode: Literal["r"],
        dparams: dict[str, int],
    ) -> NDArray:
        """
        Open an existing B2ND array with read-only memory mapping.

        Args:
            urlpath (str): Local B2ND array path.
            mode (Literal["r"]): Read-only file mode.
            mmap_mode (Literal["r"]): Read-only mapping mode.
            dparams (dict[str, int]): Decompression parameters, including thread count.

        Returns:
            blosc2.NDArray: Open compressed array retaining its stored shape and dtype.
        """
        ...


@dataclass(frozen=True)
class CaseStorageSpec:
    """
    Discovered files for one case and its storage backend.

    Attributes:
        case_id (str): Processed case identifier.
        backend (StorageBackend): NPY, generic B2ND, or nnU-Net B2ND layout.
        volume_path (Path): Resolved image-array path.
        label_path (Path | None): Resolved segmentation path, when present.
        support_path (Path | None): Optional support-array sidecar.
        properties_path (Path | None): Optional nnU-Net properties pickle with class locations.
        foreground_index_path (Path | None): Optional generic B2ND foreground-index NPY sidecar.
    """

    case_id: str
    backend: StorageBackend
    volume_path: Path
    label_path: Path | None
    support_path: Path | None = None
    properties_path: Path | None = None
    foreground_index_path: Path | None = None


@dataclass(frozen=True)
class _StorageConvention:
    """
    Relative filenames describing one supported case layout.

    Attributes:
        backend (StorageBackend): Backend selected by this filename convention.
        volume (Path): Relative image-array path.
        label (Path): Relative segmentation path.
        support (Path | None): Optional relative support-array path.
        properties (Path | None): Optional relative nnU-Net properties path.
        foreground_index (Path | None): Optional relative foreground-index path.
    """

    backend: StorageBackend
    volume: Path
    label: Path
    support: Path | None = None
    properties: Path | None = None
    foreground_index: Path | None = None


def discover_case_storage_specs(
    root: str | Path,
    case_ids: Sequence[str],
) -> tuple[CaseStorageSpec, ...]:
    """
    Match requested cases to supported image-file conventions.

    Args:
        root (str | Path): Processed dataset directory; the root may be a symlink, but discovered child artifacts may not.
        case_ids (Sequence[str]): Nonempty case identifiers in the desired order.

    Returns:
        tuple[CaseStorageSpec, ...]: Resolved case files in requested order, with one matching image layout per case.

    Raises:
        ValueError: If the root or identifiers are invalid, storage is absent or ambiguous, or a discovered child path uses a
            symlink.
    """

    root_path = Path(root)
    try:
        resolved_root = root_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"Dataset root does not exist: {root_path}") from exc
    if not resolved_root.is_dir():
        raise ValueError(f"Dataset root is not a directory: {root_path}")
    if not case_ids:
        raise ValueError("A dataset split must contain at least one case")

    specs: list[CaseStorageSpec] = []
    for raw_case_id in case_ids:
        case_id = _case_id(raw_case_id, context="dataset split")
        conventions = _storage_conventions(case_id)
        candidates = [
            convention
            for convention in conventions
            if _artifact_present(root_path / convention.volume)
        ]
        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous storage for case {case_id!r}; found volumes for "
                f"{[str(item.volume) for item in candidates]}"
            )
        if not candidates:
            orphan_labels = [
                str(convention.label)
                for convention in conventions
                if _artifact_present(root_path / convention.label)
            ]
            if orphan_labels:
                raise ValueError(
                    f"Case {case_id!r} has label files without a matching volume: "
                    f"{orphan_labels}"
                )
            raise ValueError(
                f"No supported volume file found for case {case_id!r} under "
                f"{resolved_root}"
            )

        convention = candidates[0]
        volume_path = _resolve_discovered_path(
            root_path,
            convention.volume,
            case_id=case_id,
            kind="volume",
        )
        label_path = _resolve_optional_discovered_path(
            root_path,
            convention.label,
            case_id=case_id,
            kind="label",
        )
        support_path = _resolve_optional_discovered_path(
            root_path,
            convention.support,
            case_id=case_id,
            kind="support",
        )
        properties_path = _resolve_optional_discovered_path(
            root_path,
            convention.properties,
            case_id=case_id,
            kind="properties",
        )
        foreground_index_path = _resolve_optional_discovered_path(
            root_path,
            convention.foreground_index,
            case_id=case_id,
            kind="foreground index",
        )
        specs.append(
            CaseStorageSpec(
                case_id=case_id,
                backend=convention.backend,
                volume_path=volume_path,
                label_path=label_path,
                support_path=support_path,
                properties_path=properties_path,
                foreground_index_path=foreground_index_path,
            )
        )
    return tuple(specs)


def validate_case_storage_specs(
    specs: Sequence[CaseStorageSpec],
    *,
    expected_channels: int,
    minimum_spatial_size: int | Sequence[int] | None,
    require_labels: bool = True,
    num_classes: int | None = None,
) -> None:
    """
    Validate selected files, array layouts, and optional class bounds.

    Args:
        specs (Sequence[CaseStorageSpec]): Nonempty collection of discovered case files.
        expected_channels (int): Required positive image channel count.
        minimum_spatial_size (int | Sequence[int] | None): Optional cubic or three-axis minimum case extent.
        require_labels (bool): Require a segmentation file for every case.
        num_classes (int | None): Exclusive label upper bound; None skips label-value checks.

    Raises:
        ValueError: If a file, shape, dtype, label range, or foreground-index sidecar violates the storage contract.
    """

    if int(expected_channels) <= 0:
        raise ValueError(f"expected_channels must be positive, got {expected_channels}")
    if num_classes is not None and int(num_classes) <= 1:
        raise ValueError(f"num_classes must be greater than one, got {num_classes}")
    if not specs:
        raise ValueError("A dataset split must contain at least one case")
    minimum = _normalize_spatial_size(minimum_spatial_size)
    errors: list[str] = []
    handles = ReadOnlyB2NDHandleCache(max_cached_handles=4)
    try:
        for spec in specs:
            try:
                spatial_shape = _validate_case_storage_spec(
                    spec,
                    handles=handles,
                    expected_channels=int(expected_channels),
                    require_labels=require_labels,
                    num_classes=num_classes,
                )
                if minimum is not None and any(
                    actual < needed for actual, needed in zip(spatial_shape, minimum)
                ):
                    raise ValueError(
                        f"spatial shape {spatial_shape} is smaller than {minimum}"
                    )
                if any(size <= 0 for size in spatial_shape):
                    raise ValueError(f"nonpositive spatial shape {spatial_shape}")
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                errors.append(f"{spec.case_id}: {exc}")
            if len(errors) >= 10:
                break
    finally:
        handles.close()

    if errors:
        raise ValueError("Invalid dataset storage:\n- " + "\n- ".join(errors))


class ReadOnlyB2NDHandleCache:
    """Keep a process-local least-recently-used cache of read-only B2ND handles."""

    def __init__(self, max_cached_handles: int = 32) -> None:
        """
        Args:
            max_cached_handles (int): Maximum retained B2ND handles; clamped to at least one.
        """

        self.max_cached_handles = max(1, int(max_cached_handles))
        self._pid: int | None = None
        self._handles: OrderedDict[str, Any] = OrderedDict()

    def get(self, path: Path) -> Any:
        """
        Open or reuse a B2ND array in the current process.

        Forked workers discard inherited handles; spawned workers restore an empty cache.

        Args:
            path (Path): B2ND array path.

        Returns:
            blosc2.NDArray: Read-only memory-mapped array handle using one decompression thread.
        """

        self._ensure_current_process()
        key = str(path)
        cached = self._handles.pop(key, None)
        if cached is not None:
            self._handles[key] = cached
            return cached

        try:
            import blosc2
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "The B2ND storage backends require blosc2==4.1.2. "
                "Install the project's declared runtime requirements."
            ) from exc
        # Blosc2 4.1.2 annotates every extra keyword as dict, including mmap_mode.
        handle = cast(_ReadOnlyB2NDOpener, blosc2.open)(
            key,
            mode="r",
            mmap_mode="r",
            dparams={"nthreads": 1},
        )
        self._handles[key] = handle
        while len(self._handles) > self.max_cached_handles:
            _, evicted = self._handles.popitem(last=False)
            _close_if_supported(evicted)
        return handle

    def close(self) -> None:
        """
        Release all cached handles and reset process ownership.

        Call close() when a handle provides it, then drop the reference to release logical mappings.
        """

        while self._handles:
            _, handle = self._handles.popitem(last=False)
            _close_if_supported(handle)
        self._pid = None

    def __getstate__(self) -> dict[str, Any]:
        """
        Serialize cache capacity without open B2ND handles.

        Returns:
            dict[str, int | None | OrderedDict]: Capacity, unset process ID, and empty handle cache.
        """

        return {
            "max_cached_handles": self.max_cached_handles,
            "_pid": None,
            "_handles": OrderedDict(),
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """
        Restore cache capacity with no live handles.

        Args:
            state (Mapping[str, object]): Serialized cache state containing max_cached_handles.
        """

        self.max_cached_handles = int(state["max_cached_handles"])
        self._pid = None
        self._handles = OrderedDict()

    @property
    def open_handle_count(self) -> int:
        """
        Count B2ND handles owned by the current process.

        Returns:
            int: Number of cached handles after resetting inherited state.
        """

        self._ensure_current_process()
        return len(self._handles)

    def _ensure_current_process(self) -> None:
        """Discard inherited handles when the process ID changes."""

        pid = os.getpid()
        if self._pid == pid:
            return
        self.close()
        self._pid = pid


class ForegroundIndexCache:
    """Cache validated, memory-mapped foreground-index sidecars per process."""

    def __init__(self, max_cached_cases: int = 32) -> None:
        """
        Args:
            max_cached_cases (int): Maximum mapped sidecars; clamped to at least one.
        """

        self.max_cached_cases = max(1, int(max_cached_cases))
        self._pid: int | None = None
        self._entries: OrderedDict[tuple[str, int], np.memmap] = OrderedDict()

    def get(
        self,
        path: Path,
        *,
        voxel_count: int,
    ) -> np.memmap:
        """
        Open or reuse a validated foreground-index sidecar.

        Args:
            path (Path): NPY sidecar with sorted flat foreground indices.
            voxel_count (int): Total voxels in the associated label array.

        Returns:
            np.memmap: Read-only uint32 array of shape (N,), with strictly increasing indices below voxel_count. N may be zero.
        """

        self._ensure_current_process()
        key = (str(path), int(voxel_count))
        cached = self._entries.pop(key, None)
        if cached is not None:
            self._entries[key] = cached
            return cached

        try:
            indices = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Invalid foreground-index sidecar: {path}") from exc
        try:
            _validate_foreground_indices(
                indices,
                voxel_count=int(voxel_count),
                path=path,
            )
        except Exception:
            _close_numpy_mapping(indices)
            raise

        self._entries[key] = indices
        while len(self._entries) > self.max_cached_cases:
            _, evicted = self._entries.popitem(last=False)
            _close_numpy_mapping(evicted)
        return indices

    def close(self) -> None:
        """Close all foreground-index mappings and reset process ownership."""

        while self._entries:
            _, indices = self._entries.popitem(last=False)
            _close_numpy_mapping(indices)
        self._pid = None

    def __getstate__(self) -> dict[str, Any]:
        """
        Serialize cache capacity without mapped sidecars.

        Returns:
            dict[str, int | None | OrderedDict]: Capacity, unset process ID, and empty sidecar cache.
        """

        return {
            "max_cached_cases": self.max_cached_cases,
            "_pid": None,
            "_entries": OrderedDict(),
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """
        Restore foreground-cache capacity with no open mappings.

        Args:
            state (Mapping[str, object]): Serialized state containing max_cached_cases.
        """

        self.max_cached_cases = int(state["max_cached_cases"])
        self._pid = None
        self._entries = OrderedDict()

    @property
    def cached_case_count(self) -> int:
        """
        Count sidecars mapped in the current process.

        Returns:
            int: Current cache size after resetting inherited state.
        """

        self._ensure_current_process()
        return len(self._entries)

    def _ensure_current_process(self) -> None:
        """Close inherited sidecar mappings when the process ID changes."""

        pid = os.getpid()
        if self._pid == pid:
            return
        self.close()
        self._pid = pid


class B2NDPatchReader:
    """Sample and decode paired patches from generic XYZC/XYZ B2ND arrays."""

    def __init__(
        self,
        crop_size: int | Sequence[int],
        *,
        in_channels: int,
        num_classes: int,
        max_cached_handles: int = 32,
        positive_probability: float = 0.5,
        background_attempts: int = 512,
        max_cached_foreground: int = 32,
    ) -> None:
        """
        Args:
            crop_size (int | Sequence[int]): Positive cubic size or three patch extents in stored spatial-axis order.
            in_channels (int): Required image channel count C.
            num_classes (int): Segmentation class count, including background; labels use [0, num_classes).
            max_cached_handles (int): Maximum open B2ND handles per process; clamped to at least one.
            positive_probability (float): Probability of requesting a foreground crop, in [0, 1].
            background_attempts (int): Maximum background rejection draws before exact search; clamped to at least one.
            max_cached_foreground (int): Maximum mapped foreground-index sidecars per process.
        """

        normalized_crop_size = _normalize_spatial_size(crop_size)
        assert normalized_crop_size is not None
        self.crop_size = normalized_crop_size
        self.in_channels, self.num_classes = _reader_dimensions(
            in_channels, num_classes
        )
        if not 0.0 <= float(positive_probability) <= 1.0:
            raise ValueError("positive_probability must be in [0, 1]")
        self.positive_probability = float(positive_probability)
        self.background_attempts = max(1, int(background_attempts))
        self.handles = ReadOnlyB2NDHandleCache(max_cached_handles)
        self.foreground_indices = ForegroundIndexCache(max_cached_foreground)
        self.last_sampling_kind: str | None = None
        self.last_crop_start: tuple[int, int, int] | None = None

    def read_case(self, spec: CaseStorageSpec) -> tuple[np.ndarray, np.ndarray]:
        """
        Select a foreground or background crop and decode its paired arrays.

        Foreground uses sidecar indices when available, otherwise label-plane reservoir sampling. Failed background rejection
        draws trigger an exact label-plane search. Image data is decoded only for the selected patch.

        Args:
            spec (CaseStorageSpec): Generic B2ND image, labels, and optional flat foreground-index sidecar.

        Returns:
            tuple[np.ndarray, np.ndarray]: Contiguous float32 image (X, Y, Z, C) and uint8 labels (X, Y, Z). Spatial extents
                equal the crop size; undersized axes receive centered zero padding.
        """

        if spec.backend != B2ND_BACKEND or spec.label_path is None:
            raise ValueError(f"Case {spec.case_id!r} is not a generic b2nd case")
        volume_store = self.handles.get(spec.volume_path)
        label_store = self.handles.get(spec.label_path)
        spatial_shape = _validate_generic_b2nd_stores(
            spec,
            volume_store,
            label_store,
            in_channels=self.in_channels,
        )
        indices = (
            self.foreground_indices.get(
                spec.foreground_index_path,
                voxel_count=math.prod(spatial_shape),
            )
            if spec.foreground_index_path is not None
            else None
        )

        want_positive = random.random() < self.positive_probability
        start: tuple[int, int, int] | None = None
        sampling_kind: str | None = None
        if want_positive and indices is not None and indices.size:
            start = self._sample_positive_start(
                indices,
                spatial_shape,
                label_store=label_store,
            )
            sampling_kind = "positive"
        elif want_positive and indices is None:
            start = self._sample_positive_start_exact(label_store, spatial_shape)
            if start is not None:
                sampling_kind = "positive_exact"
        if start is None:
            start = self._sample_background_start(label_store, spatial_shape)
            if start is not None:
                sampling_kind = "background"
        if start is None:
            start = self._sample_background_start_exact(
                label_store,
                spatial_shape,
            )
            if start is not None:
                sampling_kind = "background_exact"
        if start is None and not want_positive and indices is not None and indices.size:
            start = self._sample_positive_start(
                indices,
                spatial_shape,
                label_store=label_store,
            )
            sampling_kind = "positive_fallback"
        elif start is None and not want_positive and indices is None:
            start = self._sample_positive_start_exact(label_store, spatial_shape)
            if start is not None:
                sampling_kind = "positive_exact_fallback"
        if start is None:
            start = _sample_unconstrained_start(spatial_shape, self.crop_size)
            sampling_kind = "random_fallback"

        self.last_sampling_kind = sampling_kind
        self.last_crop_start = start
        return _read_generic_b2nd_patch(
            volume_store,
            label_store,
            spatial_shape=spatial_shape,
            crop_size=self.crop_size,
            start=start,
            in_channels=self.in_channels,
            num_classes=self.num_classes,
        )

    def close(self) -> None:
        """Release compressed-array handles and foreground-index mappings."""

        self.handles.close()
        self.foreground_indices.close()

    def _sample_positive_start(
        self,
        indices: np.ndarray,
        spatial_shape: Sequence[int],
        *,
        label_store: Any,
    ) -> tuple[int, int, int]:
        """
        Draw a foreground sidecar index and clamp the crop around it.

        Args:
            indices (np.ndarray): Nonempty integer flat-index array with shape (N,).
            spatial_shape (Sequence[int]): Three spatial extents in stored axis order.
            label_store (blosc2.NDArray): Open uint8 label store with shape (X, Y, Z).

        Returns:
            tuple[int, int, int]: Source crop origin in stored spatial-axis order; undersized axes start at zero.

        Raises:
            ValueError: If the sampled sidecar coordinate points to background.
        """

        flat_index = int(indices[random.randrange(int(indices.size))])
        center = _unravel_flat_index(flat_index, spatial_shape)
        value = int(np.asarray(label_store[center]).reshape(()).item())
        if value == 0:
            raise ValueError(
                "Foreground-index sidecar contains a background coordinate"
            )
        return _start_from_center(center, spatial_shape, self.crop_size)

    def _sample_positive_start_exact(
        self,
        label_store: Any,
        spatial_shape: Sequence[int],
    ) -> tuple[int, int, int] | None:
        """
        Uniformly sample foreground while decoding one label plane at a time.

        Args:
            label_store (blosc2.NDArray): Open uint8 label store with shape (X, Y, Z).
            spatial_shape (Sequence[int]): Three spatial extents in stored axis order.

        Returns:
            tuple[int, int, int] | None: Source crop origin, or None if no eligible voxel is found.
        """

        selected: tuple[int, int, int] | None = None
        seen = 0
        for x in range(int(spatial_shape[0])):
            plane = np.asarray(label_store[(x, slice(None), slice(None))])
            _validate_decoded_labels(plane, num_classes=self.num_classes)
            candidates = np.flatnonzero(plane != 0)
            count = int(candidates.size)
            if count == 0:
                continue
            if random.randrange(seen + count) < count:
                local = int(candidates[random.randrange(count)])
                y, z = divmod(local, int(spatial_shape[2]))
                selected = (x, y, z)
            seen += count
        if selected is None:
            return None
        return _start_from_center(selected, spatial_shape, self.crop_size)

    def _sample_background_start(
        self,
        label_store: Any,
        spatial_shape: Sequence[int],
    ) -> tuple[int, int, int] | None:
        """
        Try background centers in the valid padded crop-center box.

        Padding is treated as background. The number of scalar label reads is bounded by background_attempts.

        Args:
            label_store (blosc2.NDArray): Open uint8 label store with shape (X, Y, Z).
            spatial_shape (Sequence[int]): Three spatial extents in stored axis order.

        Returns:
            tuple[int, int, int] | None: Source crop origin, or None if no eligible voxel is found.
        """

        padded_shape, pad_before, lower, upper = _padded_sampling_geometry(
            spatial_shape,
            self.crop_size,
        )
        for _ in range(self.background_attempts):
            padded_center = tuple(
                random.randrange(low, high) for low, high in zip(lower, upper)
            )
            source_center = tuple(
                coordinate - before
                for coordinate, before in zip(padded_center, pad_before)
            )
            inside = all(
                0 <= coordinate < int(actual)
                for coordinate, actual in zip(source_center, spatial_shape)
            )
            if inside:
                value = int(np.asarray(label_store[source_center]).reshape(()).item())
                if value != 0:
                    continue
            return _start_from_padded_center(
                padded_center,
                spatial_shape,
                self.crop_size,
            )
        return None

    def _sample_background_start_exact(
        self,
        label_store: Any,
        spatial_shape: Sequence[int],
    ) -> tuple[int, int, int] | None:
        """
        Uniformly sample eligible background centers with label-plane scans.

        The search includes padded background and visits only the valid center box. It preserves background sampling when an
        eligible center exists after bounded rejection draws fail.

        Args:
            label_store (blosc2.NDArray): Open uint8 label store with shape (X, Y, Z).
            spatial_shape (Sequence[int]): Three spatial extents in stored axis order.

        Returns:
            tuple[int, int, int] | None: Source crop origin, or None if no eligible voxel is found.
        """

        _padded_shape, pad_before, lower, upper = _padded_sampling_geometry(
            spatial_shape,
            self.crop_size,
        )
        y_padded = np.arange(lower[1], upper[1], dtype=np.int64)
        z_padded = np.arange(lower[2], upper[2], dtype=np.int64)
        y_source = y_padded - pad_before[1]
        z_source = z_padded - pad_before[2]
        y_inside = (y_source >= 0) & (y_source < int(spatial_shape[1]))
        z_inside = (z_source >= 0) & (z_source < int(spatial_shape[2]))
        y_positions = np.flatnonzero(y_inside)
        z_positions = np.flatnonzero(z_inside)

        selected: tuple[int, int, int] | None = None
        seen = 0
        for padded_x in range(lower[0], upper[0]):
            source_x = padded_x - pad_before[0]
            background = np.ones(
                (int(y_padded.size), int(z_padded.size)),
                dtype=bool,
            )
            if (
                0 <= source_x < int(spatial_shape[0])
                and y_positions.size
                and z_positions.size
            ):
                y_start = int(y_source[int(y_positions[0])])
                y_stop = int(y_source[int(y_positions[-1])]) + 1
                z_start = int(z_source[int(z_positions[0])])
                z_stop = int(z_source[int(z_positions[-1])]) + 1
                decoded = np.asarray(
                    label_store[
                        (
                            int(source_x),
                            slice(y_start, y_stop),
                            slice(z_start, z_stop),
                        )
                    ]
                )
                background[np.ix_(y_positions, z_positions)] = decoded == 0

            count = int(background.sum())
            if count == 0:
                continue
            if random.randrange(seen + count) < count:
                candidates = np.flatnonzero(background)
                local = int(candidates[random.randrange(count)])
                y_offset, z_offset = divmod(local, int(z_padded.size))
                selected = (
                    int(padded_x),
                    int(y_padded[y_offset]),
                    int(z_padded[z_offset]),
                )
            seen += count

        if selected is None:
            return None
        return _start_from_padded_center(
            selected,
            spatial_shape,
            self.crop_size,
        )


class B2NDFullCaseReader:
    """Decode complete generic B2ND cases at their stored spatial size."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        max_cached_handles: int = 32,
    ) -> None:
        """
        Args:
            in_channels (int): Required image channel count C.
            num_classes (int): Segmentation class count, including background; labels use [0, num_classes).
            max_cached_handles (int): Maximum open B2ND handles per process; clamped to at least one.
        """

        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.handles = ReadOnlyB2NDHandleCache(max_cached_handles)

    def read_case(self, spec: CaseStorageSpec) -> tuple[np.ndarray, np.ndarray]:
        """
        Decode an entire generic B2ND image and segmentation.

        Args:
            spec (CaseStorageSpec): Generic B2ND case with both image and label paths.

        Returns:
            tuple[np.ndarray, np.ndarray]: Contiguous float32 image (X, Y, Z, C) and uint8 labels (X, Y, Z), preserving native
                spatial dimensions.
        """

        if spec.backend != B2ND_BACKEND or spec.label_path is None:
            raise ValueError(f"Case {spec.case_id!r} is not a generic b2nd case")
        volume_store = self.handles.get(spec.volume_path)
        label_store = self.handles.get(spec.label_path)
        _validate_generic_b2nd_stores(
            spec,
            volume_store,
            label_store,
            in_channels=self.in_channels,
        )
        volume = np.asarray(volume_store[tuple(slice(None) for _ in range(4))])
        label = np.asarray(label_store[tuple(slice(None) for _ in range(3))])
        _validate_decoded_labels(label, num_classes=self.num_classes)
        return (
            np.ascontiguousarray(volume, dtype=np.float32),
            np.ascontiguousarray(label, dtype=np.uint8),
        )

    def close(self) -> None:
        """Release the full-case reader's cached array handles."""

        self.handles.close()


class NnUNetB2NDFullCaseReader:
    """Decode full nnU-Net B2ND cases into channel-last image and label arrays."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        max_cached_handles: int = 32,
    ) -> None:
        """
        Args:
            in_channels (int): Required image channel count C.
            num_classes (int): Segmentation class count, including background; labels use [0, num_classes).
            max_cached_handles (int): Maximum open B2ND handles per process; clamped to at least one.
        """

        self.in_channels, self.num_classes = _reader_dimensions(
            in_channels, num_classes
        )
        self.handles = ReadOnlyB2NDHandleCache(max_cached_handles)

    def read_case(self, spec: CaseStorageSpec) -> tuple[np.ndarray, np.ndarray]:
        """
        Decode native nnU-Net arrays and map ignore labels to background.

        Args:
            spec (CaseStorageSpec): nnU-Net B2ND case storing images as CDHW and labels as 1DHW.

        Returns:
            tuple[np.ndarray, np.ndarray]: Contiguous float32 image (D, H, W, C) and uint8 labels (D, H, W). Native spatial
                order and size are preserved; ignore values -1 become zero.

        Raises:
            ValueError: If the backend, array layout, image finiteness, or label values are invalid.
        """

        if spec.backend != NNUNET_B2ND_BACKEND or spec.label_path is None:
            raise ValueError(
                f"Case {spec.case_id!r} is not a complete nnunet_b2nd case"
            )
        volume_store = self.handles.get(spec.volume_path)
        label_store = self.handles.get(spec.label_path)
        _validate_nnunet_b2nd_stores(
            spec,
            volume_store,
            label_store,
            in_channels=self.in_channels,
        )
        full_index = tuple(slice(None) for _ in range(4))
        volume_cdhw = np.asarray(volume_store[full_index])
        label_1dhw = np.asarray(label_store[full_index])
        volume = np.moveaxis(volume_cdhw, 0, -1).astype(np.float32, copy=False)
        if not np.all(np.isfinite(volume)):
            raise ValueError(
                f"B2ND volume for {spec.case_id!r} contains non-finite values"
            )
        label = _canonicalize_nnunet_label(
            label_1dhw,
            num_classes=self.num_classes,
        )
        return np.ascontiguousarray(volume), np.ascontiguousarray(label)

    def close(self) -> None:
        """Release cached full-volume nnU-Net array handles."""

        self.handles.close()


class NnUNetClassLocationsCache:
    """Cache nnU-Net foreground coordinates by case and process."""

    def __init__(self, max_cached_cases: int = 4) -> None:
        """
        Args:
            max_cached_cases (int): Maximum retained properties records; clamped to at least one.
        """

        self.max_cached_cases = max(1, int(max_cached_cases))
        self._pid: int | None = None
        self._entries: OrderedDict[str, tuple[np.ndarray, ...]] = OrderedDict()

    def get(
        self,
        properties_path: Path,
    ) -> tuple[np.ndarray, ...]:
        """
        Load and normalize nonempty foreground class locations.

        Empty class entries are skipped before shape and dtype checks. Spatial bounds are checked later by the patch reader.

        Args:
            properties_path (Path): nnU-Net properties pickle containing class_locations.

        Returns:
            tuple[np.ndarray, ...]: Contiguous int32 arrays of shape (N, 3), one per nonempty foreground entry, with columns
                (D, H, W). The stored zero channel column is removed.
        """

        self._ensure_current_process()
        key = str(properties_path)
        cached = self._entries.pop(key, None)
        if cached is not None:
            self._entries[key] = cached
            return cached

        try:
            with properties_path.open("rb") as stream:
                properties = _RestrictedNumpyUnpickler(stream).load()
        except (OSError, pickle.UnpicklingError, EOFError, ValueError) as exc:
            raise ValueError(
                f"Invalid nnU-Net properties file: {properties_path}"
            ) from exc
        if not isinstance(properties, Mapping):
            raise ValueError(f"nnU-Net properties must be a mapping: {properties_path}")
        class_locations = properties.get("class_locations")
        if not isinstance(class_locations, Mapping):
            raise ValueError(
                f"nnU-Net properties lack class_locations mapping: {properties_path}"
            )

        normalized: list[np.ndarray] = []
        for class_key, raw_locations in class_locations.items():
            if not _foreground_class_key(class_key):
                continue
            locations = np.asarray(raw_locations)
            # Absent classes use empty float arrays; skip them before shape checks.
            if locations.size == 0:
                continue
            if locations.ndim != 2 or locations.shape[1] != 4:
                raise ValueError(
                    f"class_locations entries must have shape [N, 4]: {properties_path}"
                )
            if not np.issubdtype(locations.dtype, np.integer):
                raise ValueError(
                    f"class_locations must contain integer coordinates: {properties_path}"
                )
            if np.any(locations[:, 0] != 0):
                raise ValueError(
                    f"class_locations channel coordinates must all be zero: {properties_path}"
                )
            int32_info = np.iinfo(np.int32)
            if np.any(locations[:, 1:] < int32_info.min) or np.any(
                locations[:, 1:] > int32_info.max
            ):
                raise ValueError(
                    f"class_locations exceed int32 coordinate range: {properties_path}"
                )
            normalized.append(np.ascontiguousarray(locations[:, 1:], dtype=np.int32))
        result = tuple(normalized)
        self._entries[key] = result
        while len(self._entries) > self.max_cached_cases:
            self._entries.popitem(last=False)
        return result

    def close(self) -> None:
        """Drop cached coordinate arrays and reset process ownership."""

        self._entries.clear()
        self._pid = None

    def __getstate__(self) -> dict[str, Any]:
        """
        Serialize properties-cache capacity without coordinate arrays.

        Returns:
            dict[str, int | None | OrderedDict]: Capacity, unset process ID, and empty coordinate cache.
        """

        return {
            "max_cached_cases": self.max_cached_cases,
            "_pid": None,
            "_entries": OrderedDict(),
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """
        Restore properties-cache capacity with no cached arrays.

        Args:
            state (Mapping[str, object]): Serialized state containing max_cached_cases.
        """

        self.max_cached_cases = int(state["max_cached_cases"])
        self._pid = None
        self._entries = OrderedDict()

    @property
    def cached_case_count(self) -> int:
        """
        Count cases with coordinates cached in the current process.

        Returns:
            int: Current cache size after resetting inherited state.
        """

        self._ensure_current_process()
        return len(self._entries)

    def _ensure_current_process(self) -> None:
        """Discard inherited coordinates when the process ID changes."""

        pid = os.getpid()
        if self._pid == pid:
            return
        self.close()
        self._pid = pid


class NnUNetB2NDPatchReader:
    """Sample training patches directly from nnU-Net CDHW/1DHW B2ND stores."""

    def __init__(
        self,
        crop_size: int | Sequence[int],
        *,
        in_channels: int,
        num_classes: int,
        max_cached_handles: int = 32,
        positive_probability: float = 0.5,
        background_attempts: int = 64,
        max_cached_properties: int = 4,
    ) -> None:
        """
        Args:
            crop_size (int | Sequence[int]): Positive cubic size or three patch extents in stored spatial-axis order.
            in_channels (int): Required image channel count C.
            num_classes (int): Segmentation class count, including background; labels use [0, num_classes).
            max_cached_handles (int): Maximum open B2ND handles per process; clamped to at least one.
            positive_probability (float): Probability of requesting a foreground crop, in [0, 1].
            background_attempts (int): Maximum background rejection draws before foreground or unconstrained fallback.
            max_cached_properties (int): Maximum cached nnU-Net class-location records per process.
        """

        normalized_crop_size = _normalize_spatial_size(crop_size)
        assert normalized_crop_size is not None
        self.crop_size = normalized_crop_size
        self.in_channels, self.num_classes = _reader_dimensions(
            in_channels, num_classes
        )
        if not 0.0 <= float(positive_probability) <= 1.0:
            raise ValueError("positive_probability must be in [0, 1]")
        self.positive_probability = float(positive_probability)
        self.background_attempts = max(1, int(background_attempts))
        self.handles = ReadOnlyB2NDHandleCache(max_cached_handles)
        self.class_locations = NnUNetClassLocationsCache(max_cached_properties)
        self.last_sampling_kind: str | None = None
        self.last_crop_start: tuple[int, int, int] | None = None

    def read_case(self, spec: CaseStorageSpec) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample an nnU-Net case and decode one aligned training patch.

        Args:
            spec (CaseStorageSpec): nnU-Net B2ND image, labels, and optional class-location properties.

        Returns:
            tuple[np.ndarray, np.ndarray]: Contiguous float32 image (D, H, W, C) and uint8 labels (D, H, W) at the crop size.
                Undersized axes receive centered zero padding; ignore labels -1 become background.
        """

        if spec.backend != NNUNET_B2ND_BACKEND or spec.label_path is None:
            raise ValueError(f"Case {spec.case_id!r} is not an nnunet_b2nd case")
        volume_store = self.handles.get(spec.volume_path)
        label_store = self.handles.get(spec.label_path)
        spatial_shape = self._validate_open_stores(spec, volume_store, label_store)

        want_positive = random.random() < self.positive_probability
        start: tuple[int, int, int] | None = None
        sampling_kind: str | None = None
        if want_positive:
            start = self._sample_positive_start(spec, label_store, spatial_shape)
            if start is not None:
                sampling_kind = "positive"
        if start is None:
            start = self._sample_background_start(label_store, spatial_shape)
            if start is not None:
                sampling_kind = "background"
        if start is None and not want_positive:
            start = self._sample_positive_start(spec, label_store, spatial_shape)
            if start is not None:
                sampling_kind = "positive_fallback"
        if start is None:
            start = self._sample_unconstrained_start(spatial_shape)
            sampling_kind = "random_fallback"

        self.last_sampling_kind = sampling_kind
        self.last_crop_start = start
        volume, label = self._read_patch(
            volume_store,
            label_store,
            spatial_shape=spatial_shape,
            start=start,
        )
        return volume, label

    def close(self) -> None:
        """Release nnU-Net array handles and cached class locations."""

        self.handles.close()
        self.class_locations.close()

    def _validate_open_stores(
        self,
        spec: CaseStorageSpec,
        volume_store: Any,
        label_store: Any,
    ) -> tuple[int, int, int]:
        """
        Check nnU-Net image and label layouts before sampling.

        Args:
            spec (CaseStorageSpec): Selected paths and backend for one case.
            volume_store (blosc2.NDArray): Open float32 image store with shape (C, D, H, W).
            label_store (blosc2.NDArray): Open integer label store with shape (1, D, H, W), allowing ignore value -1.

        Returns:
            tuple[int, int, int]: Shared spatial shape in (D, H, W) order.
        """

        return _validate_nnunet_b2nd_stores(
            spec,
            volume_store,
            label_store,
            in_channels=self.in_channels,
        )

    def _sample_positive_start(
        self,
        spec: CaseStorageSpec,
        label_store: Any,
        spatial_shape: Sequence[int],
    ) -> tuple[int, int, int] | None:
        """
        Select a stored foreground location or scan labels when properties are absent.

        Properties sampling chooses a nonempty foreground entry uniformly, then one stored coordinate uniformly. Exact fallback
        samples all positive label voxels.

        Args:
            spec (CaseStorageSpec): Selected paths and backend for one case.
            label_store (blosc2.NDArray): Open integer label store with shape (1, D, H, W), allowing ignore value -1.
            spatial_shape (Sequence[int]): Three spatial extents in stored axis order.

        Returns:
            tuple[int, int, int] | None: Source crop origin, or None if no eligible voxel is found.
        """

        properties_path = spec.properties_path
        if properties_path is None:
            return self._sample_positive_start_exact(label_store, spatial_shape)
        entries = self.class_locations.get(properties_path)
        if not entries:
            return None
        spatial = np.asarray(spatial_shape, dtype=np.int64)
        for locations in entries:
            if np.any(locations < 0) or np.any(locations >= spatial[None, :]):
                raise ValueError(
                    f"class_locations contains coordinates outside shape "
                    f"{tuple(spatial_shape)}: {properties_path}"
                )

        # Choose a class first, then one of its stored, subsampled locations.
        locations = entries[random.randrange(len(entries))]
        location = locations[random.randrange(int(locations.shape[0]))]
        center = tuple(int(value) for value in location)
        return self._start_from_center(center, spatial_shape)

    def _sample_positive_start_exact(
        self,
        label_store: Any,
        spatial_shape: Sequence[int],
    ) -> tuple[int, int, int] | None:
        """
        Reservoir-sample positive nnU-Net labels one plane at a time.

        Args:
            label_store (blosc2.NDArray): Open integer label store with shape (1, D, H, W), allowing ignore value -1.
            spatial_shape (Sequence[int]): Three spatial extents in stored axis order.

        Returns:
            tuple[int, int, int] | None: Source crop origin, or None if no eligible voxel is found.
        """

        selected: tuple[int, int, int] | None = None
        seen = 0
        for d in range(int(spatial_shape[0])):
            plane = np.asarray(label_store[(0, d, slice(None), slice(None))])
            _raise_for_invalid_labels(
                plane,
                num_classes=self.num_classes,
                allow_ignore=True,
                context="B2ND labels",
            )
            candidates = np.flatnonzero(plane > 0)
            count = int(candidates.size)
            if count == 0:
                continue
            if random.randrange(seen + count) < count:
                local = int(candidates[random.randrange(count)])
                h, w = divmod(local, int(spatial_shape[2]))
                selected = (d, h, w)
            seen += count
        if selected is None:
            return None
        return self._start_from_center(selected, spatial_shape)

    def _sample_background_start(
        self,
        label_store: Any,
        spatial_shape: Sequence[int],
    ) -> tuple[int, int, int] | None:
        """
        Try random source voxels labeled background or ignore.

        Accepted labels are zero and -1. The sampled source center is clamped to a valid crop origin.

        Args:
            label_store (blosc2.NDArray): Open integer label store with shape (1, D, H, W), allowing ignore value -1.
            spatial_shape (Sequence[int]): Three spatial extents in stored axis order.

        Returns:
            tuple[int, int, int] | None: Source crop origin, or None if no eligible voxel is found.
        """

        for _ in range(self.background_attempts):
            center = tuple(
                random.randrange(int(axis_size)) for axis_size in spatial_shape
            )
            value = int(np.asarray(label_store[(0, *center)]).reshape(()).item())
            if value in (-1, 0):
                return self._start_from_center(center, spatial_shape)
        return None

    def _sample_unconstrained_start(
        self,
        spatial_shape: Sequence[int],
    ) -> tuple[int, int, int]:
        """
        Sample a crop origin without consulting labels.

        Args:
            spatial_shape (Sequence[int]): Three spatial extents in stored axis order.

        Returns:
            tuple[int, int, int]: Source crop origin in stored spatial-axis order; undersized axes start at zero.
        """

        return cast(
            tuple[int, int, int],
            tuple(
                random.randint(0, max(0, int(actual) - requested))
                for actual, requested in zip(spatial_shape, self.crop_size)
            ),
        )

    def _start_from_center(
        self,
        center: Sequence[int],
        spatial_shape: Sequence[int],
    ) -> tuple[int, int, int]:
        """
        Clamp the crop around a proposed nnU-Net source center.

        Args:
            center (Sequence[int]): Source voxel coordinates in (D, H, W) order.
            spatial_shape (Sequence[int]): Three spatial extents in stored axis order.

        Returns:
            tuple[int, int, int]: Source crop origin in stored spatial-axis order; undersized axes start at zero.
        """

        return cast(
            tuple[int, int, int],
            tuple(
                0
                if int(actual) <= requested
                else min(
                    max(0, int(coordinate) - requested // 2),
                    int(actual) - requested,
                )
                for coordinate, actual, requested in zip(
                    center, spatial_shape, self.crop_size
                )
            ),
        )

    def _read_patch(
        self,
        volume_store: Any,
        label_store: Any,
        *,
        spatial_shape: Sequence[int],
        start: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Decode a channel-first nnU-Net patch and center-pad undersized axes.

        Args:
            volume_store (blosc2.NDArray): Open float32 image store with shape (C, D, H, W).
            label_store (blosc2.NDArray): Open integer label store with shape (1, D, H, W), allowing ignore value -1.
            spatial_shape (Sequence[int]): Three spatial extents in stored axis order.
            start (Sequence[int]): Source crop origin in (D, H, W) order.

        Returns:
            tuple[np.ndarray, np.ndarray]: Contiguous float32 image (D, H, W, C) and uint8 labels (D, H, W) at the crop size.
                Undersized axes receive centered zero padding; ignore labels -1 become background.
        """

        source_slices: list[slice] = []
        destination_slices: list[slice] = []
        for actual, requested, origin in zip(spatial_shape, self.crop_size, start):
            actual = int(actual)
            if actual >= requested:
                source_slices.append(slice(int(origin), int(origin) + requested))
                destination_slices.append(slice(0, requested))
            else:
                before = (requested - actual) // 2
                source_slices.append(slice(0, actual))
                destination_slices.append(slice(before, before + actual))

        source_index = (slice(None), *source_slices)
        volume_cdhw = np.asarray(volume_store[source_index])
        label_1dhw = np.asarray(label_store[source_index])
        volume_decoded = np.moveaxis(volume_cdhw, 0, -1).astype(np.float32, copy=False)
        label_decoded = _canonicalize_nnunet_label(
            label_1dhw,
            num_classes=self.num_classes,
        )

        if tuple(volume_decoded.shape[:3]) == self.crop_size:
            return np.ascontiguousarray(volume_decoded), np.ascontiguousarray(
                label_decoded
            )

        volume = np.zeros(self.crop_size + (self.in_channels,), dtype=np.float32)
        label = np.zeros(self.crop_size, dtype=np.uint8)
        destination_index = tuple(destination_slices)
        volume[destination_index + (slice(None),)] = volume_decoded
        label[destination_index] = label_decoded
        return volume, label


def _validate_nnunet_b2nd_stores(
    spec: CaseStorageSpec,
    volume_store: Any,
    label_store: Any,
    *,
    in_channels: int,
) -> tuple[int, int, int]:
    """
    Check nnU-Net array shapes and dtypes.

    Args:
        spec (CaseStorageSpec): Selected paths and backend for one case.
        volume_store (blosc2.NDArray): Open float32 image store with shape (C, D, H, W).
        label_store (blosc2.NDArray): Open integer label store with shape (1, D, H, W), allowing ignore value -1.
        in_channels (int): Required image channel count C.

    Returns:
        tuple[int, int, int]: Shared spatial shape (D, H, W).
    """

    spatial_shape = _validate_nnunet_b2nd_volume(
        spec, volume_store, in_channels=in_channels
    )
    label_shape = tuple(int(value) for value in label_store.shape)
    if len(label_shape) != 4 or label_shape[0] != 1:
        raise ValueError(
            f"B2ND label for {spec.case_id!r} must be 1DHW, got {label_shape}"
        )
    if spatial_shape != label_shape[1:]:
        raise ValueError(
            f"B2ND shape mismatch for {spec.case_id!r}: "
            f"{volume_store.shape} vs {label_shape}"
        )
    label_dtype = np.dtype(label_store.dtype)
    if not np.issubdtype(label_dtype, np.integer):
        raise TypeError(
            f"B2ND label for {spec.case_id!r} must be integer, got {label_store.dtype}"
        )
    return spatial_shape


def _canonicalize_nnunet_label(
    label_1dhw: np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    """
    Remove the nnU-Net label channel and replace ignore values with zero.

    Args:
        label_1dhw (np.ndarray): Integer labels with shape (1, D, H, W), allowing ignore value -1.
        num_classes (int): Segmentation class count, including background; labels use [0, num_classes).

    Returns:
        np.ndarray: Uint8 categorical labels with shape (D, H, W).
    """

    label = np.squeeze(label_1dhw, axis=0)
    _raise_for_invalid_labels(
        label, num_classes=num_classes, allow_ignore=True, context="B2ND labels"
    )
    return np.where(label == -1, 0, label).astype(np.uint8, copy=False)


def _validate_generic_b2nd_stores(
    spec: CaseStorageSpec,
    volume_store: Any,
    label_store: Any,
    *,
    in_channels: int,
) -> tuple[int, int, int]:
    """
    Check generic B2ND image/label alignment and dtypes.

    Args:
        spec (CaseStorageSpec): Selected paths and backend for one case.
        volume_store (blosc2.NDArray): Open float32 image store with shape (X, Y, Z, C).
        label_store (blosc2.NDArray): Open uint8 label store with shape (X, Y, Z).
        in_channels (int): Required image channel count C.

    Returns:
        tuple[int, int, int]: Shared spatial shape (X, Y, Z).
    """

    spatial_shape = _validate_generic_b2nd_volume(
        spec, volume_store, in_channels=in_channels
    )
    label_shape = tuple(int(value) for value in label_store.shape)
    if len(label_shape) != 3:
        raise ValueError(
            f"B2ND label for {spec.case_id!r} must be XYZ, got {label_shape}"
        )
    if spatial_shape != label_shape:
        raise ValueError(
            f"B2ND shape mismatch for {spec.case_id!r}: "
            f"{volume_store.shape} vs {label_shape}"
        )
    if np.dtype(label_store.dtype) != np.dtype(np.uint8):
        raise TypeError(
            f"B2ND label for {spec.case_id!r} must be uint8, got {label_store.dtype}"
        )
    return label_shape


def _read_generic_b2nd_patch(
    volume_store: Any,
    label_store: Any,
    *,
    spatial_shape: Sequence[int],
    crop_size: tuple[int, int, int],
    start: Sequence[int],
    in_channels: int,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode a generic B2ND patch and pad smaller axes with zeros.

    Args:
        volume_store (blosc2.NDArray): Open float32 image store with shape (X, Y, Z, C).
        label_store (blosc2.NDArray): Open uint8 label store with shape (X, Y, Z).
        spatial_shape (Sequence[int]): Three spatial extents in stored axis order.
        crop_size (tuple[int, int, int]): Requested (X, Y, Z) patch extents.
        start (Sequence[int]): Source crop origin in (X, Y, Z) order.
        in_channels (int): Required image channel count C.
        num_classes (int): Segmentation class count, including background; labels use [0, num_classes).

    Returns:
        tuple[np.ndarray, np.ndarray]: Contiguous float32 image (X, Y, Z, C) and uint8 labels (X, Y, Z). Spatial extents equal
            the crop size; undersized axes receive centered zero padding.
    """

    source_slices: list[slice] = []
    destination_slices: list[slice] = []
    for actual, requested, origin in zip(spatial_shape, crop_size, start):
        actual = int(actual)
        if actual >= requested:
            source_slices.append(slice(int(origin), int(origin) + requested))
            destination_slices.append(slice(0, requested))
        else:
            before = (requested - actual) // 2
            source_slices.append(slice(0, actual))
            destination_slices.append(slice(before, before + actual))

    spatial_index = tuple(source_slices)
    volume_decoded = np.asarray(volume_store[spatial_index + (slice(None),)])
    label_decoded = np.asarray(label_store[spatial_index])
    _validate_decoded_labels(label_decoded, num_classes=num_classes)

    if tuple(label_decoded.shape) == crop_size:
        return (
            np.ascontiguousarray(volume_decoded, dtype=np.float32),
            np.ascontiguousarray(label_decoded, dtype=np.uint8),
        )

    volume = np.zeros(crop_size + (int(in_channels),), dtype=np.float32)
    label = np.zeros(crop_size, dtype=np.uint8)
    destination_index = tuple(destination_slices)
    volume[destination_index + (slice(None),)] = volume_decoded
    label[destination_index] = label_decoded
    return volume, label


def _validate_decoded_labels(label: np.ndarray, *, num_classes: int) -> None:
    """
    Reject decoded generic B2ND labels outside the class range.

    Args:
        label (np.ndarray): Decoded labels of any shape.
        num_classes (int): Segmentation class count, including background; labels use [0, num_classes).

    Raises:
        ValueError: If a label is negative or reaches num_classes.
    """

    _raise_for_invalid_labels(
        label, num_classes=num_classes, allow_ignore=False, context="B2ND labels"
    )


def _sample_unconstrained_start(
    spatial_shape: Sequence[int],
    crop_size: Sequence[int],
) -> tuple[int, int, int]:
    """
    Draw each crop origin uniformly from its valid axis interval.

    Args:
        spatial_shape (Sequence[int]): Three spatial extents in stored axis order.
        crop_size (Sequence[int]): Three requested patch extents in stored axis order.

    Returns:
        tuple[int, int, int]: Source crop origin in stored spatial-axis order; undersized axes start at zero.
    """

    return cast(
        tuple[int, int, int],
        tuple(
            random.randint(0, max(0, int(actual) - int(requested)))
            for actual, requested in zip(spatial_shape, crop_size)
        ),
    )


def _padded_sampling_geometry(
    spatial_shape: Sequence[int],
    crop_size: Sequence[int],
) -> tuple[
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
]:
    """
    Compute the padded shape and valid crop-center intervals.

    Args:
        spatial_shape (Sequence[int]): Three spatial extents in stored axis order.
        crop_size (Sequence[int]): Three requested patch extents in stored axis order.

    Returns:
        tuple[tuple[int, int, int], ...]: (padded_shape, pad_before, lower, upper), each a three-axis integer tuple. Padding is
            centered; lower is inclusive and upper is exclusive.
    """

    padded_shape = tuple(
        max(int(actual), int(requested))
        for actual, requested in zip(spatial_shape, crop_size)
    )
    pad_before = tuple(
        (padded - int(actual)) // 2
        for actual, padded in zip(spatial_shape, padded_shape)
    )
    lower = tuple(int(requested) // 2 for requested in crop_size)
    upper = tuple(
        padded - (int(requested) - int(requested) // 2) + 1
        for padded, requested in zip(padded_shape, crop_size)
    )
    return (
        cast(tuple[int, int, int], padded_shape),
        cast(tuple[int, int, int], pad_before),
        cast(tuple[int, int, int], lower),
        cast(tuple[int, int, int], upper),
    )


def _start_from_padded_center(
    padded_center: Sequence[int],
    spatial_shape: Sequence[int],
    crop_size: Sequence[int],
) -> tuple[int, int, int]:
    """
    Convert a valid padded center into a source-array crop origin.

    Args:
        padded_center (Sequence[int]): Center coordinates after centered spatial padding.
        spatial_shape (Sequence[int]): Three spatial extents in stored axis order.
        crop_size (Sequence[int]): Three requested patch extents in stored axis order.

    Returns:
        tuple[int, int, int]: Source crop origin in stored spatial-axis order; undersized axes start at zero.
    """

    return cast(
        tuple[int, int, int],
        tuple(
            0 if int(actual) < int(requested) else int(center) - int(requested) // 2
            for actual, requested, center in zip(
                spatial_shape,
                crop_size,
                padded_center,
            )
        ),
    )


def _start_from_center(
    center: Sequence[int],
    spatial_shape: Sequence[int],
    crop_size: Sequence[int],
) -> tuple[int, int, int]:
    """
    Clamp a source-centered crop to each valid axis interval.

    Args:
        center (Sequence[int]): Proposed center in source spatial coordinates.
        spatial_shape (Sequence[int]): Three spatial extents in stored axis order.
        crop_size (Sequence[int]): Three requested patch extents in stored axis order.

    Returns:
        tuple[int, int, int]: Source crop origin in stored spatial-axis order; undersized axes start at zero.
    """

    return cast(
        tuple[int, int, int],
        tuple(
            0
            if int(actual) <= int(requested)
            else min(
                max(0, int(coordinate) - int(requested) // 2),
                int(actual) - int(requested),
            )
            for coordinate, actual, requested in zip(center, spatial_shape, crop_size)
        ),
    )


def _unravel_flat_index(
    flat_index: int,
    spatial_shape: Sequence[int],
) -> tuple[int, int, int]:
    """
    Convert a C-order flat voxel index to three spatial coordinates.

    Args:
        flat_index (int): Index into the flattened label array.
        spatial_shape (Sequence[int]): Three spatial extents in stored axis order.

    Returns:
        tuple[int, int, int]: Coordinates in the same spatial-axis order as spatial_shape.
    """

    _, y_extent, z_extent = (int(value) for value in spatial_shape)
    yz_extent = y_extent * z_extent
    x = int(flat_index) // yz_extent
    remainder = int(flat_index) - x * yz_extent
    y = remainder // z_extent
    z = remainder - y * z_extent
    return x, y, z


def _validate_foreground_indices(
    indices: np.ndarray,
    *,
    voxel_count: int,
    path: Path,
) -> None:
    """
    Validate sorted uint32 indices in bounded-size chunks.

    Args:
        indices (np.ndarray): Foreground sidecar expected to have shape (N,) and uint32 dtype; empty arrays are valid.
        voxel_count (int): Positive label-array size representable by uint32 voxel indices.
        path (Path): Sidecar path used in validation errors.

    Raises:
        ValueError: If rank, dtype, representable size, ordering, or index bounds are invalid.
    """

    if indices.ndim != 1 or indices.dtype != np.dtype(np.uint32):
        raise ValueError(
            f"Foreground-index sidecar must be one-dimensional uint32: {path}"
        )
    if voxel_count <= 0 or voxel_count - 1 > np.iinfo(np.uint32).max:
        raise ValueError(
            f"Foreground-index uint32 contract cannot represent {voxel_count} voxels"
        )
    previous: int | None = None
    block_size = 1_000_000
    for start in range(0, int(indices.size), block_size):
        stop = start + block_size
        block = np.asarray(indices[start:stop])
        first = int(block[0])
        last = int(block[-1])
        if previous is not None and first <= previous:
            raise ValueError(
                f"Foreground-index sidecar must be strictly increasing: {path}"
            )
        if np.any(block[1:] <= block[:-1]):
            raise ValueError(
                f"Foreground-index sidecar must be strictly increasing: {path}"
            )
        if last >= int(voxel_count):
            raise ValueError(
                f"Foreground-index sidecar exceeds the label shape: {path}"
            )
        previous = last


def _storage_conventions(case_id: str) -> tuple[_StorageConvention, ...]:
    """
    Build the supported relative filenames for a case.

    Args:
        case_id (str): Processed case identifier.

    Returns:
        tuple[_StorageConvention, ...]: NPY, generic B2ND, and two nnU-Net directory conventions.
    """

    support = Path("support") / f"{case_id}_support.npy"
    return (
        _StorageConvention(
            backend=NPY_BACKEND,
            volume=Path("vol") / f"{case_id}_vol.npy",
            label=Path("seg") / f"{case_id}_seg.npy",
            support=support,
        ),
        _StorageConvention(
            backend=B2ND_BACKEND,
            volume=Path("vol") / f"{case_id}_vol.b2nd",
            label=Path("seg") / f"{case_id}_seg.b2nd",
            support=support,
            foreground_index=Path("sampling") / f"{case_id}_foreground.npy",
        ),
        _StorageConvention(
            backend=NNUNET_B2ND_BACKEND,
            volume=Path("b2nd") / f"{case_id}.b2nd",
            label=Path("b2nd") / f"{case_id}_seg.b2nd",
            properties=Path("b2nd") / f"{case_id}.pkl",
        ),
        _StorageConvention(
            backend=NNUNET_B2ND_BACKEND,
            volume=Path("nnUNetPlans_3d_fullres") / f"{case_id}.b2nd",
            label=Path("nnUNetPlans_3d_fullres") / f"{case_id}_seg.b2nd",
            properties=Path("nnUNetPlans_3d_fullres") / f"{case_id}.pkl",
        ),
    )


def _artifact_present(path: Path) -> bool:
    """
    Detect files and symlinks when matching storage layouts.

    Args:
        path (Path): Candidate artifact path.

    Returns:
        bool: True for a regular file or symlink, including a broken symlink.
    """

    return path.is_file() or path.is_symlink()


def _resolve_discovered_path(
    root: Path,
    relative: Path,
    *,
    case_id: str,
    kind: str,
) -> Path:
    """
    Resolve a child artifact while rejecting symlinks below the dataset root.

    Args:
        root (Path): Dataset root, which may itself be a symlink.
        relative (Path): Artifact path relative to root.
        case_id (str): Case identifier used in validation errors.
        kind (str): Artifact description used in validation errors.

    Returns:
        Path: Resolved absolute artifact path; existence is validated separately.
    """

    candidate = root / relative
    child = root
    for part in relative.parts:
        child = child / part
        if child.is_symlink():
            raise ValueError(
                f"Discovered {kind} for case {case_id!r} must not use child symlinks"
            )
    return candidate.resolve()


def _resolve_optional_discovered_path(
    root: Path,
    relative: Path | None,
    *,
    case_id: str,
    kind: str,
) -> Path | None:
    """
    Resolve an optional artifact when its path is present.

    Args:
        root (Path): Dataset root.
        relative (Path | None): Optional artifact path relative to root.
        case_id (str): Case identifier used in validation errors.
        kind (str): Artifact description used in validation errors.

    Returns:
        Path | None: Resolved path, or None when the convention or artifact is absent.
    """

    if relative is None or not _artifact_present(root / relative):
        return None
    return _resolve_discovered_path(
        root,
        relative,
        case_id=case_id,
        kind=kind,
    )


def _validate_case_storage_spec(
    spec: CaseStorageSpec,
    *,
    handles: ReadOnlyB2NDHandleCache,
    expected_channels: int,
    require_labels: bool,
    num_classes: int | None,
) -> tuple[int, int, int]:
    """
    Validate one case using its selected backend.

    Args:
        spec (CaseStorageSpec): Selected paths and backend for one case.
        handles (ReadOnlyB2NDHandleCache): Shared read-only compressed-array cache for validation.
        expected_channels (int): Required image channel count.
        require_labels (bool): Reject cases without a segmentation file.
        num_classes (int | None): Exclusive label upper bound; None skips label-value checks.

    Returns:
        tuple[int, int, int]: Case spatial extents in stored axis order.
    """

    if not spec.volume_path.is_file():
        raise ValueError(f"missing volume file {spec.volume_path}")
    if require_labels and spec.label_path is None:
        raise ValueError("supervised case has no label file")
    if spec.label_path is not None and not spec.label_path.is_file():
        raise ValueError(f"missing label file {spec.label_path}")
    for kind, path in (
        ("support", spec.support_path),
        ("properties", spec.properties_path),
        ("foreground index", spec.foreground_index_path),
    ):
        if path is not None and not path.is_file():
            raise ValueError(f"missing {kind} file {path}")

    if spec.backend == NPY_BACKEND:
        return _validate_npy_case(
            spec,
            expected_channels=expected_channels,
            num_classes=num_classes,
        )

    volume_store = handles.get(spec.volume_path)
    if spec.backend == B2ND_BACKEND:
        if spec.label_path is None:
            if spec.foreground_index_path is not None:
                raise ValueError("unlabeled B2ND case has a foreground index")
            return _validate_generic_b2nd_volume(
                spec,
                volume_store,
                in_channels=expected_channels,
            )
        label_store = handles.get(spec.label_path)
        spatial_shape = _validate_generic_b2nd_stores(
            spec,
            volume_store,
            label_store,
            in_channels=expected_channels,
        )
        _validate_store_label_values(
            label_store,
            spatial_shape=spatial_shape,
            num_classes=num_classes,
            nnunet=False,
        )
        if spec.foreground_index_path is not None:
            indices = np.load(
                spec.foreground_index_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            try:
                _validate_foreground_indices(
                    indices,
                    voxel_count=math.prod(spatial_shape),
                    path=spec.foreground_index_path,
                )
            finally:
                _close_numpy_mapping(indices)
        return spatial_shape

    if spec.backend == NNUNET_B2ND_BACKEND:
        if spec.label_path is None:
            return _validate_nnunet_b2nd_volume(
                spec,
                volume_store,
                in_channels=expected_channels,
            )
        label_store = handles.get(spec.label_path)
        spatial_shape = _validate_nnunet_b2nd_stores(
            spec,
            volume_store,
            label_store,
            in_channels=expected_channels,
        )
        _validate_store_label_values(
            label_store,
            spatial_shape=spatial_shape,
            num_classes=num_classes,
            nnunet=True,
        )
        return spatial_shape

    raise ValueError(f"unsupported storage backend {spec.backend!r}")


def _validate_npy_case(
    spec: CaseStorageSpec,
    *,
    expected_channels: int,
    num_classes: int | None,
) -> tuple[int, int, int]:
    """
    Check NPY image/label layout and release temporary mappings.

    Args:
        spec (CaseStorageSpec): Selected paths and backend for one case.
        expected_channels (int): Required last-axis image channel count.
        num_classes (int | None): Exclusive label upper bound; None skips label-value checks.

    Returns:
        tuple[int, int, int]: Image spatial extents (X, Y, Z). Images must be floating point; present labels must be aligned
            integer maps.
    """

    volume: np.ndarray | None = None
    label: np.ndarray | None = None
    try:
        volume = cast(
            np.ndarray, np.load(spec.volume_path, mmap_mode="r", allow_pickle=False)
        )
        if volume.ndim != 4 or volume.shape[-1] != expected_channels:
            raise ValueError(
                "invalid volume shape: expected XYZC with "
                f"C={expected_channels}, got {volume.shape}"
            )
        if not np.issubdtype(volume.dtype, np.floating):
            raise TypeError(f"volume must be floating point, got {volume.dtype}")
        spatial_shape = cast(
            tuple[int, int, int], tuple(int(value) for value in volume.shape[:3])
        )
        if spec.label_path is None:
            return spatial_shape
        label = cast(
            np.ndarray, np.load(spec.label_path, mmap_mode="r", allow_pickle=False)
        )
        if label.shape != spatial_shape:
            raise ValueError(
                f"volume/label shape mismatch: {volume.shape} vs {label.shape}"
            )
        if not np.issubdtype(label.dtype, np.integer):
            raise TypeError(f"label must be integer, got {label.dtype}")
        _validate_numpy_label_values(label, num_classes=num_classes)
        return spatial_shape
    finally:
        _close_numpy_mapping(label)
        _close_numpy_mapping(volume)


def _validate_generic_b2nd_volume(
    spec: CaseStorageSpec,
    volume_store: Any,
    *,
    in_channels: int,
) -> tuple[int, int, int]:
    """
    Check a generic B2ND image for XYZC float32 storage.

    Args:
        spec (CaseStorageSpec): Selected paths and backend for one case.
        volume_store (blosc2.NDArray): Open float32 image store with shape (X, Y, Z, C).
        in_channels (int): Required image channel count C.

    Returns:
        tuple[int, int, int]: Image spatial shape (X, Y, Z).
    """

    volume_shape = tuple(int(value) for value in volume_store.shape)
    if len(volume_shape) != 4 or volume_shape[-1] != int(in_channels):
        raise ValueError(
            f"B2ND volume for {spec.case_id!r} must be XYZC with "
            f"C={int(in_channels)}, got {volume_shape}"
        )
    if np.dtype(volume_store.dtype) != np.dtype(np.float32):
        raise TypeError(
            f"B2ND volume for {spec.case_id!r} must be float32, "
            f"got {volume_store.dtype}"
        )
    return volume_shape[:3]


def _validate_nnunet_b2nd_volume(
    spec: CaseStorageSpec,
    volume_store: Any,
    *,
    in_channels: int,
) -> tuple[int, int, int]:
    """
    Check an nnU-Net B2ND image for CDHW float32 storage.

    Args:
        spec (CaseStorageSpec): Selected paths and backend for one case.
        volume_store (blosc2.NDArray): Open float32 image store with shape (C, D, H, W).
        in_channels (int): Required image channel count C.

    Returns:
        tuple[int, int, int]: Image spatial shape (D, H, W).
    """

    volume_shape = tuple(int(value) for value in volume_store.shape)
    if len(volume_shape) != 4 or volume_shape[0] != int(in_channels):
        raise ValueError(
            f"B2ND volume for {spec.case_id!r} must be CDHW with "
            f"C={int(in_channels)}, got {volume_shape}"
        )
    if np.dtype(volume_store.dtype) != np.dtype(np.float32):
        raise TypeError(
            f"B2ND volume for {spec.case_id!r} must be float32, "
            f"got {volume_store.dtype}"
        )
    return volume_shape[1:]


def _validate_numpy_label_values(
    label: np.ndarray,
    *,
    num_classes: int | None,
) -> None:
    """
    Check class bounds in slabs of at most 32 label planes.

    Args:
        label (np.ndarray): Integer label map with shape (X, Y, Z).
        num_classes (int | None): Exclusive label upper bound; None skips label-value checks.
    """

    if num_classes is None:
        return
    for start in range(0, int(label.shape[0]), 32):
        stop = start + 32
        block = np.asarray(label[start:stop])
        _raise_for_invalid_labels(block, num_classes=num_classes, allow_ignore=False)


def _validate_store_label_values(
    label_store: Any,
    *,
    spatial_shape: Sequence[int],
    num_classes: int | None,
    nnunet: bool,
) -> None:
    """
    Decode label slabs and check the configured categorical range.

    Args:
        label_store (blosc2.NDArray): Generic labels (X, Y, Z) or nnU-Net labels (1, D, H, W).
        spatial_shape (Sequence[int]): Three spatial extents in stored axis order.
        num_classes (int | None): Exclusive label upper bound; None skips label-value checks.
        nnunet (bool): Select channel-first labels and allow nnU-Net ignore value -1.
    """

    if num_classes is None:
        return
    for start in range(0, int(spatial_shape[0]), 32):
        stop = min(start + 32, int(spatial_shape[0]))
        index = (
            (0, slice(start, stop), slice(None), slice(None))
            if nnunet
            else (slice(start, stop), slice(None), slice(None))
        )
        block = np.asarray(label_store[index])
        _raise_for_invalid_labels(
            block,
            num_classes=num_classes,
            allow_ignore=nnunet,
        )


def _raise_for_invalid_labels(
    label: np.ndarray,
    *,
    num_classes: int,
    allow_ignore: bool,
    context: str = "labels",
) -> None:
    """
    Reject label values outside the configured interval.

    Args:
        label (np.ndarray): Label values of any shape.
        num_classes (int): Segmentation class count, including background; labels use [0, num_classes).
        allow_ignore (bool): Use -1 as the lower bound when true; otherwise use zero.
        context (str): Label-source description used in validation errors.

    Raises:
        ValueError: If a value falls below the allowed lower bound or reaches num_classes.
    """

    minimum = -1 if allow_ignore else 0
    invalid = (label < minimum) | (label >= int(num_classes))
    if np.any(invalid):
        values = np.unique(label[invalid]).tolist()
        expected = (
            f"-1 or [0, {int(num_classes) - 1}]"
            if allow_ignore
            else f"[0, {int(num_classes) - 1}]"
        )
        raise ValueError(f"Invalid {context} {values}; expected {expected}")


def _case_id(value: object, *, context: str) -> str:
    """
    Validate an identifier before constructing case artifact paths.

    Args:
        value (object): Unvalidated identifier; valid values are nonempty strings without path components or surrounding
            whitespace.
        context (str): Identifier-source description used in validation errors.

    Returns:
        str: Validated case identifier.
    """

    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or Path(value).name != value
        or value in {".", ".."}
    ):
        raise ValueError(f"{context} has an invalid case_id")
    return value


def _foreground_class_key(value: object) -> bool:
    """
    Determine whether an nnU-Net class key contains a positive label ID.

    Args:
        value (object): Class key to inspect; Python/NumPy integers and nested lists or tuples are recognized.

    Returns:
        bool: True when a positive integer ID is present; booleans and unsupported values return false.
    """

    match value:
        case bool():
            return False
        case int() | np.integer():
            return int(value) > 0
        case tuple() | list():
            return any(_foreground_class_key(component) for component in value)
        case _:
            return False


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    """Restrict nnU-Net properties pickles to the required NumPy globals."""

    def find_class(self, module: str, name: str) -> Any:
        """
        Resolve only the dtype and array-buffer reconstruction globals.

        Args:
            module (str): Module name requested by the pickle stream.
            name (str): Global name requested by the pickle stream.

        Returns:
            type[np.dtype] | Callable: NumPy dtype class or the permitted _frombuffer function.

        Raises:
            pickle.UnpicklingError: If the stream requests any other global.
        """

        match (module, name):
            case ("numpy", "dtype"):
                return np.dtype
            case ("numpy._core.numeric", "_frombuffer"):
                return getattr(np, "_core").numeric._frombuffer
            case _:
                raise pickle.UnpicklingError(
                    f"Forbidden global in nnU-Net properties pickle: {module}.{name}"
                )


def _normalize_spatial_size(
    value: int | Sequence[int] | None,
) -> tuple[int, int, int] | None:
    """
    Normalize a cubic size or three-axis sequence.

    Args:
        value (int | Sequence[int] | None): Spatial size to normalize, or None to leave it unspecified.

    Returns:
        tuple[int, int, int] | None: Three positive integer extents, or None.

    Raises:
        ValueError: If the normalized size does not contain exactly three positive axes.
    """

    if value is None:
        return None
    normalized = (
        (int(value),) * 3 if isinstance(value, int) else tuple(int(v) for v in value)
    )
    if len(normalized) != 3 or any(axis <= 0 for axis in normalized):
        raise ValueError(
            f"spatial size must contain three positive values, got {normalized}"
        )
    return normalized


def _reader_dimensions(in_channels: int, num_classes: int) -> tuple[int, int]:
    """
    Validate channel and class counts for uint8 B2ND readers.

    Args:
        in_channels (int): Required image channel count C.
        num_classes (int): Segmentation class count, including background; labels use [0, num_classes).

    Returns:
        tuple[int, int]: (in_channels, num_classes) as integers, with positive channels and 2 to 256 classes.

    Raises:
        ValueError: If channels are nonpositive or the class count falls outside [2, 256].
    """

    channels, classes = int(in_channels), int(num_classes)
    if channels <= 0:
        raise ValueError(f"in_channels must be positive, got {in_channels}")
    if not 2 <= classes <= 256:
        raise ValueError(
            f"B2ND categorical labels require num_classes in [2, 256], got {num_classes}"
        )
    return channels, classes


def _close_if_supported(handle: Any) -> None:
    """
    Call an object's close method when it is available.

    Args:
        handle (object): Storage handle or None; objects without a callable close are ignored.
    """

    close = getattr(handle, "close", None)
    if callable(close):
        close()


def _close_numpy_mapping(array: Any) -> None:
    """
    Release the backing mmap of a NumPy array when present.

    Args:
        array (np.ndarray | None): NumPy array or memory map; ordinary arrays and None require no action.
    """

    _close_if_supported(getattr(array, "_mmap", None))


__all__ = [
    "B2ND_BACKEND",
    "B2NDFullCaseReader",
    "B2NDPatchReader",
    "CaseStorageSpec",
    "ForegroundIndexCache",
    "NNUNET_B2ND_BACKEND",
    "NPY_BACKEND",
    "NnUNetClassLocationsCache",
    "NnUNetB2NDFullCaseReader",
    "NnUNetB2NDPatchReader",
    "ReadOnlyB2NDHandleCache",
    "discover_case_storage_specs",
    "validate_case_storage_specs",
]
