import random
import warnings
from collections import OrderedDict
from typing import Sequence, cast

import numpy as np
from scipy.ndimage import rotate

ImagePair = list[np.ndarray]
TransformInput = np.ndarray | ImagePair


class Base:
    """Identity transform with a shared sampling interface for aligned arrays."""

    def sample(self, *shape: int) -> list[int]:
        """
        Preserve the spatial shape for the identity transform.

        Args:
            *shape (int): Leading spatial extents, normally X, Y, and Z.

        Returns:
            list[int]: Input extents in their original order.
        """
        return list(shape)

    def tf(self, img: np.ndarray, k: int = 0) -> np.ndarray:
        """
        Return one array without applying a transformation.

        Args:
            img (np.ndarray): Image with shape (X, Y, Z, C) or label map with shape (X, Y, Z).
            k (int): Position in the image/label pair; unused by this operation.

        Returns:
            np.ndarray: The original array with unchanged shape and dtype.
        """
        return img

    def __call__(
        self, img: TransformInput, dim: int = 3, reuse: bool = False
    ) -> TransformInput:
        """
        Sample parameters once and apply them to an array or aligned pair.

        Args:
            img (TransformInput): Image array (X, Y, Z, C) or aligned [image, label] pair with labels (X, Y, Z).
            dim (int): Number of leading axes used to sample spatial parameters.
            reuse (bool): Reuse sampled spatial parameters; intensity changes still draw new factors when applied.

        Returns:
            TransformInput: Transformed array or list of transformed arrays, in input order. Subclasses determine the output
                shape and dtype.
        """
        is_array = isinstance(img, np.ndarray)
        if not reuse:
            im: np.ndarray = img if is_array else img[0]
            shape = im.shape[:dim]
            self.sample(*shape)

        if not is_array:
            return [self.tf(x, k) for k, x in enumerate(img)]

        return self.tf(img)

    def __str__(self) -> str:
        """
        Describe the identity transform.

        Returns:
            str: Identity transform name.
        """

        return "Identity()"


Identity = Base


class Compose(Base):
    """Apply paired NumPy transforms in sequence."""

    def __init__(self, ops: Base | Sequence[Base]) -> None:
        """
        Args:
            ops (Base | Sequence[Base]): One transform or a sequence in application order.
        """

        self.ops: tuple[Base, ...] = (ops,) if isinstance(ops, Base) else tuple(ops)

    def sample(self, *shape: int) -> list[int]:
        """
        Sample each transform using the preceding output shape.

        Args:
            *shape (int): Leading spatial extents, normally X, Y, and Z.

        Returns:
            list[int]: Spatial extents after all transforms.
        """

        shape_list = list(shape)
        for op in self.ops:
            shape_list = op.sample(*shape_list)
        return shape_list

    def tf(self, img: np.ndarray, k: int = 0) -> np.ndarray:
        """
        Apply every transform to one member of a sampled pair.

        Args:
            img (np.ndarray): Image with shape (X, Y, Z, C) or label map with shape (X, Y, Z).
            k (int): Pair position passed to each transform.

        Returns:
            np.ndarray: Array after all transforms; their order determines shape and dtype.
        """

        result = img
        for op in self.ops:
            result = op.tf(result, k)
        return result

    def __call__(
        self, img: TransformInput, dim: int = 3, reuse: bool = False
    ) -> TransformInput:
        """
        Run each transform on the current array or aligned pair.

        Args:
            img (TransformInput): Image array (X, Y, Z, C) or aligned [image, label] pair with labels (X, Y, Z).
            dim (int): Number of leading axes used to sample spatial parameters.
            reuse (bool): Reuse sampled spatial parameters; intensity changes still draw new factors when applied.

        Returns:
            TransformInput: Transformed image or image/label list.
        """

        result = img
        for op in self.ops:
            result = op(result, dim=dim, reuse=reuse)
        return result

    def __str__(self) -> str:
        """
        Describe the ordered transform pipeline.

        Returns:
            str: Transform names and configurations in application order.
        """

        ops = ", ".join([str(op) for op in self.ops])
        return f"Compose([{ops}])"


class RandCrop3D(Base):
    """Crop aligned arrays at a uniformly sampled spatial origin."""

    def __init__(self, size: int | tuple[int, int, int] | list[int]) -> None:
        """
        Args:
            size (int | tuple[int, int, int] | list[int]): Positive cubic extent or three patch extents in (X, Y, Z) order.
        """

        self.size = [size, size, size] if isinstance(size, int) else list(size)
        if len(self.size) != 3 or any(int(value) <= 0 for value in self.size):
            raise ValueError(
                f"Crop size must contain three positive values, got {self.size}"
            )
        self.size = [int(value) for value in self.size]
        self.start: list[int] | None = None

    def sample(self, *shape: int) -> list[int]:
        """
        Choose a crop origin within each spatial axis.

        Args:
            *shape (int): Exactly three input extents in (X, Y, Z) order.

        Returns:
            list[int]: Configured crop extents in (X, Y, Z) order.

        Raises:
            ValueError: If the input is not three-dimensional or is smaller than the crop.
        """

        if len(shape) != 3:
            raise ValueError(f"Expected three spatial dimensions, got {shape}")
        if any(actual < requested for actual, requested in zip(shape, self.size)):
            raise ValueError(
                f"Crop size {tuple(self.size)} exceeds spatial shape {tuple(shape)}"
            )
        self.start = [
            random.randint(0, actual - requested)
            for actual, requested in zip(shape, self.size)
        ]
        return self.size.copy()

    def tf(self, img: np.ndarray, k: int = 0) -> np.ndarray:
        """
        Slice one array at the sampled crop origin.

        Args:
            img (np.ndarray): Image with shape (X, Y, Z, C) or label map with shape (X, Y, Z).
            k (int): Position in the image/label pair; unused by this operation.

        Returns:
            np.ndarray: View with spatial shape tuple(self.size), preserving trailing channels and dtype.
        """

        assert self.start is not None
        slices = tuple(
            slice(start, start + size) for start, size in zip(self.start, self.size)
        )
        return img[slices]

    def __str__(self) -> str:
        """
        Describe the configured random crop.

        Returns:
            str: Crop transform name and spatial extents.
        """

        return f"RandCrop3D({self.size})"


class PadToSize3D(Base):
    """Center-pad aligned arrays with zeros to meet a minimum spatial size."""

    def __init__(self, size: int | tuple[int, int, int] | list[int]) -> None:
        """
        Args:
            size (int | tuple[int, int, int] | list[int]): Positive cubic minimum or three minimum extents in (X, Y, Z) order.
        """

        self.size = [size, size, size] if isinstance(size, int) else list(size)
        if len(self.size) != 3 or any(int(value) <= 0 for value in self.size):
            raise ValueError(
                f"Padding size must contain three positive values, got {self.size}"
            )
        self.size = [int(value) for value in self.size]
        self.pad_width: tuple[tuple[int, int], ...] | None = None

    def sample(self, *shape: int) -> list[int]:
        """
        Compute padding for each undersized spatial axis.

        Odd padding totals place the extra voxel at the end of the axis.

        Args:
            *shape (int): Exactly three positive input extents in (X, Y, Z) order.

        Returns:
            list[int]: Per-axis maximum of the input and requested extents.

        Raises:
            ValueError: If there are not exactly three positive spatial extents.
        """

        if len(shape) != 3 or any(int(value) <= 0 for value in shape):
            raise ValueError(f"Expected three positive spatial dimensions, got {shape}")

        spatial_padding: list[tuple[int, int]] = []
        padded_shape: list[int] = []
        for actual, requested in zip(shape, self.size):
            total = max(0, requested - int(actual))
            before = total // 2
            spatial_padding.append((before, total - before))
            padded_shape.append(max(int(actual), requested))
        self.pad_width = tuple(spatial_padding)
        return padded_shape

    def tf(self, img: np.ndarray, k: int = 0) -> np.ndarray:
        """
        Apply the sampled zero padding to an image or label.

        Args:
            img (np.ndarray): Image with shape (X, Y, Z, C) or label map with shape (X, Y, Z).
            k (int): Position in the image/label pair; unused by this operation.

        Returns:
            np.ndarray: Array with padded spatial axes and unchanged channels and dtype; returns img when no padding is needed.

        Raises:
            ValueError: If the input rank is neither three nor four.
        """

        if img.ndim not in (3, 4):
            raise ValueError(f"Expected an XYZ or XYZC array, got shape {img.shape}")
        assert self.pad_width is not None
        pad_width = self.pad_width + (((0, 0),) if img.ndim == 4 else ())
        if not any(before or after for before, after in self.pad_width):
            return img
        return np.pad(img, pad_width, mode="constant", constant_values=0)

    def __str__(self) -> str:
        """
        Describe the minimum spatial size.

        Returns:
            str: Padding transform name and requested extents.
        """

        return f"PadToSize3D({self.size})"


class RandCrop3DByPosNegLabel(RandCrop3D):
    """Sample paired crops using foreground and background label voxels."""

    def __init__(
        self,
        size: int | tuple[int, int, int] | list[int],
        pos: float = 1.0,
        neg: float = 1.0,
        image_threshold: float | None = None,
        allow_smaller: bool = False,
        fg_cache_size: int = 32,
        bg_sample_attempts: int = 512,
        bg_fallback_chunk_voxels: int = 1_000_000,
    ) -> None:
        """
        Args:
            size (int | tuple[int, int, int] | list[int]): Positive cubic extent or three patch extents in (X, Y, Z) order.
            pos (float): Nonnegative foreground weight; pos + neg must be positive.
            neg (float): Nonnegative background weight; foreground is requested with probability pos / (pos + neg).
            image_threshold (float | None): Optional threshold that at least one image channel must exceed at a background
                sampling voxel.
            allow_smaller (bool): Shrink crop axes when the case is smaller than the requested size.
            fg_cache_size (int): Maximum cached foreground-index arrays; zero disables caching.
            bg_sample_attempts (int): Maximum background rejection draws before an exact search; clamped to at least one.
            bg_fallback_chunk_voxels (int): Maximum voxels examined per exact-search chunk; clamped to at least one.
        """

        super().__init__(size=size)
        if pos < 0 or neg < 0:
            raise ValueError(
                f"pos and neg must be nonnegative, got pos={pos} neg={neg}."
            )
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.requested_size = self.size.copy()
        self.pos_ratio = pos / (pos + neg)
        self.image_threshold = image_threshold
        self.allow_smaller = bool(allow_smaller)
        self.fg_cache_size = max(0, int(fg_cache_size))
        self.bg_sample_attempts = max(1, int(bg_sample_attempts))
        self.bg_fallback_chunk_voxels = max(1, int(bg_fallback_chunk_voxels))
        self._fg_cache: OrderedDict[tuple[str, tuple[int, ...], str], np.ndarray] = (
            OrderedDict()
        )
        self._warned: set[str] = set()

    def _resolve_crop_size(self, shape: Sequence[int]) -> list[int]:
        """
        Adjust the requested crop to the available spatial shape.

        Args:
            shape (Sequence[int]): Three spatial extents in (X, Y, Z) order.

        Returns:
            list[int]: Effective (X, Y, Z) crop extents.

        Raises:
            ValueError: If an input axis is too small and allow_smaller is false.
        """

        crop_size = self.requested_size.copy()
        for idx, (requested, actual) in enumerate(zip(crop_size, shape)):
            if actual < requested:
                if not self.allow_smaller:
                    raise ValueError(
                        "The size of the proposed random crop ROI is larger than the image size, "
                        f"got ROI size {tuple(self.requested_size)} and label image size {tuple(shape)} respectively."
                    )
                crop_size[idx] = actual
        return crop_size

    def _center_bounds(self, shape: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the interval of crop centers that keep the crop in bounds.

        Args:
            shape (Sequence[int]): Three spatial extents in (X, Y, Z) order.

        Returns:
            tuple[np.ndarray, np.ndarray]: (lower, upper): int64 arrays of shape (3,), with inclusive lower bounds and exclusive
                upper bounds.
        """

        crop_size = np.asarray(self.size, dtype=np.int64)
        spatial_shape = np.asarray(shape, dtype=np.int64)
        valid_start = np.floor_divide(crop_size, 2)
        valid_end = np.subtract(spatial_shape + 1, crop_size / 2.0).astype(np.int64)
        valid_end = np.maximum(valid_end, valid_start + 1)
        return valid_start, valid_end

    def _correct_crop_center(
        self, center: Sequence[int], shape: Sequence[int]
    ) -> list[int]:
        """
        Clamp a proposed center to the crop-center interval.

        Args:
            center (Sequence[int]): Proposed voxel coordinates in (X, Y, Z) order.
            shape (Sequence[int]): Three spatial extents in (X, Y, Z) order.

        Returns:
            list[int]: In-bounds crop center, which may differ from the sampled voxel.
        """

        valid_start, valid_end = self._center_bounds(shape)
        return [
            int(min(max(coord, start), end - 1))
            for coord, start, end in zip(center, valid_start, valid_end)
        ]

    @staticmethod
    def _unravel_index3(flat_index: int, shape: Sequence[int]) -> list[int]:
        """
        Convert a C-order flat index to three voxel coordinates.

        Args:
            flat_index (int): Flattened voxel index.
            shape (Sequence[int]): Three spatial extents in (X, Y, Z) order.

        Returns:
            list[int]: Voxel coordinates in (X, Y, Z) order.
        """

        flat_index = int(flat_index)
        width_depth = int(shape[1]) * int(shape[2])
        h = flat_index // width_depth
        rem = flat_index - h * width_depth
        w = rem // int(shape[2])
        d = rem - w * int(shape[2])
        return [int(h), int(w), int(d)]

    def _label_cache_key(
        self,
        label: np.ndarray,
    ) -> tuple[str, tuple[int, ...], str] | None:
        """
        Identify a file-backed label by filename, shape, and dtype.

        Args:
            label (np.ndarray): Categorical label map with shape (X, Y, Z).

        Returns:
            tuple[str, tuple[int, ...], str] | None: Cache key, or None when the array has no filename metadata.
        """

        filename = getattr(label, "filename", None)
        if filename is None:
            return None
        return (str(filename), tuple(int(x) for x in label.shape), str(label.dtype))

    def _foreground_indices(self, label: np.ndarray) -> np.ndarray:
        """
        Find nonzero label voxels and cache file-backed results.

        Args:
            label (np.ndarray): Categorical label map with shape (X, Y, Z).

        Returns:
            np.ndarray: C-order flat indices with shape (N,). Uses uint32 when the label size fits, otherwise np.intp.
        """

        key = self._label_cache_key(label)
        if key is not None:
            cached = self._fg_cache.pop(key, None)
            if cached is not None:
                self._fg_cache[key] = cached
                return cached

        fg_indices = np.flatnonzero(label != 0)
        if label.size <= np.iinfo(np.uint32).max:
            fg_indices = fg_indices.astype(np.uint32, copy=False)

        if key is not None and self.fg_cache_size > 0:
            self._fg_cache[key] = fg_indices
            while len(self._fg_cache) > self.fg_cache_size:
                self._fg_cache.popitem(last=False)

        return fg_indices

    def _passes_image_threshold(
        self,
        image: np.ndarray,
        center: tuple[int, int, int],
    ) -> bool:
        """
        Check image eligibility at a proposed background voxel.

        Args:
            image (np.ndarray): Image with shape (X, Y, Z, C).
            center (tuple[int, int, int]): Voxel coordinates in (X, Y, Z) order.

        Returns:
            bool: True if thresholding is disabled or at least one channel exceeds the threshold.
        """

        return self.image_threshold is None or bool(
            np.any(image[center] > self.image_threshold)
        )

    def _sample_positive_center(
        self,
        fg_indices: np.ndarray,
        shape: Sequence[int],
    ) -> list[int]:
        """
        Select a foreground voxel uniformly and clamp its crop center.

        Args:
            fg_indices (np.ndarray): Nonempty integer array of C-order foreground indices, shape (N,).
            shape (Sequence[int]): Three spatial extents in (X, Y, Z) order.

        Returns:
            list[int]: In-bounds (X, Y, Z) center near the selected foreground voxel.
        """

        flat_index = fg_indices[random.randrange(fg_indices.size)]
        center = self._unravel_index3(int(flat_index), shape)
        return self._correct_crop_center(center, shape)

    def _sample_background_center_fast(
        self,
        image: np.ndarray,
        label: np.ndarray,
        shape: Sequence[int],
    ) -> list[int] | None:
        """
        Search the valid center box with bounded background rejection draws.

        Args:
            image (np.ndarray): Image with shape (X, Y, Z, C).
            label (np.ndarray): Categorical label map with shape (X, Y, Z).
            shape (Sequence[int]): Three spatial extents in (X, Y, Z) order.

        Returns:
            list[int] | None: Background center passing the intensity threshold, or None if all attempts fail.
        """

        valid_start, valid_end = self._center_bounds(shape)
        for _ in range(self.bg_sample_attempts):
            center = cast(
                tuple[int, int, int],
                tuple(
                    random.randrange(int(valid_start[idx]), int(valid_end[idx]))
                    for idx in range(3)
                ),
            )
            if label[center] != 0 or not self._passes_image_threshold(image, center):
                continue
            return [int(coord) for coord in center]
        return None

    def _sample_background_flat_index_chunked(
        self,
        image: np.ndarray,
        label: np.ndarray,
    ) -> int | None:
        """
        Uniformly sample eligible background voxels using chunked reservoir sampling.

        Args:
            image (np.ndarray): Image with shape (X, Y, Z, C).
            label (np.ndarray): Categorical label map with shape (X, Y, Z).

        Returns:
            int | None: C-order flat index, or None when no background voxel passes the optional intensity threshold.

        Raises:
            ValueError: If thresholding is enabled and the image is not four-dimensional.
        """

        flat_label = label.reshape(-1)
        flat_image = None
        if self.image_threshold is not None:
            if image.ndim != 4:
                raise ValueError(
                    "image_threshold requires image with shape (H, W, D, C), "
                    f"got image shape {image.shape}."
                )
            flat_image = image.reshape((-1, image.shape[-1]))

        selected: int | None = None
        seen = 0
        for start in range(0, label.size, self.bg_fallback_chunk_voxels):
            end = min(start + self.bg_fallback_chunk_voxels, label.size)
            mask = np.asarray(flat_label[start:end]) == 0
            if flat_image is not None:
                mask &= np.any(
                    np.asarray(flat_image[start:end]) > self.image_threshold,
                    axis=-1,
                )
            count = int(mask.sum())
            if count == 0:
                continue
            if random.randrange(seen + count) < count:
                local_indices = np.flatnonzero(mask)
                selected = start + int(local_indices[random.randrange(count)])
            seen += count
        return selected

    def _sample_background_center_exact(
        self,
        image: np.ndarray,
        label: np.ndarray,
        shape: Sequence[int],
    ) -> list[int] | None:
        """
        Search all eligible background voxels and clamp the sampled center.

        Clamping can move a sampled background voxel onto a foreground center.

        Args:
            image (np.ndarray): Image with shape (X, Y, Z, C).
            label (np.ndarray): Categorical label map with shape (X, Y, Z).
            shape (Sequence[int]): Three spatial extents in (X, Y, Z) order.

        Returns:
            list[int] | None: In-bounds (X, Y, Z) center, or None when no eligible voxel exists.
        """

        flat_index = self._sample_background_flat_index_chunked(image, label)
        if flat_index is None:
            return None
        center = self._unravel_index3(flat_index, shape)
        return self._correct_crop_center(center, shape)

    def _warn_once(self, code: str, message: str) -> None:
        """
        Emit a warning only once for each local code.

        Args:
            code (str): Warning identifier scoped to this transform.
            message (str): Warning text passed to warnings.warn.
        """

        if code in self._warned:
            return
        warnings.warn(message, UserWarning, stacklevel=3)
        self._warned.add(code)

    def _sample_start_from_pair(self, image: np.ndarray, label: np.ndarray) -> None:
        """
        Choose and store a crop origin from the aligned image and labels.

        Missing foreground falls back to background sampling. Background rejection failures trigger an exact search, then
        foreground fallback when available.

        Args:
            image (np.ndarray): Image with shape (X, Y, Z, C).
            label (np.ndarray): Categorical label map with shape (X, Y, Z).

        Raises:
            ValueError: If spatial shapes disagree, the crop is invalid, or no sampling voxel exists.
        """

        if label.ndim != 3:
            raise ValueError(f"Expected label with shape (H, W, D), got {label.shape}.")
        if image.shape[:3] != label.shape[:3]:
            raise ValueError(
                "Image and label spatial shapes must match, "
                f"got image shape {image.shape} and label shape {label.shape}."
            )

        shape = tuple(int(x) for x in label.shape[:3])
        self.size = self._resolve_crop_size(shape)
        fg_indices = self._foreground_indices(label)
        has_fg = fg_indices.size > 0
        want_pos = random.random() < self.pos_ratio

        center: list[int] | None = None
        if want_pos and has_fg:
            center = self._sample_positive_center(fg_indices, shape)
        else:
            if want_pos:
                self._warn_once(
                    "no_foreground",
                    "No foreground voxels found; falling back to background sampling.",
                )
            center = self._sample_background_center_fast(image, label, shape)
            if center is None:
                center = self._sample_background_center_exact(image, label, shape)
            if center is None:
                if not has_fg:
                    raise ValueError("No sampling location available.")
                self._warn_once(
                    "no_background",
                    "No valid background voxels found; falling back to foreground sampling.",
                )
                center = self._sample_positive_center(fg_indices, shape)

        assert center is not None
        self.start = [coord - size // 2 for coord, size in zip(center, self.size)]

    def sample(self, *shape: int) -> list[int]:
        """
        Sample an unconstrained origin when only a spatial shape is available.

        Args:
            *shape (int): Exactly three input extents in (X, Y, Z) order.

        Returns:
            list[int]: Effective crop extents after applying allow_smaller.
        """

        self.size = self._resolve_crop_size(shape)
        return super().sample(*shape)

    def __call__(
        self, img: TransformInput, dim: int = 3, reuse: bool = False
    ) -> TransformInput:
        """
        Crop an image or sample a foreground/background crop from a pair.

        Args:
            img (TransformInput): Image array (X, Y, Z, C) or aligned [image, label] pair with labels (X, Y, Z).
            dim (int): Number of leading axes used for image-only sampling.
            reuse (bool): Reuse the previous crop origin and extents.

        Returns:
            TransformInput: Cropped array or image/label list with original dtypes and the effective crop shape.
        """

        if reuse or isinstance(img, np.ndarray):
            return super().__call__(img, dim=dim, reuse=reuse)

        image, label = img
        self._sample_start_from_pair(image=image, label=label)
        return [self.tf(x, k) for k, x in enumerate(img)]

    def __str__(self) -> str:
        """
        Describe the foreground/background crop configuration.

        Returns:
            str: Requested size, foreground probability, threshold, and size policy.
        """

        return (
            "RandCrop3DByPosNegLabel("
            f"size={self.requested_size}, pos_ratio={self.pos_ratio:.3f}, "
            f"image_threshold={self.image_threshold}, allow_smaller={self.allow_smaller})"
        )


class RandomRotation(Base):
    """Rotate an aligned pair in one randomly selected spatial plane."""

    def __init__(self, angle_spectrum: int = 10) -> None:
        """
        Args:
            angle_spectrum (int): Maximum absolute angle in degrees; both endpoints are included.
        """

        self.angle_spectrum = angle_spectrum
        self.axes = ((0, 1), (1, 2), (0, 2))
        self.axes_buffer: tuple[int, int] | None = None
        self.angle_buffer: int | None = None

    def sample(self, *shape: int) -> list[int]:
        """
        Choose one spatial plane and an integer rotation angle.

        Args:
            *shape (int): Leading spatial extents, normally X, Y, and Z.

        Returns:
            list[int]: Input spatial extents, unchanged by rotation.
        """

        self.axes_buffer = self.axes[random.randint(0, 2)]
        self.angle_buffer = random.randint(-self.angle_spectrum, self.angle_spectrum)
        return list(shape)

    def tf(self, img: np.ndarray, k: int = 0) -> np.ndarray:
        """
        Rotate at fixed extent with zero fill outside the image.

        Args:
            img (np.ndarray): Image with shape (X, Y, Z, C) or label map with shape (X, Y, Z).
            k (int): Four-dimensional images at index zero use linear interpolation; all other inputs use nearest neighbors.

        Returns:
            np.ndarray: Rotated array with the original shape and dtype. Label interpolation preserves categorical values.
        """

        fill_value = 0
        assert self.axes_buffer is not None
        assert self.angle_buffer is not None

        if img.ndim == 4 and k == 0:
            result = np.empty_like(img)
            channels = img.shape[3]
            for c in range(channels):
                result[:, :, :, c] = rotate(
                    img[:, :, :, c],
                    self.angle_buffer,
                    axes=self.axes_buffer,
                    reshape=False,
                    order=1,
                    mode="constant",
                    cval=fill_value,
                    prefilter=False,
                )
            return result
        return rotate(
            img,
            self.angle_buffer,
            axes=self.axes_buffer,
            reshape=False,
            order=0,
            mode="constant",
            cval=fill_value,
            prefilter=False,
        )

    def __str__(self) -> str:
        """
        Describe the sampled rotation.

        Returns:
            str: Current spatial plane and rotation angle.
        """

        return f"RandomRotation(axes={self.axes_buffer}, angle={self.angle_buffer})"


class RandomFlip(Base):
    """Flip each spatial axis independently with probability one half."""

    def __init__(self) -> None:
        """Initialize the three spatial axes with no active flips."""

        self.axis = (0, 1, 2)
        self.flip_state = 0

    def sample(self, *shape: int) -> list[int]:
        """
        Choose uniformly among the eight spatial flip combinations.

        Args:
            *shape (int): Leading spatial extents, normally X, Y, and Z.

        Returns:
            list[int]: Input spatial extents, unchanged by flipping.
        """

        self.flip_state = random.randint(0, 7)
        return list(shape)

    def tf(self, img: np.ndarray, k: int = 0) -> np.ndarray:
        """
        Apply the sampled flips to one aligned array.

        Args:
            img (np.ndarray): Image with shape (X, Y, Z, C) or label map with shape (X, Y, Z).
            k (int): Position in the image/label pair; unused by this operation.

        Returns:
            np.ndarray: View with unchanged shape and dtype; flipped axes may have negative strides.
        """

        if self.flip_state & 1:
            img = np.flip(img, axis=self.axis[0])
        if self.flip_state & 2:
            img = np.flip(img, axis=self.axis[1])
        if self.flip_state & 4:
            img = np.flip(img, axis=self.axis[2])
        return img


class RandomIntensityChange(Base):
    """Perturb image intensities while preserving all-zero background voxels."""

    def __init__(
        self,
        factor: tuple[float, float],
        *,
        shared_across_modalities: bool = False,
    ) -> None:
        """
        Args:
            factor (tuple[float, float]): Nonnegative (shift, scale) half-ranges, centered at zero and one respectively.
            shared_across_modalities (bool): Apply the first sampled scale/shift pair to every image channel.
        """

        shift, scale = factor
        self.shift = shift
        self.scale = scale
        self.shared_across_modalities = bool(shared_across_modalities)
        self.shift_range = (-self.shift, self.shift)
        self.scale_range = (1.0 - self.scale, 1.0 + self.scale)

    def tf(self, img: np.ndarray, k: int = 0) -> np.ndarray:
        """
        Apply sampled scale and shift factors to image foreground.

        Factors are drawn for every channel before shared mode selects the first pair, so shared mode preserves the random
        stream.

        Args:
            img (np.ndarray): Image with shape (X, Y, Z, C) or label map with shape (X, Y, Z).
            k (int): Pair position; labels at index one pass through unchanged.

        Returns:
            np.ndarray: Array with unchanged shape and dtype. Voxels that are zero in every image channel remain zero.
        """

        if k == 1:
            return img
        if img.ndim != 4:
            raise ValueError(f"Expected image in XYZC order, got shape {img.shape}")
        channels = img.shape[-1]
        foreground_mask = np.any(img != 0, axis=-1, keepdims=True)
        shift_factors = np.random.uniform(
            self.shift_range[0],
            self.shift_range[1],
            size=(1, 1, 1, channels),
        )
        scale_factors = np.random.uniform(
            self.scale_range[0],
            self.scale_range[1],
            size=(1, 1, 1, channels),
        )
        if self.shared_across_modalities:
            shift_factors = shift_factors[..., :1]
            scale_factors = scale_factors[..., :1]
        transformed = img * scale_factors + shift_factors
        return np.where(foreground_mask, transformed, 0).astype(img.dtype, copy=False)


class NumpyType(Base):
    """Convert paired arrays to configured NumPy dtypes."""

    def __init__(self, types: tuple[np.dtype, np.dtype], num: int = -1) -> None:
        """
        Args:
            types (Sequence[numpy.typing.DTypeLike]): Nonempty dtype specifications in pair order; extra inputs reuse the final
                dtype.
            num (int): Convert only indices below num when positive; zero or negative values convert every input.
        """

        self.types = tuple(np.dtype(dtype) for dtype in types)
        self.num = num

    def tf(self, img: np.ndarray, k: int = 0) -> np.ndarray:
        """
        Cast one paired array to its configured dtype.

        Args:
            img (np.ndarray): Input image or label array of any shape.
            k (int): Pair position used to select a dtype and apply the num limit.

        Returns:
            np.ndarray: Array with the original shape and selected dtype, or img when excluded by num.
        """

        if self.num > 0 and k >= self.num:
            return img

        dtype_idx = min(k, len(self.types) - 1)
        return img.astype(self.types[dtype_idx], copy=False)

    def __str__(self) -> str:
        """
        Describe the configured output dtypes.

        Returns:
            str: Dtype names in pair order.
        """

        s = ", ".join([str(s) for s in self.types])
        return f"NumpyType(({s}))"


def get_train_transforms(
    crop_size: int | tuple[int, int, int] = 80,
    *,
    shared_intensity_across_modalities: bool = False,
) -> Compose:
    """
    Build padding, label-guided cropping, and training augmentation.

    Args:
        crop_size (int | tuple[int, int, int]): Cubic or (X, Y, Z) output patch extents.
        shared_intensity_across_modalities (bool): Use the first sampled intensity scale/shift pair for every channel.

    Returns:
        Compose: Paired pipeline producing float32 images (X, Y, Z, C) and int64 labels (X, Y, Z) at the requested crop size.
    """
    size = cast(
        tuple[int, int, int],
        (crop_size,) * 3 if isinstance(crop_size, int) else tuple(crop_size),
    )
    post_crop = get_train_post_crop_transforms(
        shared_intensity_across_modalities=shared_intensity_across_modalities,
    )
    return Compose(
        [
            PadToSize3D(size=size),
            RandCrop3DByPosNegLabel(size=size),
            *post_crop.ops,
        ]
    )


def get_train_post_crop_transforms(
    *,
    shared_intensity_across_modalities: bool = False,
) -> Compose:
    """
    Build rotation, flipping, intensity changes, and dtype conversion.

    Args:
        shared_intensity_across_modalities (bool): Use the first sampled intensity scale/shift pair for every channel.

    Returns:
        Compose: Paired augmentation for existing patches; returns float32 images (X, Y, Z, C) and int64 labels (X, Y, Z)
            without changing shape.
    """

    return Compose(
        [
            RandomRotation(angle_spectrum=10),
            RandomFlip(),
            RandomIntensityChange(
                factor=(0.1, 0.1),
                shared_across_modalities=shared_intensity_across_modalities,
            ),
            NumpyType(types=(np.dtype(np.float32), np.dtype(np.int64))),
        ]
    )


def get_test_transforms() -> Compose:
    """
    Build dtype conversion for held-out cases.

    Returns:
        Compose: Paired transform returning float32 images (X, Y, Z, C) and int64 labels (X, Y, Z) with native shapes.
    """
    return Compose([NumpyType(types=(np.dtype(np.float32), np.dtype(np.int64)))])


def get_val_transforms() -> Compose:
    """
    Build dtype conversion for validation cases.

    Returns:
        Compose: Paired transform returning float32 images (X, Y, Z, C) and int64 labels (X, Y, Z) with native shapes.
    """
    return get_test_transforms()
