from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from source.dataset import storage_backend
from source.dataset.augmentation import (
    Compose,
    get_test_transforms,
    get_train_post_crop_transforms,
    get_train_transforms,
    get_val_transforms,
)
from source.dataset.contracts import (
    read_split_names,
    validate_case_arrays,
)
from source.dataset.storage_backend import (
    B2ND_BACKEND,
    NNUNET_B2ND_BACKEND,
    B2NDFullCaseReader,
    B2NDPatchReader,
    CaseStorageSpec,
    NnUNetB2NDFullCaseReader,
    NnUNetB2NDPatchReader,
    _close_numpy_mapping,
    discover_case_storage_specs,
    validate_case_storage_specs,
)


class BaseDataset(Dataset):
    """Load aligned images and segmentation labels from processed cases."""

    def __init__(
        self,
        root: str,
        file_path: str,
        num_classes: int = 4,
        transform: Compose | None = None,
        max_cached_handles: int = 32,
        minimum_spatial_size: int | tuple[int, int, int] | None = None,
        in_channels: int = 4,
        one_hot_targets: bool = True,
    ) -> None:
        """
        Args:
            root (str): Processed dataset root containing a supported NPY or B2ND layout.
            file_path (str): Text case list with one identifier per nonblank line.
            num_classes (int): Segmentation class count K, including background.
            transform (Compose | None): Paired transform for XYZC images and XYZ labels. A transform must be set before
                sampling.
            max_cached_handles (int): Maximum retained handles per NumPy cache or B2ND reader.
            minimum_spatial_size (int | tuple[int, int, int] | None): Optional cubic or (X, Y, Z) minimum case size; undersized
                cases fail validation.
            in_channels (int): Required image channel count C.
            one_hot_targets (bool): Return float32 one-hot targets when true, or int64 categorical maps when false.
        """

        self.num_classes = num_classes
        self.transform = transform
        self.in_channels = int(in_channels)
        self.one_hot_targets = bool(one_hot_targets)
        self._max_cached_handles = max(1, int(max_cached_handles))
        self._volume_cache: OrderedDict[str, np.memmap] = OrderedDict()
        self._label_cache: OrderedDict[str, np.memmap] = OrderedDict()
        self._full_b2nd_reader: B2NDFullCaseReader | None = None
        self._nnunet_full_b2nd_reader: NnUNetB2NDFullCaseReader | None = None

        self.names = self._load_txt_file(file_path)
        self._storage_specs: tuple[CaseStorageSpec, ...] = discover_case_storage_specs(
            root, self.names
        )
        validate_case_storage_specs(
            self._storage_specs,
            expected_channels=self.in_channels,
            minimum_spatial_size=minimum_spatial_size,
            require_labels=True,
            num_classes=self.num_classes,
        )
        self.volpaths = [str(spec.volume_path) for spec in self._storage_specs]
        self.segpaths = [
            str(spec.label_path)
            for spec in self._storage_specs
            if spec.label_path is not None
        ]
        if any(spec.backend == B2ND_BACKEND for spec in self._storage_specs):
            self._full_b2nd_reader = B2NDFullCaseReader(
                in_channels=self.in_channels,
                num_classes=self.num_classes,
                max_cached_handles=max_cached_handles,
            )
        if any(spec.backend == NNUNET_B2ND_BACKEND for spec in self._storage_specs):
            self._nnunet_full_b2nd_reader = NnUNetB2NDFullCaseReader(
                in_channels=self.in_channels,
                num_classes=self.num_classes,
                max_cached_handles=max_cached_handles,
            )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Read, transform, and convert a case to tensors.

        Args:
            index (int): Position in the dataset case list.

        Returns:
            tuple[torch.Tensor, torch.Tensor, str]: (volume, target, name). The CPU float32 image has shape (C, X, Y, Z).
                Targets are CPU float32 one-hot tensors of shape (K, X, Y, Z), or CPU int64 maps of shape (X, Y, Z) when
                one_hot_targets is false. K is num_classes.
        """

        volume, label, name = self._load_data(index)
        return self._transform_and_convert(volume, label, name)

    def _transform_and_convert(
        self,
        volume: np.ndarray,
        label: np.ndarray,
        name: str,
        *,
        transform: Compose | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Transform an aligned pair and arrange contiguous CPU tensors.

        Args:
            volume (np.ndarray): Image with shape (X, Y, Z, C).
            label (np.ndarray): Categorical labels with shape (X, Y, Z).
            name (str): Case identifier returned with the tensors.
            transform (Compose | None): Paired transform override; None uses the dataset transform.

        Returns:
            tuple[torch.Tensor, torch.Tensor, str]: (volume, target, name). The CPU float32 image has shape (C, X, Y, Z).
                Targets are CPU float32 one-hot tensors of shape (K, X, Y, Z), or CPU int64 maps of shape (X, Y, Z) when
                one_hot_targets is false. K is num_classes.
        """

        transform = self.transform if transform is None else transform
        assert transform is not None

        volume, label = transform([volume, label])

        volume = volume.astype(np.float32, copy=False)
        label = label.astype(np.int64, copy=False)
        validate_case_arrays(
            volume,
            label,
            num_classes=self.num_classes,
            expected_channels=self.in_channels,
        )

        volume = np.ascontiguousarray(np.moveaxis(volume, -1, 0))
        if self.one_hot_targets:
            label = self._one_hot_encoding(label, self.num_classes)
        label = np.ascontiguousarray(label)

        volume_tensor = torch.from_numpy(volume)
        label_tensor = torch.from_numpy(label)

        return volume_tensor, label_tensor, name

    def __len__(self) -> int:
        """
        Count cases in the dataset.

        Returns:
            int: Number of loaded case identifiers.
        """

        return len(self.names)

    def _load_txt_file(self, file_path: str) -> list[str]:
        """
        Read the sorted case membership for this dataset.

        Args:
            file_path (str): Path to a nonempty text case list.

        Returns:
            list[str]: Validated, unique identifiers in sorted order.
        """

        return read_split_names(file_path)

    def _load_data(self, index: int) -> tuple[np.ndarray, np.ndarray, str]:
        """
        Read one image and label pair from its selected backend.

        Args:
            index (int): Position in the dataset case list.

        Returns:
            tuple[np.ndarray, np.ndarray, str]: (volume, label, name), with shapes (X, Y, Z, C) and (X, Y, Z). NPY arrays retain
                their floating/integer dtypes and memory mappings; B2ND readers decode float32 images and uint8 labels.
                Spatial-axis order is preserved.
        """
        spec = self._storage_specs[index]
        match spec.backend:
            case storage_backend.B2ND_BACKEND:
                reader = self._full_b2nd_reader
                assert reader is not None
                volume, label = reader.read_case(spec)
            case storage_backend.NNUNET_B2ND_BACKEND:
                nnunet_reader = self._nnunet_full_b2nd_reader
                assert nnunet_reader is not None
                volume, label = nnunet_reader.read_case(spec)
            case _:
                volume = self._get_volume_handle(index)
                label = self._get_label_handle(index)
        name = self.names[index]

        return volume, label, name

    def _get_volume_handle(self, index: int) -> np.memmap:
        """
        Open or reuse a copy-on-write image mapping.

        Args:
            index (int): Position in the image path list.

        Returns:
            np.memmap: Floating image with shape (X, Y, Z, C) and its stored dtype.
        """

        return self._get_cached_handle(self.volpaths[index], self._volume_cache)

    def _get_label_handle(self, index: int) -> np.memmap:
        """
        Open or reuse a copy-on-write segmentation mapping.

        Args:
            index (int): Position in the label path list.

        Returns:
            np.memmap: Integer labels with shape (X, Y, Z) and their stored dtype.
        """

        return self._get_cached_handle(self.segpaths[index], self._label_cache)

    def _get_cached_handle(
        self,
        path: str,
        cache: OrderedDict[str, np.memmap],
    ) -> np.memmap:
        """
        Refresh an NPY mapping in the least-recently-used cache.

        Opening beyond the cache limit closes the oldest mapping.

        Args:
            path (str): NPY array path.
            cache (OrderedDict[str, np.memmap]): Image or label cache ordered by recent use.

        Returns:
            np.memmap: Copy-on-write mapping with the stored shape and dtype.
        """

        cached = cache.pop(path, None)
        if cached is not None:
            cache[path] = cached
            return cached

        handle = np.load(path, mmap_mode="c")
        cache[path] = handle
        if len(cache) > self._max_cached_handles:
            _, evicted = cache.popitem(last=False)
            _close_numpy_mapping(evicted)
        return handle

    def close(self) -> None:
        """Release cached NumPy mappings and full-case B2ND readers."""

        for cache in (self._volume_cache, self._label_cache):
            while cache:
                _, handle = cache.popitem(last=False)
                _close_numpy_mapping(handle)
        for name in ("_full_b2nd_reader", "_nnunet_full_b2nd_reader"):
            reader = getattr(self, name, None)
            if reader is not None:
                reader.close()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        """Attempt to release cached storage during object cleanup."""

        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self) -> dict[str, object]:
        """
        Copy dataset state with empty NumPy mapping caches.

        Returns:
            dict[str, object]: Dataset attributes for pickling; nested B2ND readers clear their own caches when serialized.
        """

        state = self.__dict__.copy()
        state["_volume_cache"] = OrderedDict()
        state["_label_cache"] = OrderedDict()
        return state

    def _one_hot_encoding(self, label: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Expand categorical labels into float32 class channels.

        Args:
            label (np.ndarray): Integer map with shape (X, Y, Z) and values in [0, num_classes).
            num_classes (int): Number of output class channels K.

        Returns:
            np.ndarray: Contiguous float32 one-hot target with shape (K, X, Y, Z).
        """
        one_hot = np.zeros((num_classes,) + label.shape, dtype=np.float32)
        np.put_along_axis(one_hot, label[None, ...], 1.0, axis=0)
        return one_hot


class BratsTrainDataset(BaseDataset):
    """Sample training patches with spatial and intensity augmentation."""

    def __init__(
        self,
        root: str,
        file_path: str,
        num_classes: int = 4,
        crop_size: int | tuple[int, int, int] = 80,
        max_cached_handles: int = 32,
        in_channels: int = 4,
        *,
        shared_intensity_across_modalities: bool = False,
    ) -> None:
        """
        Args:
            root (str): Processed dataset root containing a supported NPY or B2ND layout.
            file_path (str): Text case list with one identifier per nonblank line.
            num_classes (int): Segmentation class count K, including background.
            crop_size (int | tuple[int, int, int]): Cubic or (X, Y, Z) patch size; smaller case axes receive centered zero
                padding.
            max_cached_handles (int): Maximum retained handles per NumPy cache or B2ND reader.
            in_channels (int): Required image channel count C.
            shared_intensity_across_modalities (bool): Apply the first sampled scale/shift pair to every image channel.
        """

        transform = get_train_transforms(
            crop_size=crop_size,
            shared_intensity_across_modalities=shared_intensity_across_modalities,
        )
        self._post_crop_transform = get_train_post_crop_transforms(
            shared_intensity_across_modalities=shared_intensity_across_modalities,
        )
        super().__init__(
            root,
            file_path,
            num_classes=num_classes,
            transform=transform,
            max_cached_handles=max_cached_handles,
            minimum_spatial_size=1,
            in_channels=in_channels,
        )
        self._b2nd_reader = NnUNetB2NDPatchReader(
            crop_size,
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            max_cached_handles=max_cached_handles,
            positive_probability=0.5,
        )
        self._generic_b2nd_reader = B2NDPatchReader(
            crop_size,
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            max_cached_handles=max_cached_handles,
            positive_probability=0.5,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Read and augment one sampled training patch.

        Args:
            index (int): Position in the dataset case list.

        Returns:
            tuple[torch.Tensor, torch.Tensor, str]: (volume, target, name): CPU float32 image (C, X, Y, Z), CPU float32 one-hot
                target (K, X, Y, Z), and case identifier. Spatial dimensions equal the configured crop size.
        """

        spec = self._storage_specs[index]
        match spec.backend:
            case storage_backend.NNUNET_B2ND_BACKEND:
                volume, label = self._b2nd_reader.read_case(spec)
            case storage_backend.B2ND_BACKEND:
                volume, label = self._generic_b2nd_reader.read_case(spec)
            case _:
                return super().__getitem__(index)
        return self._transform_and_convert(
            volume,
            label,
            self.names[index],
            transform=self._post_crop_transform,
        )

    def close(self) -> None:
        """Release full-case readers, patch readers, and NumPy mappings."""

        super().close()
        for name in ("_b2nd_reader", "_generic_b2nd_reader"):
            reader = getattr(self, name, None)
            if reader is not None:
                reader.close()


class BratsValidationDataset(BaseDataset):
    """Load validation cases with their stored spatial dimensions."""

    def __init__(
        self,
        root: str,
        file_path: str,
        num_classes: int = 4,
        max_cached_handles: int = 32,
        crop_size: int | tuple[int, int, int] | None = None,
        in_channels: int = 4,
        *,
        one_hot_targets: bool = True,
    ) -> None:
        """
        Args:
            root (str): Processed dataset root containing a supported NPY or B2ND layout.
            file_path (str): Text case list with one identifier per nonblank line.
            num_classes (int): Segmentation class count K, including background.
            max_cached_handles (int): Maximum retained handles per NumPy cache or B2ND reader.
            crop_size (int | tuple[int, int, int] | None): Optional minimum allowed case size; validation checks this extent
                without cropping.
            in_channels (int): Required image channel count C.
            one_hot_targets (bool): Return float32 one-hot targets when true, or int64 categorical maps when false.
        """

        transform = get_val_transforms()
        super().__init__(
            root,
            file_path,
            num_classes=num_classes,
            transform=transform,
            max_cached_handles=max_cached_handles,
            minimum_spatial_size=crop_size,
            in_channels=in_channels,
            one_hot_targets=one_hot_targets,
        )


class BratsTestDataset(BaseDataset):
    """Load held-out cases with their stored spatial dimensions."""

    def __init__(
        self,
        root: str,
        file_path: str,
        num_classes: int = 4,
        max_cached_handles: int = 32,
        crop_size: int | tuple[int, int, int] | None = None,
        in_channels: int = 4,
        *,
        one_hot_targets: bool = True,
    ) -> None:
        """
        Args:
            root (str): Processed dataset root containing a supported NPY or B2ND layout.
            file_path (str): Text case list with one identifier per nonblank line.
            num_classes (int): Segmentation class count K, including background.
            max_cached_handles (int): Maximum retained handles per NumPy cache or B2ND reader.
            crop_size (int | tuple[int, int, int] | None): Optional minimum allowed case size; validation checks this extent
                without cropping.
            in_channels (int): Required image channel count C.
            one_hot_targets (bool): Return float32 one-hot targets when true, or int64 categorical maps when false.
        """

        transform = get_test_transforms()
        super().__init__(
            root,
            file_path,
            num_classes=num_classes,
            transform=transform,
            max_cached_handles=max_cached_handles,
            minimum_spatial_size=crop_size,
            in_channels=in_channels,
            one_hot_targets=one_hot_targets,
        )
