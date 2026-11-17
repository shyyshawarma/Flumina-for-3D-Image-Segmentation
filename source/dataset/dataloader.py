from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence, TypedDict, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from source.dataset.contracts import (
    ensure_disjoint_splits,
    read_split_names,
    resolve_split_file,
)
from source.dataset.dataset import (
    BratsTestDataset,
    BratsTrainDataset,
    BratsValidationDataset,
)
from source.dataset.runtime_contracts import (
    DatasetRuntimeContract,
    get_dataset_contract,
)
from source.dataset.selection import CaseSelection, resolve_supervised_selection
from source.paths import resolve_dataset_root

CaseSample = tuple[torch.Tensor, torch.Tensor, str]
EvaluationCaseSample = tuple[torch.Tensor, torch.Tensor, str, dict[str, Any]]

_TEXT_METADATA_FIELDS = (
    "patient_id",
    "treatment_status",
    "source_role",
    "phase",
    "treatment_phase",
)

_VALIDATION_NUM_WORKERS = 2
_TEST_NUM_WORKERS = 2


def _capped_evaluation_num_workers(
    default: int,
    contract: DatasetRuntimeContract,
) -> int:
    """
    Limit evaluation workers using the dataset contract.

    Args:
        default (int): Worker count requested by the loader factory.
        contract (DatasetRuntimeContract): Dataset settings containing the optional worker cap.

    Returns:
        int: Default worker count limited by the cap, when present.
    """

    cap = contract.native_evaluation_num_workers_cap
    return default if cap is None else min(default, cap)


def _read_metadata_object(path: Path, *, description: str) -> Mapping[str, Any]:
    """
    Read a JSON object with file-specific validation errors.

    Args:
        path (Path): UTF-8 JSON file.
        description (str): Metadata source name used in error messages.

    Returns:
        dict[str, Any]: Parsed JSON object; nested JSON values are preserved.

    Raises:
        ValueError: If the file cannot be decoded or its top-level value is not an object.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {description}: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{description.capitalize()} must be a JSON object: {path}")
    return payload


def _metadata_case_name(value: object, *, context: str) -> str:
    """
    Validate a processed case identifier from metadata.

    Args:
        value (object): Unvalidated identifier; valid values are nonempty strings without path components or surrounding
            whitespace.
        context (str): Record description used in validation errors.

    Returns:
        str: Validated case identifier.

    Raises:
        ValueError: If value is not a valid case identifier.
    """

    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or Path(value).name != value
        or value in {".", ".."}
    ):
        raise ValueError(f"{context} has an invalid processed case identifier")
    return value


def _metadata_spacing(value: object, *, case_name: str) -> tuple[float, float, float]:
    """
    Validate three positive physical voxel spacings.

    Args:
        value (object): Unvalidated spacing; valid values are three-element lists or tuples of finite, positive numbers,
            excluding booleans.
        case_name (str): Case identifier used in validation errors.

    Returns:
        tuple[float, float, float]: Voxel spacing in millimeters, preserving the stored spatial-axis order.

    Raises:
        ValueError: If spacing has an invalid type, length, or component.
    """

    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(
            isinstance(component, bool)
            or not isinstance(component, (int, float))
            or not math.isfinite(float(component))
            or float(component) <= 0.0
            for component in value
        )
    ):
        raise ValueError(
            f"Spacing for {case_name!r} must contain three finite positive numbers"
        )
    return cast(
        tuple[float, float, float], tuple(float(component) for component in value)
    )


def _metadata_from_record(
    record: Mapping[str, Any],
    *,
    case_name: str,
) -> dict[str, Any]:
    """
    Extract supported identity and spacing fields from a metadata record.

    Args:
        record (Mapping[str, Any]): Source record; unrecognized keys are ignored.
        case_name (str): Case identifier used in validation errors.

    Returns:
        dict[str, str | bool | tuple[float, float, float]]: Present text identity fields, optional is_preoperative boolean, and
            validated spacing.
    """

    metadata: dict[str, Any] = {}
    for field in _TEXT_METADATA_FIELDS:
        match record.get(field):
            case None:
                continue
            case str() as value:
                metadata[field] = value
            case _:
                raise ValueError(
                    f"Metadata field {field!r} for {case_name!r} must be a string"
                )
    if record.get("is_preoperative") is not None:
        marker = record["is_preoperative"]
        if not isinstance(marker, bool):
            raise ValueError(
                f"Metadata field 'is_preoperative' for {case_name!r} must be boolean"
            )
        metadata["is_preoperative"] = marker
    raw_spacing = record.get("spacing")
    if raw_spacing is not None:
        metadata["spacing"] = _metadata_spacing(raw_spacing, case_name=case_name)
    return metadata


def load_case_metadata(
    dataset_root: str | Path,
    *,
    dataset_name: str | None = None,
    case_names: Sequence[str] = (),
) -> dict[str, dict[str, Any]]:
    """
    Read case metadata and fill absent spacing from the dataset contract.

    Args:
        dataset_root (str | Path): Processed dataset directory containing optional name_mapping.json.
        dataset_name (str | None): Registered dataset identifier for default spacing; None attempts lookup using the directory
            name.
        case_names (Sequence[str]): Additional selected cases to populate when a dataset contract is available.

    Returns:
        dict[str, dict[str, str | bool | tuple[float, float, float]]]: Metadata keyed by processed case name. Record spacing
            takes precedence over the registered default; without a recognized dataset, absent spacing stays absent.
    """

    root = Path(dataset_root)
    metadata_by_name: dict[str, dict[str, Any]] = {}
    mapping_path = root / "name_mapping.json"
    if mapping_path.is_file():
        payload = _read_metadata_object(
            mapping_path, description="dataset name mapping"
        )
        samples = payload.get("samples")
        if not isinstance(samples, list):
            raise ValueError(
                f"Dataset name mapping must contain a 'samples' list: {mapping_path}"
            )
        for index, sample in enumerate(samples):
            if not isinstance(sample, Mapping):
                raise ValueError(
                    f"Dataset name mapping sample {index} must be an object: {mapping_path}"
                )
            processed_name = _metadata_case_name(
                sample.get("processed_name"),
                context=f"Dataset name mapping sample {index}",
            )
            if processed_name in metadata_by_name:
                raise ValueError(
                    "Dataset name mapping contains duplicate processed_name "
                    f"{processed_name!r}: {mapping_path}"
                )
            metadata_by_name[processed_name] = _metadata_from_record(
                sample, case_name=processed_name
            )

    selected_names = tuple(case_names)

    contract: DatasetRuntimeContract | None
    if dataset_name is not None:
        contract = get_dataset_contract(dataset_name)
    else:
        try:
            contract = get_dataset_contract(root.name)
        except ValueError:
            contract = None
    if contract is not None:
        for case_name in (*metadata_by_name, *selected_names):
            metadata_by_name.setdefault(case_name, {}).setdefault(
                "spacing", contract.canonical_spacing
            )
    return metadata_by_name


class DatasetPathConfig(Protocol):
    """Read-only dataset identity and paths used to resolve case membership."""

    @property
    def data_path(self) -> str:
        """
        Returns:
            str: Shared data parent or processed dataset root.
        """
        ...

    @property
    def dataset_name(self) -> str:
        """
        Returns:
            str: Registered dataset name, key, or alias.
        """
        ...


class LoaderConfig(DatasetPathConfig, Protocol):
    """Read-only configuration consumed by supervised DataLoader factories."""

    @property
    def num_classes(self) -> int:
        """
        Returns:
            int: Segmentation class count including background.
        """
        ...

    @property
    def in_channels(self) -> int:
        """
        Returns:
            int: Expected input image channel count.
        """
        ...

    @property
    def crop_size(self) -> int | tuple[int, int, int]:
        """
        Returns:
            int | tuple[int, int, int]: Cubic or three-axis training patch extents.
        """
        ...

    @property
    def batch_size(self) -> int:
        """
        Returns:
            int: Requested cases or patches per batch.
        """
        ...

    @property
    def seed(self) -> int:
        """
        Returns:
            int: Base seed for loader generators and worker initialization.
        """
        ...

    @property
    def num_workers(self) -> int:
        """
        Returns:
            int: Configured number of loader worker processes.
        """
        ...

    @property
    def persistent_workers(self) -> bool:
        """
        Returns:
            bool: Whether worker processes persist between loader iterations.
        """
        ...

    @property
    def prefetch_factor(self) -> int:
        """
        Returns:
            int: Number of batches prefetched by each worker.
        """
        ...


class LoaderKwargs(TypedDict, total=False):
    """Worker, seeding, and memory options passed to a PyTorch DataLoader."""

    pin_memory: bool
    num_workers: int
    generator: torch.Generator
    worker_init_fn: Callable[[int], None]
    persistent_workers: bool
    prefetch_factor: int


def validate_evaluation_metadata(
    dataset_name: str,
    case_names: Sequence[str],
    metadata_by_name: Mapping[str, Mapping[str, Any]],
) -> None:
    """
    Check spacing and dataset-specific identity before evaluation.

    Args:
        dataset_name (str): Registered dataset name, key, or alias.
        case_names (Sequence[str]): Selected cases whose metadata must be present.
        metadata_by_name (Mapping[str, Mapping[str, Any]]): Case metadata with spacing; BraTS-2025-GLI also requires patient_id
            and treatment_status.

    Raises:
        TypeError: If metadata for a selected case is not a mapping.
        ValueError: If spacing is missing or invalid, or required patient/treatment identity is missing or invalid.
    """

    contract = get_dataset_contract(dataset_name)
    missing_spacing: list[str] = []
    missing_identity: list[str] = []
    for case_name in case_names:
        raw_metadata = metadata_by_name.get(case_name)
        match raw_metadata:
            case None:
                metadata: Mapping[str, Any] = {}
            case Mapping():
                metadata = raw_metadata
            case _:
                raise TypeError(
                    f"Evaluation metadata for {case_name!r} must be a mapping"
                )

        spacing = metadata.get("spacing")
        if spacing is None:
            missing_spacing.append(f"{case_name}:spacing")
        else:
            _metadata_spacing(spacing, case_name=case_name)

        if contract.dataset_name == "BraTS-2025-GLI":
            patient_id = metadata.get("patient_id")
            if patient_id is None:
                missing_identity.append(f"{case_name}:patient_id")
            elif not isinstance(patient_id, str):
                raise ValueError(
                    f"Evaluation patient_id for {case_name!r} must be a string"
                )
            elif not patient_id.strip():
                missing_identity.append(f"{case_name}:patient_id")

            treatment_status = metadata.get("treatment_status")
            if treatment_status is None:
                missing_identity.append(f"{case_name}:treatment_status")
            elif not isinstance(treatment_status, str):
                raise ValueError(
                    f"Evaluation treatment_status for {case_name!r} must be a string"
                )
            else:
                normalized_status = treatment_status.strip().casefold()
                if normalized_status in {"", "unknown", "na", "n/a"}:
                    missing_identity.append(f"{case_name}:treatment_status")
                elif normalized_status not in {"preoperative", "postoperative"}:
                    raise ValueError(
                        f"Evaluation treatment_status for {case_name!r} must be "
                        "'preoperative' or 'postoperative'"
                    )

    if missing_spacing:
        examples = ", ".join(missing_spacing[:8])
        raise ValueError(
            "Missing required physical voxel spacing "
            f"for {contract.dataset_name}: {examples}. Load selected cases with "
            "load_case_metadata(..., dataset_name=...) so the dataset runtime "
            "contract supplies canonical spacing; image and label arrays do not "
            "need reprocessing. Registered evaluation never assumes unit spacing."
        )
    if missing_identity:
        examples = ", ".join(missing_identity[:8])
        raise ValueError(
            "Missing required identity evaluation metadata "
            f"for {contract.dataset_name}: {examples}. Supply patient and treatment "
            "identity in name_mapping.json; image/label arrays are unchanged."
        )


class _OrderedNamesMixin:
    """Provide an explicit case order to a file-based dataset constructor."""

    _ordered_names: tuple[str, ...]

    def _load_txt_file(self, file_path: str) -> list[str]:
        """
        Use the stored selection as the dataset membership.

        Args:
            file_path (str): Unused membership path from the dataset interface.

        Returns:
            list[str]: Copy of the selected case identifiers in their supplied order.
        """

        return list(self._ordered_names)


class _SelectedBratsTrainDataset(_OrderedNamesMixin, BratsTrainDataset):
    """Training patches drawn from an explicitly ordered case selection."""

    def __init__(
        self,
        root: str,
        file_path: str | Path,
        names: Sequence[str],
        num_classes: int,
        crop_size: int | tuple[int, int, int],
        *,
        shared_intensity_across_modalities: bool,
        in_channels: int,
    ) -> None:
        """
        Args:
            root (str): Processed dataset root.
            file_path (str | Path): Membership path retained for dataset construction; names supplies the actual order.
            names (Sequence[str]): Selected case identifiers in iteration order.
            num_classes (int): Segmentation class count K, including background.
            crop_size (int | tuple[int, int, int]): Cubic or (X, Y, Z) training patch extents.
            shared_intensity_across_modalities (bool): Use a shared sampled scale/shift pair for all image channels.
            in_channels (int): Expected input channel count C.
        """

        self._ordered_names = tuple(names)
        super().__init__(
            root,
            str(file_path),
            num_classes,
            crop_size,
            shared_intensity_across_modalities=shared_intensity_across_modalities,
            in_channels=in_channels,
        )


class _MetadataMixin(_OrderedNamesMixin):
    """Attach a copy of case metadata to each selected evaluation sample."""

    def __init__(
        self,
        root: str,
        file_path: str | Path,
        names: Sequence[str],
        num_classes: int,
        *,
        crop_size: int | tuple[int, int, int] | None,
        in_channels: int,
        one_hot_targets: bool = True,
        metadata_by_name: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        """
        Args:
            root (str): Processed dataset root.
            file_path (str | Path): Membership path retained for dataset construction; names supplies the actual order.
            names (Sequence[str]): Selected case identifiers in iteration order.
            num_classes (int): Segmentation class count K, including background.
            crop_size (int | tuple[int, int, int] | None): Optional minimum accepted case size; evaluation retains native
                dimensions.
            in_channels (int): Expected input channel count C.
            one_hot_targets (bool): Return float32 one-hot targets when true, or int64 categorical maps when false.
            metadata_by_name (Mapping[str, Mapping[str, Any]] | None): Metadata to copy by case identifier; omitted cases
                receive an empty dictionary.
        """

        self._ordered_names = tuple(names)
        self._metadata_by_name = {
            str(name): dict(metadata)
            for name, metadata in (metadata_by_name or {}).items()
        }
        # Concrete subclasses place their evaluation dataset next in the MRO.
        cast(BratsValidationDataset | BratsTestDataset, super()).__init__(
            root,
            str(file_path),
            num_classes,
            crop_size=crop_size,
            in_channels=in_channels,
            one_hot_targets=one_hot_targets,
        )

    def __getitem__(self, index: int) -> EvaluationCaseSample:
        """
        Load an evaluation case and its physical metadata.

        Args:
            index (int): Position in the selected case list.

        Returns:
            EvaluationCaseSample: (volume, target, name, metadata). CPU float32 images have shape (C, X, Y, Z). Targets are CPU
                float32 one-hot tensors (K, X, Y, Z) or int64 maps (X, Y, Z). Metadata is copied for this sample.
        """

        volume, target, name = cast(
            BratsValidationDataset | BratsTestDataset, super()
        ).__getitem__(index)
        return volume, target, name, dict(self._metadata_by_name.get(name, {}))


class _SelectedBratsValidationDataset(_MetadataMixin, BratsValidationDataset):
    """Selected validation cases with native shapes and case metadata."""


class _SelectedBratsTestDataset(_MetadataMixin, BratsTestDataset):
    """Selected test cases with native shapes and case metadata."""


def _dataset_root(args: DatasetPathConfig) -> Path:
    """
    Resolve the configured dataset under a shared or direct root.

    Args:
        args (DatasetPathConfig): Dataset root and registered dataset identity.

    Returns:
        Path: Processed root for the configured dataset.
    """

    dataset_name = get_dataset_contract(args.dataset_name).dataset_name
    return resolve_dataset_root(
        args.data_path,
        dataset_name,
        direct_dataset_name=dataset_name,
    )


def _seed_worker(_worker_id: int) -> None:
    """
    Seed NumPy and Python from PyTorch and limit worker CPU threads.

    The PyTorch seed is reduced modulo 2**32 for NumPy compatibility.

    Args:
        _worker_id (int): Worker identifier; already incorporated into torch.initial_seed().
    """

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.set_num_threads(1)


def _build_loader_kwargs(
    args: LoaderConfig,
    generator: torch.Generator,
    *,
    prefetch_factor: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
) -> LoaderKwargs:
    """
    Prepare worker, prefetch, and memory settings for a DataLoader.

    Args:
        args (LoaderConfig): Dataset paths, random seed, and DataLoader settings.
        generator (torch.Generator): Generator controlling sample order and worker base seeds.
        prefetch_factor (int | None): Positive batches-per-worker override; None uses the configured value.
        num_workers (int | None): Worker-count override; None uses the configured value.
        pin_memory (bool | None): Pin-memory override; None enables it when CUDA is available.

    Returns:
        LoaderKwargs: DataLoader options; worker initialization, persistence, and prefetch are included only when workers are
            enabled.

    Raises:
        ValueError: If the worker count is negative or the prefetch factor is nonpositive.
    """

    effective_num_workers = int(
        args.num_workers if num_workers is None else num_workers
    )
    if effective_num_workers < 0:
        raise ValueError(
            f"num_workers must be nonnegative, got {effective_num_workers}"
        )
    effective_prefetch_factor = int(
        args.prefetch_factor if prefetch_factor is None else prefetch_factor
    )
    if effective_prefetch_factor <= 0:
        raise ValueError(
            f"prefetch_factor must be positive, got {effective_prefetch_factor}"
        )
    kwargs: LoaderKwargs = {
        "pin_memory": (
            torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
        ),
        "num_workers": effective_num_workers,
        "generator": generator,
    }
    if effective_num_workers > 0:
        kwargs["worker_init_fn"] = _seed_worker
        kwargs["persistent_workers"] = bool(args.persistent_workers)
        kwargs["prefetch_factor"] = effective_prefetch_factor
    return kwargs


def collate_variable_shape_cases(
    batch: list[CaseSample | EvaluationCaseSample],
) -> list[CaseSample | EvaluationCaseSample]:
    """
    Keep evaluation samples separate to preserve their native shapes.

    Args:
        batch (list[CaseSample | EvaluationCaseSample]): Nonempty sample list. Images are float32 tensors (C, X, Y, Z); targets
            are float32 one-hot tensors (K, X, Y, Z) or int64 maps (X, Y, Z). Each sample includes its name and optionally
            metadata.

    Returns:
        list[CaseSample | EvaluationCaseSample]: The original sample list, without stacking tensors.

    Raises:
        ValueError: If batch is empty.
    """
    if not batch:
        raise ValueError("Cannot collate an empty evaluation batch")
    return batch


def _resolve_and_validate_splits(
    args: DatasetPathConfig,
    roles: tuple[str, ...],
) -> tuple[Path, dict[str, Path]]:
    """
    Resolve requested split files and reject case overlap.

    Args:
        args (DatasetPathConfig): Dataset root and registered dataset identity.
        roles (tuple[str, ...]): Requested roles among train, validation, and test.

    Returns:
        tuple[Path, dict[str, Path]]: Processed dataset root and case-list paths keyed by the requested roles.

    Raises:
        ValueError: If a role is unsupported, a case list is invalid, or selected splits overlap.
    """

    root = _dataset_root(args)
    split_roles = {"train": "train", "validation": "val", "test": "test"}
    invalid_roles = sorted(set(roles) - set(split_roles))
    if invalid_roles:
        raise ValueError(f"Invalid loader split roles: {invalid_roles}")
    files = {role: resolve_split_file(root, role=split_roles[role]) for role in roles}
    ensure_disjoint_splits(
        **{name: read_split_names(path) for name, path in files.items()}
    )
    return root, files


def validate_training_splits(args: DatasetPathConfig) -> None:
    """
    Validate all configured split lists before opening case arrays.

    Args:
        args (DatasetPathConfig): Dataset root and registered dataset identity.

    Raises:
        ValueError: If split configuration or membership is invalid, including overlap between splits.
        FileNotFoundError: If a required split file is missing.
    """

    _resolve_and_validate_splits(args, ("train", "validation", "test"))


def setup_train_val_dataloaders(
    args: LoaderConfig,
    *,
    shared_intensity_across_modalities: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    Build training and validation loaders for labeled cases.

    Args:
        args (LoaderConfig): Dataset paths, random seed, and DataLoader settings.
        shared_intensity_across_modalities (bool): Use one sampled intensity scale/shift pair for every image channel.

    Returns:
        tuple[DataLoader, DataLoader]: (train_loader, validation_loader). Training batches contain float32 images
            (B, C, X, Y, Z), float32 one-hot targets (B, K, X, Y, Z), and names. Validation uses B = 1 and native shapes, with
            names and collated metadata; targets are one-hot or int64 (1, X, Y, Z) according to the dataset contract. Tensors
            are on CPU.
    """
    if int(args.batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    contract = get_dataset_contract(args.dataset_name)
    if not contract.trainable:
        raise ValueError(f"Dataset {contract.dataset_name!r} is evaluation-only")
    root, files = _resolve_and_validate_splits(args, ("train", "validation"))
    train_selection = resolve_supervised_selection(
        args.data_path,
        contract.dataset_name,
        files["train"].resolve(),
        direct_dataset_name=contract.dataset_name,
    )
    validation_selection = resolve_supervised_selection(
        args.data_path,
        contract.dataset_name,
        files["validation"].resolve(),
        direct_dataset_name=contract.dataset_name,
    )
    validation_metadata = load_case_metadata(
        root,
        dataset_name=contract.dataset_name,
        case_names=validation_selection.included,
    )
    validate_evaluation_metadata(
        contract.dataset_name,
        validation_selection.included,
        validation_metadata,
    )
    train_dataset = _SelectedBratsTrainDataset(
        str(root),
        train_selection.list_path,
        train_selection.included,
        args.num_classes,
        args.crop_size,
        shared_intensity_across_modalities=shared_intensity_across_modalities,
        in_channels=contract.in_channels,
    )
    validation_dataset = _SelectedBratsValidationDataset(
        str(root),
        validation_selection.list_path,
        validation_selection.included,
        args.num_classes,
        crop_size=None,
        in_channels=contract.in_channels,
        one_hot_targets=not contract.categorical_evaluation_targets,
        metadata_by_name=validation_metadata,
    )
    train_generator = torch.Generator().manual_seed(int(args.seed))
    validation_generator = torch.Generator().manual_seed(int(args.seed) + 1)
    validation_num_workers = _capped_evaluation_num_workers(
        _VALIDATION_NUM_WORKERS,
        contract,
    )
    return (
        DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            **_build_loader_kwargs(args, train_generator),
        ),
        DataLoader(
            validation_dataset,
            batch_size=1,
            shuffle=False,
            **_build_loader_kwargs(
                args,
                validation_generator,
                prefetch_factor=1,
                num_workers=validation_num_workers,
                pin_memory=contract.native_evaluation_pin_memory,
            ),
        ),
    )


def setup_evaluation_dataloader(
    args: LoaderConfig,
    *,
    dataset_name: str,
    membership_file: str | Path,
    batch_size: int | None = None,
    prefetch_factor: int | None = None,
    direct_dataset_name: str | None = None,
) -> tuple[DataLoader, CaseSelection]:
    """
    Build an evaluation loader from the labeled part of a requested list.

    Args:
        args (LoaderConfig): Dataset paths, random seed, and DataLoader settings.
        dataset_name (str): Registered name, key, or alias of the dataset to evaluate.
        membership_file (str | Path): Case-list key or relative/absolute file path.
        batch_size (int | None): Requested cases per batch; None uses args.batch_size, then the dataset cap applies.
        prefetch_factor (int | None): Requested batches prefetched per worker; the dataset cap applies when workers are enabled.
        direct_dataset_name (str | None): Identity authorizing args.data_path as a direct root with an arbitrary directory name.

    Returns:
        tuple[DataLoader, CaseSelection]: Evaluation loader and membership report. Each batch is a list of
            (volume, target, name, metadata) samples: CPU float32 images (C, X, Y, Z), and CPU float32 one-hot targets
            (K, X, Y, Z) or int64 maps (X, Y, Z). Spatial shapes remain native.
    """
    effective_batch_size = int(args.batch_size if batch_size is None else batch_size)
    if effective_batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {effective_batch_size}")
    contract = get_dataset_contract(dataset_name)
    configured_direct_name = (
        None
        if direct_dataset_name is None
        else get_dataset_contract(direct_dataset_name).dataset_name
    )
    evaluation_prefetch_factor = prefetch_factor
    evaluation_num_workers = _capped_evaluation_num_workers(
        _TEST_NUM_WORKERS,
        contract,
    )
    if contract.native_evaluation_batch_size_cap is not None:
        effective_batch_size = min(
            effective_batch_size,
            contract.native_evaluation_batch_size_cap,
        )
    if (
        evaluation_num_workers > 0
        and contract.native_evaluation_prefetch_factor_cap is not None
    ):
        requested_prefetch = int(
            args.prefetch_factor
            if evaluation_prefetch_factor is None
            else evaluation_prefetch_factor
        )
        evaluation_prefetch_factor = min(
            requested_prefetch,
            contract.native_evaluation_prefetch_factor_cap,
        )
    selection = resolve_supervised_selection(
        args.data_path,
        contract.dataset_name,
        membership_file,
        direct_dataset_name=configured_direct_name,
    )
    root = resolve_dataset_root(
        args.data_path,
        contract.dataset_name,
        direct_dataset_name=configured_direct_name,
    )
    metadata_by_name = load_case_metadata(
        root,
        dataset_name=contract.dataset_name,
        case_names=selection.included,
    )
    validate_evaluation_metadata(
        contract.dataset_name,
        selection.included,
        metadata_by_name,
    )
    dataset = _SelectedBratsTestDataset(
        str(root),
        selection.list_path,
        selection.included,
        contract.num_classes,
        crop_size=None,
        in_channels=contract.in_channels,
        one_hot_targets=not contract.categorical_evaluation_targets,
        metadata_by_name=metadata_by_name,
    )
    generator = torch.Generator().manual_seed(int(args.seed) + 2)
    return (
        DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=False,
            collate_fn=collate_variable_shape_cases,
            **_build_loader_kwargs(
                args,
                generator,
                prefetch_factor=evaluation_prefetch_factor,
                num_workers=evaluation_num_workers,
                pin_memory=contract.native_evaluation_pin_memory,
            ),
        ),
        selection,
    )


def setup_test_dataloader(
    args: LoaderConfig,
    *,
    batch_size: int | None = None,
    prefetch_factor: int | None = None,
) -> DataLoader:
    """
    Build evaluation loading for the configured held-out membership.

    Args:
        args (LoaderConfig): Loader settings with optional membership_file; otherwise test.txt is used.
        batch_size (int | None): Requested cases per batch before applying the dataset cap.
        prefetch_factor (int | None): Requested worker prefetch factor before applying the dataset cap.

    Returns:
        DataLoader: Batches of separate (volume, target, name, metadata) samples. CPU images are float32 (C, X, Y, Z); targets
            are float32 one-hot (K, X, Y, Z) or int64 (X, Y, Z), with native spatial shapes.
    """

    contract = get_dataset_contract(args.dataset_name)
    configured_membership = getattr(args, "membership_file", None)
    if configured_membership is None:
        _, files = _resolve_and_validate_splits(args, ("test",))
        configured_membership = files["test"].resolve()
    loader, _ = setup_evaluation_dataloader(
        args,
        dataset_name=contract.dataset_name,
        membership_file=configured_membership,
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        direct_dataset_name=contract.dataset_name,
    )
    return loader


def setup_dataloader(args: LoaderConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build all supervised loaders after checking split disjointness.

    Args:
        args (LoaderConfig): Dataset paths, random seed, and DataLoader settings.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Training, validation, and test loaders. Training stacks augmented patches;
            validation stacks one native case with metadata; test batches contain separate native cases with metadata. Images
            and targets are CPU tensors.
    """
    _resolve_and_validate_splits(args, ("train", "validation", "test"))
    train_loader, validation_loader = setup_train_val_dataloaders(args)
    return train_loader, validation_loader, setup_test_dataloader(args)


__all__ = [
    "collate_variable_shape_cases",
    "load_case_metadata",
    "setup_dataloader",
    "setup_evaluation_dataloader",
    "setup_test_dataloader",
    "setup_train_val_dataloaders",
    "validate_evaluation_metadata",
    "validate_training_splits",
]
