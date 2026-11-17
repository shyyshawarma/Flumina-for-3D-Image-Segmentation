from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping


@dataclass(frozen=True, slots=True)
class DatasetRuntimeContract:
    """
    Dataset identity, channel layout, and evaluation defaults.

    Attributes:
        dataset_key (str): Canonical configuration key.
        dataset_name (str): Physical dataset directory and CLI name.
        logical_name (str): Alternative logical dataset name.
        num_classes (int): Segmentation class count including background.
        in_channels (int): Number of input image channels.
        channel_names (tuple[str, ...]): Channel meanings in array order.
        canonical_spacing (tuple[float, float, float]): Default voxel spacing in millimeters, in stored spatial-axis order.
        trainable (bool): Whether supervised training is supported.
        label_schema_id (str): Identifier for the categorical label mapping.
        categorical_evaluation_targets (bool): Use int64 label maps during evaluation instead of float32 one-hot targets.
        native_evaluation_batch_size_cap (int | None): Maximum cases per evaluation batch, or None for no cap.
        native_evaluation_num_workers_cap (int | None): Maximum evaluation workers, or None for no cap.
        native_evaluation_prefetch_factor_cap (int | None): Maximum prefetched batches per worker, or None for no cap.
        native_evaluation_pin_memory (bool | None): Evaluation pin-memory override; None uses CUDA availability.
    """

    dataset_key: str
    dataset_name: str
    logical_name: str
    num_classes: int
    in_channels: int
    channel_names: tuple[str, ...]
    canonical_spacing: tuple[float, float, float]
    trainable: bool
    label_schema_id: str
    categorical_evaluation_targets: bool = False
    native_evaluation_batch_size_cap: int | None = None
    native_evaluation_num_workers_cap: int | None = 2
    native_evaluation_prefetch_factor_cap: int | None = None
    native_evaluation_pin_memory: bool | None = None

    @property
    def num_foreground_classes(self) -> int:
        """
        Count classes excluding background.

        Returns:
            int: Number of segmentation classes minus one.
        """

        return self.num_classes - 1


_BRATS_CHANNELS: Final[tuple[str, ...]] = ("FLAIR", "T1ce", "T1", "T2")
_BRATS_CANONICAL_SPACING: Final[tuple[float, float, float]] = (1.0, 1.0, 1.0)
_AMOS_CANONICAL_SPACING: Final[tuple[float, float, float]] = (
    2.0,
    0.712890625,
    0.712890625,
)
_MAMA_MIA_CANONICAL_SPACING: Final[tuple[float, float, float]] = (2.0, 0.7, 0.7)
_KITS23_CANONICAL_SPACING: Final[tuple[float, float, float]] = (
    1.0,
    0.78125,
    0.78125,
)
_LITS_CANONICAL_SPACING: Final[tuple[float, float, float]] = (
    1.0,
    0.767578125,
    0.767578125,
)


DATASET_CONTRACTS: Final[tuple[DatasetRuntimeContract, ...]] = (
    DatasetRuntimeContract(
        dataset_key="brats2025_gli",
        dataset_name="BraTS-2025-GLI",
        logical_name="BraTS2025GLI",
        num_classes=5,
        in_channels=4,
        channel_names=_BRATS_CHANNELS,
        canonical_spacing=_BRATS_CANONICAL_SPACING,
        trainable=True,
        label_schema_id="brats2025_gli_labels",
    ),
    DatasetRuntimeContract(
        dataset_key="amos2022",
        dataset_name="AMOS-2022",
        logical_name="AMOS2022",
        num_classes=16,
        in_channels=1,
        channel_names=("either_CT_or_MR",),
        canonical_spacing=_AMOS_CANONICAL_SPACING,
        trainable=True,
        label_schema_id="amos2022_labels",
        categorical_evaluation_targets=True,
        native_evaluation_batch_size_cap=1,
        native_evaluation_num_workers_cap=1,
        native_evaluation_prefetch_factor_cap=1,
        native_evaluation_pin_memory=False,
    ),
    DatasetRuntimeContract(
        dataset_key="mama_mia",
        dataset_name="MAMA-MIA",
        logical_name="MAMAMIA",
        num_classes=2,
        in_channels=1,
        channel_names=("first_post_contrast_T1",),
        canonical_spacing=_MAMA_MIA_CANONICAL_SPACING,
        trainable=True,
        label_schema_id="mama_mia_tumor_labels",
        categorical_evaluation_targets=True,
        native_evaluation_batch_size_cap=1,
        native_evaluation_pin_memory=False,
    ),
    DatasetRuntimeContract(
        dataset_key="kits2023",
        dataset_name="KiTS-2023",
        logical_name="KiTS2023",
        num_classes=4,
        in_channels=1,
        channel_names=("contrast_enhanced_CT",),
        canonical_spacing=_KITS23_CANONICAL_SPACING,
        trainable=True,
        label_schema_id="kits2023_kidney_tumor_cyst_labels",
        categorical_evaluation_targets=True,
        native_evaluation_batch_size_cap=1,
        native_evaluation_pin_memory=False,
    ),
    DatasetRuntimeContract(
        dataset_key="lits",
        dataset_name="LiTS",
        logical_name="LiTS",
        num_classes=3,
        in_channels=1,
        channel_names=("contrast_enhanced_CT",),
        canonical_spacing=_LITS_CANONICAL_SPACING,
        trainable=True,
        label_schema_id="lits_liver_tumor_labels",
        categorical_evaluation_targets=True,
        native_evaluation_batch_size_cap=1,
        native_evaluation_pin_memory=False,
    ),
)

DATASET_NAMES: Final[tuple[str, ...]] = tuple(
    contract.dataset_name for contract in DATASET_CONTRACTS
)
TRAINABLE_DATASET_NAMES: Final[tuple[str, ...]] = tuple(
    contract.dataset_name for contract in DATASET_CONTRACTS if contract.trainable
)


def _alias_token(value: str) -> str:
    """
    Remove case and punctuation differences from a dataset alias.

    Args:
        value (str): Dataset name, key, or alias.

    Returns:
        str: Case-folded alphanumeric lookup token.
    """

    return "".join(character for character in value.casefold() if character.isalnum())


_CONTRACT_BY_NAME: Final[Mapping[str, DatasetRuntimeContract]] = MappingProxyType(
    {contract.dataset_name: contract for contract in DATASET_CONTRACTS}
)
_CONTRACT_BY_ALIAS: Final[Mapping[str, DatasetRuntimeContract]] = MappingProxyType(
    {
        _alias_token(alias): contract
        for contract in DATASET_CONTRACTS
        for alias in (
            contract.dataset_key,
            contract.dataset_name,
            contract.logical_name,
        )
    }
)

DATASET_CONTRACT_BY_NAME: Final[Mapping[str, DatasetRuntimeContract]] = (
    _CONTRACT_BY_NAME
)


def get_dataset_contract(dataset: str) -> DatasetRuntimeContract:
    """
    Look up a dataset by key, logical name, or directory name.

    Args:
        dataset (str): Dataset identifier; case and punctuation are ignored.

    Returns:
        DatasetRuntimeContract: Immutable runtime settings for the matching dataset.

    Raises:
        ValueError: If the identifier does not match a registered dataset.
    """

    try:
        return _CONTRACT_BY_ALIAS[_alias_token(dataset)]
    except (AttributeError, KeyError) as exc:
        choices = ", ".join(DATASET_NAMES)
        raise ValueError(
            f"Unknown dataset {dataset!r}; expected one of: {choices}"
        ) from exc


def normalize_dataset_name(dataset: str) -> str:
    """
    Resolve the registered directory and CLI name.

    Args:
        dataset (str): Dataset key, name, or alias.

    Returns:
        str: Canonical physical dataset name.
    """

    return get_dataset_contract(dataset).dataset_name


def normalize_dataset_key(dataset: str) -> str:
    """
    Resolve the registered configuration key.

    Args:
        dataset (str): Dataset key, name, or alias.

    Returns:
        str: Canonical lowercase dataset key.
    """

    return get_dataset_contract(dataset).dataset_key
