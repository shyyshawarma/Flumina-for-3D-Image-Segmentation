from typing import TYPE_CHECKING

from source.dataset.contracts import (
    MODALITY_ORDER,
    channel_last_to_first,
    xyzc_to_cxyz,
)
from source.dataset.storage_backend import (
    B2ND_BACKEND,
    NNUNET_B2ND_BACKEND,
    NPY_BACKEND,
    B2NDFullCaseReader,
    B2NDPatchReader,
    CaseStorageSpec,
    ForegroundIndexCache,
    NnUNetB2NDFullCaseReader,
    discover_case_storage_specs,
    validate_case_storage_specs,
)

if TYPE_CHECKING:
    from source.dataset.dataloader import (
        load_case_metadata,
        setup_dataloader,
        setup_evaluation_dataloader,
        setup_test_dataloader,
        setup_train_val_dataloaders,
        validate_evaluation_metadata,
        validate_training_splits,
    )

_LAZY_DATALOADER_EXPORTS = frozenset(
    {
        "load_case_metadata",
        "setup_dataloader",
        "setup_evaluation_dataloader",
        "setup_test_dataloader",
        "setup_train_val_dataloaders",
        "validate_evaluation_metadata",
        "validate_training_splits",
    }
)


def __getattr__(name: str) -> object:
    """
    Import and cache a public loader function on first access.

    Args:
        name (str): Name of the requested module attribute.

    Returns:
        Callable: Requested metadata validator or loader factory.

    Raises:
        AttributeError: If the name is not a lazy loader export.
    """

    if name in _LAZY_DATALOADER_EXPORTS:
        from source.dataset import dataloader

        value = getattr(dataloader, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """
    List eager and lazy module attributes.

    Returns:
        list[str]: Sorted attribute names, including loader exports.
    """

    return sorted(set(globals()).union(_LAZY_DATALOADER_EXPORTS))


__all__ = [
    "MODALITY_ORDER",
    "B2ND_BACKEND",
    "B2NDFullCaseReader",
    "B2NDPatchReader",
    "NNUNET_B2ND_BACKEND",
    "NnUNetB2NDFullCaseReader",
    "NPY_BACKEND",
    "CaseStorageSpec",
    "ForegroundIndexCache",
    "channel_last_to_first",
    "discover_case_storage_specs",
    "load_case_metadata",
    "setup_dataloader",
    "setup_evaluation_dataloader",
    "setup_test_dataloader",
    "setup_train_val_dataloaders",
    "validate_evaluation_metadata",
    "validate_case_storage_specs",
    "validate_training_splits",
    "xyzc_to_cxyz",
]
