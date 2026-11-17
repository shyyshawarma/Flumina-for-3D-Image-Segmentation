from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from source.dataset.runtime_contracts import get_dataset_contract
from source.paths import resolve_dataset_root


@dataclass(frozen=True, slots=True)
class CaseSelection:
    """
    Ordered membership report for supervised case selection.

    Attributes:
        dataset_name (str): Canonical dataset name.
        list_key (str): Requested case-list filename stem.
        list_path (Path): Requested case-list path.
        labeled_list_path (Path): Dataset labeled.txt path.
        requested (tuple[str, ...]): All requested identifiers in file order.
        included (tuple[str, ...]): Requested identifiers present in labeled.txt.
        excluded (tuple[str, ...]): Requested identifiers absent from labeled.txt.
        requires_labels (bool): Whether this selection requires segmentation labels.
    """

    dataset_name: str
    list_key: str
    list_path: Path
    labeled_list_path: Path
    requested: tuple[str, ...]
    included: tuple[str, ...]
    excluded: tuple[str, ...]
    requires_labels: bool = True

    @property
    def requested_count(self) -> int:
        """
        Count requested cases.

        Returns:
            int: Length of the requested membership.
        """

        return len(self.requested)

    @property
    def included_count(self) -> int:
        """
        Count selected labeled cases.

        Returns:
            int: Length of the included membership.
        """

        return len(self.included)

    @property
    def excluded_count(self) -> int:
        """
        Count requested cases without labels.

        Returns:
            int: Length of the excluded membership.
        """

        return len(self.excluded)


def read_case_list(path: str | Path) -> tuple[str, ...]:
    """
    Read case identifiers while preserving file order.

    Args:
        path (str | Path): UTF-8 text file with one identifier per nonblank line.

    Returns:
        tuple[str, ...]: Nonempty case identifiers in their original order.

    Raises:
        FileNotFoundError: If the list file is missing.
        ValueError: If the list is empty or contains duplicates or path-like identifiers.
    """

    list_path = Path(path)
    if not list_path.is_file():
        raise FileNotFoundError(f"Required case list not found: {list_path}")
    names = tuple(
        line.strip()
        for line in list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not names:
        raise ValueError(f"Case list is empty: {list_path}")
    if any(Path(name).name != name or name in {".", ".."} for name in names):
        raise ValueError(f"Case list contains an invalid identifier: {list_path}")
    if len(set(names)) != len(names):
        raise ValueError(f"Case list contains duplicate identifiers: {list_path}")
    return names


def resolve_case_list_path(
    dataset_root: str | Path,
    list_path_or_key: str | Path,
) -> Path:
    """
    Resolve a case-list key or text-file path.

    Args:
        dataset_root (str | Path): Base directory for relative paths.
        list_path_or_key (str | Path): List key such as test, or a path; .txt is added when no suffix is present.

    Returns:
        Path: Existing case-list path.

    Raises:
        FileNotFoundError: If the resolved path is not a file.
    """

    candidate = Path(list_path_or_key)
    if not candidate.suffix:
        candidate = candidate.with_suffix(".txt")
    if not candidate.is_absolute():
        candidate = Path(dataset_root) / candidate
    if not candidate.is_file():
        raise FileNotFoundError(f"Required case list not found: {candidate}")
    return candidate


def read_dataset_case_list(
    data_root: str | Path,
    dataset_name: str,
    list_path_or_key: str | Path,
    *,
    direct_dataset_name: str | None = None,
) -> tuple[Path, tuple[str, ...]]:
    """
    Resolve a registered dataset and read one case list.

    Args:
        data_root (str | Path): Shared dataset parent or configured direct dataset root.
        dataset_name (str): Registered dataset name, key, or alias.
        list_path_or_key (str | Path): Case-list key or relative/absolute text-file path.
        direct_dataset_name (str | None): Identity used to authorize a direct dataset root with an arbitrary directory name.

    Returns:
        tuple[Path, tuple[str, ...]]: Case-list path and validated identifiers in file order.
    """

    contract = get_dataset_contract(dataset_name)
    configured_direct_name = (
        None
        if direct_dataset_name is None
        else get_dataset_contract(direct_dataset_name).dataset_name
    )
    root = resolve_dataset_root(
        data_root,
        contract.dataset_name,
        direct_dataset_name=configured_direct_name,
    )
    path = resolve_case_list_path(root, list_path_or_key)
    return path, read_case_list(path)


def resolve_supervised_selection(
    data_root: str | Path,
    dataset_name: str,
    list_path_or_key: str | Path,
    *,
    direct_dataset_name: str | None = None,
) -> CaseSelection:
    """
    Select requested cases that appear in labeled.txt.

    Args:
        data_root (str | Path): Shared dataset parent or configured direct dataset root.
        dataset_name (str): Registered dataset name, key, or alias.
        list_path_or_key (str | Path): Case-list key or relative/absolute text-file path.
        direct_dataset_name (str | None): Identity used to authorize a direct dataset root with an arbitrary directory name.

    Returns:
        CaseSelection: Requested, included, and excluded identifiers, preserving requested order.

    Raises:
        ValueError: If no requested case has a label or the dataset root cannot identify the requested dataset.
    """

    contract = get_dataset_contract(dataset_name)
    configured_direct_name = (
        None
        if direct_dataset_name is None
        else get_dataset_contract(direct_dataset_name).dataset_name
    )
    root = resolve_dataset_root(
        data_root,
        contract.dataset_name,
        direct_dataset_name=configured_direct_name,
    )
    list_path = resolve_case_list_path(root, list_path_or_key)
    labeled_path = resolve_case_list_path(root, "labeled")
    requested = read_case_list(list_path)
    labeled = frozenset(read_case_list(labeled_path))
    included = tuple(name for name in requested if name in labeled)
    excluded = tuple(name for name in requested if name not in labeled)
    if not included:
        raise ValueError(f"No labeled cases from {list_path} are available")
    return CaseSelection(
        dataset_name=contract.dataset_name,
        list_key=list_path.stem,
        list_path=list_path,
        labeled_list_path=labeled_path,
        requested=requested,
        included=included,
        excluded=excluded,
    )


__all__ = [
    "CaseSelection",
    "read_case_list",
    "read_dataset_case_list",
    "resolve_case_list_path",
    "resolve_supervised_selection",
]
