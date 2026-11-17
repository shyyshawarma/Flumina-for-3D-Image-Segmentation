from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

from source.dataset.contracts import resolve_split_file
from source.dataset.dataloader import DatasetPathConfig, load_case_metadata
from source.dataset.runtime_contracts import (
    TRAINABLE_DATASET_NAMES,
    get_dataset_contract,
)
from source.dataset.selection import CaseSelection, resolve_supervised_selection
from source.paths import resolve_dataset_root


@dataclass(frozen=True, slots=True)
class EvaluationTask:
    """
    Evaluation cohort and its output filenames.

    Args:
        training_dataset_name (str): Dataset used to train the model.
        evaluated_dataset_name (str): Dataset containing the evaluation cases.
        list_key (str): Evaluation cohort name.
        membership_path (Path): Fixed relative membership-list path.
        aggregate_filename (str): Output filename for aggregate metrics.
    """

    training_dataset_name: str
    evaluated_dataset_name: str
    list_key: str
    membership_path: Path
    aggregate_filename: str

    @property
    def long_form_filename(self) -> str:
        """
        Returns:
            str: Per-case CSV filename derived from the aggregate filename.
        """

        return derive_long_form_filename(self.aggregate_filename)


def derive_long_form_filename(aggregate_filename: str | Path) -> str:
    """
    Derive a per-case CSV filename from an aggregate filename.

    Args:
        aggregate_filename (str | Path): Aggregate output filename or path.

    Returns:
        str: Aggregate stem followed by "_per_case.csv".
    """

    return f"{Path(aggregate_filename).stem}_per_case.csv"


def _task(
    training_dataset_name: str,
    evaluated_dataset_name: str,
    list_key: str,
    aggregate_filename: str,
) -> EvaluationTask:
    """
    Build an evaluation task using list_key as the membership stem.

    Args:
        training_dataset_name (str): Training dataset name.
        evaluated_dataset_name (str): Evaluation dataset name.
        list_key (str): Membership-list stem and cohort name.
        aggregate_filename (str): Aggregate output filename.

    Returns:
        EvaluationTask: Task with a relative <list_key>.txt membership path.
    """

    return EvaluationTask(
        training_dataset_name=training_dataset_name,
        evaluated_dataset_name=evaluated_dataset_name,
        list_key=list_key,
        membership_path=Path(f"{list_key}.txt"),
        aggregate_filename=aggregate_filename,
    )


_EVALUATION_PLANS: Final[Mapping[str, tuple[EvaluationTask, ...]]] = MappingProxyType(
    {
        "AMOS-2022": (
            _task(
                "AMOS-2022",
                "AMOS-2022",
                "test",
                "test_results.csv",
            ),
        ),
        "MAMA-MIA": (_task("MAMA-MIA", "MAMA-MIA", "test", "test_results.csv"),),
        "KiTS-2023": (_task("KiTS-2023", "KiTS-2023", "test", "test_results.csv"),),
        "LiTS": (_task("LiTS", "LiTS", "test", "test_results.csv"),),
        "BraTS-2025-GLI": tuple(
            _task("BraTS-2025-GLI", "BraTS-2025-GLI", key, filename)
            for key, filename in (
                ("test", "test_results.csv"),
                ("test_preoperative", "test_results_preoperative.csv"),
                ("test_postoperative", "test_results_postoperative.csv"),
            )
        ),
    }
)


def build_evaluation_plan(training_dataset_name: str) -> tuple[EvaluationTask, ...]:
    """
    Resolve the evaluation tasks for a trainable dataset.

    Args:
        training_dataset_name (str): Canonical dataset name or accepted alias.

    Returns:
        tuple[EvaluationTask, ...]: Ordered tasks with canonical dataset names.
    """

    contract = get_dataset_contract(training_dataset_name)
    if not contract.trainable:
        choices = ", ".join(TRAINABLE_DATASET_NAMES)
        raise ValueError(
            f"Dataset {contract.dataset_name!r} is evaluation-only; "
            f"training dataset must be one of: {choices}"
        )
    return _EVALUATION_PLANS[contract.dataset_name]


def _selected_training_memberships(
    args: DatasetPathConfig,
) -> tuple[CaseSelection, CaseSelection]:
    """
    Resolve labeled training and validation memberships.

    Args:
        args (DatasetPathConfig): Dataset name and processed data path.

    Returns:
        tuple[CaseSelection, CaseSelection]: Training and validation selections, in that order.
    """

    contract = get_dataset_contract(args.dataset_name)
    root = resolve_dataset_root(
        args.data_path,
        contract.dataset_name,
        direct_dataset_name=contract.dataset_name,
    )
    train_path = resolve_split_file(root, role="train")
    validation_path = resolve_split_file(root, role="val")
    return (
        resolve_supervised_selection(
            args.data_path,
            contract.dataset_name,
            train_path.resolve(),
            direct_dataset_name=contract.dataset_name,
        ),
        resolve_supervised_selection(
            args.data_path,
            contract.dataset_name,
            validation_path.resolve(),
            direct_dataset_name=contract.dataset_name,
        ),
    )


def resolve_evaluation_selection(
    *,
    data_path: str | Path,
    dataset_name: str,
    membership_file: str | Path,
    direct_dataset_name: str | None = None,
) -> CaseSelection:
    """
    Resolve labeled evaluation cases within a canonical test cohort.

    Args:
        data_path (str | Path): Shared data parent or processed dataset root.
        dataset_name (str): Evaluation dataset name or alias.
        membership_file (str | Path): Requested evaluation membership list.
        direct_dataset_name (str | None): Identity of an arbitrarily named root.

    Returns:
        CaseSelection: Requested cohort with unlabeled cases excluded.
    """

    contract = get_dataset_contract(dataset_name)
    root = resolve_dataset_root(
        data_path,
        contract.dataset_name,
        direct_dataset_name=direct_dataset_name,
    )
    canonical_path = resolve_split_file(root, role="test").resolve()

    canonical = resolve_supervised_selection(
        data_path,
        contract.dataset_name,
        canonical_path,
        direct_dataset_name=direct_dataset_name,
    )
    selected = resolve_supervised_selection(
        data_path,
        contract.dataset_name,
        membership_file,
        direct_dataset_name=direct_dataset_name,
    )
    outside = sorted(set(selected.requested) - set(canonical.requested))
    if outside:
        raise ValueError(
            "Evaluation membership contains cases outside the canonical test "
            f"cohort: {outside[:5]}"
        )
    return selected


def _reject_case_overlap(
    selection: CaseSelection,
    *,
    train: CaseSelection,
    validation: CaseSelection,
) -> None:
    """
    Reject labeled evaluation cases present in training or validation.

    Args:
        selection (CaseSelection): Evaluation selection.
        train (CaseSelection): Training selection.
        validation (CaseSelection): Validation selection.
    """

    evaluation = set(selection.included)
    for role, other in (("train", train), ("validation", validation)):
        overlap = sorted(evaluation.intersection(other.included))
        if overlap:
            raise ValueError(
                f"Planned evaluation membership {selection.list_path} overlaps "
                f"the selected {role} cohort: {overlap[:5]}"
            )


def _patient_ids(
    selection: CaseSelection,
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    role: str,
) -> set[str]:
    """
    Collect required patient identifiers for the selected cases.

    Args:
        selection (CaseSelection): Cohort whose included cases need identifiers.
        metadata (Mapping[str, Mapping[str, Any]]): Per-case metadata containing nonempty patient_id strings.
        role (str): Cohort name used in missing-identifier errors.

    Returns:
        set[str]: Unique patient identifiers with surrounding whitespace removed.
    """

    patients: set[str] = set()
    missing: list[str] = []
    for case_name in selection.included:
        patient_id = metadata.get(case_name, {}).get("patient_id")
        if not isinstance(patient_id, str) or not patient_id.strip():
            missing.append(case_name)
        else:
            patients.add(patient_id.strip())
    if missing:
        raise ValueError(
            f"Cannot verify patient-level separation for {role}; missing patient_id "
            f"for cases: {missing[:5]}"
        )
    return patients


def preflight_evaluation_plan(
    args: DatasetPathConfig,
    tasks: Sequence[EvaluationTask] | None = None,
) -> None:
    """
    Check cohort boundaries and train/validation/evaluation separation.

    Args:
        args (DatasetPathConfig): Dataset name and processed data path.
        tasks (Sequence[EvaluationTask] | None): Evaluation tasks; None builds the dataset's default plan. BraTS cohorts are
            also checked by patient.
    """

    training_contract = get_dataset_contract(args.dataset_name)
    planned_tasks = tuple(
        build_evaluation_plan(training_contract.dataset_name)
        if tasks is None
        else tasks
    )
    train_selection, validation_selection = _selected_training_memberships(args)
    train_validation_overlap = sorted(
        set(train_selection.included).intersection(validation_selection.included)
    )
    if train_validation_overlap:
        raise ValueError(
            "Selected train and validation cohorts overlap: "
            f"{train_validation_overlap[:5]}"
        )

    metadata: Mapping[str, Mapping[str, Any]] = {}
    train_patients: set[str] = set()
    validation_patients: set[str] = set()
    check_patient_overlap = training_contract.dataset_name == "BraTS-2025-GLI"
    if check_patient_overlap:
        metadata = load_case_metadata(
            resolve_dataset_root(
                args.data_path,
                training_contract.dataset_name,
                direct_dataset_name=training_contract.dataset_name,
            )
        )
        train_patients = _patient_ids(train_selection, metadata, role="training cohort")
        validation_patients = _patient_ids(
            validation_selection, metadata, role="validation cohort"
        )
        patient_overlap = sorted(train_patients.intersection(validation_patients))
        if patient_overlap:
            raise ValueError(
                "Selected train and validation cohorts overlap by patient_id: "
                f"{patient_overlap[:5]}"
            )

    for task in planned_tasks:
        evaluated_contract = get_dataset_contract(task.evaluated_dataset_name)
        evaluation_selection = resolve_evaluation_selection(
            data_path=args.data_path,
            dataset_name=evaluated_contract.dataset_name,
            membership_file=task.membership_path,
            direct_dataset_name=training_contract.dataset_name,
        )
        if evaluated_contract.dataset_name != training_contract.dataset_name:
            continue
        _reject_case_overlap(
            evaluation_selection,
            train=train_selection,
            validation=validation_selection,
        )
        if not check_patient_overlap:
            continue
        evaluation_patients = _patient_ids(
            evaluation_selection,
            metadata,
            role=f"evaluation membership {evaluation_selection.list_path}",
        )
        for role, selected_patients in (
            ("training", train_patients),
            ("validation", validation_patients),
        ):
            overlap = sorted(evaluation_patients.intersection(selected_patients))
            if overlap:
                raise ValueError(
                    f"Planned evaluation membership {evaluation_selection.list_path} "
                    f"overlaps the selected {role} cohort by patient_id: "
                    f"{overlap[:5]}"
                )


__all__ = [
    "EvaluationTask",
    "build_evaluation_plan",
    "derive_long_form_filename",
    "preflight_evaluation_plan",
    "resolve_evaluation_selection",
]
