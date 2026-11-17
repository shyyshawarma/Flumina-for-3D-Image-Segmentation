from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class EvaluationProtocol(str, Enum):
    """Supported global and lesion-based metric families."""

    LESION_WISE = "lesion_wise"
    CONNECTED_COMPONENT_LESION_WISE = "connected_component_lesion_wise"
    GLOBAL = "global"


@dataclass(frozen=True, slots=True)
class RegionDefinition:
    """
    A binary evaluation region formed by a union of class labels.

    Args:
        name (str): Region name used in metric records and CSV columns.
        labels (tuple[int, ...]): Categorical labels included in the region.
    """

    name: str
    labels: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    """
    Dataset labels, evaluation regions, and metric policy.

    Args:
        dataset_name (str): Canonical dataset name.
        protocol_id (str): Evaluation protocol identifier.
        source_url (str): Reference describing the source evaluation protocol.
        protocol_status (str): Relationship to the source protocol.
        label_names (tuple[tuple[int, str], ...]): Ordered label ID/name pairs, including background.
        regions (tuple[RegionDefinition, ...]): Ordered binary regions to evaluate.
        protocol (EvaluationProtocol): Global or lesion-based evaluation family.
        metrics (tuple[str, ...]): Metric identifiers included in the protocol.
        selection_metric (str): Metric used to select checkpoints.
        selection_regions (tuple[str, ...]): Regions included in checkpoint scoring.
        selection_proxy_id (str): Identifier for the checkpoint selection policy.
        description (str): Human-readable description of the evaluation policy.
        metric_scope (str): Evaluation unit, such as subject.
        lesion_matching_method (str): Matching policy, or not_applicable.
        lesion_dilation_iterations (int): Dilation count for lesion matching.
        minimum_lesion_volume_mm3 (float): Strict lower reference-lesion volume bound in cubic millimetres.
        lesion_matching_threshold (float): Minimum score accepted as a lesion match.
        nsd_tolerance_mm (float): Default surface Dice tolerance in millimetres.
        nsd_tolerances_mm (tuple[float, ...]): Additional registered surface Dice tolerances in millimetres.
        aggregation (str): Reduction policy for case records.
        empty_region_policy (str): Policy for empty prediction/reference regions.
    """

    dataset_name: str
    protocol_id: str
    source_url: str
    protocol_status: str
    label_names: tuple[tuple[int, str], ...]
    regions: tuple[RegionDefinition, ...]
    protocol: EvaluationProtocol
    metrics: tuple[str, ...]
    selection_metric: str
    selection_regions: tuple[str, ...]
    selection_proxy_id: str
    description: str
    metric_scope: str
    lesion_matching_method: str = "not_applicable"
    lesion_dilation_iterations: int = 0
    minimum_lesion_volume_mm3: float = 0.0
    lesion_matching_threshold: float = 1.0e-6
    nsd_tolerance_mm: float = 1.0
    nsd_tolerances_mm: tuple[float, ...] = ()
    aggregation: str = "case_macro"
    empty_region_policy: str = "both_empty_perfect_one_sided_failure"

    @property
    def num_classes(self) -> int:
        """
        Count registered output classes, including background.

        Returns:
            int: Number of label ID/name pairs.
        """

        return len(self.label_names)

    @property
    def selection_proxy_description(self) -> str:
        """
        Describe the checkpoint selection score.

        Returns:
            str: Selection metric, region names, and case-macro aggregation policy.
        """

        regions = "/".join(self.selection_regions)
        return (
            f"case-macro mean {self.selection_metric} over {regions}; "
            "checkpoint selection by validation Dice"
        )


_FOUR_CLASS_GLI_LABELS = (
    (0, "background"),
    (1, "non-enhancing tumor core (NETC)"),
    (2, "surrounding non-enhancing FLAIR hyperintensity (SNFH)"),
    (3, "enhancing tumor (ET)"),
)
_FIVE_CLASS_GLI_LABELS = _FOUR_CLASS_GLI_LABELS + ((4, "resection cavity (RC)"),)
_AMOS_LABELS = (
    (0, "background"),
    (1, "spleen"),
    (2, "right kidney"),
    (3, "left kidney"),
    (4, "gall bladder"),
    (5, "esophagus"),
    (6, "liver"),
    (7, "stomach"),
    (8, "aorta"),
    (9, "postcava"),
    (10, "pancreas"),
    (11, "right adrenal gland"),
    (12, "left adrenal gland"),
    (13, "duodenum"),
    (14, "bladder"),
    (15, "prostate/uterus"),
)
_MAMA_MIA_LABELS = (
    (0, "background"),
    (1, "tumor"),
)
_KITS23_LABELS = (
    (0, "background"),
    (1, "kidney"),
    (2, "tumor"),
    (3, "cyst"),
)
_LITS_LABELS = (
    (0, "background"),
    (1, "liver"),
    (2, "tumor"),
)

_CLASSIC_REGIONS = (
    RegionDefinition("WT", (1, 2, 3)),
    RegionDefinition("TC", (1, 3)),
    RegionDefinition("ET", (3,)),
)
_POST_TREATMENT_REGIONS = _CLASSIC_REGIONS + (
    RegionDefinition("NETC", (1,)),
    RegionDefinition("SNFH", (2,)),
    RegionDefinition("RC", (4,)),
)
_AMOS_REGIONS = tuple(
    RegionDefinition(name, (label,)) for label, name in _AMOS_LABELS[1:]
)
_MAMA_MIA_REGIONS = (RegionDefinition("Tumor", (1,)),)
_KITS23_REGIONS = (
    RegionDefinition("Kidney and Masses", (1, 2, 3)),
    RegionDefinition("Masses", (2, 3)),
    RegionDefinition("Tumor", (2,)),
)
_LITS_REGIONS = (
    RegionDefinition("Liver", (1, 2)),
    RegionDefinition("Tumor", (2,)),
)


def _case_macro_dice_task(
    dataset_name: str,
    protocol_id: str,
    source_url: str,
    label_names: tuple[tuple[int, str], ...],
    regions: tuple[RegionDefinition, ...],
    description: str,
    *,
    protocol_status: str,
) -> TaskDefinition:
    """
    Define whole-volume Dice evaluation with equal case weighting.

    Args:
        dataset_name (str): Canonical dataset name.
        protocol_id (str): Evaluation protocol identifier.
        source_url (str): Reference describing the source evaluation protocol.
        label_names (tuple[tuple[int, str], ...]): Ordered label ID/name pairs.
        regions (tuple[RegionDefinition, ...]): Binary regions to score.
        description (str): Description of the evaluation policy.
        protocol_status (str): Relationship to the source evaluation protocol.

    Returns:
        TaskDefinition: Immutable Dice task selecting checkpoints across all registered regions.
    """

    return TaskDefinition(
        dataset_name=dataset_name,
        protocol_id=protocol_id,
        source_url=source_url,
        protocol_status=protocol_status,
        label_names=label_names,
        regions=regions,
        protocol=EvaluationProtocol.GLOBAL,
        metrics=("dice",),
        selection_metric="dice",
        selection_regions=tuple(region.name for region in regions),
        selection_proxy_id="unimed.case_macro_mean_dice",
        description=description,
        metric_scope="subject",
    )


_TASKS = (
    _case_macro_dice_task(
        dataset_name="BraTS-2025-GLI",
        protocol_id="brats-2025-gli-case-macro-dice",
        source_url="https://www.synapse.org/Synapse:syn64153130/wiki/632724",
        label_names=_FIVE_CLASS_GLI_LABELS,
        regions=_POST_TREATMENT_REGIONS,
        description=(
            "Whole-volume collective pre/post WT/TC/ET/NETC/SNFH/RC Dice "
            "computed per case and averaged across cases."
        ),
        protocol_status="output_compatible_unverified_matching",
    ),
    _case_macro_dice_task(
        dataset_name="AMOS-2022",
        protocol_id="amos-2022-case-macro-dice",
        source_url="https://amos22.grand-challenge.org/",
        label_names=_AMOS_LABELS,
        regions=_AMOS_REGIONS,
        description=(
            "Whole-volume Dice for each of the 15 AMOS abdominal organs, "
            "computed per case and averaged across cases."
        ),
        protocol_status="official_source_derived",
    ),
    _case_macro_dice_task(
        dataset_name="MAMA-MIA",
        protocol_id="mama-mia-local-case-macro-dice",
        source_url="https://doi.org/10.1038/s41597-025-04707-4",
        label_names=_MAMA_MIA_LABELS,
        regions=_MAMA_MIA_REGIONS,
        description=(
            "Whole-volume Tumor Dice computed per case and averaged across "
            "cases. This checkpoint score uses only Dice; "
            "it does not reproduce the official NormHD, performance, or "
            "subgroup-fairness composite."
        ),
        protocol_status="internal_proxy",
    ),
    _case_macro_dice_task(
        dataset_name="KiTS-2023",
        protocol_id="kits-2023-local-case-macro-dice",
        source_url=(
            "https://github.com/neheller/kits23/blob/main/"
            "kits23/configuration/labels.py"
        ),
        label_names=_KITS23_LABELS,
        regions=_KITS23_REGIONS,
        description=(
            "Whole-volume Dice for the official hierarchical Kidney-and-Masses, "
            "Masses, and Tumor evaluation classes, computed per case and "
            "averaged across cases. This checkpoint-selection Dice "
            "proxy and does not reproduce the official surface-Dice ranking."
        ),
        protocol_status="internal_proxy",
    ),
    _case_macro_dice_task(
        dataset_name="LiTS",
        protocol_id="lits-local-case-macro-dice",
        source_url=(
            "https://github.com/PatrickChrist/lits-challenge-scoring/blob/"
            "master/program/evaluate.py"
        ),
        label_names=_LITS_LABELS,
        regions=_LITS_REGIONS,
        description=(
            "Whole-volume Dice for the official Liver (liver plus tumor) and "
            "Tumor masks, computed per case and averaged across cases. This is "
            "checkpoint-selection Dice score does not reproduce "
            "the complete official multi-metric, lesion-detection, tumor-burden, "
            "or challenge-ranking procedure."
        ),
        protocol_status="internal_proxy",
    ),
)

TASKS = {task.dataset_name: task for task in _TASKS}
CANONICAL_DATASET_NAMES = tuple(TASKS)


def normalize_dataset_name(dataset_name: str) -> str:
    """
    Normalize case and separators for dataset lookup.

    Args:
        dataset_name (str): Nonempty dataset name containing letters, digits, hyphens, underscores, or whitespace.

    Returns:
        str: Case-folded alphanumeric name with common separators removed.
    """

    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("Dataset name must be a nonempty string")
    normalized = re.sub(r"[-_\s]+", "", dataset_name.strip()).casefold()
    if not normalized.isalnum():
        raise ValueError(f"Unknown segmentation dataset: {dataset_name!r}")
    return normalized


_BY_NORMALIZED_NAME = {
    normalize_dataset_name(task.dataset_name): task for task in _TASKS
}


def get_task(dataset_name: str) -> TaskDefinition:
    """
    Resolve a registered dataset's evaluation policy.

    Args:
        dataset_name (str): Dataset name with optional case and separator changes.

    Returns:
        TaskDefinition: Immutable evaluation definition for the dataset.
    """

    task = _BY_NORMALIZED_NAME.get(normalize_dataset_name(dataset_name))
    if task is None:
        supported = ", ".join(CANONICAL_DATASET_NAMES)
        raise ValueError(
            f"Unknown segmentation dataset {dataset_name!r}; supported: {supported}"
        )
    return task


def canonical_dataset_name(dataset_name: str) -> str:
    """
    Resolve the canonical spelling of a supported dataset.

    Args:
        dataset_name (str): Dataset name with optional case and separator changes.

    Returns:
        str: Canonical dataset name.
    """

    return get_task(dataset_name).dataset_name


__all__ = [
    "CANONICAL_DATASET_NAMES",
    "EvaluationProtocol",
    "RegionDefinition",
    "TASKS",
    "TaskDefinition",
    "canonical_dataset_name",
    "get_task",
    "normalize_dataset_name",
]
