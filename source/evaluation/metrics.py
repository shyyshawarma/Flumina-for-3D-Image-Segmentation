from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, cast

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from source.evaluation.surface_distance import (
    compute_robust_hausdorff,
    compute_surface_dice_at_tolerance,
    compute_surface_distances as compute_surfel_distances,
)


def _binary_pair(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert aligned spatial masks to boolean arrays.

    Args:
        prediction (numpy.ndarray): Predicted mask with one or more spatial axes; nonzero entries are foreground.
        target (numpy.ndarray): Reference mask with the same shape as prediction.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Boolean prediction and reference arrays, each with the input shape.
    """

    pred = np.asarray(prediction, dtype=bool)
    ref = np.asarray(target, dtype=bool)
    if pred.shape != ref.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {pred.shape} vs {ref.shape}"
        )
    if pred.ndim < 1:
        raise ValueError("Binary metrics require at least one spatial dimension")
    return pred, ref


def _spacing_tuple(spacing: Sequence[float], ndim: int) -> tuple[float, ...]:
    """
    Validate voxel spacing in array-axis order.

    Args:
        spacing (Sequence[float]): One finite, positive spacing per spatial axis.
        ndim (int): Required number of spacing values.

    Returns:
        tuple[float, ...]: The ndim spacing values converted to Python floats.
    """

    try:
        values = tuple(float(value) for value in spacing)
    except TypeError as exc:
        raise ValueError("Spacing must be a sequence of positive numbers") from exc
    if len(values) != ndim:
        raise ValueError(f"Expected {ndim} spacing values, got {values}")
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(f"Spacing values must be finite and positive, got {values}")
    return values


def _surface_pair(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    """
    Validate the grid for surface and lesion metrics.

    Args:
        prediction (numpy.ndarray): Predicted mask with shape (S0, S1, S2).
        target (numpy.ndarray): Reference mask on the same grid.
        spacing (Sequence[float]): Three finite, positive voxel spacings (s0, s1, s2), in millimetres and in array-axis order.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, tuple[float, float, float]]: Boolean prediction and reference masks with shape (S0,
            S1, S2), followed by the validated spacing triple.
    """
    pred, ref = _binary_pair(prediction, target)
    if pred.ndim != 3:
        raise ValueError(f"Surface metrics require 3D masks, got {pred.shape}")
    return pred, ref, _spacing_tuple(spacing, 3)


def case_diagonal_mm(shape: Sequence[int], spacing: Sequence[float]) -> float:
    """
    Compute the physical diagonal of a complete processed volume.

    Args:
        shape (Sequence[int]): Full processed spatial dimensions in stored array axis order.
        spacing (Sequence[float]): Validated positive voxel spacings for the same axes, in millimetres.

    Returns:
        float: Square root of the summed squared axis lengths, using shape times spacing, in millimetres.
    """
    return float(np.linalg.norm(np.asarray(shape, dtype=np.float64) * spacing))


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    """
    Compute binary Dice over all spatial axes.

    Args:
        prediction (numpy.ndarray): Predicted mask with one or more spatial axes; nonzero entries are foreground.
        target (numpy.ndarray): Reference mask with the same shape as prediction.

    Returns:
        float: Dice in [0, 1]. Two empty masks score 1; one empty mask scores 0.
    """

    pred, ref = _binary_pair(prediction, target)
    pred_count = int(np.count_nonzero(pred))
    ref_count = int(np.count_nonzero(ref))
    denominator = pred_count + ref_count
    if denominator == 0:
        return 1.0
    intersection = int(np.count_nonzero(pred & ref))
    return float(2.0 * intersection / denominator)


def precision_score(prediction: np.ndarray, target: np.ndarray) -> float:
    """
    Compute the fraction of predicted foreground that matches the reference.

    Args:
        prediction (numpy.ndarray): Predicted mask with one or more spatial axes; nonzero entries are foreground.
        target (numpy.ndarray): Reference mask with the same shape as prediction.

    Returns:
        float: Precision in [0, 1]. Two empty masks score 1; one empty mask scores 0.
    """

    pred, ref = _binary_pair(prediction, target)
    predicted_positive = int(np.count_nonzero(pred))
    if predicted_positive == 0:
        return 1.0 if not np.any(ref) else 0.0
    true_positive = int(np.count_nonzero(pred & ref))
    return float(true_positive / predicted_positive)


def recall_score(prediction: np.ndarray, target: np.ndarray) -> float:
    """
    Compute the fraction of reference foreground recovered by the prediction.

    Args:
        prediction (numpy.ndarray): Predicted mask with one or more spatial axes; nonzero entries are foreground.
        target (numpy.ndarray): Reference mask with the same shape as prediction.

    Returns:
        float: Recall in [0, 1]. Two empty masks score 1; one empty mask scores 0.
    """

    pred, ref = _binary_pair(prediction, target)
    reference_positive = int(np.count_nonzero(ref))
    if reference_positive == 0:
        return 1.0 if not np.any(pred) else 0.0
    true_positive = int(np.count_nonzero(pred & ref))
    return float(true_positive / reference_positive)


def surface_distances(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute directed distances between binary mask surfaces.

    Args:
        prediction (numpy.ndarray): Predicted mask with shape (S0, S1, S2).
        target (numpy.ndarray): Reference mask on the same grid.
        spacing (Sequence[float]): Three finite, positive voxel spacings (s0, s1, s2), in millimetres and in array-axis order.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Sorted float64 distances in millimetres, from prediction to reference with shape
            (N_pred,), then from reference to prediction with shape (N_gt,). Each length is the source surface's surfel count.
            An empty source gives an empty array; a nonempty source and empty destination give infinity.
    """

    pred, ref, physical_spacing = _surface_pair(prediction, target, spacing)
    distances = compute_surfel_distances(ref, pred, physical_spacing)
    return (
        distances["distances_pred_to_gt"],
        distances["distances_gt_to_pred"],
    )


def hd95_distance(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> float:
    """
    Compute the maximum of the two area-weighted 95th-percentile distances.

    Args:
        prediction (numpy.ndarray): Predicted mask on the complete processed volume with shape (S0, S1, S2).
        target (numpy.ndarray): Reference mask on the same grid.
        spacing (Sequence[float]): Three finite, positive voxel spacings (s0, s1, s2), in millimetres and in array-axis order.

    Returns:
        float: HD95 in millimetres, zero for two empty masks, or the full volume's physical diagonal for one empty mask.
    """

    pred, ref, physical_spacing = _surface_pair(prediction, target, spacing)
    pred_nonempty = bool(np.any(pred))
    ref_nonempty = bool(np.any(ref))
    if not pred_nonempty and not ref_nonempty:
        return 0.0
    if pred_nonempty != ref_nonempty:
        return case_diagonal_mm(pred.shape, physical_spacing)
    distances = compute_surfel_distances(ref, pred, physical_spacing)
    return float(
        compute_robust_hausdorff(
            distances,
            percent=95.0,
        )
    )


def normalized_surface_dice(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    tolerance: float = 1.0,
) -> float:
    """
    Compute area-weighted surface Dice within a distance tolerance.

    Args:
        prediction (numpy.ndarray): Predicted mask with shape (S0, S1, S2).
        target (numpy.ndarray): Reference mask on the same grid.
        spacing (Sequence[float]): Three finite, positive voxel spacings (s0, s1, s2), in millimetres and in array-axis order.
        tolerance (float): Maximum surface distance in millimetres counted as overlap; the threshold is inclusive.

    Returns:
        float: Surface Dice in [0, 1]. Two empty masks score 1; one empty mask scores 0.
    """

    pred, ref, physical_spacing = _surface_pair(prediction, target, spacing)
    threshold = float(tolerance)
    pred_nonempty = bool(np.any(pred))
    ref_nonempty = bool(np.any(ref))
    if not pred_nonempty and not ref_nonempty:
        return 1.0
    if pred_nonempty != ref_nonempty:
        return 0.0
    distances = compute_surfel_distances(ref, pred, physical_spacing)
    return float(
        compute_surface_dice_at_tolerance(
            distances,
            tolerance_mm=threshold,
        )
    )


@dataclass(frozen=True, slots=True)
class LesionWiseScores:
    """
    Lesion scores and detection counts for one binary region.

    Args:
        dice (float): Lesion Dice after false-positive and missed-lesion penalties.
        hd95 (float): Penalized mean lesion HD95 in millimetres.
        true_positive (int): Eligible reference lesions matched to predictions.
        false_positive (int): Predicted components matching no reference lesion, regardless of reference lesion volume.
        false_negative (int): Eligible reference lesions without a prediction.
        eligible_ground_truth (int): Reference lesions above the volume threshold.
    """

    dice: float
    hd95: float
    true_positive: int
    false_positive: int
    false_negative: int
    eligible_ground_truth: int


@dataclass(frozen=True, slots=True)
class ConnectedComponentLesionScores:
    """
    Scores from one-to-one matching of connected lesions.

    Args:
        dice (float): Matched Dice sum divided by TP + FP + FN, or 1 when empty.
        surface_dice (tuple[tuple[float, float], ...]): Ordered pairs of tolerance in millimetres and the corresponding
            penalized surface Dice.
        true_positive (int): Matched prediction/reference pairs.
        false_positive (int): Unmatched predicted components.
        false_negative (int): Unmatched reference components.
    """

    dice: float
    surface_dice: tuple[tuple[float, float], ...]
    true_positive: int
    false_positive: int
    false_negative: int

    def surface_dice_at(self, tolerance_mm: float) -> float:
        """
        Look up surface Dice by an exact tolerance value.

        Args:
            tolerance_mm (float): Registered surface tolerance in millimetres.

        Returns:
            float: Surface Dice stored for that tolerance.
        """

        tolerance = float(tolerance_mm)
        for registered, value in self.surface_dice:
            if registered == tolerance:
                return value
        available = ", ".join(str(item[0]) for item in self.surface_dice)
        raise KeyError(f"No NSD score at {tolerance} mm; available: {available}")


def _combined_ground_truth_components(
    target: np.ndarray,
    *,
    dilation_iterations: int,
) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Merge reference lesions connected after dilation.

    Args:
        target (numpy.ndarray): Boolean reference mask with shape (S0, S1, S2).
        dilation_iterations (int): Dilation iterations using 18-connectivity; values below one repeat dilation until the mask
            stops changing.

    Returns:
        tuple[numpy.ndarray, int, numpy.ndarray]: Int32 component labels with the input shape, the number of 26-connected
            dilated components, and the boolean dilation structure with shape (3, 3, 3). Labels are restricted to the original
            reference foreground.
    """

    full_connectivity = np.ones((3,) * target.ndim, dtype=bool)
    dilation_structure = ndimage.generate_binary_structure(target.ndim, 2)
    dilated = ndimage.binary_dilation(
        target,
        structure=dilation_structure,
        iterations=dilation_iterations,
    )
    dilated_labels, component_count = cast(
        tuple[NDArray[np.signedinteger], int],
        ndimage.label(
            dilated,
            structure=full_connectivity,
        ),
    )
    combined = np.zeros(target.shape, dtype=np.int32)
    for component in range(1, component_count + 1):
        combined[target & (dilated_labels == component)] = component
    return combined, int(component_count), dilation_structure


def lesion_wise_scores(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    dilation_iterations: int = 3,
    minimum_lesion_volume_mm3: float = 50.0,
) -> LesionWiseScores:
    """
    Compute lesion Dice and HD95 with detection penalties.

    Reference lesions connected after 18-connected dilation are merged. Each merged lesion is compared with the union of
    predicted 26-connected components intersecting its dilated extent. Missed eligible lesions and unmatched predictions
    receive the physical diagonal of the complete processed volume as their HD95 penalty.

    Args:
        prediction (numpy.ndarray): Predicted mask on the complete processed volume with shape (S0, S1, S2).
        target (numpy.ndarray): Reference mask on the same grid.
        spacing (Sequence[float]): Three finite, positive voxel spacings (s0, s1, s2), in millimetres and in array-axis order.
        dilation_iterations (int): Reference dilation iterations; values below one repeat dilation until the mask stops
            changing.
        minimum_lesion_volume_mm3 (float): Strict lower bound for eligible reference lesion volume in cubic millimetres.

    Returns:
        LesionWiseScores: Scores divided by eligible reference lesions plus false positives, and detection counts. References at
            or below the volume bound are omitted, but predictions matching them are exempt from false-positive penalties. A
            zero denominator gives Dice 1 and HD95 0.
    """

    pred, ref, physical_spacing = _surface_pair(prediction, target, spacing)
    iterations = int(dilation_iterations)
    volume_threshold = float(minimum_lesion_volume_mm3)
    penalty = case_diagonal_mm(pred.shape, physical_spacing)

    combined_gt, gt_count, dilation_structure = _combined_ground_truth_components(
        ref,
        dilation_iterations=iterations,
    )
    full_connectivity = np.ones((3, 3, 3), dtype=bool)
    predicted_components, predicted_count = cast(
        tuple[NDArray[np.signedinteger], int],
        ndimage.label(
            pred,
            structure=full_connectivity,
        ),
    )

    matched_prediction_ids: set[int] = set()
    eligible_dice: list[float] = []
    eligible_hd95: list[float] = []
    true_positive = 0
    false_negative = 0
    voxel_volume = float(np.prod(physical_spacing))

    for component in range(1, gt_count + 1):
        gt_lesion = combined_gt == component
        gt_extent = cast(
            NDArray[np.bool_],
            ndimage.binary_dilation(
                gt_lesion,
                structure=dilation_structure,
                iterations=iterations,
            ),
        )
        intersecting = np.unique(predicted_components[gt_extent])
        intersecting = intersecting[intersecting > 0]
        matched_prediction_ids.update(int(value) for value in intersecting)
        predicted_lesion = np.isin(predicted_components, intersecting)

        lesion_volume = float(np.count_nonzero(gt_lesion) * voxel_volume)
        if lesion_volume <= volume_threshold:
            continue
        if intersecting.size:
            true_positive += 1
        else:
            false_negative += 1
        eligible_dice.append(dice_score(predicted_lesion, gt_lesion))
        eligible_hd95.append(
            hd95_distance(
                predicted_lesion,
                gt_lesion,
                spacing=physical_spacing,
            )
        )

    all_prediction_ids = set(range(1, int(predicted_count) + 1))
    false_positive = len(all_prediction_ids - matched_prediction_ids)
    denominator = len(eligible_dice) + false_positive
    if denominator == 0:
        lesion_dice = 1.0
        lesion_hd95 = 0.0
    else:
        lesion_dice = float(sum(eligible_dice) / denominator)
        lesion_hd95 = float(
            (sum(eligible_hd95) + false_positive * penalty) / denominator
        )

    return LesionWiseScores(
        dice=lesion_dice,
        hd95=lesion_hd95,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        eligible_ground_truth=len(eligible_dice),
    )


def connected_component_lesion_scores(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    nsd_tolerances_mm: Sequence[float] = (0.5, 1.0),
    matching_threshold: float = 1.0e-6,
) -> ConnectedComponentLesionScores:
    """
    Match 26-connected lesions one to one and apply detection penalties.

    The assignment minimizes a Dice-based cost, then discards assigned pairs whose Dice is below matching_threshold.

    Args:
        prediction (numpy.ndarray): Predicted mask with shape (S0, S1, S2).
        target (numpy.ndarray): Reference mask on the same grid.
        spacing (Sequence[float]): Three finite, positive voxel spacings (s0, s1, s2), in millimetres and in array-axis order.
        nsd_tolerances_mm (Sequence[float]): Distinct surface distance thresholds in millimetres, retained in the supplied
            order.
        matching_threshold (float): Inclusive minimum Dice for an assigned pair to count as a match.

    Returns:
        ConnectedComponentLesionScores: Matched Dice and surface Dice sums divided by TP + FP + FN, with lesion counts.
            Unmatched lesions contribute zero; two empty masks receive scores of 1.
    """

    pred, ref, physical_spacing = _surface_pair(prediction, target, spacing)
    threshold = float(matching_threshold)
    tolerances = tuple(float(value) for value in nsd_tolerances_mm)

    connectivity = np.ones((3, 3, 3), dtype=bool)
    pred_components, pred_count = cast(
        tuple[NDArray[np.signedinteger], int],
        ndimage.label(pred, structure=connectivity),
    )
    ref_components, ref_count = cast(
        tuple[NDArray[np.signedinteger], int],
        ndimage.label(ref, structure=connectivity),
    )
    pred_count = int(pred_count)
    ref_count = int(ref_count)

    if pred_count == 0 or ref_count == 0:
        empty_score = 1.0 if pred_count == ref_count else 0.0
        return ConnectedComponentLesionScores(
            dice=empty_score,
            surface_dice=tuple((value, empty_score) for value in tolerances),
            true_positive=0,
            false_positive=pred_count,
            false_negative=ref_count,
        )

    pred_sizes = np.bincount(pred_components.ravel(), minlength=pred_count + 1)
    ref_sizes = np.bincount(ref_components.ravel(), minlength=ref_count + 1)
    dice_matrix = np.zeros((ref_count, pred_count), dtype=np.float64)
    overlap_mask = (pred_components > 0) & (ref_components > 0)
    encoded_pairs = ref_components[overlap_mask].astype(np.int64) * (
        pred_count + 1
    ) + pred_components[overlap_mask].astype(np.int64)
    encoded, intersections = np.unique(encoded_pairs, return_counts=True)
    for pair, intersection in zip(encoded.tolist(), intersections.tolist()):
        ref_id, pred_id = divmod(int(pair), pred_count + 1)
        denominator = int(ref_sizes[ref_id] + pred_sizes[pred_id])
        dice_matrix[ref_id - 1, pred_id - 1] = 2.0 * int(intersection) / denominator

    from scipy.optimize import linear_sum_assignment

    cost = np.where(dice_matrix >= threshold, 1.0 - dice_matrix, 1.0 + 1.0e-6)
    ref_indices, pred_indices = linear_sum_assignment(cost)
    matches = tuple(
        (int(ref_index) + 1, int(pred_index) + 1)
        for ref_index, pred_index in zip(ref_indices.tolist(), pred_indices.tolist())
        if dice_matrix[ref_index, pred_index] >= threshold
    )

    true_positive = len(matches)
    false_positive = pred_count - true_positive
    false_negative = ref_count - true_positive
    denominator = true_positive + false_positive + false_negative

    dice_sum = sum(dice_matrix[ref_id - 1, pred_id - 1] for ref_id, pred_id in matches)
    nsd_sums = {tolerance: 0.0 for tolerance in tolerances}
    for ref_id, pred_id in matches:
        ref_lesion = ref_components == ref_id
        pred_lesion = pred_components == pred_id
        distances = compute_surfel_distances(ref_lesion, pred_lesion, physical_spacing)
        for tolerance in tolerances:
            nsd_sums[tolerance] += compute_surface_dice_at_tolerance(
                distances, tolerance_mm=tolerance
            )

    return ConnectedComponentLesionScores(
        dice=float(dice_sum / denominator),
        surface_dice=tuple(
            (tolerance, float(nsd_sums[tolerance] / denominator))
            for tolerance in tolerances
        ),
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )


__all__ = [
    "ConnectedComponentLesionScores",
    "LesionWiseScores",
    "case_diagonal_mm",
    "connected_component_lesion_scores",
    "dice_score",
    "hd95_distance",
    "lesion_wise_scores",
    "normalized_surface_dice",
    "precision_score",
    "recall_score",
    "surface_distances",
]
