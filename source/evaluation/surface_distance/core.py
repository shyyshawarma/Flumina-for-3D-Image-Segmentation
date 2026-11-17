from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from source.evaluation.surface_distance import lookup_tables


def compute_surface_distances(mask_gt, mask_pred, spacing_mm):
    """
    Compute nearest surface distances and marching-cubes surfel areas.

    Args:
        mask_gt (numpy.ndarray): Boolean reference mask with shape (S0, S1, S2).
        mask_pred (numpy.ndarray): Boolean prediction on the same grid.
        spacing_mm (Sequence[float]): Voxel spacing (s0, s1, s2), in millimetres and in the same order as the three array axes.

    Returns:
        dict[str, numpy.ndarray]: Float64 arrays for both directed surfaces. distances_gt_to_pred and surfel_areas_gt have shape
            (N_gt,); distances_pred_to_gt and surfel_areas_pred have shape (N_pred,). Distances are sorted in ascending order
            and measured in millimetres; areas are in square millimetres and follow their distance order. An empty source
            surface has empty arrays. Distances from a nonempty surface to an empty one are infinite.
    """

    # Scale the surfel areas for all 256 binary 2x2x2 neighborhoods.
    neighbour_code_to_surface_area = np.zeros([256])
    for code in range(256):
        normals = np.array(lookup_tables.neighbour_code_to_normals[code])
        sum_area = 0
        for normal_idx in range(normals.shape[0]):
            # Scale each normal component by its perpendicular voxel-face area.
            n = np.zeros([3])
            n[0] = normals[normal_idx, 0] * spacing_mm[1] * spacing_mm[2]
            n[1] = normals[normal_idx, 1] * spacing_mm[0] * spacing_mm[2]
            n[2] = normals[normal_idx, 2] * spacing_mm[0] * spacing_mm[1]
            area = np.linalg.norm(n)
            sum_area += area
        neighbour_code_to_surface_area[code] = sum_area

    # Restrict processing to the bounding box of both masks.
    mask_all = mask_gt | mask_pred
    bbox_min = np.zeros(3, np.int64)
    bbox_max = np.zeros(3, np.int64)

    # Project foreground onto array axis 0.
    proj_0 = np.max(np.max(mask_all, axis=2), axis=1)
    idx_nonzero_0 = np.nonzero(proj_0)[0]
    if len(idx_nonzero_0) == 0:  # pylint: disable=g-explicit-length-test
        return {
            "distances_gt_to_pred": np.array([]),
            "distances_pred_to_gt": np.array([]),
            "surfel_areas_gt": np.array([]),
            "surfel_areas_pred": np.array([]),
        }

    bbox_min[0] = np.min(idx_nonzero_0)
    bbox_max[0] = np.max(idx_nonzero_0)

    # Project foreground onto array axis 1.
    proj_1 = np.max(np.max(mask_all, axis=2), axis=0)
    idx_nonzero_1 = np.nonzero(proj_1)[0]
    bbox_min[1] = np.min(idx_nonzero_1)
    bbox_max[1] = np.max(idx_nonzero_1)

    # Project foreground onto array axis 2.
    proj_2 = np.max(np.max(mask_all, axis=1), axis=0)
    idx_nonzero_2 = np.nonzero(proj_2)[0]
    bbox_min[2] = np.min(idx_nonzero_2)
    bbox_max[2] = np.max(idx_nonzero_2)

    # Pad the upper end of each cropped axis by one voxel so the 2x2x2 correlation includes the full boundary.
    cropmask_gt = np.zeros((bbox_max - bbox_min) + 2, np.uint8)
    cropmask_pred = np.zeros((bbox_max - bbox_min) + 2, np.uint8)

    bbox_slices = (
        slice(bbox_min[0], bbox_max[0] + 1),
        slice(bbox_min[1], bbox_max[1] + 1),
        slice(bbox_min[2], bbox_max[2] + 1),
    )
    cropmask_gt[0:-1, 0:-1, 0:-1] = mask_gt[bbox_slices]

    cropmask_pred[0:-1, 0:-1, 0:-1] = mask_pred[bbox_slices]

    # Encode local occupancy at voxel corners, half a voxel below each center.
    kernel = np.array([[[128, 64], [32, 16]], [[8, 4], [2, 1]]])
    neighbour_code_map_gt = cast(
        NDArray[np.uint8],
        ndimage.correlate(
            cropmask_gt.astype(np.uint8), kernel, mode="constant", cval=0
        ),
    )
    neighbour_code_map_pred = cast(
        NDArray[np.uint8],
        ndimage.correlate(
            cropmask_pred.astype(np.uint8), kernel, mode="constant", cval=0
        ),
    )

    # Mixed neighborhoods contain the mask boundary.
    borders_gt = (neighbour_code_map_gt != 0) & (neighbour_code_map_gt != 255)
    borders_pred = (neighbour_code_map_pred != 0) & (neighbour_code_map_pred != 255)

    # Measure distance to each surface, using infinity when it is empty.
    if borders_gt.any():
        distmap_gt = cast(
            NDArray[np.float64],
            ndimage.distance_transform_edt(~borders_gt, sampling=spacing_mm),
        )
    else:
        distmap_gt = np.inf * np.ones(borders_gt.shape)

    if borders_pred.any():
        distmap_pred = cast(
            NDArray[np.float64],
            ndimage.distance_transform_edt(~borders_pred, sampling=spacing_mm),
        )
    else:
        distmap_pred = np.inf * np.ones(borders_pred.shape)

    # Recover physical surfel areas from the neighborhood codes.
    surface_area_map_gt = neighbour_code_to_surface_area[neighbour_code_map_gt]
    surface_area_map_pred = neighbour_code_to_surface_area[neighbour_code_map_pred]

    # Sample distance to the opposite surface at each source surfel.
    distances_gt_to_pred = distmap_pred[borders_gt]
    distances_pred_to_gt = distmap_gt[borders_pred]
    surfel_areas_gt = surface_area_map_gt[borders_gt]
    surfel_areas_pred = surface_area_map_pred[borders_pred]

    # Keep each surfel area aligned with its sorted distance.
    if distances_gt_to_pred.shape != (0,):
        sorted_surfels_gt = np.array(sorted(zip(distances_gt_to_pred, surfel_areas_gt)))
        distances_gt_to_pred = sorted_surfels_gt[:, 0]
        surfel_areas_gt = sorted_surfels_gt[:, 1]

    if distances_pred_to_gt.shape != (0,):
        sorted_surfels_pred = np.array(
            sorted(zip(distances_pred_to_gt, surfel_areas_pred))
        )
        distances_pred_to_gt = sorted_surfels_pred[:, 0]
        surfel_areas_pred = sorted_surfels_pred[:, 1]

    return {
        "distances_gt_to_pred": distances_gt_to_pred,
        "distances_pred_to_gt": distances_pred_to_gt,
        "surfel_areas_gt": surfel_areas_gt,
        "surfel_areas_pred": surfel_areas_pred,
    }


def compute_average_surface_distance(surface_distances):
    """
    Compute the area-weighted mean distance in each direction.

    Args:
        surface_distances (dict[str, numpy.ndarray]): Output of compute_surface_distances, containing distances_gt_to_pred and
            surfel_areas_gt with shape (N_gt,), and distances_pred_to_gt and surfel_areas_pred with shape (N_pred,).

    Returns:
        tuple[numpy.float64, numpy.float64]: Mean distances in millimetres from reference to prediction, then prediction to
            reference. An empty source gives NaN; only an empty destination gives infinity.
    """
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]
    average_distance_gt_to_pred = np.sum(
        distances_gt_to_pred * surfel_areas_gt
    ) / np.sum(surfel_areas_gt)
    average_distance_pred_to_gt = np.sum(
        distances_pred_to_gt * surfel_areas_pred
    ) / np.sum(surfel_areas_pred)
    return (average_distance_gt_to_pred, average_distance_pred_to_gt)


def compute_robust_hausdorff(surface_distances, percent):
    """
    Compute the maximum area-weighted distance percentile across directions.

    Args:
        surface_distances (dict[str, numpy.ndarray]): Sorted distances and aligned surfel areas from compute_surface_distances;
            reference arrays have shape (N_gt,), and prediction arrays have shape (N_pred,).
        percent (float): Requested cumulative surface-area percentile in [0, 100].

    Returns:
        numpy.float64 | float: Symmetric percentile distance in millimetres. An empty surface in either direction produces
            infinity.
    """
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]
    if len(distances_gt_to_pred) > 0:  # pylint: disable=g-explicit-length-test
        surfel_areas_cum_gt = np.cumsum(surfel_areas_gt) / np.sum(surfel_areas_gt)
        idx = np.searchsorted(surfel_areas_cum_gt, percent / 100.0)
        perc_distance_gt_to_pred = distances_gt_to_pred[
            min(idx, len(distances_gt_to_pred) - 1)
        ]
    else:
        perc_distance_gt_to_pred = np.inf

    if len(distances_pred_to_gt) > 0:  # pylint: disable=g-explicit-length-test
        surfel_areas_cum_pred = np.cumsum(surfel_areas_pred) / np.sum(surfel_areas_pred)
        idx = np.searchsorted(surfel_areas_cum_pred, percent / 100.0)
        perc_distance_pred_to_gt = distances_pred_to_gt[
            min(idx, len(distances_pred_to_gt) - 1)
        ]
    else:
        perc_distance_pred_to_gt = np.inf

    return max(perc_distance_gt_to_pred, perc_distance_pred_to_gt)


def compute_surface_overlap_at_tolerance(surface_distances, tolerance_mm):
    """
    Compute each surface's area fraction within an inclusive tolerance.

    Args:
        surface_distances (dict[str, numpy.ndarray]): Directed distances and aligned surfel areas from
            compute_surface_distances; reference arrays have shape (N_gt,), and prediction arrays have shape (N_pred,).
        tolerance_mm (float): Maximum distance in millimetres counted as overlap.

    Returns:
        tuple[numpy.float64, numpy.float64]: Overlap fractions for the reference surface, then the predicted surface. A fraction
            is NaN when its source surface is empty.
    """
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]
    rel_overlap_gt = np.sum(
        surfel_areas_gt[distances_gt_to_pred <= tolerance_mm]
    ) / np.sum(surfel_areas_gt)
    rel_overlap_pred = np.sum(
        surfel_areas_pred[distances_pred_to_gt <= tolerance_mm]
    ) / np.sum(surfel_areas_pred)
    return (rel_overlap_gt, rel_overlap_pred)


def compute_surface_dice_at_tolerance(surface_distances, tolerance_mm):
    """
    Compute surface Dice from area within an inclusive distance tolerance.

    Args:
        surface_distances (dict[str, numpy.ndarray]): Directed distances and aligned surfel areas from
            compute_surface_distances; reference arrays have shape (N_gt,), and prediction arrays have shape (N_pred,).
        tolerance_mm (float): Maximum distance in millimetres counted as overlap.

    Returns:
        numpy.float64: Combined overlapping area divided by total surface area, in [0, 1], or NaN when both surfaces are empty.
    """
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]
    overlap_gt = np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance_mm])
    overlap_pred = np.sum(surfel_areas_pred[distances_pred_to_gt <= tolerance_mm])
    surface_dice = (overlap_gt + overlap_pred) / (
        np.sum(surfel_areas_gt) + np.sum(surfel_areas_pred)
    )
    return surface_dice


def compute_dice_coefficient(mask_gt, mask_pred):
    """
    Compute volumetric Dice for two boolean masks.

    Args:
        mask_gt (numpy.ndarray): Boolean reference mask with shape (S0, S1, S2).
        mask_pred (numpy.ndarray): Boolean prediction on the same grid.

    Returns:
        numpy.float64 | float: Dice in [0, 1], or NaN when both masks are empty.
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.nan
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum
