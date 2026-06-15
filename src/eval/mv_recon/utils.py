import numpy as np
from scipy.spatial import cKDTree as KDTree
import torch


def completion_ratio(gt_points, rec_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(np.float32))
    return comp_ratio


def accuracy(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(gt_points)
    distances, idx = gt_points_kd_tree.query(rec_points, workers=-1)
    acc = np.mean(distances)

    acc_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals[idx] * rec_normals, axis=-1)
        normal_dot = np.abs(normal_dot)

        return acc, acc_median, np.mean(normal_dot), np.median(normal_dot)

    return acc, acc_median


def completion(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(rec_points)
    distances, idx = gt_points_kd_tree.query(gt_points, workers=-1)
    comp = np.mean(distances)
    comp_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals * rec_normals[idx], axis=-1)
        normal_dot = np.abs(normal_dot)

        return comp, comp_median, np.mean(normal_dot), np.median(normal_dot)

    return comp, comp_median


def compute_chamfer_distance(points_pred, points_gt, max_dist=None):
    """Symmetric Chamfer distance between two point clouds.

    For every predicted point we take the distance to its nearest ground-truth
    point (the accuracy term) and vice versa (the completeness term); the Chamfer
    distance returned here is the sum of the two mean distances. When ``max_dist``
    is given, per-point distances are clipped to it so that a few gross outliers
    cannot dominate the average.
    """
    points_pred = np.asarray(points_pred)
    points_gt = np.asarray(points_gt)
    if len(points_pred) == 0 or len(points_gt) == 0:
        return float("nan")

    dist_pred_to_gt, _ = KDTree(points_gt).query(points_pred, workers=-1)
    dist_gt_to_pred, _ = KDTree(points_pred).query(points_gt, workers=-1)

    if max_dist is not None:
        dist_pred_to_gt = np.clip(dist_pred_to_gt, None, max_dist)
        dist_gt_to_pred = np.clip(dist_gt_to_pred, None, max_dist)

    return float(dist_pred_to_gt.mean() + dist_gt_to_pred.mean())


def compute_iou(pred_vox, target_vox):
    # Get voxel indices
    v_pred_indices = [voxel.grid_index for voxel in pred_vox.get_voxels()]
    v_target_indices = [voxel.grid_index for voxel in target_vox.get_voxels()]

    # Convert to sets for set operations
    v_pred_filled = set(tuple(np.round(x, 4)) for x in v_pred_indices)
    v_target_filled = set(tuple(np.round(x, 4)) for x in v_target_indices)

    # Compute intersection and union
    intersection = v_pred_filled & v_target_filled
    union = v_pred_filled | v_target_filled

    # Compute IoU
    iou = len(intersection) / len(union)
    return iou
