"""Long3D long-sequence 3D reconstruction evaluation for OVGGT.

For every Long3D scene the model is run over the (key-framed) video, the predicted
per-frame point maps are fused into a single point cloud, that cloud is aligned to
the ground-truth point cloud, and accuracy / completeness / normal-consistency /
Chamfer metrics are reported.

Long3D provides no camera poses or intrinsics, so the predicted cloud and the GT
cloud live in different (and arbitrarily scaled) coordinate frames. Alignment is
therefore done in three steps:

  1. Global scale: estimate a single scale factor from the relative spatial extent
     (90th-percentile radius of the largest cluster) of each cloud.
  2. RANSAC global registration on FPFH features for a coarse rigid alignment.
  3. ICP refinement (coarse then fine) for the final pose.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import gc
import time
import argparse
import os.path as osp

import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from accelerate import Accelerator

from add_ckpt_path import add_path_to_dust3r


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Long3D 3D reconstruction evaluation for OVGGT", add_help=False
    )
    parser.add_argument("--weights", type=str, default="", help="path to model checkpoint")
    parser.add_argument("--model_name", type=str, default="OVGGT", choices=["OVGGT", "VGGT"])
    parser.add_argument("--root", type=str, default="./data/Long3D", help="Long3D dataset root")
    parser.add_argument("--output_dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--scene", type=str, default=None,
        help="evaluate a single scene (e.g. 'Lecture_hall'); default: all scenes",
    )
    parser.add_argument("--size", type=int, default=518, help="resolution preset: 224, 512 or 518")
    parser.add_argument("--kf_every", type=int, default=1, help="keyframe sampling stride")
    parser.add_argument("--max_frames", type=int, default=None, help="cap on frames per scene")
    parser.add_argument(
        "--use_proj", action="store_true",
        help="unproject predicted depth into points instead of using the point-map head",
    )
    parser.add_argument("--max_token_budget", type=int, default=200000, help="model token budget")
    parser.add_argument(
        "--camera_budget", type=int, default=384, help="camera-head KV cache budget"
    )

    # Point-cloud post-processing / alignment.
    parser.add_argument(
        "--voxel_size", type=float, default=0.05,
        help="voxel size (metres) for downsampling before alignment",
    )
    parser.add_argument(
        "--max_points", type=int, default=50_000_000,
        help="randomly subsample the raw predicted cloud to at most this many points",
    )
    parser.add_argument(
        "--sor_neighbors", type=int, default=50,
        help="statistical outlier removal: number of neighbours",
    )
    parser.add_argument(
        "--sor_std_ratio", type=float, default=1.0,
        help="statistical outlier removal: std-ratio threshold",
    )
    return parser


def set_seed(seed=42):
    """Seed every RNG used in the pipeline for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    o3d.utility.random.seed(seed)


def lightweight_collate(batch):
    """Keep only the fields needed for inference, dropping anything memory-heavy."""
    views = batch[0]
    essential_keys = {"img", "label", "instance", "dataset", "idx"}
    return [{k: v for k, v in view.items() if k in essential_keys} for view in views]


def process_predictions_to_pointcloud(preds, batch, args):
    """Convert per-frame model predictions into a single (points, colors) cloud."""
    if args.model_name == "OVGGT":
        from ovggt.utils.pose_enc import pose_encoding_to_extri_intri
        from ovggt.utils.geometry import unproject_depth_map_to_point_map
    elif args.model_name == "VGGT":
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map

    pts_all = []
    colors_all = []

    for i, pred in enumerate(preds):
        if "pts3d" in pred:
            pts = pred["pts3d"]
        elif "pts3d_in_other_view" in pred:
            pts = pred["pts3d_in_other_view"]
        elif "world_points" in pred:
            pts = pred["world_points"]
        elif args.use_proj and "camera_pose" in pred and "depth" in pred:
            # Reconstruct points by unprojecting the predicted depth map.
            pose_enc = pred["camera_pose"].unsqueeze(0).unsqueeze(0)  # (1, 1, C)
            depth_map = pred["depth"].unsqueeze(0).unsqueeze(0)       # (1, 1, H, W)
            img_shape = batch[i]["img"].shape[-2:]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, img_shape)
            point_map = unproject_depth_map_to_point_map(
                depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
            )
            pts = point_map[0]  # (H, W, 3)
        else:
            raise KeyError(f"No point data available in prediction keys: {pred.keys()}")

        if isinstance(pts, torch.Tensor):
            pts = pts.detach().cpu().numpy()
        while pts.ndim > 3:
            pts = pts[0]

        # Per-frame color from the input image.
        img = batch[i]["img"]
        if isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img[0]
            img = img.permute(1, 2, 0).detach().cpu().numpy()  # (H, W, 3)
        else:
            img = np.array(img)

        # Normalise color to [0, 1] regardless of the input range.
        if img.max() > 2.0:
            img = img / 255.0
        elif img.min() < -0.01:
            img = (img + 1.0) / 2.0
        img = np.clip(img, 0.0, 1.0)

        # Center-crop the image to the point-map resolution if they differ.
        H_img, W_img = img.shape[:2]
        H_pts, W_pts = pts.shape[:2]
        if (H_img != H_pts or W_img != W_pts) and H_img >= H_pts and W_img >= W_pts:
            cx, cy = W_img // 2, H_img // 2
            l, t = cx - W_pts // 2, cy - H_pts // 2
            img = img[t:t + H_pts, l:l + W_pts]

        pts_flat = pts.reshape(-1, 3)
        colors_flat = img.reshape(-1, 3)

        valid = np.isfinite(pts_flat).all(axis=1)
        pts_all.append(pts_flat[valid])
        colors_all.append(colors_flat[valid])

    return np.concatenate(pts_all, axis=0), np.concatenate(colors_all, axis=0)


def get_statistical_scale(pcd):
    """Estimate a cloud's spatial scale as the 90th-percentile radius of its
    largest cluster.

    DBSCAN (with a density-adaptive radius) isolates the dominant structure so
    that distant drift / noise does not inflate the estimate.
    """
    pts = np.asarray(pcd.points)
    if len(pts) > 50000:
        idx = np.random.choice(len(pts), 50000, replace=False)
        pcd_sub = pcd.select_by_index(idx)
    else:
        pcd_sub = pcd

    try:
        # Estimate local density (distance to the 5th neighbour) to make the
        # DBSCAN radius scale-independent.
        pcd_tree = o3d.geometry.KDTreeFlann(pcd_sub)
        check_pts = np.asarray(pcd_sub.points)
        stride = max(1, len(check_pts) // 1000)
        distances = []
        for i in range(0, len(check_pts), stride):
            _, _, d = pcd_tree.search_knn_vector_3d(check_pts[i], 5)
            distances.append(np.sqrt(d[-1]))

        if distances:
            eps_val = np.median(distances) * 2.0
            labels = np.array(pcd_sub.cluster_dbscan(eps=eps_val, min_points=10))
            if labels.max() >= 0:
                counts = np.bincount(labels[labels >= 0])
                largest = np.where(labels == np.argmax(counts))[0]
                if len(largest) > 500:
                    pts = np.asarray(pcd_sub.points)[largest]
    except Exception as e:
        print(f"  [scale] clustering failed ({e}); using raw points")
        pts = np.asarray(pcd_sub.points)

    if len(pts) == 0:
        return 1.0
    center = pts.mean(axis=0)
    dists = np.linalg.norm(pts - center, axis=1)
    return np.percentile(dists, 90)


def smart_ror_filter(pcd, name, radius):
    """Apply radius outlier removal, but revert it on sparse clouds.

    On a sparse scene (e.g. Lecture Hall) ROR would delete most valid points, so
    if fewer than 80% of points survive we keep the original cloud and flag the
    scene as sparse. Returns ``(filtered_pcd, is_sparse)``.
    """
    if len(pcd.points) == 0:
        return pcd, False

    pcd_clean, _ = pcd.remove_radius_outlier(nb_points=5, radius=radius)
    survival_rate = len(pcd_clean.points) / len(pcd.points)

    if survival_rate < 0.8:
        print(f"  {name}: survival {survival_rate:.1%} < 80% -> sparse scene, reverting ROR")
        return pcd, True

    removed = len(pcd.points) - len(pcd_clean.points)
    print(f"  {name}: removed {removed:,} points (survival {survival_rate:.1%})")
    return pcd_clean, False


def prepare_for_registration(pcd, voxel_size, max_points=150_000):
    """Downsample, cap point count, and compute FPFH features for RANSAC."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    if len(pcd_down.points) > max_points:
        pcd_down = pcd_down.random_down_sample(max_points / len(pcd_down.points))

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return pcd_down, pcd_fpfh


def main(args):
    set_seed(42)
    add_path_to_dust3r(args.weights)
    from eval.mv_recon.data_long3d import Long3D
    from eval.mv_recon.utils import accuracy, completion, compute_chamfer_distance

    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    elif args.size == 518:
        resolution = (518, 392)
    else:
        raise NotImplementedError(f"Unsupported size: {args.size}")

    datasets_all = {
        "Long3D": Long3D(
            split="test",
            ROOT=args.root,
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=args.kf_every,
            max_frames=args.max_frames,
            test_id=args.scene,
            lightweight=True,
        ),
    }

    accelerator = Accelerator()
    device = accelerator.device

    if args.model_name == "OVGGT":
        from ovggt.models.ovggt import OVGGT
        model = OVGGT(total_budget=args.max_token_budget, camera_budget=args.camera_budget)
    elif args.model_name == "VGGT":
        from vggt.models.vggt import VGGT
        model = VGGT()
    else:
        raise NotImplementedError(f"Unsupported model: {args.model_name}")

    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    model = model.to(device)
    del ckpt

    os.makedirs(args.output_dir, exist_ok=True)

    # ICP correspondence thresholds (metres).
    icp_coarse_thresh = 1.0
    icp_fine_thresh = 0.2

    print(
        f"Config | voxel_size={args.voxel_size} max_points={args.max_points:,} "
        f"SOR(nb={args.sor_neighbors}, std={args.sor_std_ratio}) "
        f"ICP(coarse={icp_coarse_thresh}m, fine={icp_fine_thresh}m)"
    )

    with torch.no_grad():
        for name_data, dataset in datasets_all.items():
            save_path = osp.join(args.output_dir, name_data)
            os.makedirs(save_path, exist_ok=True)
            log_file = osp.join(save_path, f"logs_{accelerator.process_index}.txt")

            acc_all = comp_all = nc_all = nc1_all = nc2_all = chamfer_all = 0.0
            run_time = 0.0
            frame_num = 0
            run_peak_alloc = 0

            with accelerator.split_between_processes(list(range(len(dataset)))) as idxs:
                for data_idx in tqdm(idxs):
                    frames = lightweight_collate([dataset[data_idx]])
                    label = frames[0]["label"]
                    if isinstance(label, (list, tuple)):
                        label = label[0]
                    scene_label = label.split("/")[0]

                    print(f"\n=== Scene: {scene_label} | Frames: {len(frames)} ===")
                    gt_pcd_path = dataset.get_ground_truth_pcd_path(scene_label)

                    # 1. Inference.
                    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                    with torch.cuda.amp.autocast(dtype=dtype):
                        torch.cuda.reset_peak_memory_stats()
                        start_inference = time.time()
                        results = model.inference(frames)
                        torch.cuda.synchronize()
                        total_inference_time = time.time() - start_inference
                        batch_peak_alloc = torch.cuda.max_memory_allocated()

                    run_peak_alloc = max(run_peak_alloc, batch_peak_alloc)
                    run_time += total_inference_time
                    frame_num += len(frames)
                    print(
                        f"Inference: {total_inference_time:.2f}s "
                        f"({len(frames) / total_inference_time:.2f} FPS), "
                        f"peak GPU {batch_peak_alloc / 1024**3:.2f} GB"
                    )

                    # 2. Fuse predictions into a raw point cloud.
                    start_process = time.time()
                    pts_flat, colors_flat = process_predictions_to_pointcloud(results.ress, frames, args)
                    process_time = time.time() - start_process
                    del results
                    torch.cuda.empty_cache()
                    gc.collect()

                    pcd_pred_raw = o3d.geometry.PointCloud()
                    pcd_pred_raw.points = o3d.utility.Vector3dVector(pts_flat)
                    pcd_pred_raw.colors = o3d.utility.Vector3dVector(colors_flat)
                    raw_point_count = len(pts_flat)
                    del pts_flat, colors_flat
                    gc.collect()
                    print(f"Fused {raw_point_count:,} raw points ({process_time:.2f}s)")

                    # 3. Subsample as a safety valve against memory blow-ups.
                    if args.max_points is not None and raw_point_count > args.max_points:
                        pcd_pred_raw = pcd_pred_raw.random_down_sample(args.max_points / raw_point_count)
                        print(f"Subsampled to {len(pcd_pred_raw.points):,} points")

                    # 4. Statistical outlier removal before scale estimation.
                    pts_before_sor = len(pcd_pred_raw.points)
                    pcd_pred_clean, _ = pcd_pred_raw.remove_statistical_outlier(
                        nb_neighbors=args.sor_neighbors, std_ratio=args.sor_std_ratio
                    )
                    pts_after_sor = len(pcd_pred_clean.points)
                    print(f"SOR: {pts_before_sor:,} -> {pts_after_sor:,} points")
                    del pcd_pred_raw
                    gc.collect()

                    # 5. Load GT and estimate a global scale factor.
                    pcd_gt = o3d.io.read_point_cloud(gt_pcd_path)
                    p90_gt = get_statistical_scale(pcd_gt)
                    p90_pred = get_statistical_scale(pcd_pred_clean)
                    scale_factor = p90_gt / p90_pred if p90_pred > 1e-6 else 1.0
                    print(f"Scale: GT_p90={p90_gt:.4f} Pred_p90={p90_pred:.4f} -> factor={scale_factor:.4f}")
                    pcd_pred_clean.scale(scale_factor, center=pcd_pred_clean.get_center())

                    # 6. Voxel downsample both clouds to metric resolution.
                    pcd_pred_ds = pcd_pred_clean.voxel_down_sample(args.voxel_size)
                    pcd_gt_ds = pcd_gt.voxel_down_sample(args.voxel_size)
                    print(
                        f"Downsampled: pred {len(pcd_pred_clean.points):,}->{len(pcd_pred_ds.points):,}, "
                        f"GT {len(pcd_gt.points):,}->{len(pcd_gt_ds.points):,}"
                    )
                    del pcd_pred_clean, pcd_gt
                    gc.collect()

                    # 7. Radius outlier removal (auto-reverted on sparse scenes).
                    pcd_pred_ds, sparse_pred = smart_ror_filter(pcd_pred_ds, "Pred", args.voxel_size * 4)
                    pcd_gt_ds, sparse_gt = smart_ror_filter(pcd_gt_ds, "GT", args.voxel_size * 4)
                    is_sparse_scene = sparse_pred or sparse_gt

                    # 8. Alignment: RANSAC global registration + ICP refinement.
                    start_alignment = time.time()
                    reg_voxel_size = 0.05
                    source_down, src_fpfh = prepare_for_registration(pcd_pred_ds, reg_voxel_size)
                    target_down, tgt_fpfh = prepare_for_registration(pcd_gt_ds, reg_voxel_size)
                    distance_threshold = 1.5 if is_sparse_scene else reg_voxel_size * 5

                    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                        source_down, target_down, src_fpfh, tgt_fpfh,
                        mutual_filter=True,
                        max_correspondence_distance=distance_threshold,
                        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
                        ransac_n=3,
                        checkers=[
                            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
                            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                        ],
                        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(10000000, 500),
                    )
                    print(f"RANSAC: fitness={result_ransac.fitness:.4f} RMSE={result_ransac.inlier_rmse:.4f}")

                    pcd_pred_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                    pcd_gt_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

                    reg_coarse = o3d.pipelines.registration.registration_icp(
                        pcd_pred_ds, pcd_gt_ds, icp_coarse_thresh, result_ransac.transformation,
                        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
                    )
                    reg_fine = o3d.pipelines.registration.registration_icp(
                        pcd_pred_ds, pcd_gt_ds, icp_fine_thresh, reg_coarse.transformation,
                        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
                    )
                    alignment_time = time.time() - start_alignment
                    print(
                        f"ICP: coarse fitness={reg_coarse.fitness:.4f}, "
                        f"fine fitness={reg_fine.fitness:.4f} RMSE={reg_fine.inlier_rmse:.4f} "
                        f"({alignment_time:.2f}s)"
                    )

                    pcd_pred_aligned = pcd_pred_ds.transform(reg_fine.transformation)

                    o3d.io.write_point_cloud(osp.join(save_path, f"{scene_label}_pred_aligned.ply"), pcd_pred_aligned)
                    o3d.io.write_point_cloud(osp.join(save_path, f"{scene_label}_gt_ds.ply"), pcd_gt_ds)

                    # 9. Metrics.
                    pcd_pred_aligned.estimate_normals()
                    pcd_gt_ds.estimate_normals()
                    gt_normal = np.asarray(pcd_gt_ds.normals)
                    pred_normal = np.asarray(pcd_pred_aligned.normals)

                    acc, acc_med, nc1, nc1_med = accuracy(
                        pcd_gt_ds.points, pcd_pred_aligned.points, gt_normal, pred_normal
                    )
                    comp, comp_med, nc2, nc2_med = completion(
                        pcd_gt_ds.points, pcd_pred_aligned.points, gt_normal, pred_normal
                    )
                    chamfer_dist = compute_chamfer_distance(
                        np.asarray(pcd_pred_aligned.points), np.asarray(pcd_gt_ds.points), max_dist=4.0
                    )
                    nc = (nc1 + nc2) / 2
                    nc_med = (nc1_med + nc2_med) / 2

                    log_str = (
                        f"Scene: {scene_label}, "
                        f"Acc: {acc:.8f}, Comp: {comp:.8f}, "
                        f"NC: {nc:.8f}, NC1: {nc1:.8f}, NC2: {nc2:.8f}, Chamfer: {chamfer_dist:.8f} - "
                        f"Acc_med: {acc_med:.8f}, Comp_med: {comp_med:.8f}, "
                        f"NC_med: {nc_med:.8f}, NC1_med: {nc1_med:.8f}, NC2_med: {nc2_med:.8f}"
                    )
                    print(log_str)
                    with open(log_file, "a") as f:
                        print(log_str, file=f)

                    acc_all += acc
                    comp_all += comp
                    nc_all += nc
                    nc1_all += nc1
                    nc2_all += nc2
                    chamfer_all += chamfer_dist

                    del pcd_pred_aligned, pcd_gt_ds, gt_normal, pred_normal
                    torch.cuda.empty_cache()
                    gc.collect()

                num_scenes = len(idxs)

            accelerator.wait_for_everyone()

            # Per-process summary (run with a single process for global averages).
            if accelerator.is_main_process and num_scenes > 0:
                avg_fps = frame_num / run_time if run_time > 0 else 0
                with open(osp.join(save_path, "logs_all.txt"), "w") as f:
                    f.write(
                        f"voxel_size={args.voxel_size}, max_points={args.max_points:,}, "
                        f"SOR(nb={args.sor_neighbors}, std={args.sor_std_ratio}), "
                        f"ICP(coarse={icp_coarse_thresh}m, fine={icp_fine_thresh}m)\n"
                    )
                    f.write(f"Average FPS: {avg_fps:.2f}\n")
                    f.write(f"Total time: {run_time:.2f}s, Total frames: {frame_num}\n")
                    f.write(f"Peak memory: {run_peak_alloc / 1024**3:.2f} GB allocated\n\n")
                    f.write("Average metrics ({} scenes):\n".format(num_scenes))
                    f.write(f"  Accuracy:   {acc_all / num_scenes:.4f}\n")
                    f.write(f"  Completion: {comp_all / num_scenes:.4f}\n")
                    f.write(f"  NC:         {nc_all / num_scenes:.4f}\n")
                    f.write(f"  NC1:        {nc1_all / num_scenes:.4f}\n")
                    f.write(f"  NC2:        {nc2_all / num_scenes:.4f}\n")
                    f.write(f"  Chamfer:    {chamfer_all / num_scenes:.4f}\n")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    print(args)
    main(args)
