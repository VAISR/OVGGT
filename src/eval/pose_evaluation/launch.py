import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import math
import cv2
import numpy as np
import torch
import argparse

from copy import deepcopy
from eval.pose_evaluation.metadata import dataset_metadata
from eval.pose_evaluation.utils import *

from accelerate import PartialState
from ovggt.models.ovggt import OVGGT

from tqdm import tqdm
import time

def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        help="path to the model weights",
        default="",
    )

    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument(
        "--no_crop", type=bool, default=True, help="whether to crop input data"
    )

    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="sintel",
        choices=list(dataset_metadata.keys()),
    )
    parser.add_argument("--size", type=int, default="224")

    parser.add_argument(
        "--pose_eval_stride", default=1, type=int, help="stride for pose evaluation"
    )
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument(
        "--full_seq",
        action="store_true",
        default=False,
        help="use full sequence for pose evaluation",
    )
    parser.add_argument(
        "--seq_list",
        nargs="+",
        default=None,
        help="list of sequences for pose evaluation",
    )

    parser.add_argument("--revisit", type=int, default=1)
    parser.add_argument("--freeze_state", action="store_true", default=False)
    parser.add_argument("--solve_pose", action="store_true", default=False)
    return parser


def eval_pose_estimation(args, model, save_dir=None):
    metadata = dataset_metadata.get(args.eval_dataset)
    img_path = metadata["img_path"]
    mask_path = metadata["mask_path"]

    ate_mean, rpe_trans_mean, rpe_rot_mean = eval_pose_estimation_dist(
        args, model, save_dir=save_dir, img_path=img_path, mask_path=mask_path
    )
    return ate_mean, rpe_trans_mean, rpe_rot_mean


def eval_pose_estimation_dist(args, model, img_path, save_dir=None, mask_path=None):

    metadata = dataset_metadata.get(args.eval_dataset)
    anno_path = metadata.get("anno_path", None)

    seq_list = args.seq_list
    if seq_list is None:
        if metadata.get("full_seq", False):
            args.full_seq = True
        else:
            seq_list = metadata.get("seq_list", [])
        if args.full_seq:
            seq_list = os.listdir(img_path)
            seq_list = [
                seq for seq in seq_list if os.path.isdir(os.path.join(img_path, seq))
            ]
        seq_list = sorted(seq_list)

    if save_dir is None:
        save_dir = args.output_dir

    distributed_state = PartialState()
    model.to(distributed_state.device)
    device = distributed_state.device

    with distributed_state.split_between_processes(seq_list) as seqs:
        ate_list = []
        rpe_trans_list = []
        rpe_rot_list = []
        load_img_size = args.size
        error_log_path = f"{save_dir}/_error_log_{distributed_state.process_index}.txt"  # Unique log file per process
        bug = False
        for seq in tqdm(seqs):
            try:
                dir_path = metadata["dir_path_func"](img_path, seq)

                # Handle skip_condition
                skip_condition = metadata.get("skip_condition", None)
                if skip_condition is not None and skip_condition(save_dir, seq):
                    continue

                mask_path_seq_func = metadata.get(
                    "mask_path_seq_func", lambda mask_path, seq: None
                )
                mask_path_seq = mask_path_seq_func(mask_path, seq)

                filelist = [
                    os.path.join(dir_path, name) for name in os.listdir(dir_path)
                ]
                filelist.sort()
                filelist = filelist[:: args.pose_eval_stride]

                images = load_and_preprocess_images(filelist).to(device)
                frames = []
                for i in range(images.shape[0]):
                    image = images[i].unsqueeze(0) 
                    frame = {
                        "img": image
                    }
                    frames.append(frame)

                start = time.time()
                predictions = {}
                with torch.no_grad():
                    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                    with torch.cuda.amp.autocast(dtype=dtype):
                        output = model.inference(frames)
                end = time.time()
                fps = len(filelist) / (end - start)
                print(f"Finished pose estimation for {args.eval_dataset} {seq: <16}, FPS: {fps:.2f}")

                all_camera_pose = []
                for res in output.ress:
                    all_camera_pose.append(res['camera_pose'].squeeze(0))
                    
                predictions["pose_enc"] = torch.stack(all_camera_pose, dim=0) # (S, 9)
                extrinsic, intrinsic = pose_encoding_to_extri_intri(
                        predictions["pose_enc"].unsqueeze(0) if predictions["pose_enc"].ndim == 2 else predictions["pose_enc"], 
                        images.shape[-2:]
                    )
                predictions["extrinsic"] = extrinsic.squeeze(0)  # (S, 3, 4)
                predictions["intrinsic"] = intrinsic.squeeze(0) if intrinsic is not None else None # (S, 3, 3)

                # Convert (S, 3, 4) w2c extrinsics to (S, 4, 4) c2w poses
                # pose_encoding_to_extri_intri outputs w2c (world-to-camera) extrinsics,
                # but get_tum_poses / c2w_to_tumpose expects c2w (camera-to-world) convention.
                add_row = torch.tensor([0, 0, 0, 1], device=predictions["extrinsic"].device, dtype=predictions["extrinsic"].dtype).expand(predictions["extrinsic"].size(0), 1, 4)
                w2c_poses = torch.cat((predictions["extrinsic"], add_row), dim=1) # (S, 4, 4) world-to-camera
                pr_poses = torch.linalg.inv(w2c_poses) # (S, 4, 4) camera-to-world

                # Extract focal length and principal point from intrinsics for saving
                if predictions["intrinsic"] is not None:
                    focals_x = predictions["intrinsic"][:, 0, 0]
                    focals_y = predictions["intrinsic"][:, 1, 1]
                    focal = (focals_x + focals_y) / 2.0 # Average focal length
                    pp = predictions["intrinsic"][:, :2, 2] # Principal points (S, 2)
                    cam_dict = {
                        "focal": focal.cpu().numpy(),
                        "pp": pp.cpu().numpy(),
                    }
                else: # Fallback if no intrinsics are predicted
                    H, W = images.shape[-2:]
                    cam_dict = {
                        "focal": np.full(len(images), max(H, W)), # A common heuristic
                        "pp": np.tile([W/2, H/2], (len(images), 1)),
                    }

                pred_traj = get_tum_poses(pr_poses)
                os.makedirs(f"{save_dir}/{seq}", exist_ok=True)
                save_tum_poses(pr_poses, f"{save_dir}/{seq}/pred_traj.txt")
                save_focals(cam_dict, f"{save_dir}/{seq}/pred_focal.txt")
                pose_save_path = os.path.join(save_dir, f"{seq}_poses.npz")
                np.savez(
                    pose_save_path,
                    pose_enc=predictions["pose_enc"].cpu().numpy(),
                    extrinsic=predictions["extrinsic"].cpu().numpy()
                )

                print(f"Pose encoding and extrinsics saved to: {pose_save_path}")


                gt_traj_file = metadata["gt_traj_func"](img_path, anno_path, seq)
                traj_format = metadata.get("traj_format", None)

                if args.eval_dataset == "sintel":
                    gt_traj = load_traj(
                        gt_traj_file=gt_traj_file, stride=args.pose_eval_stride
                    )
                elif traj_format is not None:
                    gt_traj = load_traj(
                        gt_traj_file=gt_traj_file,
                        traj_format=traj_format,
                        stride=args.pose_eval_stride,
                    )
                else:
                    gt_traj = None

                if gt_traj is not None:
                    ate, rpe_trans, rpe_rot = eval_metrics(
                        pred_traj,
                        gt_traj,
                        seq=seq,
                        filename=f"{save_dir}/{seq}_eval_metric.txt",
                    )
                    plot_trajectory(
                        pred_traj, gt_traj, title=seq, filename=f"{save_dir}/{seq}.png"
                    )
                else:
                    ate, rpe_trans, rpe_rot = 0, 0, 0
                    bug = True

                ate_list.append(ate)
                rpe_trans_list.append(rpe_trans)
                rpe_rot_list.append(rpe_rot)

                # Write to error log after each sequence
                with open(error_log_path, "a") as f:
                    f.write(
                        f"{args.eval_dataset}-{seq: <16} | ATE: {ate:.5f}, RPE trans: {rpe_trans:.5f}, RPE rot: {rpe_rot:.5f}\n"
                    )
                    f.write(f"{ate:.5f}\n")
                    f.write(f"{rpe_trans:.5f}\n")
                    f.write(f"{rpe_rot:.5f}\n")

            except Exception as e:
                if "out of memory" in str(e):
                    # Handle OOM
                    torch.cuda.empty_cache()  # Clear the CUDA memory
                    with open(error_log_path, "a") as f:
                        f.write(
                            f"OOM error in sequence {seq}, skipping this sequence.\n"
                        )
                    print(f"OOM error in sequence {seq}, skipping...")
                elif "Degenerate covariance rank" in str(
                    e
                ) or "Eigenvalues did not converge" in str(e):
                    # Handle Degenerate covariance rank exception and Eigenvalues did not converge exception
                    with open(error_log_path, "a") as f:
                        f.write(f"Exception in sequence {seq}: {str(e)}\n")
                    print(f"Traj evaluation error in sequence {seq}, skipping.")
                else:
                    raise e  # Rethrow if it's not an expected exception

    distributed_state.wait_for_everyone()

    results = process_directory(save_dir)
    avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)

    # Write the averages to the error log (only on the main process)
    if distributed_state.is_main_process:
        with open(f"{save_dir}/_error_log.txt", "a") as f:
            # Copy the error log from each process to the main error log
            for i in range(distributed_state.num_processes):
                if not os.path.exists(f"{save_dir}/_error_log_{i}.txt"):
                    break
                with open(f"{save_dir}/_error_log_{i}.txt", "r") as f_sub:
                    f.write(f_sub.read())
            f.write(
                f"Average ATE: {avg_ate:.5f}, Average RPE trans: {avg_rpe_trans:.5f}, Average RPE rot: {avg_rpe_rot:.5f}\n"
            )

    return avg_ate, avg_rpe_trans, avg_rpe_rot


if __name__ == "__main__":
    seed = 42
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import open3d as o3d
        o3d.utility.random.seed(seed)
    except (ImportError, AttributeError):
        pass
    args = get_args_parser()
    args = args.parse_args()
    from ovggt.utils.load_fn import load_and_preprocess_images 
    from ovggt.utils.pose_enc import pose_encoding_to_extri_intri
    from dust3r.utils.image import load_images_for_eval as load_images

    args.full_seq = False
    args.no_crop = False

    print("Loading OVGGT model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OVGGT()
    checkpoint = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    print("Model loaded successfully.")

    eval_pose_estimation(args, model, save_dir=args.output_dir)