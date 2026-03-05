"""
demo_viser.py
OVGGT Inference & Visualization Pipeline
Ported from InfiniteVGGT (https://github.com/AutoLab-SAI-SJTU/InfiniteVGGT)
"""

import os
import sys
import argparse
import glob
import tempfile
import shutil
import numpy as np
import torch
import cv2
import imageio.v2 as iio
from natsort import natsorted

sys.path.append("src/")
from ovggt.models.ovggt import OVGGT
from ovggt.utils.load_fn import load_and_preprocess_images
from ovggt.utils.pose_enc import pose_encoding_to_extri_intri
from viser_utils import PointCloudViewer

def get_image_paths(seq_path, interval=1):
    """Returns list of image paths. Extracts frames if video."""
    tmp_dir = None
    if os.path.isdir(seq_path):
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.webp'}
        paths = [p for p in natsorted(glob.glob(os.path.join(seq_path, "*"))) 
                 if os.path.splitext(p)[1].lower() in exts]
        return paths[::interval], None
    
    # Video handling
    cap = cv2.VideoCapture(seq_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {seq_path}")
    
    tmp_dir = tempfile.mkdtemp()
    paths = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in range(0, total, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: break
        p = os.path.join(tmp_dir, f"{i:06d}.jpg")
        cv2.imwrite(p, frame)
        paths.append(p)
    cap.release()
    return paths, tmp_dir

def load_gt_poses(path, interval=1, max_frames=None):
    if not path or not os.path.exists(path): return None
    poses = np.loadtxt(path).reshape(-1, 4, 4)[::interval]
    if max_frames: poses = poses[:max_frames]
    return poses

def align_sim3(gt_poses, pred_centers):
    
    n = min(len(gt_poses), len(pred_centers))
    gt_p, pred_p = gt_poses[:n, :3, 3], pred_centers[:n]
    
    # Center alignment
    mu_gt, mu_pred = gt_p.mean(0), pred_p.mean(0)
    gt_c, pred_c = gt_p - mu_gt, pred_p - mu_pred
    
    # Rotation & Scale
    H = gt_c.T @ pred_c
    U, S, Vh = np.linalg.svd(H)
    R = Vh.T @ U.T
    if np.linalg.det(R) < 0:
        Vh[2] *= -1; R = Vh.T @ U.T
    
    scale = S.sum() / np.trace(gt_c.T @ gt_c) if S.sum() > 1e-6 else 1.0
    
    # Anchor to t0
    gt_t0_aligned = scale * (R @ gt_poses[0, :3, 3])
    t_anchor = pred_centers[0] - gt_t0_aligned
    
    aligned = []
    for pose in gt_poses[:n]:
        new_pose = np.eye(4, dtype=np.float32)
        new_pose[:3, :3] = R @ pose[:3, :3]
        new_pose[:3, 3] = scale * (R @ pose[:3, 3]) + t_anchor
        aligned.append(new_pose)
    return np.array(aligned)

@torch.no_grad()
def run_inference(model, img_paths, device, size=512):
    images = load_and_preprocess_images(img_paths).to(device)
    inputs = [{"img": img.unsqueeze(0)} for img in images]
    
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.amp.autocast('cuda', dtype=dtype):
        output = model.inference(inputs)

    # Unpack structured results directly to tensors
    res_keys = ['pts3d_in_other_view', 'conf', 'depth', 'depth_conf', 'camera_pose']
    raw = {k: torch.stack([r[k].squeeze(0) for r in output.ress]) for k in res_keys}
    
    # Process camera poses
    pose_enc = raw['camera_pose']
    pose_enc_in = pose_enc.unsqueeze(0) if pose_enc.ndim == 2 else pose_enc
    ext, intr = pose_encoding_to_extri_intri(pose_enc_in, images.shape[-2:])
    
    return {
        "world_points": raw['pts3d_in_other_view'].cpu().numpy(),
        "conf": raw['conf'].cpu().numpy(),
        "depth": raw['depth'].squeeze(-1).cpu().numpy(),
        "images": images.permute(0, 2, 3, 1).cpu().numpy(),
        "extrinsic": ext.squeeze(0).cpu().numpy(),
        "intrinsic": intr.squeeze(0).cpu().numpy() if intr is not None else None
    }

def save_and_format(data, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    subdirs = {k: os.path.join(out_dir, k) for k in ['depth', 'conf', 'color', 'camera']}
    for d in subdirs.values(): os.makedirs(d, exist_ok=True)

    N, H, W, _ = data["images"].shape
    
    # Vectorized Camera Math (w2c -> c2w)
    w2c = np.eye(4, dtype=np.float32)[None].repeat(N, 0)
    w2c[:, :3, :] = data["extrinsic"]
    c2w = np.linalg.inv(w2c)
    
    R_list, t_list = c2w[:, :3, :3], c2w[:, :3, 3]
    
    # Intrinsics
    if data["intrinsic"] is not None:
        focal = data["intrinsic"][:, 0, 0]
        pp = data["intrinsic"][:, :2, 2]
    else:
        focal = np.full(N, W / 2.0)
        pp = np.tile([W/2.0, H/2.0], (N, 1))

    # Saving loop
    for i in range(N):
        np.save(f"{subdirs['depth']}/{i:06d}.npy", data["depth"][i])
        np.save(f"{subdirs['conf']}/{i:06d}.npy", data["conf"][i])
        iio.imwrite(f"{subdirs['color']}/{i:06d}.png", (data["images"][i] * 255).astype(np.uint8))
        
        # Save camera dict
        intr = np.eye(3)
        intr[0,0] = intr[1,1] = focal[i]
        intr[:2,2] = pp[i]
        np.savez(f"{subdirs['camera']}/{i:06d}.npz", pose=c2w[i], intrinsics=intr)

    return {
        "R": R_list, "t": t_list, "focal": focal, "pp": pp,
        "pts": list(data["world_points"]), 
        "colors": list(data["images"]), 
        "confs": list(data["conf"])
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_path", default="./examples", help="Input folder or video")
    parser.add_argument("--gt_path", default=None, help="GT poses (optional)")
    parser.add_argument("--output_dir", default="./demo_tmp")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--vis_threshold", type=float, default=1.5)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--ckpt", default="./ckpt/checkpoints.pth", help="Path to StreamVGGT checkpoint")
    args = parser.parse_args()

    # Load Data
    print(f"Loading from {args.seq_path}...")
    img_paths, tmp_dir = get_image_paths(args.seq_path, args.frame_interval)
    
    # Load Model
    ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}, downloading from Hugging Face...")
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(
            repo_id="lch01/StreamVGGT",
            filename="checkpoints.pth",
            revision="main",
        )
        print(f"Downloaded checkpoint to {ckpt_path}")

    model = OVGGT().to(args.device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    del ckpt
    model.eval()

    # Inference & Save
    preds = run_inference(model, img_paths, args.device, args.size)
    vis_data = save_and_format(preds, args.output_dir)

    # Alignment (if GT exists)
    gt_poses = load_gt_poses(args.gt_path, args.frame_interval, len(vis_data['t']))
    if gt_poses is not None:
        gt_poses = align_sim3(gt_poses, vis_data['t'])

    # Visualization
    print(f"Launching Viser on port {args.port}...")
    viewer = PointCloudViewer(
        model, None, 
        vis_data['pts'], vis_data['colors'], vis_data['confs'],
        {"R": vis_data['R'], "t": vis_data['t'], "focal": vis_data['focal'], "pp": vis_data['pp']},
        gt_poses=gt_poses,
        device=args.device,
        vis_threshold=args.vis_threshold,
        size=args.size,
        port=args.port,
        edge_color_list=[None]*len(img_paths),
        show_camera=True
    )
    viewer.run()

    if tmp_dir: shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    main()