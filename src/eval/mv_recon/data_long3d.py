import os
import os.path as osp
import random
from collections import deque

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from eval.mv_recon.base import BaseStereoViewDataset
import eval.mv_recon.dataset_utils.cropping as cropping
from dust3r.utils.image import imread_cv2


def shuffle_deque(dq, seed=None):
    """Return a shuffled copy of a deque (optionally seeded for reproducibility)."""
    if seed is not None:
        random.seed(seed)
    shuffled_list = list(dq)
    random.shuffle(shuffled_list)
    return deque(shuffled_list)


class Long3D(BaseStereoViewDataset):
    """Long3D dataset loader for long-sequence 3D reconstruction evaluation.

    Expected directory layout::

        ROOT/
            <scene_name>/
                dense_cloud_map.pcd          # ground-truth point cloud
                images/
                    scan_images/
                        <timestamp>.jpg      # RGB frames

    Long3D only ships RGB frames and a ground-truth point cloud; it has no
    per-frame depth maps or camera poses, so those are left for the model to
    predict. With ``lightweight=True`` (the default used for evaluation) only the
    RGB frames are loaded, which keeps memory usage low on long sequences. With
    ``lightweight=False`` the loader also produces placeholder depth/pose tensors
    so it can be consumed through the standard ``BaseStereoViewDataset`` pipeline.
    """

    def __init__(
        self,
        num_seq=1,
        num_frames=100,
        test_id=None,
        full_video=False,
        kf_every=10,
        shuffle_seed=-1,
        max_frames=None,
        lightweight=True,
        *args,
        ROOT,
        **kwargs,
    ):
        self.ROOT = ROOT
        self.lightweight = lightweight
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.num_frames = num_frames
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self.shuffle_seed = shuffle_seed
        self.max_frames = max_frames

        if self.lightweight:
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

        self.load_all_scenes(ROOT)

    def __len__(self):
        return len(self.scene_list) * self.num_seq

    def __getitem__(self, idx):
        # Lightweight mode returns the RGB-only views directly; otherwise fall
        # back to the base-class pipeline (which calls ``_get_views``).
        if not self.lightweight:
            return super().__getitem__(idx)

        if self.seed:
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, "_rng"):
            self._rng = np.random.default_rng(seed=torch.initial_seed())

        resolution = self._resolutions[0]  # always use the first resolution
        return self._get_views_lightweight(idx, resolution, self._rng)

    @staticmethod
    def _extract_timestamp(filename):
        """Sort key understanding the two Long3D filename formats.

        Most scenes name frames ``<timestamp>.jpg``; a few (e.g. Classroom) use
        ``frame_<idx>_<timestamp>_undistort.png``.
        """
        if filename.startswith("frame_"):
            parts = filename.split("_")
            if len(parts) >= 3:
                try:
                    return float(parts[2])
                except (ValueError, IndexError):
                    pass
        try:
            return float(filename.split(".jpg")[0].split(".png")[0])
        except ValueError:
            return 0.0

    def _list_scene_frames(self, scene_id):
        """Return the image directory and the time-sorted frame filenames."""
        image_dir = osp.join(self.ROOT, scene_id, "images", "scan_images")
        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory not found: {image_dir}")
        images = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))]
        images.sort(key=self._extract_timestamp)
        return image_dir, images

    def _sample_frames(self, all_images, seq_idx, rng):
        """Select which frames to evaluate for one sequence."""
        if self.full_video:
            img_idxs = all_images[:: self.kf_every]
            if self.max_frames is not None:
                img_idxs = img_idxs[: self.max_frames]
            return img_idxs

        # Random subset (used for ablations rather than full-video evaluation).
        if self.shuffle_seed >= 0:
            rng_local = np.random.default_rng(self.shuffle_seed + seq_idx)
            indices = rng_local.choice(
                len(all_images), min(self.num_frames, len(all_images)), replace=False
            )
            return [all_images[i] for i in sorted(indices)]

        rng.shuffle(all_images)
        return all_images[: self.num_frames]

    def _get_views_lightweight(self, idx, resolution, rng):
        """Load RGB frames only (no depth / pose), to save memory on long sequences."""
        scene_id = self.scene_list[idx // self.num_seq]
        seq_idx = idx % self.num_seq

        image_dir, all_images = self._list_scene_frames(scene_id)
        img_idxs = self._sample_frames(all_images, seq_idx, rng)

        target_size = resolution if isinstance(resolution, tuple) else (resolution, resolution)

        views = []
        for v_idx, im_idx in enumerate(img_idxs):
            impath = osp.join(image_dir, im_idx)
            rgb_np = imread_cv2(impath)  # (H, W, 3)
            rgb_image = Image.fromarray(rgb_np).resize(target_size, Image.Resampling.LANCZOS)
            img_tensor = self.img_transform(rgb_image).unsqueeze(0)  # (1, C, H, W)

            views.append({
                "img": img_tensor,
                "label": osp.join(scene_id, im_idx),
                "instance": impath,
                "dataset": "long3d",
                "idx": v_idx,
            })

        return views

    def _get_views(self, idx, resolution, rng):
        """Load views with placeholder depth/pose for the standard pipeline.

        Long3D has no ground-truth depth or camera poses, so depth maps are filled
        with zeros, poses with the identity, and intrinsics with a rough estimate
        derived from the image size; all of these are expected to be predicted by
        the model rather than used as supervision.
        """
        scene_id = self.scene_list[idx // self.num_seq]
        seq_idx = idx % self.num_seq

        image_dir, all_images = self._list_scene_frames(scene_id)
        img_idxs = self._sample_frames(all_images, seq_idx, rng)

        views = []
        imgs_idxs = deque(img_idxs)
        if self.shuffle_seed >= 0 and not self.full_video:
            imgs_idxs = shuffle_deque(imgs_idxs, seed=self.shuffle_seed)

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()
            impath = osp.join(image_dir, im_idx)

            rgb_image = imread_cv2(impath)
            h, w = rgb_image.shape[:2]

            # Rough intrinsics estimate (no calibration is provided).
            intrinsics = np.array([
                [max(w, h) * 0.8, 0, w / 2],
                [0, max(w, h) * 0.8, h / 2],
                [0, 0, 1],
            ], dtype=np.float32)

            # Placeholder depth (zeros) and pose (identity); predicted by the model.
            depthmap = np.zeros((h, w), dtype=np.float32)
            camera_pose = np.eye(4, dtype=np.float32)

            if resolution != (224, 224):
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath
                )
            else:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, (512, 384), rng=rng, info=impath
                )
                W, H = rgb_image.size
                cx, cy = W // 2, H // 2
                crop_bbox = (cx - 112, cy - 112, cx + 112, cy + 112)
                rgb_image, depthmap, intrinsics = cropping.crop_image_depthmap(
                    rgb_image, depthmap, intrinsics, crop_bbox
                )

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset="long3d",
                label=osp.join(scene_id, im_idx),
                instance=impath,
            ))

        return views

    def load_all_scenes(self, base_dir):
        """Discover the scenes to evaluate under ``base_dir``."""
        scenes = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d not in [".cache"]
        ]

        if self.test_id is not None:
            self.scene_list = self.test_id if isinstance(self.test_id, list) else [self.test_id]
        else:
            self.scene_list = sorted(scenes)

        print(f"Found {len(self.scene_list)} scene(s) in Long3D split '{self.split}'")

    def get_ground_truth_pcd_path(self, scene_id):
        """Return the path to a scene's ground-truth point cloud."""
        pcd_path = osp.join(self.ROOT, scene_id, "dense_cloud_map.pcd")
        if not os.path.exists(pcd_path):
            raise ValueError(f"Ground truth PCD not found: {pcd_path}")
        return pcd_path
