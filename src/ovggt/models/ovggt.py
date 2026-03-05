import time
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from ovggt.models.aggregator import Aggregator
from ovggt.heads.camera_head import CameraHead
from ovggt.heads.dpt_head import DPTHead
from ovggt.heads.track_head import TrackHead
from ovggt.utils.history_anchor import HistoryAnchorConfig, HistoryAnchorManager
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple, List, Any, Callable
from dataclasses import dataclass

@dataclass
class OVGGTOutput(ModelOutput):
    ress: Optional[List[dict]] = None
    views: Optional[torch.Tensor] = None

class OVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        total_budget=200000,
        camera_budget=384,
        eviction_strategy='repr_shift_spatial',
        intra_frame_keep_ratio=1.0,
        spatial_alpha=0.5,
        importance_weight: float = 0.5,
    ):
        super().__init__()

        self.intra_frame_keep_ratio = intra_frame_keep_ratio
        self.spatial_alpha = spatial_alpha
        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            eviction_strategy=eviction_strategy,
            intra_frame_keep_ratio=intra_frame_keep_ratio,
            spatial_alpha=spatial_alpha,
        )
        self.camera_head = CameraHead(dim_in=2 * embed_dim, total_budget=camera_budget)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
        self.total_budget = total_budget
        self.eviction_strategy = eviction_strategy
        self.importance_weight = importance_weight
    


    def forward(
        self,
        views,
        query_points: torch.Tensor = None,
        history_info: Optional[dict] = None,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0
    ):
        images = torch.stack(
            [view["img"] for view in views], dim=0
        ).permute(1, 0, 2, 3, 4)    # B S C H W

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        if history_info is None:
            history_info = {"token": None}

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.track_head is not None and query_points is not None:
                track_list, vis, conf = self.track_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                predictions["track"] = track_list[-1]  # track of the last iteration
                predictions["vis"] = vis
                predictions["conf"] = conf
            predictions["images"] = images

            B, S = images.shape[:2]
            ress = []
            for s in range(S):
                res = {
                    'pts3d_in_other_view': predictions['world_points'][:, s],  # [B, H, W, 3]
                    'conf': predictions['world_points_conf'][:, s],  # [B, H, W]

                    'depth': predictions['depth'][:, s],  # [B, H, W, 1]
                    'depth_conf': predictions['depth_conf'][:, s],  # [B, H, W]
                    'camera_pose': predictions['pose_enc'][:, s, :],  # [B, 9]

                    **({'valid_mask': views[s]["valid_mask"]}
                    if 'valid_mask' in views[s] else {}),  # [B, H, W]

                    **({'track': predictions['track'][:, s],  # [B, N, 2]
                        'vis': predictions['vis'][:, s],  # [B, N]
                        'track_conf': predictions['conf'][:, s]}
                    if 'track' in predictions else {})
                }
                ress.append(res)
            return OVGGTOutput(ress=ress, views=views)  # [S] [B, C, H, W]
    
    def inference(
        self,
        frames,
        query_points: torch.Tensor = None,
        past_key_values=None,
        frame_writer: Optional[Callable[[int, dict, dict], None]] = None,
        cache_results: bool = True,
        history_anchor_strategy: str = 'coverage',
        anchor_interval: int = 250,
        max_anchors: int = 3,
        coverage_threshold: float = 0.2,
        anchor_keep_ratio: float = 0.05,
    ):
        past_key_values = [None] * self.aggregator.depth
        past_key_values_camera = [None] * self.camera_head.trunk_depth
        total_budget = self.total_budget
        importance_weight = self.importance_weight

        # Calculate tokens per frame: camera(1) + register(4) + patches
        # For 518x392 with patch_size=14: 1 + 4 + (37 * 28) = 1041
        img_h, img_w = 392, 518  # Default from resolution
        if len(frames) > 0 and 'img' in frames[0]:
            sample_img = frames[0]['img']
            if sample_img.dim() == 3:  # [C, H, W]
                img_h, img_w = sample_img.shape[1], sample_img.shape[2]
            elif sample_img.dim() == 4:  # [B, C, H, W]
                img_h, img_w = sample_img.shape[2], sample_img.shape[3]
        patch_size = self.aggregator.patch_size
        num_patches = (img_h // patch_size) * (img_w // patch_size)
        tokens_per_frame = 1 + 4 + num_patches  # camera + register + patches

        # Initialize History Anchor Manager (count-based with FIFO)
        anchor_config = HistoryAnchorConfig(
            strategy=history_anchor_strategy,
            interval=anchor_interval,
            max_anchors=max_anchors,
            coverage_threshold=coverage_threshold,
            anchor_keep_ratio=anchor_keep_ratio,
        )
        anchor_manager = HistoryAnchorManager(anchor_config, tokens_per_frame)
        anchor_manager.image_size_hw = (img_h, img_w)

        all_ress = []
        processed_frames = []

        for i, frame in enumerate(frames):
            # For fixed_interval: decision happens BEFORE inference
            fixed_interval_registered = False
            fixed_interval_is_fifo = False
            if history_anchor_strategy == 'fixed_interval':
                should_register, is_fifo, reason = anchor_manager.should_become_anchor(frame_idx=i)
                if should_register:
                    anchor_manager.register_anchor(i)
                    fixed_interval_registered = True
                    fixed_interval_is_fifo = is_fifo
                    fifo_msg = " (FIFO: oldest demoted)" if is_fifo else ""
                    print(f"[History Anchor] Frame {i} registered{fifo_msg}: {reason}")

            # Get protected anchor token count (never pause eviction)
            anchor_token_count = anchor_manager.get_protected_token_count()

            images = frame["img"].unsqueeze(0)


            aggregator_output = self.aggregator(
                images,
                past_key_values=past_key_values,
                use_cache=True,
                past_frame_idx=i,
                total_budget=total_budget,  
                anchor_token_count=anchor_token_count,
                importance_weight=importance_weight,
            )


            if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                aggregated_tokens, patch_start_idx, past_key_values = aggregator_output
            else:
                aggregated_tokens, patch_start_idx = aggregator_output

            with torch.cuda.amp.autocast(enabled=False):
                if self.camera_head is not None:
                    num_cam_iters = 4  # camera head refinement iterations per frame
                    total_anchors = anchor_manager.get_num_anchors()  # 1 (global) + history
                    camera_anchor_token_count = total_anchors * num_cam_iters

                    pose_enc, past_key_values_camera = self.camera_head(
                        aggregated_tokens,
                        past_key_values_camera=past_key_values_camera,
                        use_cache=True,
                        anchor_token_count=camera_anchor_token_count,
                    )
                    pose_enc = pose_enc[-1]
                    camera_pose = pose_enc[:, 0, :]

                    # For fixed_interval: sync camera KV cache AFTER processing
                    if fixed_interval_registered:
                        past_key_values_camera = self.camera_head.sync_anchor_change(
                            past_key_values_camera,
                            anchor_token_count=camera_anchor_token_count,
                            num_cam_iters=4,
                            is_fifo=fixed_interval_is_fifo,
                        )

                if self.depth_head is not None:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    depth = depth[:, 0]
                    depth_conf = depth_conf[:, 0]

                if self.point_head is not None:
                    pts3d, pts3d_conf = self.point_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    pts3d = pts3d[:, 0]
                    pts3d_conf = pts3d_conf[:, 0]

                if self.track_head is not None and query_points is not None:
                    track_list, vis, conf = self.track_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                    )
                    track = track_list[-1][:, 0]
                    query_points = track
                    vis = vis[:, 0]
                    track_conf = conf[:, 0]

            # For coverage strategy: decision happens AFTER depth/pose are available
            if history_anchor_strategy == 'coverage':
                should_register, is_fifo, reason, coverage = anchor_manager.should_become_anchor_coverage(
                    frame_idx=i,
                    current_depth=depth[0],  
                    current_pose=camera_pose[0],  
                )
                if should_register:
                    anchor_manager.register_anchor_coverage(i, depth[0], camera_pose[0])
                    fifo_msg = " (FIFO: oldest demoted)" if is_fifo else ""
                    print(f"[History Anchor] Frame {i} registered{fifo_msg}: {reason}")

                    # Sync camera KV cache anchor zone with Aggregator's FIFO/promotion.
                    cam_anchor_count_post = anchor_manager.get_num_anchors() * 4
                    past_key_values_camera = self.camera_head.sync_anchor_change(
                        past_key_values_camera,
                        anchor_token_count=cam_anchor_count_post,
                        num_cam_iters=4,
                        is_fifo=is_fifo,
                    )

            res_gpu = {
                "pts3d_in_other_view": pts3d,
                "conf": pts3d_conf,
                "depth": depth,
                "depth_conf": depth_conf,
                "camera_pose": camera_pose,
                **({"valid_mask": frame["valid_mask"]} if "valid_mask" in frame else {}),
                **(
                    {"track": track, "vis": vis, "track_conf": track_conf}
                    if query_points is not None
                    else {}
                ),
            }
            res_cpu = {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in res_gpu.items()
            }
            if frame_writer is not None:
                frame_writer(i, frame, res_cpu)

            if cache_results:
                all_ress.append(res_cpu)
                processed_frames.append(
                    {nk: nv.detach().cpu() if isinstance(nv, torch.Tensor) else nv for nk, nv in frame.items()}
                )

            del res_gpu
            torch.cuda.empty_cache()

        return OVGGTOutput(
            ress=all_ress if cache_results else None,
            views=processed_frames if cache_results else None,
        )