"""
History Anchor Manager for KV Cache Optimization

This module implements strategies for selecting History Anchors in
long-sequence video inference:

1. Fixed Interval: Select anchors at regular frame intervals
2. Coverage-based: Adaptive selection based on spatial coverage ratio

Key Features:
- Count-based protection: Tracks number of history anchors, not exact positions
- FIFO mechanism: When max_anchors is reached, oldest anchor is demoted
- No eviction pause: Always allows eviction to maintain low VRAM
- Coverage-based: Select anchor when scene coverage drops below threshold

History Anchors provide long-range reference points to prevent error
accumulation in streaming inference.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
import numpy as np


@dataclass
class HistoryAnchorConfig:
    """Configuration for History Anchor selection."""

    strategy: str = 'none'       # 'none', 'fixed_interval', or 'coverage'
    interval: int = 50           # Fixed interval (e.g., 25, 50, 100)
    # Maximum number of history anchors (excluding Global Anchor)
    max_anchors: int = 3
    # Coverage-based parameters
    # Register new anchor when coverage drops below this
    coverage_threshold: float = 0.4
    # Sampling ratio for speedup (0.1 = use 10% of points)
    sample_ratio: float = 0.1
    # Anchor token retention ratio
    # Ratio of anchor tokens to protect (1.0 = all, 0.25 = top 25%)
    anchor_keep_ratio: float = 1.0


def compute_coverage(
    anchor_depth: torch.Tensor,
    anchor_pose: torch.Tensor,
    current_pose: torch.Tensor,
    image_size_hw: Tuple[int, int],
    sample_ratio: float = 0.1,
) -> float:
    """
    Compute the spatial coverage ratio of anchor frame's 3D points
    when projected to the current camera view.

    This measures how much of the anchor frame's content is still visible
    in the current frame. When coverage drops below a threshold, it indicates
    significant camera movement and a new anchor should be selected.

    Steps:
    1. anchor depth → 3D world points (using pose_encoding_to_extri_intri)
    2. 3D world points → current camera coords
    3. current camera coords → current image coords (u, v)
    4. Count ratio of (u, v) falling within [0, W) x [0, H)

    Args:
        anchor_depth: Depth map of anchor frame, shape (H, W) or (H, W, 1)
        anchor_pose: Camera pose encoding of anchor frame, shape (9,)
        current_pose: Camera pose encoding of current frame, shape (9,)
        image_size_hw: Tuple of (height, width)
        sample_ratio: Fraction of points to sample for speedup (default 0.1)

    Returns:
        coverage: Float in [0.0, 1.0], ratio of anchor points visible in current view
    """
    from .pose_enc import pose_encoding_to_extri_intri
    from .geometry import closed_form_inverse_se3

    H, W = image_size_hw
    device = anchor_depth.device
    dtype = anchor_depth.dtype

    # Ensure depth is 2D
    if anchor_depth.dim() == 3:
        anchor_depth = anchor_depth.squeeze(-1)

    # Get camera parameters from pose encoding
    # pose_encoding_to_extri_intri expects shape (B, S, 9)
    anchor_pose_batched = anchor_pose.unsqueeze(0).unsqueeze(0)  # (1, 1, 9)
    current_pose_batched = current_pose.unsqueeze(0).unsqueeze(0)  # (1, 1, 9)

    anchor_extri, anchor_intri = pose_encoding_to_extri_intri(
        anchor_pose_batched, image_size_hw
    )
    current_extri, current_intri = pose_encoding_to_extri_intri(
        current_pose_batched, image_size_hw
    )

    # Remove batch dimensions: (1, 1, ...) -> (...)
    anchor_extri = anchor_extri[0, 0]  # (3, 4)
    anchor_intri = anchor_intri[0, 0]  # (3, 3)
    current_extri = current_extri[0, 0]  # (3, 4)
    current_intri = current_intri[0, 0]  # (3, 3)

    # Sample points for speedup
    total_pixels = H * W
    num_samples = max(1, int(total_pixels * sample_ratio))

    # Create valid depth mask (depth > 0)
    valid_mask = anchor_depth > 1e-6
    valid_indices = torch.nonzero(
        valid_mask.flatten(), as_tuple=False).squeeze(-1)

    if valid_indices.numel() == 0:
        return 0.0

    # Random sampling from valid points
    if valid_indices.numel() > num_samples:
        perm = torch.randperm(valid_indices.numel(), device=device)[
            :num_samples]
        sample_indices = valid_indices[perm]
    else:
        sample_indices = valid_indices
        num_samples = valid_indices.numel()

    # Convert flat indices to 2D coordinates
    v_coords = sample_indices // W  # row
    u_coords = sample_indices % W   # col

    # Get depth values at sampled points
    depths = anchor_depth[v_coords, u_coords]  # (num_samples,)

    # Unproject to camera coordinates
    fx = anchor_intri[0, 0]
    fy = anchor_intri[1, 1]
    cx = anchor_intri[0, 2]
    cy = anchor_intri[1, 2]

    x_cam = (u_coords.float() - cx) * depths / fx
    y_cam = (v_coords.float() - cy) * depths / fy
    z_cam = depths

    # Stack to form camera coordinates (num_samples, 3)
    cam_coords = torch.stack([x_cam, y_cam, z_cam], dim=-1)

    # Transform anchor camera coords to world coords
    # anchor_extri is world_to_cam, we need cam_to_world
    anchor_extri_4x4 = torch.eye(4, device=device, dtype=dtype)
    anchor_extri_4x4[:3, :4] = anchor_extri
    anchor_cam_to_world = closed_form_inverse_se3(
        anchor_extri_4x4.unsqueeze(0))[0]

    R_a2w = anchor_cam_to_world[:3, :3]  # (3, 3)
    t_a2w = anchor_cam_to_world[:3, 3]   # (3,)

    # Transform to world coordinates
    world_coords = cam_coords @ R_a2w.T + t_a2w  # (num_samples, 3)

    # Transform world coords to current camera coords
    R_c = current_extri[:3, :3]  # (3, 3)
    t_c = current_extri[:3, 3]   # (3,)

    current_cam_coords = world_coords @ R_c.T + t_c  # (num_samples, 3)

    # Project to current image plane
    fx_c = current_intri[0, 0]
    fy_c = current_intri[1, 1]
    cx_c = current_intri[0, 2]
    cy_c = current_intri[1, 2]

    # Avoid division by zero
    z_current = current_cam_coords[:, 2]
    valid_z = z_current > 1e-6

    u_proj = torch.zeros_like(z_current)
    v_proj = torch.zeros_like(z_current)

    u_proj[valid_z] = (current_cam_coords[valid_z, 0] /
                       z_current[valid_z]) * fx_c + cx_c
    v_proj[valid_z] = (current_cam_coords[valid_z, 1] /
                       z_current[valid_z]) * fy_c + cy_c

    # Check if projected points are within image bounds
    in_bounds = (
        valid_z &
        (u_proj >= 0) & (u_proj < W) &
        (v_proj >= 0) & (v_proj < H)
    )

    coverage = in_bounds.float().mean().item()

    return coverage


class HistoryAnchorManager:
    """
    Manages History Anchor selection using count-based protection with FIFO.

    This implementation supports two strategies:
    1. fixed_interval: Select anchors at regular frame intervals
    2. coverage: Adaptive selection based on spatial coverage ratio

    KV cache structure after eviction:
        [protected_tokens] + [candidate_tokens]

    protected_count = tokens_per_frame * (1 + num_history_anchors)
                      ↑                   ↑
                   Global Anchor      History Anchors

    Args:
        config: HistoryAnchorConfig with strategy parameters
        tokens_per_frame: Number of tokens per frame (including special tokens)
    """

    def __init__(self, config: HistoryAnchorConfig, tokens_per_frame: int):
        self.config = config
        self.tokens_per_frame = tokens_per_frame

        # Count-based tracking (not exact positions)
        self.num_history_anchors: int = 0

        # Frame list for logging only (FIFO maintained)
        self.history_anchor_frames: List[int] = []

        # Next anchor target frame (for fixed_interval)
        self.next_anchor_frame: int = config.interval

        # Coverage-based: store latest anchor's depth and pose
        self.latest_anchor_depth: Optional[torch.Tensor] = None
        self.latest_anchor_pose: Optional[torch.Tensor] = None
        self.image_size_hw: Optional[Tuple[int, int]] = None

    def is_eviction_paused(self) -> bool:
        """
        Check if eviction should be paused.

        Simplified version: Never pause eviction for low VRAM.

        Returns:
            Always False
        """
        return False

    def should_become_anchor(self, frame_idx: int) -> Tuple[bool, bool, str]:
        """
        Check if this frame should become a History Anchor.

        Simplified version: Direct interval-based selection without
        confidence comparison or window mechanism.

        Args:
            frame_idx: Current frame index

        Returns:
            Tuple of:
            - should_register: Whether to register a new anchor
            - is_fifo: Whether FIFO is triggered (oldest anchor demoted)
            - reason: Description of the decision
        """
        if self.config.strategy != 'fixed_interval':
            return False, False, 'disabled'

        if frame_idx != self.next_anchor_frame:
            return False, False, f'not_target_frame_{frame_idx}'

        # Update next target
        self.next_anchor_frame += self.config.interval

        # Check if FIFO is needed
        is_fifo = self.num_history_anchors >= self.config.max_anchors

        return True, is_fifo, f'interval_anchor_at_frame_{frame_idx}'

    def register_anchor(self, frame_idx: int):
        """
        Register a new History Anchor.

        If max_anchors is reached, implements FIFO:
        - num_history_anchors stays the same
        - Oldest frame is removed from tracking list
        - In practice, old anchor tokens may already be evicted

        Args:
            frame_idx: Frame index of the new anchor
        """
        # Increment count only if below max
        if self.num_history_anchors < self.config.max_anchors:
            self.num_history_anchors += 1

        # Update frame list (FIFO)
        if len(self.history_anchor_frames) >= self.config.max_anchors:
            self.history_anchor_frames.pop(0)
        self.history_anchor_frames.append(frame_idx)

    def get_protected_token_count(self) -> int:
        """
        Get total number of protected anchor tokens.

        With anchor_keep_ratio < 1.0, only a fraction of each anchor's tokens
        are protected. This reduces noise from low-confidence tokens.

        Returns:
            int: Global Anchor tokens + History Anchor tokens (scaled by keep_ratio)
        """
        # Global Anchor always fully protected
        global_anchor_tokens = self.tokens_per_frame
        # not protecting global anchor tokens to allow eviction if needed for VRAM constraints
        # global_anchor_tokens = 0

        # History Anchors: apply keep_ratio
        history_anchor_tokens = int(
            self.num_history_anchors * self.tokens_per_frame * self.config.anchor_keep_ratio
        )

        return global_anchor_tokens + history_anchor_tokens

    def get_num_anchors(self) -> int:
        """Get total number of anchors (including Global Anchor)."""
        return 1 + self.num_history_anchors

    def should_become_anchor_coverage(
        self,
        frame_idx: int,
        current_depth: torch.Tensor,
        current_pose: torch.Tensor,
    ) -> Tuple[bool, bool, str, float]:
        """
        Determine if current frame should become an anchor based on spatial coverage.

        This is called AFTER depth and pose predictions are available.

        Args:
            frame_idx: Current frame index
            current_depth: Predicted depth map of current frame, shape (H, W) or (H, W, 1)
            current_pose: Predicted camera pose of current frame, shape (9,)

        Returns:
            Tuple of:
            - should_register: Whether to register a new anchor
            - is_fifo: Whether FIFO is triggered (oldest anchor demoted)
            - reason: Description of the decision
            - coverage: Computed coverage ratio (or 1.0 for frame 0)
        """
        if self.config.strategy != 'coverage':
            return False, False, 'coverage_disabled', 1.0

        # Frame 0: Initialize with global anchor's info for later comparison
        if frame_idx == 0:
            self.latest_anchor_depth = current_depth.clone()
            self.latest_anchor_pose = current_pose.clone()
            return False, False, 'frame_0_global_anchor', 1.0

        # Compute coverage ratio
        if self.latest_anchor_depth is None or self.latest_anchor_pose is None:
            # Fallback: no anchor info, treat as needing anchor
            return True, False, 'no_anchor_info', 0.0

        if self.image_size_hw is None:
            # Infer from depth shape
            if current_depth.dim() == 3:
                H, W = current_depth.shape[0], current_depth.shape[1]
            else:
                H, W = current_depth.shape
            self.image_size_hw = (H, W)

        coverage = compute_coverage(
            self.latest_anchor_depth,
            self.latest_anchor_pose,
            current_pose,
            self.image_size_hw,
            self.config.sample_ratio,
        )

        # Coverage above threshold: no new anchor needed
        if coverage >= self.config.coverage_threshold:
            return False, False, f'coverage_{coverage:.3f}>=threshold_{self.config.coverage_threshold}', coverage

        # Coverage below threshold: need new anchor
        is_fifo = self.num_history_anchors >= self.config.max_anchors
        return True, is_fifo, f'coverage_{coverage:.3f}<threshold_{self.config.coverage_threshold}', coverage

    def register_anchor_coverage(
        self,
        frame_idx: int,
        depth: torch.Tensor,
        pose: torch.Tensor
    ):
        """
        Register a new anchor and update latest anchor information.

        For coverage-based strategy, this also updates the depth/pose
        used for subsequent coverage calculations.

        Args:
            frame_idx: Frame index of the new anchor
            depth: Depth map of the new anchor frame
            pose: Camera pose of the new anchor frame
        """
        self.register_anchor(frame_idx)
        self.latest_anchor_depth = depth.clone()
        self.latest_anchor_pose = pose.clone()

    def __repr__(self) -> str:
        base_repr = (
            f"HistoryAnchorManager("
            f"strategy={self.config.strategy}, "
            f"num_anchors={self.get_num_anchors()}, "
            f"history_frames={self.history_anchor_frames}"
        )
        if self.config.strategy == 'fixed_interval':
            return base_repr + f", next_target={self.next_anchor_frame})"
        elif self.config.strategy == 'coverage':
            return base_repr + f", threshold={self.config.coverage_threshold})"
        return base_repr + ")"
