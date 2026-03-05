"""
Token importance scoring strategies for KV cache eviction.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional


class SpatialImportanceScorer:
    """
    Spatial-aware importance scorer.

    Combines repr_shift with spatial locality:
    - High importance regions have their neighbors boosted
    - Avoids retaining isolated tokens, maintains region coherence
    """

    def __init__(
        self,
        kernel_size: int = 3,
        # Spatial smoothing weight (1-alpha = original weight)
        alpha: float = 0.3,
    ):
        """
        Args:
            kernel_size: Smoothing kernel size (fixed at 3 for Gaussian 3x3)
            alpha: Weight for spatial smoothing (0=pure repr_shift, 1=full smoothing)
        """
        self.kernel_size = kernel_size
        self.alpha = alpha
        # Lazy init for device compatibility
        self._kernel: Optional[Tensor] = None

    def _get_kernel(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Lazy initialization of Gaussian kernel."""
        if self._kernel is None or self._kernel.device != device:
            # Simple Gaussian-like kernel (normalized)
            k = torch.tensor([
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1],
            ], device=device, dtype=dtype) / 16.0
            self._kernel = k.view(1, 1, 3, 3)
        return self._kernel

    def compute(
        self,
        importance: Tensor,  # [B, N] raw repr_shift
        patch_start_idx: int = 5,  # camera(1) + register(4)
        grid_size: Tuple[int, int] = None,  # (H, W) or infer from N
    ) -> Tensor:
        """
        Compute spatially-enhanced importance scores.

        Args:
            importance: Raw importance scores [B, N]
            patch_start_idx: Index where patch tokens start (after special tokens)
            grid_size: (H, W) grid dimensions, or None to infer square grid

        Returns:
            Tensor [B, N]: Spatially-enhanced importance scores
        """
        B, N = importance.shape
        device, dtype = importance.device, importance.dtype

        # Separate special tokens and patch tokens
        special = importance[:, :patch_start_idx]
        patches = importance[:, patch_start_idx:]

        # Infer or use provided grid size
        num_patches = patches.shape[1]
        if grid_size is None:
            # Assume square grid
            H = W = int(num_patches ** 0.5)
            if H * W != num_patches:
                # Cannot infer grid, return original importance
                return importance
        else:
            H, W = grid_size

        if H * W != num_patches:
            # Grid size mismatch, return original
            return importance

        # Reshape to spatial grid [B, 1, H, W]
        I_grid = patches.reshape(B, 1, H, W)

        # Spatial smoothing
        kernel = self._get_kernel(device, dtype)
        I_smoothed = F.conv2d(I_grid, kernel, padding=1)

        # Combine: α * smoothed + (1-α) * original
        I_combined = self.alpha * I_smoothed + (1 - self.alpha) * I_grid

        # Flatten back
        I_patches = I_combined.squeeze(1).reshape(B, -1)

        # Special tokens: keep original importance
        # (they are typically protected by other logic anyway)
        return torch.cat([special, I_patches], dim=-1)


class TokenImportanceScorer:
    """
    Token importance scoring strategies for KV cache eviction.

    Supported strategies:
    - 'baseline': Keep tokens with LOW similarity to mean (diverse)
    - 'repr_shift': Keep tokens with HIGH representation shift (important)
    - 'repr_shift_spatial': repr_shift with spatial locality smoothing
    """

    VALID_STRATEGIES = ('baseline', 'repr_shift', 'repr_shift_spatial')

    def __init__(
        self,
        strategy: str = 'baseline',
        spatial_alpha: float = 0.3,
    ):
        """
        Args:
            strategy: Scoring strategy ('baseline', 'repr_shift', or 'repr_shift_spatial')
            spatial_alpha: Weight for spatial smoothing in 'repr_shift_spatial' strategy
        """
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy}. Must be one of {self.VALID_STRATEGIES}")
        self.strategy = strategy
        self.spatial_alpha = spatial_alpha

        if strategy == 'repr_shift_spatial':
            self.spatial_scorer = SpatialImportanceScorer(alpha=spatial_alpha)
        else:
            self.spatial_scorer = None

    def compute(
        self,
        k: Tensor = None,
        x_before_mlp: Tensor = None,
        mlp_output: Tensor = None,
        patch_start_idx: int = 5,
        grid_size: Tuple[int, int] = None,
    ) -> Tensor:
        """
        Compute importance scores for tokens.

        Args:
            k: Key tensor [B, H, N, D] - used for baseline strategy
            x_before_mlp: Input to MLP [B, N, C] - used for repr_shift strategy
            mlp_output: MLP output (after ls2 scaling) [B, N, C] - used for repr_shift strategy
            patch_start_idx: Index where patch tokens start - used for spatial strategy
            grid_size: (H, W) patch grid dimensions - used for spatial strategy

        Returns:
            Tensor [B, N]: Importance scores. Higher = more important = KEEP.
        """
        if self.strategy == 'baseline':
            return self._baseline(k)
        elif self.strategy == 'repr_shift':
            return self._repr_shift(x_before_mlp, mlp_output)
        elif self.strategy == 'repr_shift_spatial':
            raw = self._repr_shift(x_before_mlp, mlp_output)
            return self.spatial_scorer.compute(raw, patch_start_idx, grid_size)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _baseline(self, k: Tensor) -> Tensor:
        """
        Original baseline: keep tokens with LOW similarity (diverse).

        Computes cosine similarity between each token's key and the mean key,
        then returns 1 - similarity as the diversity score.

        Args:
            k: Key tensor [B, H, N, D]

        Returns:
            Tensor [B, N]: Diversity scores (higher = more diverse = keep)
        """
        if k is None:
            raise ValueError(
                "Key tensor 'k' is required for baseline strategy")

        # k: [B, H, N, D]
        k_norm = F.normalize(k, p=2, dim=-1)
        mean_k = k_norm.mean(dim=2, keepdim=True)  # [B, H, 1, D]
        similarity = (k_norm * mean_k).sum(dim=-1)  # [B, H, N]

        # Average across heads and compute diversity
        diversity = 1.0 - similarity.mean(dim=1)  # [B, N]
        return diversity  # Higher = more diverse = keep

    def _repr_shift(self, x_before_mlp: Tensor, mlp_output: Tensor) -> Tensor:
        """
        Representation Shift: keep tokens with HIGH shift (important).

        Measures the L2 distance between input and output of MLP block,
        which indicates how much the token's representation changed.
        Tokens with larger shifts are considered more important.

        Args:
            x_before_mlp: Input to MLP [B, N, C]
            mlp_output: MLP residual output (ls2(mlp(norm2(x)))) [B, N, C]

        Returns:
            Tensor [B, N]: Shift magnitude (higher = more important = keep)
        """
        if x_before_mlp is None or mlp_output is None:
            raise ValueError(
                "Both x_before_mlp and mlp_output are required for repr_shift strategy")

        # x_before_mlp, mlp_output: [B, N, C]
        # Compute L2 norm of the residual (shift)
        shift = mlp_output.norm(dim=-1)  # [B, N]
        return shift  # Higher = more important = keep
