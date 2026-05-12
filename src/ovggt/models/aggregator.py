# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any

from ovggt.layers import PatchEmbed
from ovggt.layers.block import Block
from ovggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from ovggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.


    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        eviction_strategy='repr_shift',
        intra_frame_keep_ratio=1.0,
        spatial_alpha=0.3,
    ):
        super().__init__()

        self.eviction_strategy = eviction_strategy
        self.intra_frame_keep_ratio = intra_frame_keep_ratio
        self.spatial_alpha = spatial_alpha

        # Patch tokens start after camera(1) + register tokens
        self._patch_start_idx = 1 + num_register_tokens
        # Store patch_size for dynamic grid size computation
        self._patch_size = patch_size
        # patch_grid_size will be computed dynamically in forward() from actual image dims
        self.patch_grid_size = None  # Placeholder, updated in forward()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                    eviction_strategy=eviction_strategy,
                    spatial_alpha=spatial_alpha,
                    patch_start_idx=self._patch_start_idx,
                    patch_grid_size=self.patch_grid_size,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).reshape(1, 1, 3, 1, 1),
                persistent=False,
            )
        self.last_scores = torch.zeros(self.depth)


    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0,
        total_budget=0,
        anchor_token_count: int = None,
        importance_weight: float = 0.5,
        window_token_count: int = 0,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        # Compute patch grid size dynamically from actual image dimensions
        grid_h = H // self._patch_size
        grid_w = W // self._patch_size
        if self.patch_grid_size != (grid_h, grid_w):
            self.patch_grid_size = (grid_h, grid_w)
            for block in self.global_blocks:
                block.patch_grid_size = (grid_h, grid_w)

        if use_cache and past_key_values[0] is not None:
            # _, _, S_true, _, _ = past_key_values[0][0].shape
            S_true = past_frame_idx + 1
        else:
            S_true = S
        
        if use_cache and S > 1:
            print(f"Use KV cache expects S=1, got S={S}")

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean.to(images.device)) / self._resnet_std.to(images.device)

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.reshape(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        if use_cache:
            camera_token_full = slice_expand_and_flatten(self.camera_token, B, S_true)
            camera_token = camera_token_full[-1:, :, :]
            
            register_token_full = slice_expand_and_flatten(self.register_token, B, S_true)
            register_token = register_token_full[-1:, :, :]
        else:
            camera_token = slice_expand_and_flatten(self.camera_token, B, S)
            register_token = slice_expand_and_flatten(self.register_token, B, S)
        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []
        current_budgets = self._calculate_dynamic_budgets(total_budget)
        scores = []

        # Track importance within current frame processing (layer to layer)
        prev_importance = None

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    if use_cache:
                        if past_key_values[global_idx] is not None:
                            k, v = past_key_values[global_idx]

                        # Determine cache budget for this layer (None = no eviction)
                        layer_budget = None if current_budgets is None else current_budgets[global_idx].item()

                        # For repr_shift: use previous layer's importance for new tokens
                        # Falls back to baseline (cosine diversity) when shapes don't match
                        tokens, global_idx, global_intermediates, new_kv, current_scores, new_importance, kept_indices = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos,
                            past_key_values_block=past_key_values[global_idx] if past_key_values[global_idx] is not None else None,
                            use_cache=True,
                            past_frame_idx=past_frame_idx,
                            cache_budget=layer_budget,
                            prev_importance=prev_importance,
                            intra_frame_keep_ratio=self.intra_frame_keep_ratio,
                            anchor_token_count=anchor_token_count,
                            importance_weight=importance_weight,
                            window_token_count=window_token_count,
                        )

                        # Pass new importance to next layer (within same frame)
                        prev_importance = new_importance

                        layer_idx = global_idx - 1
                        past_key_values[layer_idx] = new_kv
                        if current_scores is not None: 
                            scores.append(current_scores)
                        else:
                            scores.append(self.last_scores[layer_idx].item())
                    else:
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos
                        )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")
            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)
        if scores:  
            self.last_scores = torch.tensor(scores, device=self.last_scores.device, dtype=self.last_scores.dtype)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        if use_cache:      
            return output_list, self.patch_start_idx, past_key_values
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.reshape(B, S, P, 2).reshape(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):

            tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(
        self,
        tokens,
        B,
        S,
        P,
        C,
        global_idx,
        pos=None,
        past_key_values_block=None,
        use_cache=False,
        past_frame_idx=0,
        cache_budget=None,
        prev_importance=None,
        intra_frame_keep_ratio=1.0,
        anchor_token_count: int = None,
        importance_weight: float = 0.5,
        window_token_count: int = 0,
    ) -> Union[Tuple[torch.Tensor, int, List[torch.Tensor]], Tuple[torch.Tensor, int, List[torch.Tensor], List]]:
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).

        Args:
            prev_importance: Optional [B, N] accumulated importance scores for KV cache.
                Used for eviction decision when eviction_strategy='repr_shift'.

        Returns:
            When use_cache=True: (tokens, global_idx, intermediates, block_kv, scores, new_importance, kept_indices)
            Otherwise: (tokens, global_idx, intermediates)
        """

        if tokens.shape != (B, S * P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.reshape(B, S, P, 2).reshape(B, S * P, 2)

        intermediates = []
        new_importance = None
        kept_indices = None

        for _ in range(self.aa_block_size):
            if not use_cache:
                L = S * P
                frame_ids = torch.arange(L, device=tokens.device) // P  # [0,0,...,1,1,...,S-1]
                future_frame = frame_ids.unsqueeze(1) < frame_ids.unsqueeze(0)
                attn_mask = future_frame.to(tokens.dtype) * torch.finfo(tokens.dtype).min
            else:
                attn_mask = None

            scores = None
            if use_cache:
                # Skip intra-frame pruning for anchor frame (first frame)
                effective_keep_ratio = 1.0 if past_frame_idx == 0 else intra_frame_keep_ratio
                tokens, block_kv, scores, new_importance, kept_indices = self.global_blocks[global_idx](
                    tokens,
                    pos=pos,
                    attn_mask=attn_mask,
                    past_key_values=past_key_values_block,
                    use_cache=True,
                    cache_budget=cache_budget,
                    prev_importance=prev_importance,
                    intra_frame_keep_ratio=effective_keep_ratio,
                    anchor_token_count=anchor_token_count,
                    importance_weight=importance_weight,
                    window_token_count=window_token_count,
                )
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos, attn_mask=attn_mask)

            global_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

            # if self.use_causal_global:
            #     del attn_mask
        if use_cache:
            return tokens, global_idx, intermediates, block_kv, scores, new_importance, kept_indices
        return tokens, global_idx, intermediates
        
    def _calculate_dynamic_budgets(self, total_budget):
        # Handle None budget (eviction paused for History Anchor window)
        if total_budget is None:
            return None

        with torch.no_grad():
            diversity_scores = 1.0 - self.last_scores
            scaled_scores = diversity_scores / 0.5
            proportions = torch.softmax(scaled_scores, dim=0)
            if total_budget < 0:
                total_budget = 0
            budgets = proportions * total_budget

        return budgets.int()

    def sync_anchor_change(
        self,
        past_key_values,
        anchor_token_count: int,
        tokens_per_frame: int,
        anchor_keep_ratio: float,
        anchor_token_indices: Optional[torch.Tensor] = None,
        is_fifo: bool = False,
    ):
        """
        Rearrange global KV cache after registering a history anchor.

        This promotes a chunk of the newest frame's tokens into the anchor
        zone (first anchor_token_count tokens) so it is protected by eviction.
        If anchor_token_indices are provided, the promotion uses those indices
        (relative to the new frame token segment), typically ranked by point
        head confidence.

        Args:
            past_key_values: list of (k, v) per layer
            anchor_token_count: anchor zone size AFTER the change
            tokens_per_frame: number of tokens per frame (camera + register + patches)
            anchor_keep_ratio: fraction of each anchor frame to protect
            anchor_token_indices: indices within the new frame segment to promote
            is_fifo: whether FIFO demotion occurred (oldest history anchor demoted)

        Returns:
            Modified past_key_values
        """
        if past_key_values is None or anchor_token_count is None:
            return past_key_values

        if anchor_token_count <= tokens_per_frame:
            return past_key_values

        if anchor_token_indices is not None:
            anchor_chunk = anchor_token_indices.shape[-1]
        else:
            anchor_chunk = max(int(tokens_per_frame * anchor_keep_ratio), 1)
        anchor_chunk = min(anchor_chunk, tokens_per_frame)
        global_anchor_end = tokens_per_frame

        for idx in range(self.depth):
            if past_key_values[idx] is None:
                continue

            k, v = past_key_values[idx]
            N = k.shape[2]

            if N <= anchor_token_count:
                continue

            new_frame_start = N - tokens_per_frame
            if new_frame_start < 0:
                continue

            # If the new frame already overlaps the anchor zone, skip reordering.
            if new_frame_start < anchor_token_count:
                continue

            if anchor_chunk <= 0:
                continue

            new_frame_anchor_end = min(new_frame_start + anchor_chunk, N)
            if anchor_token_indices is not None:
                k_frame = k[:, :, new_frame_start:new_frame_start + tokens_per_frame, :]
                v_frame = v[:, :, new_frame_start:new_frame_start + tokens_per_frame, :]
                frame_indices = torch.arange(tokens_per_frame, device=k.device)

                if anchor_token_indices.dim() == 1:
                    anchor_token_indices_exp = anchor_token_indices.unsqueeze(0).expand(k.shape[0], -1)
                else:
                    anchor_token_indices_exp = anchor_token_indices

                selected_k = []
                selected_v = []
                remaining_k = []
                remaining_v = []

                for b in range(k.shape[0]):
                    selected = anchor_token_indices_exp[b]
                    selected = selected[(selected >= 0) & (selected < tokens_per_frame)]
                    if selected.numel() == 0:
                        continue

                    mask = torch.ones(tokens_per_frame, device=k.device, dtype=torch.bool)
                    mask[selected] = False
                    remaining = frame_indices[mask]

                    selected_k.append(k_frame[b:b + 1, :, selected, :])
                    selected_v.append(v_frame[b:b + 1, :, selected, :])
                    remaining_k.append(k_frame[b:b + 1, :, remaining, :])
                    remaining_v.append(v_frame[b:b + 1, :, remaining, :])

                if not selected_k:
                    continue

                k_selected = torch.cat(selected_k, dim=0)
                v_selected = torch.cat(selected_v, dim=0)
                k_remaining = torch.cat(remaining_k, dim=0)
                v_remaining = torch.cat(remaining_v, dim=0)
            else:
                k_selected = k[:, :, new_frame_start:new_frame_anchor_end, :]
                v_selected = v[:, :, new_frame_start:new_frame_anchor_end, :]
                k_remaining = k[:, :, new_frame_anchor_end:new_frame_start + tokens_per_frame, :]
                v_remaining = v[:, :, new_frame_anchor_end:new_frame_start + tokens_per_frame, :]

            if is_fifo:
                demote_start = global_anchor_end
                demote_end = min(global_anchor_end + anchor_chunk, anchor_token_count)
                if demote_end <= demote_start:
                    continue

                k_new = torch.cat(
                    [
                        k[:, :, :demote_start, :],
                        k[:, :, demote_end:anchor_token_count, :],
                        k_selected,
                        k[:, :, anchor_token_count:new_frame_start, :],
                        k_remaining,
                        k[:, :, new_frame_start + tokens_per_frame:, :],
                        k[:, :, demote_start:demote_end, :],
                    ],
                    dim=2,
                )
                v_new = torch.cat(
                    [
                        v[:, :, :demote_start, :],
                        v[:, :, demote_end:anchor_token_count, :],
                        v_selected,
                        v[:, :, anchor_token_count:new_frame_start, :],
                        v_remaining,
                        v[:, :, new_frame_start + tokens_per_frame:, :],
                        v[:, :, demote_start:demote_end, :],
                    ],
                    dim=2,
                )
            else:
                old_anchor_end = max(anchor_token_count - anchor_chunk, global_anchor_end)
                if old_anchor_end > new_frame_start:
                    continue

                k_new = torch.cat(
                    [
                        k[:, :, :old_anchor_end, :],
                        k_selected,
                        k[:, :, old_anchor_end:new_frame_start, :],
                        k_remaining,
                        k[:, :, new_frame_start + tokens_per_frame:, :],
                    ],
                    dim=2,
                )
                v_new = torch.cat(
                    [
                        v[:, :, :old_anchor_end, :],
                        v_selected,
                        v[:, :, old_anchor_end:new_frame_start, :],
                        v_remaining,
                        v[:, :, new_frame_start + tokens_per_frame:, :],
                    ],
                    dim=2,
                )

            past_key_values[idx] = (k_new, v_new)

        return past_key_values


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.reshape(B * S, *combined.shape[2:])
    return combined
