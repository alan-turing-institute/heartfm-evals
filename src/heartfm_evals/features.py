"""Feature extraction utilities for frozen backbone models.

Provides preprocessing and feature extraction functions for:
- 2D spatial features (DINOv3 multi-layer, SAM2 Hiera)
- 3D volume features (DINOv3, CineMA, SAM v1)

All functions operate on frozen backbones and return CPU tensors.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from heartfm_evals.constants import (
    GRID_SIZE,
    SAX_TARGET_DEPTH,
    imagenet_normalize,
)


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_slice(image_2d: torch.Tensor) -> torch.Tensor:
    """Prepare a (H, W) [0,1] tensor for DINOv3.  Returns (1, 3, H, W)."""
    x = image_2d.unsqueeze(0).repeat(3, 1, 1)
    x = imagenet_normalize(x)
    return x.unsqueeze(0)


# ── 2D Spatial Feature Extraction ──────────────────────────────────────────────
@torch.inference_mode()
def extract_multilayer_features(
    backbone: nn.Module,
    image_2d: torch.Tensor,
    layer_indices: tuple[int, ...] = (3, 6, 9, 11),
    device: torch.device | None = None,
) -> torch.Tensor:
    """Extract and concatenate features from multiple backbone layers for one slice.

    Args:
        backbone: Frozen DINOv3 backbone.
        image_2d: (H, W) tensor in [0, 1].
        layer_indices: Which intermediate layers to extract.
        device: Device for inference.

    Returns:
        Concatenated feature tensor (embed_dim * n_layers, h_patches, w_patches).
    """
    img = preprocess_slice(image_2d)  # (1, 3, H, W)
    if device is not None:
        img = img.to(device)
    feats = backbone.get_intermediate_layers(
        img, n=list(layer_indices), reshape=True, norm=True
    )
    # Each feat: (1, embed_dim, h, w) — concatenate along channel dim
    cat = torch.cat(feats, dim=1)  # (1, embed_dim*n_layers, h, w)
    return cat.squeeze(0).cpu()  # (C, h, w)


# ── 3D Volume Helpers ──────────────────────────────────────────────────────────
def _pad_volume_z(vol: torch.Tensor, target_depth: int) -> tuple[torch.Tensor, int]:
    """Pad or truncate the last (z) dimension to *target_depth*.

    Returns (padded_volume, actual_slices_before_padding).
    """
    n_slices = int(vol.shape[-1])
    if n_slices > target_depth:
        vol = vol[..., :target_depth]
        n_slices = target_depth
    elif vol.shape[-1] < target_depth:
        vol = F.pad(vol, (0, target_depth - vol.shape[-1]), mode="constant", value=0.0)
    return vol, n_slices


# ── 3D DINOv3 Volume Feature Extraction ───────────────────────────────────────
@torch.inference_mode()
def extract_dino_volume_features(
    backbone: nn.Module,
    sax_volume: torch.Tensor,
    layer_indices: tuple[int, ...],
    device: torch.device | None = None,
    target_depth: int = SAX_TARGET_DEPTH,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
    """Extract per-layer DINOv3 features for a SAX volume, stacked along z.

    For each slice, runs the frozen backbone and extracts features at the
    specified layers.  Features from all slices are stacked to form 3D volumes.
    Follows the same z-padding convention as CineMA (pad to *target_depth*).

    Args:
        backbone: Frozen DINOv3 backbone in eval mode.
        sax_volume: (1, H, W, z) tensor in [0, 1].
        layer_indices: Which intermediate layers to extract.
        device: Device for inference.
        target_depth: Pad z to this depth.

    Returns:
        Tuple of (features_dict, padded_image, n_slices):
            features_dict: ``{f"layer_{idx}": (embed_dim, g, g, target_depth)}``
            padded_image: ``(1, H, W, target_depth)``
            n_slices: actual number of slices before padding.
    """
    vol = sax_volume  # (1, H, W, z)
    n_slices = int(vol.shape[-1])

    if n_slices > target_depth:
        vol = vol[..., :target_depth]
        n_slices = target_depth
    if vol.shape[-1] < target_depth:
        vol = F.pad(vol, (0, target_depth - vol.shape[-1]), mode="constant", value=0.0)

    padded_image = vol  # (1, H, W, target_depth)

    per_layer: dict[int, list[torch.Tensor]] = {idx: [] for idx in layer_indices}

    for z in range(target_depth):
        image_2d = vol[0, :, :, z]  # (H, W)
        img = preprocess_slice(image_2d)  # (1, 3, H, W)
        if device is not None:
            img = img.to(device)

        feats = backbone.get_intermediate_layers(
            img, n=list(layer_indices), reshape=True, norm=True
        )
        for i, idx in enumerate(layer_indices):
            per_layer[idx].append(feats[i].squeeze(0).cpu())  # (embed_dim, h, w)

    features_dict: dict[str, torch.Tensor] = {}
    for idx in layer_indices:
        features_dict[f"layer_{idx}"] = torch.stack(per_layer[idx], dim=-1)
        # shape: (embed_dim, grid_h, grid_w, target_depth)

    return features_dict, padded_image, n_slices


# ── 3D CineMA Volume Feature Extraction ───────────────────────────────────────
@torch.inference_mode()
def extract_cinema_volume_features(
    backbone: nn.Module,
    sax_volume: torch.Tensor,
    device: torch.device | None = None,
    target_depth: int = SAX_TARGET_DEPTH,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
    """Extract CineMA conv-encoder skips and ViT features for a SAX volume.

    Runs the frozen CineMA encoder (``enc_down_dict["sax"]`` + ``encoder``)
    and returns the multi-scale conv skips together with the reshaped ViT
    output.  These are the ingredients needed by ``CineMAUNetRDecoder``.

    Args:
        backbone: Frozen CineMA model (``CineMA.from_pretrained()``).
        sax_volume: ``(1, H, W, z)`` tensor in [0, 1].
        device: Device for inference.
        target_depth: Pad z to this depth.

    Returns:
        Tuple of ``(features_dict, padded_image, n_slices)``:
            features_dict: ``{"conv_skip_0": ..., ..., "vit_features": ...}``
            padded_image: ``(1, H, W, target_depth)``
            n_slices: actual number of slices before padding.
    """
    vol, n_slices = _pad_volume_z(sax_volume, target_depth)

    # CineMA expects (B, 1, H, W, Z) — add batch dim
    batch_input = vol.unsqueeze(0)  # (1, 1, H, W, Z)
    if device is not None:
        batch_input = batch_input.to(device=device, dtype=torch.float32)

    # 1. Conv encoder: multi-scale skips + patch-embedded tokens
    skips_list, x_view = backbone.enc_down_dict["sax"](batch_input, mask=None)
    # skips_list: list of (1, ch, *spatial), x_view: (1, n_patches, embed_dim)

    # 2. Shared ViT encoder
    x = backbone.encoder(x_view)  # (1, 1+n_patches, embed_dim)
    x = x[:, 1:]  # strip cls token → (1, n_patches, embed_dim)

    # 3. Reshape ViT output to spatial grid
    grid_size = backbone.enc_down_dict["sax"].patch_embed.grid_size
    x = x.permute(0, 2, 1)  # (1, embed_dim, n_patches)
    vit_feat = x.reshape(1, x.shape[1], *grid_size)  # (1, embed_dim, gx, gy, gz)

    # Build return dict — move everything to CPU
    features_dict: dict[str, torch.Tensor] = {}
    for i, skip in enumerate(skips_list):
        features_dict[f"conv_skip_{i}"] = skip.squeeze(0).cpu()
    features_dict["vit_features"] = vit_feat.squeeze(0).cpu()

    return features_dict, vol, n_slices


# ── 3D SAM Volume Feature Extraction ──────────────────────────────────────────
@torch.inference_mode()
def extract_sam_volume_features(
    sam_model: nn.Module,
    processor,
    sax_volume: torch.Tensor,
    layer_indices: tuple[int, ...],
    device: torch.device | None = None,
    target_depth: int = SAX_TARGET_DEPTH,
    grid_size: int = GRID_SIZE,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
    """Extract per-layer SAM ViT features for a SAX volume, stacked along z.

    For each slice, runs the frozen SAM vision encoder with
    ``output_hidden_states=True`` and extracts intermediate hidden states at
    the specified layers.  The 64x64 feature maps are downsampled to
    *grid_size* x *grid_size* so the output is compatible with
    ``DINOv3UNetRDecoder``.

    Args:
        sam_model: Frozen ``SamModel`` in eval mode.
        processor: ``SamImageProcessor`` for image pre-processing.
        sax_volume: ``(1, H, W, z)`` tensor in [0, 1].
        layer_indices: Which intermediate ViT layers to extract.
        device: Device for inference.
        target_depth: Pad z to this depth.
        grid_size: Downsample spatial dims to this size (default 12).

    Returns:
        Tuple of ``(features_dict, padded_image, n_slices)``:
            features_dict:
                ``{f"layer_{idx}": (embed_dim, gs, gs, target_depth)}``
            padded_image: ``(1, H, W, target_depth)``
            n_slices: actual number of slices before padding.
    """
    vol, n_slices = _pad_volume_z(sax_volume, target_depth)

    per_layer: dict[int, list[torch.Tensor]] = {idx: [] for idx in layer_indices}

    for z in range(target_depth):
        image_2d = vol[0, :, :, z]  # (H, W)

        # Grayscale [0,1] → uint8 → RGB PIL (same pipeline as SAM notebook)
        img_np = (image_2d.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        pil = Image.fromarray(img_np, mode="L").convert("RGB")

        proc = processor(images=pil, return_tensors="pt")
        pixel_values = proc["pixel_values"]
        if device is not None:
            pixel_values = pixel_values.to(device)

        # Get intermediate hidden states from SAM vision encoder
        enc_out = sam_model.vision_encoder(pixel_values, output_hidden_states=True)
        hidden_states = enc_out.hidden_states  # tuple of (B, 64, 64, 768)

        for idx in layer_indices:
            feat = hidden_states[idx]  # (1, 64, 64, 768) — channels-last
            feat = feat.permute(0, 3, 1, 2)  # (1, 768, 64, 64)
            # Downsample to match DINOv3 grid size
            feat = F.interpolate(
                feat,
                size=(grid_size, grid_size),
                mode="bilinear",
                align_corners=False,
            )
            per_layer[idx].append(feat.squeeze(0).cpu())  # (768, gs, gs)

    features_dict: dict[str, torch.Tensor] = {}
    for idx in layer_indices:
        features_dict[f"layer_{idx}"] = torch.stack(per_layer[idx], dim=-1)
        # shape: (embed_dim, grid_size, grid_size, target_depth)

    return features_dict, vol, n_slices


# ── 2D CineMA Slice Feature Extraction ────────────────────────────────────────
@torch.inference_mode()
def extract_cinema_2d_feature_volume(
    backbone: nn.Module,
    sax_volume: torch.Tensor,
    device: torch.device | None = None,
    target_depth: int = SAX_TARGET_DEPTH,
) -> tuple[torch.Tensor, int]:
    """Extract CineMA ViT features for a SAX volume as a spatial feature volume.

    Runs the frozen CineMA encoder via ``feature_forward`` and reshapes the
    token output into a spatial feature volume ``(C, gx, gy, gz)``.

    Args:
        backbone: Frozen CineMA model.
        sax_volume: ``(1, H, W, z)`` tensor in [0, 1].
        device: Device for inference.
        target_depth: Pad z to this depth.

    Returns:
        Tuple of ``(feat_vol, n_slices)``:
            feat_vol: ``(C, gx, gy, gz)`` feature volume.
            n_slices: actual number of slices before padding.
    """
    vol, n_slices = _pad_volume_z(sax_volume, target_depth)

    batch = {"sax": vol.unsqueeze(0)}  # (1, 1, H, W, Z)
    if device is not None:
        batch = {"sax": batch["sax"].to(device=device, dtype=torch.float32)}

    tokens = backbone.feature_forward(batch)["sax"]  # (1, n_tokens, C)

    b, n_tokens, c = tokens.shape
    gx, gy, gz = backbone.enc_down_dict["sax"].patch_embed.grid_size

    feat_vol = (
        tokens.reshape(b, gx, gy, gz, c).permute(0, 4, 1, 2, 3).contiguous()
    )  # (1, C, gx, gy, gz)
    return feat_vol.squeeze(0).cpu(), n_slices


# ── 2D SAM2 Slice Feature Extraction ──────────────────────────────────────────
@torch.inference_mode()
def extract_sam2_2d_features(
    sam2_model: nn.Module,
    image_processor,
    image_2d: torch.Tensor,
    layer_indices: tuple[int, ...],
    device: torch.device | None = None,
    grid_size: int = GRID_SIZE,
) -> torch.Tensor:
    """Extract multi-layer SAM2 Hiera features for a single 2D slice.

    Uses ``output_hidden_states=True`` to extract intermediate Hiera block
    outputs at the specified layer indices, then downsamples each to
    *grid_size* × *grid_size* and concatenates along the channel dimension.
    This mirrors ``extract_sam_volume_features`` for the 2D case.

    Args:
        sam2_model: Frozen SAM2 model in eval mode.
        image_processor: SAM2 processor for image pre-processing.
        image_2d: (H, W) tensor in [0, 1].
        layer_indices: Which intermediate Hiera block outputs to extract.
        device: Device for inference.
        grid_size: Spatial size to downsample each feature map to.

    Returns:
        Feature tensor ``(embed_dim * n_layers, grid_size, grid_size)``
        with layer features concatenated along the channel dimension.
    """
    img_np = (image_2d.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    pil = Image.fromarray(img_np, mode="L").convert("RGB")

    proc = image_processor(images=pil, return_tensors="pt")
    pixel_values = proc["pixel_values"]
    if device is not None:
        pixel_values = pixel_values.to(device)

    enc_out = sam2_model.vision_encoder(pixel_values, output_hidden_states=True)
    hidden_states = enc_out.hidden_states  # tuple of (1, H', W', C) — channels-last

    feats = []
    for idx in layer_indices:
        feat = hidden_states[idx]  # (1, H', W', C)
        feat = feat.permute(0, 3, 1, 2)  # (1, C, H', W')
        feat = F.interpolate(
            feat, size=(grid_size, grid_size), mode="bilinear", align_corners=False
        )
        feats.append(feat.squeeze(0).cpu())  # (C, grid_size, grid_size)

    return torch.cat(feats, dim=0)  # (C * n_layers, grid_size, grid_size)
