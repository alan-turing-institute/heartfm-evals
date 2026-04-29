"""Backbone loading utilities.

Provides a unified :func:`load_backbone` factory that returns a frozen backbone
model together with a metadata dict (``embed_dim``, ``n_layers``,
``layer_indices``, …) for every supported backbone family.

Supported backbone types:

* ``"dinov3"`` – DINOv3 ViT loaded via local ``torch.hub``.
* ``"cinema"`` – CineMA 3-D cardiac ViT from HuggingFace.
* ``"sam"``   – SAM v1 (used for classification).
* ``"sam2"``  – SAM 2.1 Hiera (used for segmentation).
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# ── DINOv3 model configs ──────────────────────────────────────────────────────
DINOV3_CONFIGS: dict[str, dict[str, Any]] = {
    "dinov3_vits16": {
        "embed_dim": 384,
        "n_layers": 12,
        "layer_indices": (3, 6, 9, 11),
    },
    "dinov3_vitb16": {
        "embed_dim": 768,
        "n_layers": 12,
        "layer_indices": (3, 6, 9, 11),
    },
    "dinov3_vitl16": {
        "embed_dim": 1024,
        "n_layers": 24,
        "layer_indices": (5, 11, 17, 23),
        # vitl16 hub validates a hash in the filename
        "weights_filename": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    },
}

# ── SAM 2.1 Hiera configs ────────────────────────────────────────────────────
# hidden_states from transformers includes the initial patch embedding as
# index 0, so hidden_states[i+1] is the output of block i.
#
# embed_dim:       Stage 3 channel count — kept for backward compatibility.
# cls_embed_dim:   Stage 4 channel count — used for classification (GAP of final block).
# layer_indices:   One representative block per stage (Stage 1–4), giving
#                  truly multi-scale features for segmentation.
# stage_embed_dims: Channel counts at each of the 4 layer_indices
#                   (Stage 1, Stage 2, Stage 3 end, Stage 4 end).
#
# Stage layout (block counts verified empirically):
#   tiny       12 total: Stage1=[1](256×256,C=96) Stage2=[2–3](128×128,C=192)
#                        Stage3=[4–10](64×64,C=384) Stage4=[11–12](32×32,C=768)
#   small      16 total: Stage1=[1](256×256,C=96) Stage2=[2–3](128×128,C=192)
#                        Stage3=[4–14](64×64,C=384) Stage4=[15–16](32×32,C=768)
#   base-plus  24 total: Stage1=[1–2](256×256,C=112) Stage2=[3–5](128×128,C=224)
#                        Stage3=[6–21](64×64,C=448) Stage4=[22–24](32×32,C=896)
#   large      48 total: Stage1=[1–2](256×256,C=144) Stage2=[3–8](128×128,C=288)
#                        Stage3=[9–44](64×64,C=576) Stage4=[45–48](32×32,C=1152)
SAM2_CONFIGS: dict[str, dict[str, Any]] = {
    "facebook/sam2.1-hiera-tiny": {
        "embed_dim": 384,
        "layer_indices": (1, 3, 10, 12),
        "stage_embed_dims": (96, 192, 384, 768),
    },
    "facebook/sam2.1-hiera-small": {
        "embed_dim": 384,
        "layer_indices": (1, 3, 14, 16),
        "stage_embed_dims": (96, 192, 384, 768),
    },
    "facebook/sam2.1-hiera-base-plus": {
        "embed_dim": 448,
        "layer_indices": (2, 5, 21, 24),
        "stage_embed_dims": (112, 224, 448, 896),
    },
    "facebook/sam2.1-hiera-large": {
        "embed_dim": 576,
        "layer_indices": (2, 8, 44, 48),
        "stage_embed_dims": (144, 288, 576, 1152),
    },
}


# ── Public helpers ─────────────────────────────────────────────────────────────


def _freeze(model: nn.Module) -> nn.Module:
    """Set model to eval mode and freeze all parameters."""
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_backbone(
    backbone_type: str,
    device: torch.device,
    *,
    # DINOv3 options
    dinov3_model_name: str = "dinov3_vits16",
    dinov3_repo_dir: str = "models/dinov3/",
    dinov3_weights_path: str | None = None,
    # SAM v1 options
    sam_model_id: str = "facebook/sam-vit-base",
    # SAM2 options
    sam2_model_id: str = "facebook/sam2.1-hiera-base-plus",
    # Shared HuggingFace options
    hf_cache_dir: str | Path = "model_weights/hf",
    auto_download: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load and freeze a backbone, returning ``(model, info_dict)``.

    Parameters
    ----------
    backbone_type:
        One of ``"dinov3"``, ``"cinema"``, ``"sam"``, ``"sam2"``.
    device:
        Target device for the model.

    Returns
    -------
    model:
        Frozen backbone on *device* in eval mode.
    info:
        Dict with at least ``"embed_dim"`` and ``"backbone_type"``.
        May also contain ``"n_layers"``, ``"layer_indices"``,
        ``"sam_image_processor"``, etc.
    """
    hf_cache_dir = Path(hf_cache_dir)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    if backbone_type == "dinov3":
        return _load_dinov3(
            dinov3_model_name, dinov3_repo_dir, dinov3_weights_path, device
        )
    if backbone_type == "cinema":
        return _load_cinema(hf_cache_dir, auto_download, device)
    if backbone_type == "sam":
        return _load_sam(sam_model_id, hf_cache_dir, auto_download, device)
    if backbone_type == "sam2":
        return _load_sam2(sam2_model_id, hf_cache_dir, auto_download, device)

    msg = f"Unknown backbone_type: {backbone_type!r}"
    raise ValueError(msg)


# ── Private loaders ────────────────────────────────────────────────────────────


def _load_dinov3(
    model_name: str,
    repo_dir: str,
    weights_path: str | None,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    cfg = DINOV3_CONFIGS[model_name]

    if weights_path is None:
        fname = cfg.get("weights_filename", f"{model_name}.pth")
        candidates = glob.glob(f"model_weights/{fname}")
        if not candidates:
            candidates = glob.glob(f"model_weights/{model_name}*.pth")
        weights_path = candidates[0] if candidates else None

    if weights_path is not None and not Path(weights_path).exists():
        weights_path = None

    backbone = torch.hub.load(
        repo_dir, model_name, source="local", weights=weights_path
    )
    _freeze(backbone).to(device)

    return backbone, {
        "backbone_type": "dinov3",
        "model_name": model_name,
        "embed_dim": cfg["embed_dim"],
        "n_layers": cfg["n_layers"],
        "layer_indices": cfg["layer_indices"],
    }


def _load_cinema(
    hf_cache_dir: Path, auto_download: bool, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    from cinema import CineMA

    backbone = CineMA.from_pretrained(
        cache_dir=str(hf_cache_dir),
        local_files_only=not auto_download,
    )
    embed_dim: int = backbone.enc_down_dict["sax"].patch_embed.proj.out_features
    enc_conv_chans = tuple(
        b.patch_embed.conv.out_channels
        for b in backbone.enc_down_dict["sax"].conv_blocks
    )
    _freeze(backbone).to(device)

    return backbone, {
        "backbone_type": "cinema",
        "model_name": "cinema_pretrained",
        "embed_dim": embed_dim,
        "enc_conv_chans": enc_conv_chans,
    }


def _load_sam(
    model_id: str, hf_cache_dir: Path, auto_download: bool, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    from transformers import SamImageProcessor, SamModel

    processor = SamImageProcessor.from_pretrained(
        model_id,
        cache_dir=str(hf_cache_dir),
        local_files_only=not auto_download,
    )
    backbone = SamModel.from_pretrained(
        model_id,
        cache_dir=str(hf_cache_dir),
        local_files_only=not auto_download,
    )
    embed_dim: int = backbone.config.vision_config.hidden_size
    _freeze(backbone).to(device)

    return backbone, {
        "backbone_type": "sam",
        "model_name": model_id.split("/")[-1].replace("-", "_"),
        "embed_dim": embed_dim,
        "layer_indices": (2, 5, 8, 11),
        "sam_image_processor": processor,
    }


def _load_sam2(
    model_id: str, hf_cache_dir: Path, auto_download: bool, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    from transformers import Sam2Model, Sam2Processor

    cfg = SAM2_CONFIGS[model_id]

    processor = Sam2Processor.from_pretrained(
        model_id,
        cache_dir=str(hf_cache_dir),
        local_files_only=not auto_download,
    )
    backbone = Sam2Model.from_pretrained(
        model_id,
        cache_dir=str(hf_cache_dir),
        local_files_only=not auto_download,
    )
    _freeze(backbone).to(device)

    return backbone, {
        "backbone_type": "sam2",
        "model_name": model_id.split("/")[-1].replace(".", "_").replace("-", "_"),
        "embed_dim": cfg["embed_dim"],
        "cls_embed_dim": cfg["cls_embed_dim"],
        "layer_indices": cfg["layer_indices"],
        "stage_embed_dims": cfg["stage_embed_dims"],
        "sam2_processor": processor,
    }
