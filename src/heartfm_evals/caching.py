"""Feature caching utilities.

Provides functions to pre-extract and cache features from frozen backbones,
and Dataset classes to load cached features at training time.

Caching patterns:
- 2D slice features: one ``.pt`` file per slice (for linear_probe / conv_decoder)
- 3D volume features: one ``.pt`` file per patient+frame (for UNetR decoders)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from heartfm_evals.constants import GRID_SIZE, SAX_TARGET_DEPTH
from heartfm_evals.features import (
    _pad_volume_z,
    extract_cinema_2d_feature_volume,
    extract_cinema_volume_features,
    extract_dino_volume_features,
    extract_multilayer_features,
    extract_sam2_2d_features,
    extract_sam_volume_features,
)


# ── 2D Slice Feature Caching ──────────────────────────────────────────────────
def cache_features(
    backbone: nn.Module,
    cinema_dataset,
    cache_dir: Path,
    layer_indices: tuple[int, ...] = (3, 6, 9, 11),
    device: torch.device | None = None,
) -> list[dict]:
    """Pre-extract and cache multi-layer features for all slices in a CineMA dataset.

    Saves one .pt file per 2D slice containing {"features": Tensor, "label": Tensor}.
    Returns a manifest (list of dicts) with paths and metadata.

    The actual files are stored in a subdirectory of ``cache_dir`` whose name encodes
    ``layer_indices`` (e.g. ``layers_3-6-9-11``). This means that changing
    ``layer_indices`` and reusing the same ``cache_dir`` will write to a different
    subdirectory, preventing silent reuse of incompatible cached tensors.

    Args:
        backbone: Frozen DINOv3 backbone in eval mode.
        cinema_dataset: CineMA EndDiastoleEndSystoleDataset.
        cache_dir: Root directory under which layer-specific subdirectories are created.
        layer_indices: Which intermediate layers to extract.
        device: Device for inference.

    Returns:
        List of dicts with keys: path, pid, is_ed, z_idx.
    """
    layers_tag = "layers_" + "-".join(str(i) for i in sorted(layer_indices))
    cache_dir = Path(cache_dir) / layers_tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for sample_idx in tqdm(range(len(cinema_dataset)), desc="Caching features"):
        sample = cinema_dataset[sample_idx]
        image_3d = sample["sax_image"]  # (1, H, W, z)
        label_3d = sample["sax_label"]  # (1, H, W, z)
        n_slices = sample["n_slices"]
        pid = sample["pid"]
        is_ed = sample["is_ed"]
        frame = "ed" if is_ed else "es"

        for z in range(n_slices):
            fname = f"{pid}_{frame}_z{z:02d}.pt"
            fpath = cache_dir / fname

            if fpath.exists():
                manifest.append({"path": fpath, "pid": pid, "is_ed": is_ed, "z_idx": z})
                continue

            image_2d = image_3d[0, :, :, z]  # (H, W)
            label_2d = label_3d[0, :, :, z]  # (H, W)

            feats = extract_multilayer_features(
                backbone, image_2d, layer_indices=layer_indices, device=device
            )

            torch.save({"features": feats, "label": label_2d.long()}, fpath)
            manifest.append({"path": fpath, "pid": pid, "is_ed": is_ed, "z_idx": z})

    return manifest


# ── 3D DINOv3 Volume Feature Caching ──────────────────────────────────────────
def cache_dino_volume_features(
    backbone: nn.Module,
    cinema_dataset,
    cache_dir: Path,
    layer_indices: tuple[int, ...] = (3, 6, 9, 11),
    device: torch.device | None = None,
    target_depth: int = SAX_TARGET_DEPTH,
) -> list[dict]:
    """Cache volume-level DINOv3 features for all patients/frames.

    Saves one ``.pt`` file per patient+frame containing per-layer features,
    the padded image, padded label, and the actual number of slices.

    Args:
        backbone: Frozen DINOv3 backbone in eval mode.
        cinema_dataset: CineMA ``EndDiastoleEndSystoleDataset``.
        cache_dir: Root cache directory.
        layer_indices: Which intermediate layers to extract.
        device: Device for inference.
        target_depth: Pad z to this depth.

    Returns:
        List of dicts with keys: ``path``, ``pid``, ``is_ed``, ``n_slices``.
    """
    layers_tag = "layers_" + "-".join(str(i) for i in sorted(layer_indices))
    out_dir = Path(cache_dir) / layers_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for sample_idx in tqdm(range(len(cinema_dataset)), desc="Caching volume features"):
        sample = cinema_dataset[sample_idx]
        image_3d = sample["sax_image"]  # (1, H, W, z)
        label_3d = sample["sax_label"]  # (1, H, W, z)
        n_slices = int(sample["n_slices"])
        pid = sample["pid"]
        is_ed = sample["is_ed"]
        frame = "ed" if is_ed else "es"

        fname = f"{pid}_{frame}.pt"
        fpath = out_dir / fname

        if fpath.exists():
            manifest.append(
                {"path": fpath, "pid": pid, "is_ed": is_ed, "n_slices": n_slices}
            )
            continue

        features_dict, padded_image, actual_slices = extract_dino_volume_features(
            backbone, image_3d, layer_indices, device, target_depth
        )

        # Pad label along z (padded region stays 0 = background)
        label = label_3d
        if label.shape[-1] > target_depth:
            label = label[..., :target_depth]
        elif label.shape[-1] < target_depth:
            label = F.pad(
                label, (0, target_depth - label.shape[-1]), mode="constant", value=0.0
            )

        save_dict: dict = {
            "image": padded_image,  # (1, H, W, target_depth)
            "label": label.long(),  # (1, H, W, target_depth)
            "n_slices": actual_slices,
        }
        save_dict.update(features_dict)

        torch.save(save_dict, fpath)
        manifest.append(
            {"path": fpath, "pid": pid, "is_ed": is_ed, "n_slices": actual_slices}
        )

    return manifest


# ── 3D CineMA Volume Feature Caching ──────────────────────────────────────────
def cache_cinema_volume_features(
    backbone: nn.Module,
    cinema_dataset,
    cache_dir: Path,
    device: torch.device | None = None,
    target_depth: int = SAX_TARGET_DEPTH,
) -> list[dict]:
    """Cache volume-level CineMA encoder features for all patients/frames.

    Saves one ``.pt`` file per patient+frame containing conv skips, ViT
    features, the padded image, padded label, and the actual slice count.

    Args:
        backbone: Frozen CineMA model in eval mode.
        cinema_dataset: CineMA ``EndDiastoleEndSystoleDataset``.
        cache_dir: Root cache directory.
        device: Device for inference.
        target_depth: Pad z to this depth.

    Returns:
        List of dicts with keys: ``path``, ``pid``, ``is_ed``, ``n_slices``.
    """
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for sample_idx in tqdm(
        range(len(cinema_dataset)), desc="Caching CineMA volume features"
    ):
        sample = cinema_dataset[sample_idx]
        image_3d = sample["sax_image"]  # (1, H, W, z)
        label_3d = sample["sax_label"]  # (1, H, W, z)
        n_slices = int(sample["n_slices"])
        pid = sample["pid"]
        is_ed = sample["is_ed"]
        frame = "ed" if is_ed else "es"

        fname = f"{pid}_{frame}.pt"
        fpath = out_dir / fname

        if fpath.exists():
            manifest.append(
                {"path": fpath, "pid": pid, "is_ed": is_ed, "n_slices": n_slices}
            )
            continue

        features_dict, padded_image, actual_slices = extract_cinema_volume_features(
            backbone, image_3d, device, target_depth
        )

        # Pad label along z
        label, _ = _pad_volume_z(label_3d, target_depth)

        save_dict: dict = {
            "image": padded_image,  # (1, H, W, target_depth)
            "label": label.long(),  # (1, H, W, target_depth)
            "n_slices": actual_slices,
        }
        save_dict.update(features_dict)

        torch.save(save_dict, fpath)
        manifest.append(
            {"path": fpath, "pid": pid, "is_ed": is_ed, "n_slices": actual_slices}
        )

    return manifest


# ── 3D SAM Volume Feature Caching ─────────────────────────────────────────────
def cache_sam_volume_features(
    sam_model: nn.Module,
    processor,
    cinema_dataset,
    cache_dir: Path,
    layer_indices: tuple[int, ...] = (2, 5, 8, 11),
    device: torch.device | None = None,
    target_depth: int = SAX_TARGET_DEPTH,
    grid_size: int = GRID_SIZE,
) -> list[dict]:
    """Cache volume-level SAM ViT features for all patients/frames.

    Saves one ``.pt`` file per patient+frame containing per-layer features,
    the padded image, padded label, and the actual number of slices.

    Args:
        sam_model: Frozen ``SamModel`` in eval mode.
        processor: ``SamImageProcessor`` for image pre-processing.
        cinema_dataset: CineMA ``EndDiastoleEndSystoleDataset``.
        cache_dir: Root cache directory.
        layer_indices: Which intermediate ViT layers to extract.
        device: Device for inference.
        target_depth: Pad z to this depth.
        grid_size: Downsample spatial dims to this size (default 12).

    Returns:
        List of dicts with keys: ``path``, ``pid``, ``is_ed``, ``n_slices``.
    """
    layers_tag = "layers_" + "-".join(str(i) for i in sorted(layer_indices))
    out_dir = Path(cache_dir) / layers_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for sample_idx in tqdm(
        range(len(cinema_dataset)), desc="Caching SAM volume features"
    ):
        sample = cinema_dataset[sample_idx]
        image_3d = sample["sax_image"]  # (1, H, W, z)
        label_3d = sample["sax_label"]  # (1, H, W, z)
        n_slices = int(sample["n_slices"])
        pid = sample["pid"]
        is_ed = sample["is_ed"]
        frame = "ed" if is_ed else "es"

        fname = f"{pid}_{frame}.pt"
        fpath = out_dir / fname

        if fpath.exists():
            manifest.append(
                {"path": fpath, "pid": pid, "is_ed": is_ed, "n_slices": n_slices}
            )
            continue

        features_dict, padded_image, actual_slices = extract_sam_volume_features(
            sam_model,
            processor,
            image_3d,
            layer_indices,
            device,
            target_depth,
            grid_size,
        )

        # Pad label along z
        label, _ = _pad_volume_z(label_3d, target_depth)

        save_dict: dict = {
            "image": padded_image,  # (1, H, W, target_depth)
            "label": label.long(),  # (1, H, W, target_depth)
            "n_slices": actual_slices,
        }
        save_dict.update(features_dict)

        torch.save(save_dict, fpath)
        manifest.append(
            {"path": fpath, "pid": pid, "is_ed": is_ed, "n_slices": actual_slices}
        )

    return manifest


# ── 2D CineMA Slice Feature Caching ───────────────────────────────────────────
def cache_cinema_2d_features(
    backbone: nn.Module,
    cinema_dataset,
    cache_dir: Path,
    device: torch.device | None = None,
) -> list[dict]:
    """Cache per-slice CineMA ViT features for all slices in a dataset.

    Runs the full 3D volume through ``feature_forward``, then slices the
    resulting feature volume per z-index.

    Args:
        backbone: Frozen CineMA model in eval mode.
        cinema_dataset: CineMA ``EndDiastoleEndSystoleDataset``.
        cache_dir: Cache directory.
        device: Device for inference.

    Returns:
        List of dicts with keys: ``path``, ``pid``, ``is_ed``, ``z_idx``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for sample_idx in tqdm(
        range(len(cinema_dataset)), desc="Caching CineMA 2D features"
    ):
        sample = cinema_dataset[sample_idx]
        image_3d = sample["sax_image"]  # (1, H, W, z)
        label_3d = sample["sax_label"]  # (1, H, W, z)
        n_slices = int(sample["n_slices"])
        pid = sample["pid"]
        is_ed = sample["is_ed"]
        frame = "ed" if is_ed else "es"

        # Skip backbone forward pass if all slices are already cached
        all_cached = all(
            (cache_dir / f"{pid}_{frame}_z{z_idx:02d}.pt").exists()
            for z_idx in range(n_slices)
        )
        if all_cached:
            for z_idx in range(n_slices):
                fpath = cache_dir / f"{pid}_{frame}_z{z_idx:02d}.pt"
                manifest.append(
                    {"path": fpath, "pid": pid, "is_ed": is_ed, "z_idx": z_idx}
                )
            continue

        feat_vol, used_depth = extract_cinema_2d_feature_volume(
            backbone, image_3d, device
        )
        gz = feat_vol.shape[-1]

        for z_idx in range(n_slices):
            fname = f"{pid}_{frame}_z{z_idx:02d}.pt"
            fpath = cache_dir / fname

            if fpath.exists():
                manifest.append(
                    {"path": fpath, "pid": pid, "is_ed": is_ed, "z_idx": z_idx}
                )
                continue

            # Map original slice index to feature-volume depth index
            src_z = min(z_idx, max(used_depth - 1, 0))
            feat_z = int(round(src_z * (gz - 1) / max(used_depth - 1, 1)))

            feats_2d = feat_vol[..., feat_z]  # (C, gx, gy)
            label_2d = label_3d[0, :, :, z_idx]

            torch.save({"features": feats_2d, "label": label_2d.long()}, fpath)
            manifest.append({"path": fpath, "pid": pid, "is_ed": is_ed, "z_idx": z_idx})

    return manifest


# ── 2D SAM2 Slice Feature Caching ─────────────────────────────────────────────
def cache_sam2_2d_features(
    sam2_model: nn.Module,
    image_processor,
    cinema_dataset,
    cache_dir: Path,
    device: torch.device | None = None,
) -> list[dict]:
    """Cache per-slice SAM2 image embeddings for all slices in a dataset.

    Args:
        sam2_model: Frozen SAM2 model in eval mode.
        image_processor: SAM2 processor for image pre-processing.
        cinema_dataset: CineMA ``EndDiastoleEndSystoleDataset``.
        cache_dir: Cache directory.
        device: Device for inference.

    Returns:
        List of dicts with keys: ``path``, ``pid``, ``is_ed``, ``z_idx``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for sample_idx in tqdm(range(len(cinema_dataset)), desc="Caching SAM2 2D features"):
        sample = cinema_dataset[sample_idx]
        image_3d = sample["sax_image"]  # (1, H, W, z)
        label_3d = sample["sax_label"]  # (1, H, W, z)
        n_slices = int(sample["n_slices"])
        pid = sample["pid"]
        is_ed = sample["is_ed"]
        frame = "ed" if is_ed else "es"

        for z_idx in range(n_slices):
            fname = f"{pid}_{frame}_z{z_idx:02d}.pt"
            fpath = cache_dir / fname

            if fpath.exists():
                manifest.append(
                    {"path": fpath, "pid": pid, "is_ed": is_ed, "z_idx": z_idx}
                )
                continue

            image_2d = image_3d[0, :, :, z_idx]  # (H, W)
            label_2d = label_3d[0, :, :, z_idx]  # (H, W)

            feats = extract_sam2_2d_features(
                sam2_model, image_processor, image_2d, device
            )

            torch.save({"features": feats, "label": label_2d.long()}, fpath)
            manifest.append({"path": fpath, "pid": pid, "is_ed": is_ed, "z_idx": z_idx})

    return manifest


# ── Cached Dataset Classes ─────────────────────────────────────────────────────
class CachedFeatureDataset(Dataset):
    """Loads pre-extracted 2D features + labels from cached .pt files."""

    def __init__(self, manifest: list[dict]):
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        entry = self.manifest[idx]
        data = torch.load(entry["path"], weights_only=True)
        return {
            "features": data["features"],  # (C, h, w)
            "label": data["label"],  # (H, W)
            "pid": entry["pid"],
        }


class CachedVolumeDataset(Dataset):
    """Loads pre-cached volume-level features from ``.pt`` files."""

    def __init__(
        self,
        manifest: list[dict],
        layer_indices: tuple[int, ...] = (3, 6, 9, 11),
    ):
        self.manifest = manifest
        self.layer_indices = layer_indices

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        entry = self.manifest[idx]
        data = torch.load(entry["path"], weights_only=True)
        result = {
            "image": data["image"],  # (1, H, W, Z)
            "label": data["label"],  # (1, H, W, Z)
            "n_slices": data["n_slices"],
            "pid": entry["pid"],
        }
        for layer_idx in self.layer_indices:
            key = f"layer_{layer_idx}"
            result[key] = data[key]  # (embed_dim, g, g, Z)
        return result


class CachedCinemaVolumeDataset(Dataset):
    """Loads pre-cached CineMA volume-level features from ``.pt`` files."""

    def __init__(
        self,
        manifest: list[dict],
        n_conv_skips: int = 3,
    ):
        self.manifest = manifest
        self.n_conv_skips = n_conv_skips

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        entry = self.manifest[idx]
        data = torch.load(entry["path"], weights_only=True)
        result: dict = {
            "image": data["image"],  # (1, H, W, Z)
            "label": data["label"],  # (1, H, W, Z)
            "n_slices": data["n_slices"],
            "pid": entry["pid"],
            "vit_features": data["vit_features"],
        }
        for i in range(self.n_conv_skips):
            result[f"conv_skip_{i}"] = data[f"conv_skip_{i}"]
        return result
