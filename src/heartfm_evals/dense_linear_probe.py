"""Dense linear probe for pixel-level segmentation using frozen DINOv3 features.

Implements the DINOv3 paper's linear evaluation protocol for semantic segmentation:
frozen backbone → multi-layer feature concatenation → bilinear upsample → 1×1 Conv2d.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH_SIZE = 16
IMAGE_SIZE = 192
NUM_CLASSES = 4
CLASS_NAMES = {0: "BG", 1: "RV", 2: "MYO", 3: "LV"}
CLASS_COLORS = {
    0: (0, 0, 0, 0),
    1: (0, 0, 1, 0.4),
    2: (0, 1, 0, 0.4),
    3: (1, 0, 0, 0.4),
}

MODEL_CONFIGS = {
    "dinov3_vits16": {"embed_dim": 384, "n_layers": 12},
    "dinov3_vitb16": {"embed_dim": 768, "n_layers": 12},
    "dinov3_vitl16": {"embed_dim": 1024, "n_layers": 24},
}

_imagenet_normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_slice(image_2d: torch.Tensor) -> torch.Tensor:
    """Prepare a (H, W) [0,1] tensor for DINOv3.  Returns (1, 3, H, W)."""
    x = image_2d.unsqueeze(0).repeat(3, 1, 1)
    x = _imagenet_normalize(x)
    return x.unsqueeze(0)


# ── Metrics ────────────────────────────────────────────────────────────────────
def dice_score(pred: np.ndarray, true: np.ndarray, class_idx: int) -> float:
    """Dice coefficient for a single class."""
    pred_c = pred == class_idx
    true_c = true == class_idx
    intersection = (pred_c & true_c).sum()
    return float(2 * intersection / (pred_c.sum() + true_c.sum() + 1e-8))


def macro_dice(pred: np.ndarray, true: np.ndarray, exclude_bg: bool = True) -> float:
    """Macro-averaged Dice across foreground classes."""
    start = 1 if exclude_bg else 0
    return float(
        np.mean([dice_score(pred, true, c) for c in range(start, NUM_CLASSES)])
    )


# ── Visualization ──────────────────────────────────────────────────────────────
def overlay_labels(
    label_map: np.ndarray,
    h: int,
    w: int,
    class_colors: dict | None = None,
) -> np.ndarray:
    """Create an RGBA overlay image from a label map of shape (h, w)."""
    if class_colors is None:
        class_colors = CLASS_COLORS
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    for cls, rgba in class_colors.items():
        mask = label_map == cls
        for ch in range(4):
            overlay[:, :, ch][mask] = rgba[ch]
    return overlay


# ── Dice Loss ──────────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation."""

    def __init__(self, smooth: float = 1.0, exclude_bg: bool = True):
        super().__init__()
        self.smooth = smooth
        self.exclude_bg = exclude_bg

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw predictions.
            targets: (B, H, W) integer class labels.
        """
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        targets_oh = F.one_hot(targets.long(), num_classes=logits.shape[1])  # (B,H,W,C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        start_c = 1 if self.exclude_bg else 0
        probs = probs[:, start_c:]
        targets_oh = targets_oh[:, start_c:]

        dims = (0, 2, 3)  # reduce over batch and spatial
        intersection = (probs * targets_oh).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets_oh.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """CrossEntropy + DiceLoss."""

    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(
            logits, targets.long()
        ) + self.dice_weight * self.dice(logits, targets)


# ── Dense Linear Probe ────────────────────────────────────────────────────────
class DenseLinearProbe(nn.Module):
    """Per-pixel linear classifier on concatenated multi-layer DINOv3 features.

    Architecture:
        frozen backbone → extract features from selected layers →
        concatenate → bilinear upsample → 1×1 Conv2d → class logits
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int = NUM_CLASSES,
        layer_indices: tuple[int, ...] = (3, 6, 9, 11),
        output_size: tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE),
    ):
        super().__init__()
        self.layer_indices = layer_indices
        self.output_size = output_size
        in_channels = embed_dim * len(layer_indices)
        self.head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Pre-extracted concatenated features (B, C, h, w)
                      where C = embed_dim * n_selected_layers, h/w = patch grid.
        Returns:
            logits: (B, num_classes, H, W) at output_size resolution.
        """
        x = F.interpolate(
            features, size=self.output_size, mode="bilinear", align_corners=False
        )
        return self.head(x)


# ── Feature Extraction ─────────────────────────────────────────────────────────
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


# ── Feature Caching ────────────────────────────────────────────────────────────
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

    Args:
        backbone: Frozen DINOv3 backbone in eval mode.
        cinema_dataset: CineMA EndDiastoleEndSystoleDataset.
        cache_dir: Directory to save cached features.
        layer_indices: Which intermediate layers to extract.
        device: Device for inference.

    Returns:
        List of dicts with keys: path, pid, is_ed, z_idx.
    """
    cache_dir = Path(cache_dir)
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


# ── Datasets ───────────────────────────────────────────────────────────────────
class CachedFeatureDataset(Dataset):
    """Loads pre-extracted features + labels from cached .pt files."""

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


class ACDCSliceDataset(Dataset):
    """Wraps a CineMA EndDiastoleEndSystoleDataset to yield individual 2D slices.

    Each item is a dict with keys: image (3, H, W), label (H, W), pid.

    Uses lazy indexing: only a lightweight (sample_idx, z, pid) index list is
    built at construction time; the actual volume tensors are loaded on demand
    inside ``__getitem__``.
    """

    def __init__(self, cinema_dataset, augment: bool = False):
        self.cinema_dataset = cinema_dataset
        # Lightweight index: list of (sample_idx, z, pid, is_ed)
        self.index: list[tuple[int, int, str, bool]] = []
        for sample_idx in range(len(cinema_dataset)):
            sample = cinema_dataset[sample_idx]
            n_slices = sample["n_slices"]
            pid = sample["pid"]
            is_ed = sample["is_ed"]
            for z in range(n_slices):
                self.index.append((sample_idx, z, pid, is_ed))
        self.augment = augment
        self._cache_key: int | None = None
        self._cache_value: dict | None = None

    def __len__(self) -> int:
        return len(self.index)

    def _load_sample(self, sample_idx: int) -> dict:
        """Load a volume sample, reusing the cached result when possible."""
        if self._cache_key != sample_idx:
            self._cache_key = sample_idx
            self._cache_value = self.cinema_dataset[sample_idx]
        return self._cache_value

    def __getitem__(self, idx: int) -> dict:
        sample_idx, z, pid, _ = self.index[idx]
        sample = self._load_sample(sample_idx)
        image_2d = sample["sax_image"][0, :, :, z]  # (H, W) in [0,1]
        label_2d = sample["sax_label"][0, :, :, z]  # (H, W)

        # Repeat to 3 channels + ImageNet normalize
        img = image_2d.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
        img = _imagenet_normalize(img)

        if self.augment:
            # Random horizontal flip (apply same to image and label)
            if torch.rand(1).item() > 0.5:
                img = img.flip(-1)
                label_2d = label_2d.flip(-1)

        return {"image": img, "label": label_2d.long(), "pid": pid}


# ── Training Utilities ─────────────────────────────────────────────────────────
def train_one_epoch(
    model: DenseLinearProbe,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch on cached features. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in dataloader:
        features = batch["features"].to(device)  # (B, C, h, w)
        labels = batch["label"].to(device)  # (B, H, W)

        logits = model(features)  # (B, num_classes, H, W)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.inference_mode()
def evaluate(
    model: DenseLinearProbe,
    dataloader,
    device: torch.device,
) -> dict:
    """Evaluate on cached features. Returns per-class Dice and macro Dice."""
    model.eval()
    all_preds = []
    all_labels = []

    for batch in dataloader:
        features = batch["features"].to(device)
        labels = batch["label"]

        logits = model(features)
        preds = logits.argmax(dim=1).cpu()  # (B, H, W)

        all_preds.append(preds.numpy())
        all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    per_class = {
        CLASS_NAMES[c]: dice_score(all_preds, all_labels, c) for c in range(NUM_CLASSES)
    }
    m_dice = macro_dice(all_preds, all_labels)

    return {"per_class_dice": per_class, "macro_dice": m_dice}
