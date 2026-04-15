"""
Dense Probe: Pixel-Level Cardiac Segmentation with CineMA on ACDC

Architecture: Frozen CineMA (SAX feature extraction) -> cached 2D slice features ->
upsample -> small conv decoder -> class logits (BG, RV, MYO, LV).
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from cinema import CineMA
from cinema.segmentation.dataset import EndDiastoleEndSystoleDataset
from monai.transforms import ScaleIntensityd
from torch.utils.data import DataLoader
from tqdm import tqdm

from heartfm_evals.dense_linear_probe import (
    CLASS_COLORS,
    CLASS_NAMES,
    IMAGE_SIZE,
    NUM_CLASSES,
    CachedFeatureDataset,
    DiceLoss,
    evaluate,
    macro_dice,
    overlay_labels,
    train_one_epoch,
)
from heartfm_evals.reproducibility import set_seed

# -- Paths --
# ACDC_DATA_DIR = Path("/home/rwood/heartfm/data-evals/acdc/")
ACDC_DATA_DIR = Path("../../../data/heartfm/processed/acdc/")  # adjust as needed


# -- CineMA loading --
HF_CACHE_DIR = Path("../../model_weights/hf")
AUTO_DOWNLOAD = True

# -- Cache --
CACHE_NAME = "cinema_pretrained"
CACHE_DIR = Path(f"../../feature_cache/{CACHE_NAME}")
SAX_TARGET_DEPTH = 16  # CineMA pretrained SAX depth

# -- Training --
BATCH_SIZE = 4
LR = 1e-3
WEIGHT_DECAY = 1e-4
N_EPOCHS = 1
PATIENCE = 10
SEED = 0

# -- Device --
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
set_seed(SEED)
print(f"Using device: {DEVICE}")
print(f"CineMA source: mathpluscode/CineMA (auto_download={AUTO_DOWNLOAD})")


# -- Data --
train_meta_df = pd.read_csv(ACDC_DATA_DIR / "train_metadata.csv")
test_meta_df = pd.read_csv(ACDC_DATA_DIR / "test_metadata.csv")

if "pathology" in train_meta_df.columns:
    val_pids = (
        train_meta_df.groupby("pathology").sample(n=2, random_state=0)["pid"].tolist()
    )
else:
    val_pids = train_meta_df.sample(frac=0.1, random_state=0)["pid"].tolist()

train_split_df = train_meta_df[~train_meta_df["pid"].isin(val_pids)].reset_index(
    drop=True
)
val_split_df = train_meta_df[train_meta_df["pid"].isin(val_pids)].reset_index(drop=True)

print(f"Train split: {len(train_split_df)} patients")
print(f"Val split:   {len(val_split_df)} patients")
print(f"Test set:    {len(test_meta_df)} patients")

transform = ScaleIntensityd(keys="sax_image", factor=1 / 255, channel_wise=False)

train_cinema = EndDiastoleEndSystoleDataset(
    data_dir=ACDC_DATA_DIR / "train",
    meta_df=train_split_df,
    views="sax",
    transform=transform,
)

val_cinema = EndDiastoleEndSystoleDataset(
    data_dir=ACDC_DATA_DIR / "train",
    meta_df=val_split_df,
    views="sax",
    transform=transform,
)

test_cinema = EndDiastoleEndSystoleDataset(
    data_dir=ACDC_DATA_DIR / "test",
    meta_df=test_meta_df,
    views="sax",
    transform=transform,
)

print(f"Train CineMA dataset: {len(train_cinema)} samples")
print(f"Val CineMA dataset:   {len(val_cinema)} samples")
print(f"Test CineMA dataset:  {len(test_cinema)} samples")


# -- Load Pretrained CineMA and Cache Slice Features --
backbone = CineMA.from_pretrained(
    cache_dir=str(HF_CACHE_DIR),
    local_files_only=not AUTO_DOWNLOAD,
)
backbone.eval().to(DEVICE)
for p in backbone.parameters():
    p.requires_grad = False

grid_size = backbone.enc_down_dict["sax"].patch_embed.grid_size
embed_dim = backbone.enc_down_dict["sax"].patch_embed.proj.out_features
print("Loaded CineMA pretrained backbone")
print(f"SAX token grid: {grid_size}, embed_dim: {embed_dim}")


@torch.inference_mode()
def extract_cinema_sax_feature_volume(
    backbone, sax_volume, device, target_depth=SAX_TARGET_DEPTH
):
    # sax_volume: (1, H, W, z) in [0,1]
    vol = sax_volume
    z = int(vol.shape[-1])
    if z > target_depth:
        vol = vol[..., :target_depth]
        z = target_depth
    if z < target_depth:
        vol = F.pad(vol, (0, target_depth - z), mode="constant", value=0.0)

    batch = {
        "sax": vol.unsqueeze(0).to(device=device, dtype=torch.float32)
    }  # (1,1,H,W,Z)
    tokens = backbone.feature_forward(batch)["sax"]  # (1, n_tokens, C)

    b, n_tokens, c = tokens.shape
    gx, gy, gz = backbone.enc_down_dict["sax"].patch_embed.grid_size
    if n_tokens != gx * gy * gz:
        raise RuntimeError(f"Unexpected token count: {n_tokens} vs grid {gx}x{gy}x{gz}")

    feat_vol = (
        tokens.reshape(b, gx, gy, gz, c).permute(0, 4, 1, 2, 3).contiguous()
    )  # (1,C,gx,gy,gz)
    return feat_vol.squeeze(0).cpu(), z


def cache_cinema_features(backbone, cinema_dataset, cache_dir, device):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for sample_idx in tqdm(range(len(cinema_dataset)), desc="Caching CineMA features"):
        sample = cinema_dataset[sample_idx]
        image_3d = sample["sax_image"]  # (1, H, W, z)
        label_3d = sample["sax_label"]  # (1, H, W, z)
        n_slices = int(sample["n_slices"])
        pid = sample["pid"]
        is_ed = sample["is_ed"]
        frame = "ed" if is_ed else "es"

        feat_vol, used_depth = extract_cinema_sax_feature_volume(
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


print("Caching training features...")
train_manifest = cache_cinema_features(
    backbone, train_cinema, CACHE_DIR / "train", DEVICE
)

print("\nCaching validation features...")
val_manifest = cache_cinema_features(backbone, val_cinema, CACHE_DIR / "val", DEVICE)

print("\nCaching test features...")
test_manifest = cache_cinema_features(backbone, test_cinema, CACHE_DIR / "test", DEVICE)

sample = torch.load(train_manifest[0]["path"], weights_only=True)
print(f"Cached train slices: {len(train_manifest)}")
print(f"Cached val slices:   {len(val_manifest)}")
print(f"Cached test slices:  {len(test_manifest)}")
print(f"Feature shape: {sample['features'].shape}")
print(f"Label shape:   {sample['label'].shape}")


# -- Decoder, Loss, and Training --
class WeightedCombinedLoss(nn.Module):
    def __init__(self, ce_weight_tensor, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight_tensor)
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.ce_weight * self.ce(
            logits, targets.long()
        ) + self.dice_weight * self.dice(logits, targets)


class DenseDecoderProbe(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes=NUM_CLASSES,
        output_size=(IMAGE_SIZE, IMAGE_SIZE),
        hidden_dim=128,
    ):
        super().__init__()
        self.output_size = output_size
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, features):
        x = F.interpolate(
            features, size=self.output_size, mode="bilinear", align_corners=False
        )
        x = self.decoder(x)
        return self.head(x)


train_ds = CachedFeatureDataset(train_manifest)
val_ds = CachedFeatureDataset(val_manifest)
test_ds = CachedFeatureDataset(test_manifest)

g = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, generator=g)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

in_channels = sample["features"].shape[0]
probe = DenseDecoderProbe(in_channels=in_channels).to(DEVICE)

class_counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
for entry in train_manifest:
    y = torch.load(entry["path"], weights_only=True)["label"]
    class_counts += torch.bincount(y.long().reshape(-1), minlength=NUM_CLASSES)

class_weights = class_counts.sum().float() / (
    NUM_CLASSES * class_counts.clamp_min(1).float()
)
class_weights[0] = class_weights[0] * 0.5
class_weights = class_weights / class_weights.mean()
criterion = WeightedCombinedLoss(
    class_weights.to(DEVICE), ce_weight=1.0, dice_weight=1.0
)

optimizer = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

print(f"Dense probe input channels: {in_channels}")
print(f"Class weights (BG/RV/MYO/LV): {class_weights.tolist()}")
print(
    f"Trainable params: {sum(p.numel() for p in probe.parameters() if p.requires_grad):,}"
)

best_val_dice = 0.0
best_epoch = 0
epochs_no_improve = 0
history = {"train_loss": [], "val_macro_dice": [], "lr": []}

for epoch in range(1, N_EPOCHS + 1):
    train_loss = train_one_epoch(probe, train_loader, criterion, optimizer, DEVICE)
    scheduler.step()

    val_metrics = evaluate(probe, val_loader, DEVICE)
    val_dice = val_metrics["macro_dice"]

    history["train_loss"].append(train_loss)
    history["val_macro_dice"].append(val_dice)
    history["lr"].append(optimizer.param_groups[0]["lr"])

    improved = val_dice > best_val_dice
    if epoch == 1 or epoch % 5 == 0 or improved:
        tag = " *" if improved else ""
        print(
            f"Epoch {epoch:3d}/{N_EPOCHS} | loss={train_loss:.4f} | val Dice={val_dice:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}{tag}"
        )

    if improved:
        best_val_dice = val_dice
        best_epoch = epoch
        epochs_no_improve = 0
        best_state = {
            k: v.detach().cpu().clone() for k, v in probe.state_dict().items()
        }
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch}. Best val Dice={best_val_dice:.4f} at epoch {best_epoch}."
            )
            break

probe.load_state_dict(best_state)
print(
    f"Restored best checkpoint from epoch {best_epoch} (val Dice={best_val_dice:.4f})"
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
ax1.plot(history["train_loss"], label="Train Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss (weighted CE + Dice)")
ax1.set_title("Training Loss")
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.plot(history["val_macro_dice"], label="Val Macro Dice", color="tab:orange")
ax2.axhline(
    best_val_dice, ls="--", color="gray", alpha=0.5, label=f"Best={best_val_dice:.4f}"
)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Macro Dice (excl. BG)")
ax2.set_title("Validation Performance")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig("cinema_training_curves.png", dpi=150)
plt.close()


# -- Test Evaluation --
test_metrics = evaluate(probe, test_loader, DEVICE)

print("Per-class Dice scores (test set):")
for name, d in test_metrics["per_class_dice"].items():
    print(f"  {name:>3s}: {d:.4f}")
print(f"\nMacro Dice (excl. BG): {test_metrics['macro_dice']:.4f}")

n_show = min(6, len(test_manifest))
show_indices = np.linspace(0, len(test_manifest) - 1, n_show, dtype=int)

fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show), dpi=150)
if n_show == 1:
    axes = axes[np.newaxis, :]

probe.eval()
with torch.inference_mode():
    for row, idx in enumerate(show_indices):
        entry = test_manifest[idx]
        data = torch.load(entry["path"], weights_only=True)
        feats = data["features"].unsqueeze(0).to(DEVICE)
        label = data["label"].numpy()

        logits = probe(feats)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        gt_overlay = overlay_labels(label, IMAGE_SIZE, IMAGE_SIZE)
        pred_overlay = overlay_labels(pred, IMAGE_SIZE, IMAGE_SIZE)

        axes[row, 0].imshow(label, cmap="tab10", vmin=0, vmax=3)
        axes[row, 0].set_title(
            f"GT Labels ({entry['pid']}, z={entry['z_idx']})", fontsize=9
        )
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt_overlay)
        axes[row, 1].set_title("Ground Truth Overlay", fontsize=9)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_overlay)
        axes[row, 2].set_title(
            f"Predicted (Dice={macro_dice(pred, label):.3f})", fontsize=9
        )
        axes[row, 2].axis("off")

legend_patches = [
    mpatches.Patch(color=CLASS_COLORS[c][:3] + (1.0,), label=CLASS_NAMES[c])
    for c in range(1, NUM_CLASSES)
]
axes[-1, 2].legend(handles=legend_patches, loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("cinema_test_predictions.png", dpi=150)
plt.close()


# -- Save Model --
save_path = Path("dense_probe_cinema_pretrained.pt")
torch.save(
    {
        "model_state_dict": probe.state_dict(),
        "backbone": "mathpluscode/CineMA",
        "in_channels": in_channels,
        "num_classes": NUM_CLASSES,
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "test_metrics": test_metrics,
        "val_pids": val_pids,
    },
    save_path,
)
print(f"Model saved to {save_path}")
