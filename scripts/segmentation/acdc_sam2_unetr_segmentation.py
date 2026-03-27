"""
Dense UNetR Probe: Pixel-Level Cardiac Segmentation with SAM 2 on ACDC

Architecture: Frozen SAM 2 Hiera image encoder -> cached per-layer intermediate
features from the main Hiera stage -> stacked along z to form pseudo-3D volumes
-> 3D UNetR decoder -> class logits (BG, RV, MYO, LV).

Layer indices are chosen from within SAM 2's main (third) Hiera stage, where all
blocks share the same spatial resolution.  Embed dims are 4x the base embed dim
for that stage.
"""

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from cinema.segmentation.dataset import EndDiastoleEndSystoleDataset
from monai.transforms import ScaleIntensityd
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Sam2Model, Sam2Processor

from heartfm_evals.dense_linear_probe import (
    CLASS_COLORS,
    CLASS_NAMES,
    IMAGE_SIZE,
    NUM_CLASSES,
    macro_dice,
    overlay_labels,
)
from heartfm_evals.dense_unetr_probe import (
    GRID_SIZE,
    SAX_TARGET_DEPTH,
    CachedVolumeDataset,
    DINOv3UNetRDecoder,
    MaskedVolumeLoss,
    evaluate_vol,
    train_one_epoch_vol,
)

_SAM2_VARIANTS = [
    "facebook/sam2.1-hiera-tiny",
    "facebook/sam2.1-hiera-small",
    "facebook/sam2.1-hiera-base-plus",
    "facebook/sam2.1-hiera-large",
]

# Per-variant configs: embed_dim at main Hiera stage (base * 4) and 4 layer
# indices evenly spaced within that stage's block range.
#   tiny:     stages [1,2,7,2], stage-2 blocks 3-9  (embed 96*4=384)
#   small:    stages [1,2,11,2], stage-2 blocks 3-13 (embed 96*4=384)
#   base+:    stages [2,3,16,3], stage-2 blocks 5-20 (embed 112*4=448)
#   large:    stages [2,6,36,4], stage-2 blocks 8-43 (embed 144*4=576)
_SAM2_CONFIGS = {
    "facebook/sam2.1-hiera-tiny": {
        "embed_dim": 384,
        "layer_indices": (3, 5, 7, 9),
    },
    "facebook/sam2.1-hiera-small": {
        "embed_dim": 384,
        "layer_indices": (3, 6, 10, 13),
    },
    "facebook/sam2.1-hiera-base-plus": {
        "embed_dim": 448,
        "layer_indices": (5, 10, 15, 20),
    },
    "facebook/sam2.1-hiera-large": {
        "embed_dim": 576,
        "layer_indices": (8, 20, 32, 43),
    },
}

parser = argparse.ArgumentParser(
    description="Dense UNetR segmentation probe with SAM 2 on ACDC"
)
parser.add_argument(
    "--model",
    default="facebook/sam2.1-hiera-base-plus",
    choices=_SAM2_VARIANTS,
    help="SAM 2.1 model variant to use",
)
args = parser.parse_args()

# -- Paths --
ACDC_DATA_DIR = Path("/home/rwood/heartfm/data-evals/acdc/")

# -- SAM 2 source --
SAM2_MODEL_ID = args.model
SAM2_LOCAL_DIR = None  # optional: Path("../../model_weights/sam2.1-hiera-base-plus")
HF_CACHE_DIR = Path("../../model_weights/hf")
AUTO_DOWNLOAD = True

# -- Backbone config --
EMBED_DIM = _SAM2_CONFIGS[SAM2_MODEL_ID]["embed_dim"]
LAYER_INDICES = _SAM2_CONFIGS[SAM2_MODEL_ID]["layer_indices"]

# -- 3D UNetR decoder config (matches CineMA segmentation.yaml) --
DEC_CHANS = (32, 64, 128, 256, 512)
DEC_PATCH_SIZE = (2, 2, 1)
DEC_SCALE_FACTOR = (2, 2, 1)
Z_PAD = SAX_TARGET_DEPTH  # 16

# -- Cache --
MODEL_SHORT_NAME = SAM2_MODEL_ID.split("/")[-1].replace("-", "_").replace(".", "_")
CACHE_DIR = Path(f"../../feature_cache/{MODEL_SHORT_NAME}_unetr3d")

# -- Training --
BATCH_SIZE = 4  # volumes use ~16x more memory than 2D slices
LR = 1e-3
WEIGHT_DECAY = 1e-4
N_EPOCHS = 20
PATIENCE = 10

# -- Device --
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Backbone: {SAM2_MODEL_ID} (embed_dim={EMBED_DIM})")
print(f"Selected layers: {LAYER_INDICES}")
print(f"Decoder: UpsampleDecoder(3D) chans={DEC_CHANS}, Z-pad={Z_PAD}")


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


# -- Feature extraction helpers --
@torch.inference_mode()
def extract_sam2_volume_features(
    sam2_model,
    processor,
    sax_volume,
    layer_indices,
    device,
    target_depth=SAX_TARGET_DEPTH,
    grid_size=GRID_SIZE,
):
    """Extract per-layer SAM 2 Hiera features for a SAX volume, stacked along z.

    Uses hidden states from within the main (third) Hiera stage so that all
    chosen layers share the same spatial resolution.  Features are interpolated
    to *grid_size* x *grid_size* to match the DINOv3UNetRDecoder input format.
    """
    vol = sax_volume  # (1, H, W, z)
    n_slices = int(vol.shape[-1])
    if n_slices > target_depth:
        vol = vol[..., :target_depth]
        n_slices = target_depth
    elif vol.shape[-1] < target_depth:
        vol = F.pad(vol, (0, target_depth - vol.shape[-1]), mode="constant", value=0.0)

    per_layer = {idx: [] for idx in layer_indices}

    for z in range(target_depth):
        image_2d = vol[0, :, :, z]  # (H, W)
        img_np = (image_2d.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        pil = Image.fromarray(img_np, mode="L").convert("RGB")

        proc = processor(images=pil, return_tensors="pt")
        pixel_values = proc["pixel_values"]
        if device is not None:
            pixel_values = pixel_values.to(device)

        # SAM 2 Hiera encoder; output_hidden_states gives one tensor per block
        enc_out = sam2_model.vision_encoder(pixel_values, output_hidden_states=True)
        hidden_states = enc_out.hidden_states  # tuple of (B, H, W, C) — channels-last

        for idx in layer_indices:
            feat = hidden_states[idx]  # (1, H, W, C)
            feat = feat.permute(0, 3, 1, 2)  # (1, C, H, W)
            feat = F.interpolate(
                feat,
                size=(grid_size, grid_size),
                mode="bilinear",
                align_corners=False,
            )
            per_layer[idx].append(feat.squeeze(0).cpu())  # (C, gs, gs)

    features_dict = {}
    for idx in layer_indices:
        features_dict[f"layer_{idx}"] = torch.stack(per_layer[idx], dim=-1)
        # shape: (embed_dim, grid_size, grid_size, target_depth)

    return features_dict, vol, n_slices


def cache_sam2_volume_features(
    sam2_model,
    processor,
    cinema_dataset,
    cache_dir,
    layer_indices,
    device,
    target_depth=SAX_TARGET_DEPTH,
    grid_size=GRID_SIZE,
):
    """Cache SAM 2 volume features for all patients/frames."""
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for sample_idx in tqdm(
        range(len(cinema_dataset)), desc="Caching SAM 2 volume features"
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

        features_dict, padded_image, actual_slices = extract_sam2_volume_features(
            sam2_model,
            processor,
            image_3d,
            layer_indices,
            device,
            target_depth,
            grid_size,
        )

        # Pad label along z
        label = label_3d
        n = int(label.shape[-1])
        if n > target_depth:
            label = label[..., :target_depth]
        elif n < target_depth:
            label = F.pad(label, (0, target_depth - n), mode="constant", value=0.0)

        save_dict = {
            "image": padded_image,
            "label": label.long(),
            "n_slices": actual_slices,
        }
        save_dict.update(features_dict)
        torch.save(save_dict, fpath)
        manifest.append(
            {"path": fpath, "pid": pid, "is_ed": is_ed, "n_slices": actual_slices}
        )

    return manifest


# -- Load SAM 2 and Cache Volume Features --
if SAM2_LOCAL_DIR is not None:
    sam2_source = str(SAM2_LOCAL_DIR)
    image_processor = Sam2Processor.from_pretrained(sam2_source)
    sam2_model = Sam2Model.from_pretrained(sam2_source).to(DEVICE).eval()
else:
    sam2_source = SAM2_MODEL_ID
    image_processor = Sam2Processor.from_pretrained(
        sam2_source,
        cache_dir=str(HF_CACHE_DIR),
        local_files_only=not AUTO_DOWNLOAD,
    )
    sam2_model = (
        Sam2Model.from_pretrained(
            sam2_source,
            cache_dir=str(HF_CACHE_DIR),
            local_files_only=not AUTO_DOWNLOAD,
        )
        .to(DEVICE)
        .eval()
    )
for p in sam2_model.parameters():
    p.requires_grad = False

print(f"Loaded SAM 2 from: {sam2_source}")
print(f"Frozen parameters: {sum(p.numel() for p in sam2_model.parameters()):,}")

print("Caching training volume features...")
train_manifest = cache_sam2_volume_features(
    sam2_model,
    image_processor,
    train_cinema,
    CACHE_DIR / "train",
    layer_indices=LAYER_INDICES,
    device=DEVICE,
    target_depth=Z_PAD,
)

print("\nCaching validation volume features...")
val_manifest = cache_sam2_volume_features(
    sam2_model,
    image_processor,
    val_cinema,
    CACHE_DIR / "val",
    layer_indices=LAYER_INDICES,
    device=DEVICE,
    target_depth=Z_PAD,
)

print("\nCaching test volume features...")
test_manifest = cache_sam2_volume_features(
    sam2_model,
    image_processor,
    test_cinema,
    CACHE_DIR / "test",
    layer_indices=LAYER_INDICES,
    device=DEVICE,
    target_depth=Z_PAD,
)

sample = torch.load(train_manifest[0]["path"], weights_only=True)
print(f"\nCached train volumes: {len(train_manifest)}")
print(f"Cached val volumes:   {len(val_manifest)}")
print(f"Cached test volumes:  {len(test_manifest)}")
print(f"Image shape:   {sample['image'].shape}")
print(f"Label shape:   {sample['label'].shape}")
for idx in LAYER_INDICES:
    print(f"Layer {idx} shape: {sample[f'layer_{idx}'].shape}")
print(f"n_slices:      {sample['n_slices']}")


# -- Model, Loss & Training --
train_ds = CachedVolumeDataset(train_manifest, layer_indices=LAYER_INDICES)
val_ds = CachedVolumeDataset(val_manifest, layer_indices=LAYER_INDICES)
test_ds = CachedVolumeDataset(test_manifest, layer_indices=LAYER_INDICES)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

probe = DINOv3UNetRDecoder(
    embed_dim=EMBED_DIM,
    layer_indices=LAYER_INDICES,
    dec_chans=DEC_CHANS,
    dec_patch_size=DEC_PATCH_SIZE,
    dec_scale_factor=DEC_SCALE_FACTOR,
    num_classes=NUM_CLASSES,
).to(DEVICE)

class_counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
for entry in train_manifest:
    data = torch.load(entry["path"], weights_only=True)
    ns = data["n_slices"]
    label_valid = data["label"][0, :, :, :ns]  # only valid slices
    class_counts += torch.bincount(label_valid.reshape(-1), minlength=NUM_CLASSES)

class_weights = class_counts.sum().float() / (
    NUM_CLASSES * class_counts.clamp_min(1).float()
)
class_weights[0] = class_weights[0] * 0.5  # reduce BG weight
class_weights = class_weights / class_weights.mean()

criterion = MaskedVolumeLoss(class_weights.to(DEVICE), ce_weight=1.0, dice_weight=1.0)
optimizer = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

print(
    f"Decoder config: chans={DEC_CHANS}, patch_size={DEC_PATCH_SIZE}, scale_factor={DEC_SCALE_FACTOR}"
)
print(f"Class weights (BG/RV/MYO/LV): {class_weights.tolist()}")
print(
    f"Trainable params: {sum(p.numel() for p in probe.parameters() if p.requires_grad):,}"
)

best_val_dice = 0.0
best_epoch = 0
epochs_no_improve = 0
history = {"train_loss": [], "val_macro_dice": [], "lr": []}

for epoch in range(1, N_EPOCHS + 1):
    train_loss = train_one_epoch_vol(
        probe,
        train_loader,
        criterion,
        optimizer,
        DEVICE,
        layer_indices=LAYER_INDICES,
    )
    scheduler.step()

    val_metrics = evaluate_vol(probe, val_loader, DEVICE, layer_indices=LAYER_INDICES)
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
                f"Early stopping at epoch {epoch}. "
                f"Best val Dice={best_val_dice:.4f} at epoch {best_epoch}."
            )
            break

probe.load_state_dict(best_state)
print(
    f"Restored best checkpoint from epoch {best_epoch} (val Dice={best_val_dice:.4f})"
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
ax1.plot(history["train_loss"], label="Train Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss (masked CE + Dice)")
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
plt.savefig(f"{MODEL_SHORT_NAME}_unetr3d_training_curves.png", dpi=150)
plt.close()


# -- Test Evaluation --
test_metrics = evaluate_vol(probe, test_loader, DEVICE, layer_indices=LAYER_INDICES)

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
    for row, vol_idx in enumerate(show_indices):
        entry = test_manifest[vol_idx]
        data = torch.load(entry["path"], weights_only=True)
        ns = data["n_slices"]
        mid_z = ns // 2  # pick middle slice

        batch_gpu = {"image": data["image"].unsqueeze(0).to(DEVICE)}
        for idx in LAYER_INDICES:
            batch_gpu[f"layer_{idx}"] = data[f"layer_{idx}"].unsqueeze(0).to(DEVICE)

        logits = probe(batch_gpu)
        pred_vol = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W, Z)
        pred = pred_vol[:, :, mid_z]
        label = data["label"][0, :, :, mid_z].numpy()

        gt_overlay = overlay_labels(label, IMAGE_SIZE, IMAGE_SIZE)
        pred_overlay = overlay_labels(pred, IMAGE_SIZE, IMAGE_SIZE)

        axes[row, 0].imshow(label, cmap="tab10", vmin=0, vmax=3)
        axes[row, 0].set_title(f"GT ({entry['pid']}, z={mid_z}/{ns})", fontsize=9)
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
plt.savefig(f"{MODEL_SHORT_NAME}_unetr3d_test_predictions.png", dpi=150)
plt.close()


# -- Save Model --
save_path = Path(f"dense_unetr3d_probe_{MODEL_SHORT_NAME}.pt")
torch.save(
    {
        "model_state_dict": probe.state_dict(),
        "model_name": MODEL_SHORT_NAME,
        "sam2_model_id": SAM2_MODEL_ID,
        "embed_dim": EMBED_DIM,
        "layer_indices": LAYER_INDICES,
        "dec_chans": DEC_CHANS,
        "dec_patch_size": DEC_PATCH_SIZE,
        "dec_scale_factor": DEC_SCALE_FACTOR,
        "num_classes": NUM_CLASSES,
        "z_pad": Z_PAD,
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "test_metrics": test_metrics,
        "val_pids": val_pids,
    },
    save_path,
)
print(f"Model saved to {save_path}")
