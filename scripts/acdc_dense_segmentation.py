"""
Dense Linear Probe: Pixel-Level Cardiac Segmentation with DINOv3 on ACDC

Architecture: Frozen DINOv3 ViT-S/16 backbone -> multi-layer feature concatenation ->
bilinear upsample -> per-pixel 1x1 Conv2d (4 classes: BG, RV, MYO, LV).
"""
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from cinema.segmentation.dataset import EndDiastoleEndSystoleDataset
from monai.transforms import ScaleIntensityd
from torch.utils.data import DataLoader

from heartfm_evals.dense_linear_probe import (
    CLASS_COLORS,
    CLASS_NAMES,
    IMAGE_SIZE,
    MODEL_CONFIGS,
    NUM_CLASSES,
    CachedFeatureDataset,
    CombinedLoss,
    DenseLinearProbe,
    cache_features,
    dice_score,
    evaluate,
    macro_dice as compute_macro_dice,
    overlay_labels,
    train_one_epoch,
)

# -- Paths --
ACDC_DATA_DIR = Path("/home/rwood/heartfm/data-evals/acdc/")
REPO_DIR = "../models/dinov3/"

# -- Backbone selection --
MODEL_NAME = "dinov3_vits16"
WEIGHTS_PATH = f"../model_weights/{MODEL_NAME}.pth"
EMBED_DIM = MODEL_CONFIGS[MODEL_NAME]["embed_dim"]
N_LAYERS = MODEL_CONFIGS[MODEL_NAME]["n_layers"]
LAYER_INDICES = (3, 6, 9, 11)

# -- Cache --
CACHE_DIR = Path(f"../feature_cache/{MODEL_NAME}")

# -- Training --
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
N_EPOCHS = 100
PATIENCE = 10

# -- Device --
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")
print(f"Backbone: {MODEL_NAME} (embed_dim={EMBED_DIM}, layers={N_LAYERS})")
print(f"Selected layers: {LAYER_INDICES} -> concat dim = {EMBED_DIM * len(LAYER_INDICES)}")


# -- Data --
train_meta_df = pd.read_csv(ACDC_DATA_DIR / "train_metadata.csv")
test_meta_df = pd.read_csv(ACDC_DATA_DIR / "test_metadata.csv")

print(f"Full training set: {len(train_meta_df)} patients")
print(f"Full test set:     {len(test_meta_df)} patients")
if "pathology" in train_meta_df.columns:
    print(f"\nPathology distribution (train):\n{train_meta_df['pathology'].value_counts().to_string()}")

if "pathology" in train_meta_df.columns:
    val_pids = (
        train_meta_df.groupby("pathology")
        .sample(n=2, random_state=0)["pid"]
        .tolist()
    )
else:
    val_pids = train_meta_df.sample(frac=0.1, random_state=0)["pid"].tolist()

train_split_df = train_meta_df[~train_meta_df["pid"].isin(val_pids)].reset_index(drop=True)
val_split_df = train_meta_df[train_meta_df["pid"].isin(val_pids)].reset_index(drop=True)

print(f"Train split: {len(train_split_df)} patients")
print(f"Val split:   {len(val_split_df)} patients")
print(f"Test set:    {len(test_meta_df)} patients")
print(f"\nVal patient IDs: {val_pids}")

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


# -- Load Backbone and Cache Features --
backbone = torch.hub.load(REPO_DIR, MODEL_NAME, source="local", weights=WEIGHTS_PATH)
backbone.eval()
backbone.to(DEVICE)
for p in backbone.parameters():
    p.requires_grad = False
print(f"Loaded {MODEL_NAME} with {sum(p.numel() for p in backbone.parameters()):,} parameters (frozen)")

print("Caching training features...")
train_manifest = cache_features(
    backbone, train_cinema, CACHE_DIR / "train",
    layer_indices=LAYER_INDICES, device=DEVICE,
)

print("\nCaching validation features...")
val_manifest = cache_features(
    backbone, val_cinema, CACHE_DIR / "val",
    layer_indices=LAYER_INDICES, device=DEVICE,
)

print("\nCaching test features...")
test_manifest = cache_features(
    backbone, test_cinema, CACHE_DIR / "test",
    layer_indices=LAYER_INDICES, device=DEVICE,
)

print(f"\nCached: {len(train_manifest)} train, {len(val_manifest)} val, {len(test_manifest)} test slices")

sample = torch.load(train_manifest[0]["path"], weights_only=True)
print(f"Feature shape: {sample['features'].shape}")
print(f"Label shape:   {sample['label'].shape}")

expected_channels = EMBED_DIM * len(LAYER_INDICES)
assert sample["features"].shape[0] == expected_channels, (
    f"Expected {expected_channels} channels, got {sample['features'].shape[0]}"
)
print("Shape check passed!")


# -- DataLoaders --
train_ds = CachedFeatureDataset(train_manifest)
val_ds = CachedFeatureDataset(val_manifest)
test_ds = CachedFeatureDataset(test_manifest)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {len(train_ds)} slices, {len(train_loader)} batches")
print(f"Val:   {len(val_ds)} slices, {len(val_loader)} batches")
print(f"Test:  {len(test_ds)} slices, {len(test_loader)} batches")


# -- Model, Loss, Optimizer --
probe = DenseLinearProbe(
    embed_dim=EMBED_DIM,
    num_classes=NUM_CLASSES,
    layer_indices=LAYER_INDICES,
).to(DEVICE)

criterion = CombinedLoss(ce_weight=1.0, dice_weight=1.0)
optimizer = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

n_params = sum(p.numel() for p in probe.parameters() if p.requires_grad)
print(f"Dense linear probe: {n_params:,} trainable parameters")
print("Loss: CE + Dice")
print(f"Optimizer: AdamW (lr={LR}, wd={WEIGHT_DECAY})")
print(f"Scheduler: CosineAnnealing (T_max={N_EPOCHS})")


# -- Training Loop --
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
    if epoch % 5 == 0 or epoch == 1 or improved:
        tag = " *" if improved else ""
        print(
            f"Epoch {epoch:3d}/{N_EPOCHS} | "
            f"loss={train_loss:.4f} | "
            f"val Dice={val_dice:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}{tag}"
        )

    if improved:
        best_val_dice = val_dice
        best_epoch = epoch
        epochs_no_improve = 0
        best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}. Best val Dice={best_val_dice:.4f} at epoch {best_epoch}.")
            break

probe.load_state_dict(best_state)
print(f"\nRestored best model from epoch {best_epoch} (val Dice={best_val_dice:.4f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=150)

ax1.plot(history["train_loss"], label="Train Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss (CE + Dice)")
ax1.set_title("Training Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history["val_macro_dice"], label="Val Macro Dice", color="tab:orange")
ax2.axhline(best_val_dice, ls="--", color="gray", alpha=0.5, label=f"Best={best_val_dice:.4f}")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Macro Dice (excl. BG)")
ax2.set_title("Validation Performance")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"dino_{MODEL_NAME}_training_curves.png", dpi=150)
plt.close()


# -- Test Set Evaluation --
test_metrics = evaluate(probe, test_loader, DEVICE)

print("Per-class Dice scores (test set):")
for name, d in test_metrics["per_class_dice"].items():
    print(f"  {name:>3s}: {d:.4f}")
print(f"\nMacro Dice (excl. BG): {test_metrics['macro_dice']:.4f}")

# Per-patient Dice breakdown
patient_dices = []

probe.eval()
with torch.inference_mode():
    for entry in test_manifest:
        data = torch.load(entry["path"], weights_only=True)
        feats = data["features"].unsqueeze(0).to(DEVICE)
        label = data["label"].numpy()

        logits = probe(feats)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        patient_dices.append({
            "pid": entry["pid"],
            "z_idx": entry["z_idx"],
            "macro_dice": compute_macro_dice(pred, label),
            **{CLASS_NAMES[c]: dice_score(pred, label, c) for c in range(NUM_CLASSES)},
        })

dice_df = pd.DataFrame(patient_dices)
patient_summary = dice_df.groupby("pid")[["macro_dice", "RV", "MYO", "LV"]].mean()
print("Per-patient mean Macro Dice (test set):")
print(patient_summary.round(4).to_string())
print(f"\nOverall mean ± std: {patient_summary['macro_dice'].mean():.4f} ± {patient_summary['macro_dice'].std():.4f}")


# -- Visualization --
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
        axes[row, 0].set_title(f"GT Labels ({entry['pid']}, z={entry['z_idx']})", fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt_overlay)
        axes[row, 1].set_title("Ground Truth Overlay", fontsize=9)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_overlay)
        axes[row, 2].set_title(f"Predicted (Dice={compute_macro_dice(pred, label):.3f})", fontsize=9)
        axes[row, 2].axis("off")

legend_patches = [
    mpatches.Patch(color=CLASS_COLORS[c][:3] + (1.0,), label=CLASS_NAMES[c])
    for c in range(1, NUM_CLASSES)
]
axes[-1, 2].legend(handles=legend_patches, loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(f"dino_{MODEL_NAME}_test_predictions.png", dpi=150)
plt.close()


# -- Save Model --
save_path = Path(f"dense_probe_{MODEL_NAME}.pt")
torch.save(
    {
        "model_state_dict": probe.state_dict(),
        "model_name": MODEL_NAME,
        "embed_dim": EMBED_DIM,
        "layer_indices": LAYER_INDICES,
        "num_classes": NUM_CLASSES,
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "test_metrics": test_metrics,
        "val_pids": val_pids,
    },
    save_path,
)
print(f"Model saved to {save_path}")
