"""
Dense Linear Probe: Pixel-Level Cardiac Segmentation with SAM 2 on ACDC

Architecture: Frozen SAM 2 image encoder -> cached image embeddings ->
Dropout2d -> BatchNorm2d -> per-pixel 1x1 Conv2d -> bilinear upsample
(4 classes: BG, RV, MYO, LV).

Matches the DINOv3 paper's linear evaluation protocol (Appendix D.1),
applied to SAM 2 image features.
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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
    CachedFeatureDataset,
    CombinedLoss,
    DenseLinearProbe,
    dice_score,
    evaluate,
    macro_dice,
    overlay_labels,
    train_one_epoch,
)
from heartfm_evals.reproducibility import set_seed

_SAM2_VARIANTS = [
    "facebook/sam2.1-hiera-tiny",
    "facebook/sam2.1-hiera-small",
    "facebook/sam2.1-hiera-base-plus",
    "facebook/sam2.1-hiera-large",
]

parser = argparse.ArgumentParser(
    description="Dense linear probe segmentation with SAM 2 on ACDC"
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
HF_CACHE_DIR = Path("../../model_weights/hf")
AUTO_DOWNLOAD = True

# -- Cache --
CACHE_NAME = SAM2_MODEL_ID.split("/")[-1].replace("-", "_").replace(".", "_")
CACHE_DIR = Path(f"../../feature_cache/{CACHE_NAME}")

# -- Training --
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-3
DROPOUT = 0.1
TRAIN_AUG_COPIES = 1
N_EPOCHS = 20
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
print(f"SAM 2 source: hub -> {SAM2_MODEL_ID}")


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


# -- Load SAM 2 and Cache Features --
image_processor = Sam2Processor.from_pretrained(
    SAM2_MODEL_ID,
    cache_dir=str(HF_CACHE_DIR),
    local_files_only=not AUTO_DOWNLOAD,
)
sam2_model = (
    Sam2Model.from_pretrained(
        SAM2_MODEL_ID,
        cache_dir=str(HF_CACHE_DIR),
        local_files_only=not AUTO_DOWNLOAD,
    )
    .to(DEVICE)
    .eval()
)

for p in sam2_model.parameters():
    p.requires_grad = False

print(f"Loaded SAM 2 from: {SAM2_MODEL_ID}")
print(f"Frozen parameters: {sum(p.numel() for p in sam2_model.parameters()):,}")


@torch.inference_mode()
def extract_sam2_features(sam2_model, image_processor, image_2d, device):
    # image_2d: (H, W) in [0, 1]
    img_np = (image_2d.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    pil = Image.fromarray(img_np, mode="L").convert("RGB")

    proc = image_processor(images=pil, return_tensors="pt")
    pixel_values = proc["pixel_values"].to(device)

    feats = sam2_model.get_image_embeddings(pixel_values)
    return feats[-1].squeeze(0).cpu()


def _augment_slice(image_2d, label_2d):
    # Geometry augments (label-safe)
    if torch.rand(1).item() < 0.5:
        image_2d = torch.flip(image_2d, dims=(-1,))
        label_2d = torch.flip(label_2d, dims=(-1,))
    if torch.rand(1).item() < 0.2:
        image_2d = torch.flip(image_2d, dims=(-2,))
        label_2d = torch.flip(label_2d, dims=(-2,))
    k = int(torch.randint(low=0, high=4, size=(1,)).item())
    if k > 0:
        image_2d = torch.rot90(image_2d, k=k, dims=(-2, -1))
        label_2d = torch.rot90(label_2d, k=k, dims=(-2, -1))

    # Intensity augments (image only)
    if torch.rand(1).item() < 0.6:
        gamma = float(torch.empty(1).uniform_(0.7, 1.5).item())
        image_2d = image_2d.clamp(0, 1).pow(gamma)
    if torch.rand(1).item() < 0.6:
        scale = float(torch.empty(1).uniform_(0.9, 1.1).item())
        shift = float(torch.empty(1).uniform_(-0.05, 0.05).item())
        image_2d = (image_2d * scale + shift).clamp(0, 1)

    return image_2d, label_2d


def cache_sam2_features(
    sam2_model,
    image_processor,
    cinema_dataset,
    cache_dir,
    device,
    augment=False,
    n_augments=0,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for sample_idx in tqdm(range(len(cinema_dataset)), desc="Caching SAM 2 features"):
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

            image_2d = image_3d[0, :, :, z]
            label_2d = label_3d[0, :, :, z]

            feats = extract_sam2_features(sam2_model, image_processor, image_2d, device)
            torch.save({"features": feats, "label": label_2d.long()}, fpath)
            manifest.append({"path": fpath, "pid": pid, "is_ed": is_ed, "z_idx": z})

            if augment and n_augments > 0:
                for aug_idx in range(n_augments):
                    aug_name = f"{pid}_{frame}_z{z:02d}_aug{aug_idx:02d}.pt"
                    aug_path = cache_dir / aug_name
                    if aug_path.exists():
                        manifest.append(
                            {"path": aug_path, "pid": pid, "is_ed": is_ed, "z_idx": z}
                        )
                        continue

                    aug_img, aug_lbl = _augment_slice(
                        image_2d.clone(), label_2d.clone()
                    )
                    aug_feats = extract_sam2_features(
                        sam2_model, image_processor, aug_img, device
                    )
                    torch.save(
                        {"features": aug_feats, "label": aug_lbl.long()}, aug_path
                    )
                    manifest.append(
                        {"path": aug_path, "pid": pid, "is_ed": is_ed, "z_idx": z}
                    )

    return manifest


print("Caching training features...")
train_manifest = cache_sam2_features(
    sam2_model,
    image_processor,
    train_cinema,
    CACHE_DIR / "train",
    DEVICE,
    augment=True,
    n_augments=TRAIN_AUG_COPIES,
)

print("\nCaching validation features...")
val_manifest = cache_sam2_features(
    sam2_model, image_processor, val_cinema, CACHE_DIR / "val", DEVICE
)

print("\nCaching test features...")
test_manifest = cache_sam2_features(
    sam2_model, image_processor, test_cinema, CACHE_DIR / "test", DEVICE
)

sample = torch.load(train_manifest[0]["path"], weights_only=True)
print(f"Cached train slices: {len(train_manifest)}")
print(f"Cached val slices:   {len(val_manifest)}")
print(f"Cached test slices:  {len(test_manifest)}")
print(f"Feature shape: {sample['features'].shape}")
print(f"Label shape:   {sample['label'].shape}")


# -- Define Dense Linear Probe and Train --
train_ds = CachedFeatureDataset(train_manifest)
val_ds = CachedFeatureDataset(val_manifest)
test_ds = CachedFeatureDataset(test_manifest)

g = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, generator=g)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

in_channels = sample["features"].shape[0]

# SAM 2 produces a single feature map, so we use a single-layer probe
# (embed_dim=in_channels, layer_indices=(0,)) which maps directly to the
# DINOv3 linear evaluation protocol: Dropout2d -> BN -> 1x1 Conv2d -> upsample.
probe = DenseLinearProbe(
    embed_dim=in_channels,
    num_classes=NUM_CLASSES,
    layer_indices=(0,),
    cached_layers=(0,),
    dropout=DROPOUT,
).to(DEVICE)

criterion = CombinedLoss(ce_weight=1.0, dice_weight=1.0)
optimizer = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

print(f"Dense linear probe input channels: {in_channels}")
n_params = sum(p.numel() for p in probe.parameters() if p.requires_grad)
print(f"Dense linear probe: {n_params:,} trainable parameters (includes BN + Conv)")
print("Loss: CE + Dice")
print(f"Optimizer: AdamW (lr={LR}, wd={WEIGHT_DECAY})")
print(f"Scheduler: CosineAnnealing (T_max={N_EPOCHS})")

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
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{N_EPOCHS} | loss={train_loss:.4f} | "
            f"val Dice={val_dice:.4f} | lr={lr:.2e}{tag}"
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
ax1.set_ylabel("Loss (CE + Dice)")
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
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"sam2_linear_probe_{CACHE_NAME}_training_curves_{timestamp}.png", dpi=150)
plt.close()


# -- Test Evaluation --
test_metrics = evaluate(probe, test_loader, DEVICE)

print("Per-class Dice scores (test set):")
for name, d in test_metrics["per_class_dice"].items():
    print(f"  {name:>3s}: {d:.4f}")
print(f"\nMacro Dice (excl. BG): {test_metrics['macro_dice']:.4f}")

# -- Per-Patient Dice Breakdown --
patient_dices = []

probe.eval()
with torch.inference_mode():
    for entry in test_manifest:
        data = torch.load(entry["path"], weights_only=True)
        feats = data["features"].unsqueeze(0).to(DEVICE)
        label = data["label"].numpy()

        logits = probe(feats)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        patient_dices.append(
            {
                "pid": entry["pid"],
                "z_idx": entry["z_idx"],
                "macro_dice": macro_dice(pred, label),
                **{
                    CLASS_NAMES[c]: dice_score(pred, label, c)
                    for c in range(NUM_CLASSES)
                },
            }
        )

dice_df = pd.DataFrame(patient_dices)
patient_summary = dice_df.groupby("pid")[["macro_dice", "RV", "MYO", "LV"]].mean()
print("Per-patient mean Macro Dice (test set):")
print(patient_summary.round(4).to_string())
print(
    f"\nOverall mean +/- std: "
    f"{patient_summary['macro_dice'].mean():.4f} "
    f"+/- {patient_summary['macro_dice'].std():.4f}"
)

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
    mpatches.Patch(color=(*CLASS_COLORS[c][:3], 1.0), label=CLASS_NAMES[c])
    for c in range(1, NUM_CLASSES)
]
axes[-1, 2].legend(handles=legend_patches, loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(f"sam2_linear_probe_{CACHE_NAME}_test_predictions_{timestamp}.png", dpi=150)
plt.close()


# -- Save Model --
save_path = Path(f"dense_linear_probe_{CACHE_NAME}_{timestamp}.pt")
torch.save(
    {
        "model_state_dict": probe.state_dict(),
        "sam2_model_id": SAM2_MODEL_ID,
        "embed_dim": in_channels,
        "layer_indices": (0,),
        "num_classes": NUM_CLASSES,
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "test_metrics": test_metrics,
        "val_pids": val_pids,
    },
    save_path,
)
print(f"Model saved to {save_path}")
