"""
Mask2Former Class-Embed Probe: Cardiac Segmentation with DINOv3 7B on ACDC

Architecture: Frozen DINOv3 7B backbone + frozen Mask2Former decoder (pre-trained
on ADE20K with 150 classes) -> cache query representations & mask predictions ->
train only `class_embed` Linear(2048, 5) for 4 ACDC classes + 1 void.

Because `class_embed` is independent of `mask_embed` in the M2F transformer
decoder, replacing it does not affect mask predictions or attention masks.
We cache the frozen model outputs once, then train the linear class head cheaply.
"""

import gc
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from cinema.segmentation.dataset import EndDiastoleEndSystoleDataset
from monai.transforms import ScaleIntensityd
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from heartfm_evals.dense_linear_probe import (
    CLASS_COLORS,
    CLASS_NAMES,
    IMAGE_SIZE,
    NUM_CLASSES,
    DiceLoss,
    macro_dice,
    overlay_labels,
)

# -- Paths --
ACDC_DATA_DIR = Path("/home/rwood/heartfm/data-evals/acdc/")
REPO_DIR = "../../models/dinov3/"
sys.path.append(REPO_DIR)

BACKBONE_WEIGHTS = "../../model_weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
HEAD_WEIGHTS = "../../model_weights/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth"

# -- Cache --
CACHE_DIR = Path("../../feature_cache/dinov3_vit7b16_m2f")

# -- M2F constants --
M2F_INPUT_SIZE = 512  # resize ACDC to this for the M2F forward pass
M2F_NUM_QUERIES = 100
M2F_HIDDEN_DIM = 2048
M2F_NUM_CLASSES_ACDC = NUM_CLASSES + 1  # 4 ACDC classes + 1 void (M2F convention)

# -- Training --
BATCH_SIZE = 16
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

print(f"Using device: {DEVICE}")


# ── M2F Image Transform ───────────────────────────────────────────────────────
def make_m2f_transform(resize_size: int = M2F_INPUT_SIZE):
    """Transform for M2F input: grayscale (H,W) [0,1] -> (3, resize, resize) normalized."""
    return v2.Compose(
        [
            v2.Resize((resize_size, resize_size), antialias=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


m2f_transform = make_m2f_transform()


# ── Caching ────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def cache_m2f_features(
    segmentor: nn.Module,
    cinema_dataset,
    cache_dir: Path,
    device: torch.device,
) -> list[dict]:
    """Run the frozen M2F segmentor and cache decoder_output + pred_masks per slice.

    Uses a forward hook on `predictor.post_norm` to capture the query
    representations (decoder_output) that feed into class_embed.

    Saves per slice:
        decoder_output: (num_queries, hidden_dim) = (100, 2048)
        pred_masks:     (num_queries, H/4, W/4) — frozen mask predictions
        label:          (IMAGE_SIZE, IMAGE_SIZE) — ground truth at native resolution

    Returns a manifest (list of dicts) with paths and metadata.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    # Access the M2F transformer decoder's post_norm layer
    # Path: segmentor.segmentation_model[1].predictor.post_norm
    predictor = segmentor.segmentation_model[1].predictor
    captured = {}

    def _hook_fn(_module, _input, output):
        captured["post_norm_output"] = output

    hook = predictor.post_norm.register_forward_hook(_hook_fn)

    segmentor.eval()
    for sample_idx in tqdm(range(len(cinema_dataset)), desc="Caching M2F features"):
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

            image_2d = image_3d[0, :, :, z]  # (H, W) in [0, 1]
            label_2d = label_3d[0, :, :, z]  # (H, W)

            # Prepare input: grayscale -> 3ch -> normalize -> resize -> batch
            img = image_2d.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
            img = m2f_transform(img).unsqueeze(0).to(device)  # (1, 3, 512, 512)

            # Forward pass through frozen segmentor
            output = segmentor(img)
            # output = {"pred_logits": (1, 100, 151), "pred_masks": (1, 100, H/4, W/4), ...}

            # Capture decoder_output from hook (post_norm output)
            # post_norm is called on (Q, B, C) shaped tensor, output is same shape
            # Then transposed to (B, Q, C) in forward_prediction_heads
            # Our hook captures the raw post_norm output: (Q, B, C)
            post_norm_out = captured["post_norm_output"]  # (Q, B, C)
            decoder_output = post_norm_out.transpose(0, 1).squeeze(
                0
            )  # (Q, C) = (100, 2048)

            pred_masks = output["pred_masks"].squeeze(0)  # (100, H/4, W/4)

            torch.save(
                {
                    "decoder_output": decoder_output.cpu(),
                    "pred_masks": pred_masks.cpu(),
                    "label": label_2d.long(),
                },
                fpath,
            )
            manifest.append({"path": fpath, "pid": pid, "is_ed": is_ed, "z_idx": z})

    hook.remove()
    return manifest


# ── Dataset for Cached M2F Features ───────────────────────────────────────────
class CachedM2FDataset(Dataset):
    """Loads cached decoder_output + pred_masks + labels from .pt files."""

    def __init__(self, manifest: list[dict]):
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        entry = self.manifest[idx]
        data = torch.load(entry["path"], weights_only=True)
        return {
            "decoder_output": data["decoder_output"],  # (100, 2048)
            "pred_masks": data["pred_masks"],  # (100, H/4, W/4)
            "label": data["label"],  # (H, W)
            "pid": entry["pid"],
        }


# ── Loss ───────────────────────────────────────────────────────────────────────
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


# ── M2F Postprocessing (differentiable) ───────────────────────────────────────
def m2f_postprocess(
    pred_logits: torch.Tensor,
    pred_masks: torch.Tensor,
    target_size: tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE),
) -> torch.Tensor:
    """Convert M2F query predictions to per-pixel class logits.

    Args:
        pred_logits: (B, Q, num_classes+1) raw class logits per query.
        pred_masks:  (B, Q, H, W) raw mask logits per query.
        target_size: Final spatial resolution for output.

    Returns:
        (B, num_classes, H, W) per-pixel class predictions (pre-softmax scores).
    """
    # Softmax over classes per query, drop the void (last) class
    mask_cls = F.softmax(pred_logits, dim=-1)[..., :-1]  # (B, Q, num_classes)
    mask_pred = pred_masks.sigmoid()  # (B, Q, h, w)

    # Weighted combination: per-pixel class scores
    pixel_pred = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)  # (B, C, h, w)

    # Upsample to target resolution
    pixel_pred = F.interpolate(
        pixel_pred, size=target_size, mode="bilinear", align_corners=False
    )
    return pixel_pred  # (B, NUM_CLASSES, H, W)


# ── Training ──────────────────────────────────────────────────────────────────
def train_one_epoch(
    class_embed: nn.Linear,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train class_embed for one epoch on cached M2F features. Returns mean loss."""
    class_embed.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        decoder_output = batch["decoder_output"].to(device)  # (B, 100, 2048)
        pred_masks = batch["pred_masks"].to(device)  # (B, 100, H/4, W/4)
        labels = batch["label"].to(device)  # (B, H, W)

        # Forward: class_embed -> M2F postprocessing
        pred_logits = class_embed(decoder_output)  # (B, 100, 5)
        pixel_pred = m2f_postprocess(pred_logits, pred_masks)  # (B, 4, 192, 192)

        loss = criterion(pixel_pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.inference_mode()
def evaluate(
    class_embed: nn.Linear,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate class_embed on cached M2F features. Returns per-class and macro Dice."""
    class_embed.eval()
    all_preds = []
    all_labels = []

    for batch in dataloader:
        decoder_output = batch["decoder_output"].to(device)
        pred_masks = batch["pred_masks"].to(device)
        labels = batch["label"]

        pred_logits = class_embed(decoder_output)
        pixel_pred = m2f_postprocess(pred_logits, pred_masks)
        preds = pixel_pred.argmax(dim=1).cpu()  # (B, H, W)

        all_preds.append(preds.numpy())
        all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    per_class = {
        CLASS_NAMES[c]: float(
            2
            * ((all_preds == c) & (all_labels == c)).sum()
            / ((all_preds == c).sum() + (all_labels == c).sum() + 1e-8)
        )
        for c in range(NUM_CLASSES)
    }
    m_dice = macro_dice(all_preds, all_labels)

    return {"per_class_dice": per_class, "macro_dice": m_dice}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

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


# -- Phase 1: Load Segmentor & Cache Features --
print("\n=== Phase 1: Caching M2F features ===")
print("Loading DINOv3 7B + Mask2Former segmentor...")

segmentor = torch.hub.load(
    REPO_DIR,
    "dinov3_vit7b16_ms",
    source="local",
    weights=HEAD_WEIGHTS,
    backbone_weights=BACKBONE_WEIGHTS,
)
segmentor.eval().to(DEVICE)
for p in segmentor.parameters():
    p.requires_grad = False

n_params = sum(p.numel() for p in segmentor.parameters())
print(f"Loaded segmentor with {n_params:,} parameters (all frozen)")

print("\nCaching training features...")
train_manifest = cache_m2f_features(
    segmentor, train_cinema, CACHE_DIR / "train", device=DEVICE
)
print("Caching validation features...")
val_manifest = cache_m2f_features(
    segmentor, val_cinema, CACHE_DIR / "val", device=DEVICE
)
print("Caching test features...")
test_manifest = cache_m2f_features(
    segmentor, test_cinema, CACHE_DIR / "test", device=DEVICE
)

# Verify cached shapes
sample = torch.load(train_manifest[0]["path"], weights_only=True)
print(f"\nCached train slices: {len(train_manifest)}")
print(f"Cached val slices:   {len(val_manifest)}")
print(f"Cached test slices:  {len(test_manifest)}")
print(f"decoder_output shape: {sample['decoder_output'].shape}")
print(f"pred_masks shape:     {sample['pred_masks'].shape}")
print(f"label shape:          {sample['label'].shape}")

# Unload the 7B model to free memory
del segmentor
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("Unloaded segmentor to free memory.")


# -- Phase 2: Train class_embed --
print("\n=== Phase 2: Training class_embed ===")

train_ds = CachedM2FDataset(train_manifest)
val_ds = CachedM2FDataset(val_manifest)
test_ds = CachedM2FDataset(test_manifest)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# New class_embed: Linear(2048, 5) — 4 ACDC classes + 1 void
class_embed = nn.Linear(M2F_HIDDEN_DIM, M2F_NUM_CLASSES_ACDC).to(DEVICE)
nn.init.xavier_uniform_(class_embed.weight)
nn.init.zeros_(class_embed.bias)

n_trainable = sum(p.numel() for p in class_embed.parameters())
print(f"Trainable params (class_embed): {n_trainable:,}")

# Compute class weights from training labels
class_counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
for entry in train_manifest:
    y = torch.load(entry["path"], weights_only=True)["label"]
    class_counts += torch.bincount(y.long().reshape(-1), minlength=NUM_CLASSES)

class_weights = class_counts.sum().float() / (
    NUM_CLASSES * class_counts.clamp_min(1).float()
)
class_weights[0] = class_weights[0] * 0.5  # further reduce BG weight
class_weights = class_weights / class_weights.mean()
print(f"Class weights (BG/RV/MYO/LV): {class_weights.tolist()}")

criterion = WeightedCombinedLoss(
    class_weights.to(DEVICE), ce_weight=1.0, dice_weight=1.0
)
optimizer = torch.optim.AdamW(
    class_embed.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

best_val_dice = 0.0
best_epoch = 0
epochs_no_improve = 0
history: dict[str, list] = {"train_loss": [], "val_macro_dice": [], "lr": []}

for epoch in range(1, N_EPOCHS + 1):
    train_loss = train_one_epoch(
        class_embed, train_loader, criterion, optimizer, DEVICE
    )
    scheduler.step()

    val_metrics = evaluate(class_embed, val_loader, DEVICE)
    val_dice = val_metrics["macro_dice"]

    history["train_loss"].append(train_loss)
    history["val_macro_dice"].append(val_dice)
    history["lr"].append(optimizer.param_groups[0]["lr"])

    improved = val_dice > best_val_dice
    if epoch == 1 or epoch % 5 == 0 or improved:
        tag = " *" if improved else ""
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{N_EPOCHS} | loss={train_loss:.4f} "
            f"| val Dice={val_dice:.4f} | lr={lr:.2e}{tag}"
        )

    if improved:
        best_val_dice = val_dice
        best_epoch = epoch
        epochs_no_improve = 0
        best_state = {
            k: v.detach().cpu().clone() for k, v in class_embed.state_dict().items()
        }
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best val Dice={best_val_dice:.4f} at epoch {best_epoch}."
            )
            break

class_embed.load_state_dict(best_state)
print(
    f"Restored best checkpoint from epoch {best_epoch} (val Dice={best_val_dice:.4f})"
)

# -- Training Curves --
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
plt.savefig("dino_m2f_class_embed_training_curves.png", dpi=150)
plt.close()


# -- Phase 3: Test Evaluation --
print("\n=== Phase 3: Test Evaluation ===")
test_metrics = evaluate(class_embed, test_loader, DEVICE)

print("Per-class Dice scores (test set):")
for name, d in test_metrics["per_class_dice"].items():
    print(f"  {name:>3s}: {d:.4f}")
print(f"\nMacro Dice (excl. BG): {test_metrics['macro_dice']:.4f}")

# -- Visualization --
n_show = min(6, len(test_manifest))
show_indices = np.linspace(0, len(test_manifest) - 1, n_show, dtype=int)

fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show), dpi=150)
if n_show == 1:
    axes = axes[np.newaxis, :]

class_embed.eval()
with torch.inference_mode():
    for row, idx in enumerate(show_indices):
        entry = test_manifest[idx]
        data = torch.load(entry["path"], weights_only=True)
        decoder_output = data["decoder_output"].unsqueeze(0).to(DEVICE)
        pred_masks = data["pred_masks"].unsqueeze(0).to(DEVICE)
        label = data["label"].numpy()

        pred_logits = class_embed(decoder_output)
        pixel_pred = m2f_postprocess(pred_logits, pred_masks)
        pred = pixel_pred.argmax(dim=1).squeeze(0).cpu().numpy()

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
plt.savefig("dino_m2f_class_embed_test_predictions.png", dpi=150)
plt.close()


# -- Save Model --
save_path = Path("m2f_class_embed_probe_dinov3_vit7b16.pt")
torch.save(
    {
        "model_state_dict": class_embed.state_dict(),
        "model_name": "dinov3_vit7b16",
        "hidden_dim": M2F_HIDDEN_DIM,
        "num_queries": M2F_NUM_QUERIES,
        "num_classes_acdc": NUM_CLASSES,
        "num_classes_m2f": M2F_NUM_CLASSES_ACDC,
        "m2f_input_size": M2F_INPUT_SIZE,
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "test_metrics": test_metrics,
        "val_pids": val_pids,
    },
    save_path,
)
print(f"Model saved to {save_path}")
