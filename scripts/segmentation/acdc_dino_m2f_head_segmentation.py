"""
Mask2Former Head Training: Cardiac Segmentation with DINOv3 on ACDC

Architecture: Frozen DINOv3 backbone -> trainable DINOv3_Adapter + Mask2Former
decoder head, trained end-to-end on ACDC for 4-class cardiac segmentation
(BG, RV, MYO, LV).

Supports multiple backbone sizes (ViT-S/B/L/7B). For 7B, the adapter + M2F head
can be initialized from the pre-trained ADE20K checkpoint. For smaller backbones,
the M2F head trains from random initialization.

Requires CUDA (MSDeformAttn is a CUDA-only custom op).
"""

import argparse
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from heartfm_evals.dense_linear_probe import (
    CLASS_COLORS,
    CLASS_NAMES,
    IMAGE_SIZE,
    NUM_CLASSES,
    ACDCSliceDataset,
    macro_dice,
    overlay_labels,
)

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Train Mask2Former segmentation head with DINOv3 backbone on ACDC"
)
parser.add_argument(
    "--model",
    default="dinov3_vits16",
    choices=["dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16", "dinov3_vit7b16"],
    help="DINOv3 backbone variant",
)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=15)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument(
    "--hidden-dim",
    type=int,
    default=None,
    help="M2F hidden dim override (default: auto from model config)",
)
args = parser.parse_args()

# ── Paths ──────────────────────────────────────────────────────────────────────
ACDC_DATA_DIR = Path("/home/rwood/heartfm/data-evals/acdc/")
REPO_DIR = "../../models/dinov3/"
sys.path.append(REPO_DIR)

# ── Model Configs ──────────────────────────────────────────────────────────────
M2F_MODEL_CONFIGS = {
    "dinov3_vits16": {
        "embed_dim": 384,
        "hidden_dim": 256,
        "weights": "../../model_weights/dinov3_vits16.pth",
        "has_m2f_pretrained": False,
    },
    "dinov3_vitb16": {
        "embed_dim": 768,
        "hidden_dim": 256,
        "weights": "../../model_weights/dinov3_vitb16.pth",
        "has_m2f_pretrained": False,
    },
    "dinov3_vitl16": {
        "embed_dim": 1024,
        "hidden_dim": 256,
        "weights": "../../model_weights/dinov3_vitl16.pth",
        "has_m2f_pretrained": False,
    },
    "dinov3_vit7b16": {
        "embed_dim": 4096,
        "hidden_dim": 2048,
        "weights": "../../model_weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
        "has_m2f_pretrained": True,
        "m2f_weights": (
            "../../model_weights/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth"
        ),
    },
}

# ── Constants ──────────────────────────────────────────────────────────────────
M2F_INPUT_SIZE = 512  # M2F expects this spatial resolution
M2F_NUM_CLASSES_ACDC = NUM_CLASSES  # M2F adds its own void/no-object class internally

MODEL_NAME = args.model
config = M2F_MODEL_CONFIGS[MODEL_NAME]
HIDDEN_DIM = args.hidden_dim if args.hidden_dim is not None else config["hidden_dim"]
BATCH_SIZE = args.batch_size
LR = args.lr
N_EPOCHS = args.epochs
PATIENCE = args.patience
WEIGHT_DECAY = args.weight_decay

# ── Device ─────────────────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required. The M2F pipeline uses MSDeformAttn (a CUDA-only custom op)."
    )
DEVICE = torch.device("cuda")
print(f"Using device: {DEVICE}")


# ── SyncBatchNorm → BatchNorm2d conversion ─────────────────────────────────────
def convert_sync_bn_to_bn(module: nn.Module) -> nn.Module:
    """Replace all SyncBatchNorm layers with BatchNorm2d (for single-GPU training)."""
    for name, child in module.named_children():
        if isinstance(child, nn.SyncBatchNorm):
            bn = nn.BatchNorm2d(
                child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats,
            )
            if child.affine:
                bn.weight = child.weight
                bn.bias = child.bias
            bn.running_mean = child.running_mean
            bn.running_var = child.running_var
            bn.num_batches_tracked = child.num_batches_tracked
            setattr(module, name, bn)
        else:
            convert_sync_bn_to_bn(child)
    return module


# ── Model Building ─────────────────────────────────────────────────────────────
def build_m2f_segmentor(
    model_name: str,
    hidden_dim: int,
    device: torch.device,
) -> nn.Module:
    """Build DINOv3 + Mask2Former segmentor.

    Backbone is frozen inside DINOv3_Adapter. Adapter + M2F head are trainable.
    For 7B, optionally loads pre-trained ADE20K M2F weights (strict=False to skip
    class_embed with mismatched num_classes).
    """
    from dinov3.eval.segmentation.models import build_segmentation_decoder

    cfg = M2F_MODEL_CONFIGS[model_name]

    # Load backbone
    backbone = torch.hub.load(
        REPO_DIR, model_name, source="local", weights=cfg["weights"]
    )

    # Build full segmentor: FeatureDecoder(ModuleList([DINOv3_Adapter, M2FHead]))
    segmentor = build_segmentation_decoder(
        backbone_model=backbone,
        decoder_type="m2f",
        hidden_dim=hidden_dim,
        num_classes=M2F_NUM_CLASSES_ACDC,
        autocast_dtype=torch.bfloat16,
    )

    # For 7B: load pre-trained adapter + M2F head from ADE20K
    if cfg["has_m2f_pretrained"]:
        print("Loading pre-trained M2F head weights (ADE20K)...")
        m2f_state = torch.load(
            cfg["m2f_weights"], map_location="cpu", weights_only=True
        )
        missing, unexpected = segmentor.load_state_dict(m2f_state, strict=False)
        print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        if missing:
            # Expected: class_embed has different shape (151 vs 5)
            print(
                f"  (expected: class_embed shape mismatch for {M2F_NUM_CLASSES_ACDC} classes)"
            )

    # Convert SyncBatchNorm → BatchNorm2d for single-GPU
    convert_sync_bn_to_bn(segmentor)

    segmentor = segmentor.to(device)

    # Count parameters
    n_total = sum(p.numel() for p in segmentor.parameters())
    n_trainable = sum(p.numel() for p in segmentor.parameters() if p.requires_grad)
    print(f"Total params: {n_total:,}")
    print(f"Trainable params (adapter + M2F head): {n_trainable:,}")
    print(f"Frozen params (backbone): {n_total - n_trainable:,}")

    return segmentor


# ── M2F Postprocessing (differentiable) ───────────────────────────────────────
def m2f_postprocess(
    pred_logits: torch.Tensor,
    pred_masks: torch.Tensor,
    target_size: tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE),
) -> torch.Tensor:
    """Convert M2F query predictions to per-pixel class probabilities.

    Args:
        pred_logits: (B, Q, num_classes+1) raw class logits per query.
        pred_masks:  (B, Q, H, W) raw mask logits per query.
        target_size: Final spatial resolution for output.

    Returns:
        (B, num_classes, H, W) per-pixel class probabilities.
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
    return pixel_pred  # (B, NUM_CLASSES, H, W) — probability space


# ── Probability-Aware Loss ────────────────────────────────────────────────────
class M2FProbabilityLoss(nn.Module):
    """Combined CE + Dice loss that operates on probability-space M2F output.

    m2f_postprocess produces values in probability space (softmax * sigmoid * einsum),
    so standard CrossEntropyLoss (which applies log_softmax) would double-normalize.
    Instead:
      - CE: normalize to valid distribution, then NLL on log-probs
      - Dice: compute directly on soft probabilities (no internal softmax)
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.register_buffer(
            "class_weights", class_weights if class_weights is not None else None
        )
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.eps = eps

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs: (B, C, H, W) per-pixel class probabilities from m2f_postprocess.
            targets: (B, H, W) integer class labels.
        """
        # -- CE on probabilities --
        # Normalize to a valid distribution (columns sum to 1)
        prob_sum = probs.sum(dim=1, keepdim=True).clamp(min=self.eps)
        normalized = probs / prob_sum  # (B, C, H, W)
        log_probs = torch.log(normalized + self.eps)  # (B, C, H, W)
        ce_loss = F.nll_loss(log_probs, targets.long(), weight=self.class_weights)

        # -- Dice on probabilities (no softmax) --
        targets_oh = F.one_hot(targets.long(), num_classes=probs.shape[1])  # (B,H,W,C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Exclude background (class 0)
        probs_fg = probs[:, 1:]
        targets_fg = targets_oh[:, 1:]

        dims = (0, 2, 3)  # reduce over batch and spatial
        intersection = (probs_fg * targets_fg).sum(dim=dims)
        cardinality = probs_fg.sum(dim=dims) + targets_fg.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


# ── Training ──────────────────────────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> float:
    """Train M2F segmentor for one epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        images = batch["image"].to(device)  # (B, 3, 192, 192)
        labels = batch["label"].to(device)  # (B, 192, 192)

        # Resize image to M2F input size
        images = F.interpolate(
            images,
            size=(M2F_INPUT_SIZE, M2F_INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(images)
            pixel_pred = m2f_postprocess(
                output["pred_logits"],
                output["pred_masks"],
                target_size=(IMAGE_SIZE, IMAGE_SIZE),
            )
            # pixel_pred is in probability space — use M2FProbabilityLoss
            loss = criterion(pixel_pred.float(), labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate M2F segmentor. Returns per-class and macro Dice."""
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in tqdm(dataloader, desc="Eval", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"]

        images = F.interpolate(
            images,
            size=(M2F_INPUT_SIZE, M2F_INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(images)
            pixel_pred = m2f_postprocess(
                output["pred_logits"],
                output["pred_masks"],
                target_size=(IMAGE_SIZE, IMAGE_SIZE),
            )

        preds = pixel_pred.float().argmax(dim=1).cpu().numpy()  # (B, H, W)
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    all_preds_arr = np.concatenate(all_preds)
    all_labels_arr = np.concatenate(all_labels)

    per_class = {
        CLASS_NAMES[c]: float(
            2
            * ((all_preds_arr == c) & (all_labels_arr == c)).sum()
            / ((all_preds_arr == c).sum() + (all_labels_arr == c).sum() + 1e-8)
        )
        for c in range(NUM_CLASSES)
    }
    m_dice = macro_dice(all_preds_arr, all_labels_arr)

    return {"per_class_dice": per_class, "macro_dice": m_dice}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

print(f"Model: {MODEL_NAME} | hidden_dim: {HIDDEN_DIM}")
print(f"Hyperparams: lr={LR}, batch_size={BATCH_SIZE}, epochs={N_EPOCHS}")

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

train_ds = ACDCSliceDataset(train_cinema, augment=True)
val_ds = ACDCSliceDataset(val_cinema, augment=False)
test_ds = ACDCSliceDataset(test_cinema, augment=False)

print(f"Train slices: {len(train_ds)}")
print(f"Val slices:   {len(val_ds)}")
print(f"Test slices:  {len(test_ds)}")

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# -- Build Model --
print(f"\n=== Building {MODEL_NAME} + Mask2Former segmentor ===")
segmentor = build_m2f_segmentor(MODEL_NAME, HIDDEN_DIM, DEVICE)

# -- Loss --
# Compute class weights from training labels
class_counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
for i in tqdm(range(len(train_ds)), desc="Computing class weights"):
    y = train_ds[i]["label"]
    class_counts += torch.bincount(y.reshape(-1), minlength=NUM_CLASSES)

class_weights = class_counts.sum().float() / (
    NUM_CLASSES * class_counts.clamp_min(1).float()
)
class_weights[0] = class_weights[0] * 0.5  # reduce BG weight
class_weights = class_weights / class_weights.mean()
print(f"Class weights (BG/RV/MYO/LV): {class_weights.tolist()}")

criterion = M2FProbabilityLoss(class_weights=class_weights.to(DEVICE))

# -- Optimizer --
trainable_params = [p for p in segmentor.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
scaler = torch.amp.GradScaler()

# -- Training Loop --
print(f"\n=== Training ({N_EPOCHS} epochs, patience={PATIENCE}) ===")

best_val_dice = 0.0
best_epoch = 0
epochs_no_improve = 0
best_state: dict = {}
history: dict[str, list] = {"train_loss": [], "val_macro_dice": [], "lr": []}

for epoch in range(1, N_EPOCHS + 1):
    train_loss = train_one_epoch(
        segmentor,
        train_loader,
        criterion,
        optimizer,
        scaler,
        DEVICE,
    )
    scheduler.step()

    val_metrics = evaluate(segmentor, val_loader, DEVICE)
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
            k: v.detach().cpu().clone() for k, v in segmentor.state_dict().items()
        }
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best val Dice={best_val_dice:.4f} at epoch {best_epoch}."
            )
            break

segmentor.load_state_dict(best_state)
print(
    f"Restored best checkpoint from epoch {best_epoch} (val Dice={best_val_dice:.4f})"
)

# -- Training Curves --
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
ax1.plot(history["train_loss"], label="Train Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss (CE + Dice)")
ax1.set_title(f"Training Loss ({MODEL_NAME})")
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.plot(history["val_macro_dice"], label="Val Macro Dice", color="tab:orange")
ax2.axhline(
    best_val_dice,
    ls="--",
    color="gray",
    alpha=0.5,
    label=f"Best={best_val_dice:.4f}",
)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Macro Dice (excl. BG)")
ax2.set_title(f"Validation Performance ({MODEL_NAME})")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
curves_path = f"dino_m2f_{MODEL_NAME}_training_curves.png"
plt.savefig(curves_path, dpi=150)
plt.close()
print(f"Training curves saved to {curves_path}")

# -- Test Evaluation --
print(f"\n=== Test Evaluation ({MODEL_NAME}) ===")
test_metrics = evaluate(segmentor, test_loader, DEVICE)

print("Per-class Dice scores (test set):")
for name, d in test_metrics["per_class_dice"].items():
    print(f"  {name:>3s}: {d:.4f}")
print(f"\nMacro Dice (excl. BG): {test_metrics['macro_dice']:.4f}")

# -- Visualization --
n_show = min(6, len(test_ds))
show_indices = np.linspace(0, len(test_ds) - 1, n_show, dtype=int)

fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show), dpi=150)
if n_show == 1:
    axes = axes[np.newaxis, :]

segmentor.eval()
with torch.inference_mode():
    for row, idx in enumerate(show_indices):
        sample = test_ds[idx]
        image = sample["image"].unsqueeze(0).to(DEVICE)  # (1, 3, 192, 192)
        label = sample["label"].numpy()

        image_resized = F.interpolate(
            image,
            size=(M2F_INPUT_SIZE, M2F_INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = segmentor(image_resized)
            pixel_pred = m2f_postprocess(
                output["pred_logits"],
                output["pred_masks"],
                target_size=(IMAGE_SIZE, IMAGE_SIZE),
            )
        pred = pixel_pred.float().argmax(dim=1).squeeze(0).cpu().numpy()

        gt_overlay = overlay_labels(label, IMAGE_SIZE, IMAGE_SIZE)
        pred_overlay = overlay_labels(pred, IMAGE_SIZE, IMAGE_SIZE)

        axes[row, 0].imshow(label, cmap="tab10", vmin=0, vmax=3)
        axes[row, 0].set_title(
            f"GT Labels ({sample['pid']})",
            fontsize=9,
        )
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt_overlay)
        axes[row, 1].set_title("Ground Truth Overlay", fontsize=9)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_overlay)
        axes[row, 2].set_title(
            f"Predicted (Dice={macro_dice(pred, label):.3f})",
            fontsize=9,
        )
        axes[row, 2].axis("off")

legend_patches = [
    mpatches.Patch(color=(*CLASS_COLORS[c][:3], 1.0), label=CLASS_NAMES[c])
    for c in range(1, NUM_CLASSES)
]
axes[-1, 2].legend(handles=legend_patches, loc="lower right", fontsize=8)
plt.tight_layout()
vis_path = f"dino_m2f_{MODEL_NAME}_test_predictions.png"
plt.savefig(vis_path, dpi=150)
plt.close()
print(f"Test predictions saved to {vis_path}")

# -- Save Model --
save_path = Path(f"m2f_head_{MODEL_NAME}.pt")
torch.save(
    {
        "model_state_dict": {k: v.cpu() for k, v in segmentor.state_dict().items()},
        "model_name": MODEL_NAME,
        "hidden_dim": HIDDEN_DIM,
        "num_classes_acdc": NUM_CLASSES,
        "num_classes_m2f": M2F_NUM_CLASSES_ACDC,
        "m2f_input_size": M2F_INPUT_SIZE,
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "test_metrics": test_metrics,
        "val_pids": val_pids,
        "args": vars(args),
    },
    save_path,
)
print(f"Model saved to {save_path}")
