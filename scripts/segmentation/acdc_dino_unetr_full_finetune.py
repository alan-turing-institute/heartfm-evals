"""
Dense UNetR Fine-Tuning: End-to-end DINOv3 + UNetR cardiac segmentation on ACDC.

This script keeps the same 3D UNetR decoder used by
`acdc_dino_unetr_segmentation.py`, but removes the cached frozen-feature path.
Instead, it extracts DINOv3 intermediate features on the fly so gradients can
flow through the full backbone during training.

Uses local DINOv3 weights/config from this repository. See `LICENSE-DINOv3.md`
for the DINOv3-specific license terms that apply to the backbone assets.
"""

from __future__ import annotations

import argparse
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

from heartfm_evals.dense_linear_probe import (
    CLASS_COLORS,
    CLASS_NAMES,
    IMAGE_SIZE,
    MODEL_CONFIGS,
    NUM_CLASSES,
    dice_score,
    macro_dice,
    overlay_labels,
)
from heartfm_evals.dense_unetr_probe import (
    SAX_TARGET_DEPTH,
    DINOv3UNetRDecoder,
    MaskedVolumeLoss,
    train_one_epoch_vol,
)

LAYER_INDICES = (3, 6, 9, 11)
DEC_CHANS = (32, 64, 128, 256, 512)
DEC_PATCH_SIZE = (2, 2, 1)
DEC_SCALE_FACTOR = (2, 2, 1)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def _pad_volume_z(vol: torch.Tensor, target_depth: int) -> tuple[torch.Tensor, int]:
    """Pad or truncate the final z dimension to `target_depth`."""
    n_slices = int(vol.shape[-1])
    if n_slices > target_depth:
        vol = vol[..., :target_depth]
        n_slices = target_depth
    elif n_slices < target_depth:
        vol = F.pad(vol, (0, target_depth - n_slices), mode="constant", value=0.0)
    return vol, n_slices


class RawVolumeDataset(Dataset):
    """Dataset returning padded SAX volumes without cached DINO features."""

    def __init__(self, cinema_dataset, target_depth: int = SAX_TARGET_DEPTH):
        self.cinema_dataset = cinema_dataset
        self.target_depth = target_depth

    def __len__(self) -> int:
        return len(self.cinema_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:
        sample = self.cinema_dataset[idx]
        image, n_slices = _pad_volume_z(sample["sax_image"], self.target_depth)
        label, _ = _pad_volume_z(sample["sax_label"], self.target_depth)
        return {
            "image": image.float(),
            "label": label.long(),
            "n_slices": n_slices,
            "pid": sample["pid"],
        }


class FinetunableDINOv3UNetR(nn.Module):
    """Joint model: live DINOv3 features + existing 3D UNetR decoder."""

    def __init__(
        self,
        backbone: nn.Module,
        decoder: DINOv3UNetRDecoder,
        layer_indices: tuple[int, ...] = LAYER_INDICES,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.layer_indices = layer_indices

    def _extract_volume_features(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract per-layer 3D feature volumes for a batch of padded volumes."""
        batch_size, _, _, _, depth = image.shape
        per_layer: dict[int, list[torch.Tensor]] = {idx: [] for idx in self.layer_indices}

        # ImageNet normalization constants (reshaped for broadcasting over (B, 3, H, W)).
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)

        for z in range(depth):
            slice_batch = image[:, 0, :, :, z]  # (B, H, W)

            # Preprocess each slice volume-wise: repeat to 3 channels + normalize in a single tensor op
            dino_batch = slice_batch.unsqueeze(1).repeat(1, 3, 1, 1)  # (B, 3, H, W)
            dino_batch = (dino_batch - imagenet_mean) / imagenet_std

            # Extract features with inference mode disabled to allow gradient flow
            with torch.inference_mode(False):
                feats = self.backbone.get_intermediate_layers(
                    dino_batch,
                    n=list(self.layer_indices),
                    reshape=True,
                    norm=True,
                )

            for feat_idx, layer_idx in enumerate(self.layer_indices):
                # Some DINOv3 hub paths can still surface inference tensors here.
                # Clone at the model boundary so the UNetR decoder sees normal
                # autograd-safe tensors during training.
                per_layer[layer_idx].append(feats[feat_idx].clone())

        feature_batch: dict[str, torch.Tensor] = {"image": image}
        for layer_idx in self.layer_indices:
            feature_batch[f"layer_{layer_idx}"] = torch.stack(per_layer[layer_idx], dim=-1)
        return feature_batch

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        feature_batch = self._extract_volume_features(batch["image"])
        return self.decoder(feature_batch)


@torch.no_grad()
def compute_class_weights(dataset: Dataset) -> torch.Tensor:
    """Estimate inverse-frequency class weights from valid slices only."""
    class_counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        ns = int(sample["n_slices"])
        labels = sample["label"][0, :, :, :ns]
        class_counts += torch.bincount(labels.reshape(-1), minlength=NUM_CLASSES)

    class_weights = class_counts.sum().float() / (
        NUM_CLASSES * class_counts.clamp_min(1).float()
    )
    class_weights[0] = class_weights[0] * 0.5
    return class_weights / class_weights.mean()


@torch.no_grad()
def evaluate_live_vol(
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> dict:
    """Evaluate a live-feature model without creating inference tensors."""
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in dataloader:
        batch_gpu = {
            k: v.to(device)
            for k, v in batch.items()
            if k not in {"label", "n_slices", "pid"} and isinstance(v, torch.Tensor)
        }

        labels = batch["label"]
        n_slices = batch["n_slices"]

        logits = model(batch_gpu)
        preds = logits.argmax(dim=1).cpu()

        for i in range(preds.shape[0]):
            ns = int(n_slices[i])
            for z in range(ns):
                all_preds.append(preds[i, :, :, z].numpy())
                all_labels.append(labels[i, 0, :, :, z].numpy())

    all_preds_arr = np.stack(all_preds)
    all_labels_arr = np.stack(all_labels)

    results: dict = {"per_class_dice": {}}
    for c in range(NUM_CLASSES):
        name = CLASS_NAMES.get(c, f"C{c}")
        results["per_class_dice"][name] = float(
            dice_score(all_preds_arr, all_labels_arr, c)
        )

    results["macro_dice"] = float(macro_dice(all_preds_arr, all_labels_arr))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end DINOv3 + UNetR segmentation fine-tuning on ACDC"
    )
    parser.add_argument(
        "--model",
        default="dinov3_vits16",
        choices=list(MODEL_CONFIGS.keys()),
        help="DINOv3 model variant to use",
    )
    parser.add_argument(
        "--acdc-data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "heartfm" / "processed" / "acdc",
        help="Path containing ACDC train/test folders and metadata CSVs (default: REPO_ROOT/data/acdc)",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=REPO_ROOT / "models" / "dinov3",
        help="Local DINOv3 torch.hub repo path",
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=None,
        help="Optional override for DINOv3 checkpoint path",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Volume batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--decoder-lr", type=float, default=1e-4, help="Learning rate for decoder"
    )
    parser.add_argument(
        "--backbone-lr", type=float, default=1e-5, help="Learning rate for DINO backbone"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="AdamW weight decay"
    )
    parser.add_argument(
        "--z-pad",
        type=int,
        default=SAX_TARGET_DEPTH,
        help="Pad/truncate each volume to this many slices",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Only train the decoder while still extracting live features",
    )
    return parser


def build_datasets(acdc_data_dir: Path):
    train_meta_df = pd.read_csv(acdc_data_dir / "train_metadata.csv")
    test_meta_df = pd.read_csv(acdc_data_dir / "test_metadata.csv")

    if "pathology" in train_meta_df.columns:
        val_pids = (
            train_meta_df.groupby("pathology").sample(n=2, random_state=0)["pid"].tolist()
        )
    else:
        val_pids = train_meta_df.sample(frac=0.1, random_state=0)["pid"].tolist()

    train_split_df = train_meta_df[~train_meta_df["pid"].isin(val_pids)].reset_index(
        drop=True
    )
    val_split_df = train_meta_df[train_meta_df["pid"].isin(val_pids)].reset_index(
        drop=True
    )

    transform = ScaleIntensityd(keys="sax_image", factor=1 / 255, channel_wise=False)

    train_cinema = EndDiastoleEndSystoleDataset(
        data_dir=acdc_data_dir / "train",
        meta_df=train_split_df,
        views="sax",
        transform=transform,
    )
    val_cinema = EndDiastoleEndSystoleDataset(
        data_dir=acdc_data_dir / "train",
        meta_df=val_split_df,
        views="sax",
        transform=transform,
    )
    test_cinema = EndDiastoleEndSystoleDataset(
        data_dir=acdc_data_dir / "test",
        meta_df=test_meta_df,
        views="sax",
        transform=transform,
    )

    return train_cinema, val_cinema, test_cinema, val_pids, train_split_df, val_split_df, test_meta_df


def choose_device() -> torch.device:
    #if torch.backends.mps.is_available():
    #    return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = build_parser().parse_args()

    model_name = args.model
    weights_path = args.weights_path or (REPO_ROOT / "model_weights" / f"{model_name}.pth")
    embed_dim = MODEL_CONFIGS[model_name]["embed_dim"]
    n_layers = MODEL_CONFIGS[model_name]["n_layers"]
    device = choose_device()

    print(f"Using device: {device}")
    print(f"Backbone: {model_name} (embed_dim={embed_dim}, layers={n_layers})")
    print(f"Selected layers: {LAYER_INDICES}")
    print(f"Decoder: UpsampleDecoder(3D) chans={DEC_CHANS}, Z-pad={args.z_pad}")
    print(f"Fine-tune mode: {'decoder-only' if args.freeze_backbone else 'full backbone'}")

    (
        train_cinema,
        val_cinema,
        test_cinema,
        val_pids,
        train_split_df,
        val_split_df,
        test_meta_df,
    ) = build_datasets(args.acdc_data_dir)

    print(f"Train split: {len(train_split_df)} patients")
    print(f"Val split:   {len(val_split_df)} patients")
    print(f"Test set:    {len(test_meta_df)} patients")
    print(f"Train CineMA dataset: {len(train_cinema)} samples")
    print(f"Val CineMA dataset:   {len(val_cinema)} samples")
    print(f"Test CineMA dataset:  {len(test_cinema)} samples")

    train_ds = RawVolumeDataset(train_cinema, target_depth=args.z_pad)
    val_ds = RawVolumeDataset(val_cinema, target_depth=args.z_pad)
    test_ds = RawVolumeDataset(test_cinema, target_depth=args.z_pad)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    repo_dir = args.repo_dir.resolve()
    weights_path = weights_path.resolve()

    if not (repo_dir / "hubconf.py").exists():
        msg = (
            f"DINOv3 repo not found at {repo_dir}. Pass --repo-dir with the directory "
            "containing hubconf.py."
        )
        raise FileNotFoundError(msg)

    if not weights_path.exists():
        msg = (
            f"DINOv3 weights not found at {weights_path}. Pass --weights-path with the "
            "checkpoint to load."
        )
        raise FileNotFoundError(msg)

    backbone = torch.hub.load(
        str(repo_dir), model_name, source="local", weights=str(weights_path)
    ).to(device)
    if args.freeze_backbone:
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False
    else:
        backbone.train()

    decoder = DINOv3UNetRDecoder(
        embed_dim=embed_dim,
        layer_indices=LAYER_INDICES,
        dec_chans=DEC_CHANS,
        dec_patch_size=DEC_PATCH_SIZE,
        dec_scale_factor=DEC_SCALE_FACTOR,
        num_classes=NUM_CLASSES,
    )
    probe = FinetunableDINOv3UNetR(backbone, decoder, layer_indices=LAYER_INDICES).to(
        device
    )

    class_weights = compute_class_weights(train_ds)
    criterion = MaskedVolumeLoss(class_weights.to(device), ce_weight=1.0, dice_weight=1.0)

    if args.freeze_backbone:
        optimizer = torch.optim.AdamW(
            probe.decoder.parameters(),
            lr=args.decoder_lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": list(probe.backbone.parameters()),
                    "lr": args.backbone_lr,
                },
                {
                    "params": list(probe.decoder.parameters()),
                    "lr": args.decoder_lr,
                },
            ],
            weight_decay=args.weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Class weights (BG/RV/MYO/LV): {class_weights.tolist()}")
    print(
        f"Trainable params: {sum(p.numel() for p in probe.parameters() if p.requires_grad):,}"
    )

    best_val_dice = 0.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_macro_dice": [], "lr": []}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch_vol(
            probe,
            train_loader,
            criterion,
            optimizer,
            device,
            layer_indices=LAYER_INDICES,
        )
        scheduler.step()

        val_metrics = evaluate_live_vol(probe, val_loader, device)
        val_dice = val_metrics["macro_dice"]

        history["train_loss"].append(train_loss)
        history["val_macro_dice"].append(val_dice)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        improved = val_dice > best_val_dice
        if epoch == 1 or epoch % 5 == 0 or improved:
            tag = " *" if improved else ""
            print(
                f"Epoch {epoch:3d}/{args.epochs} | loss={train_loss:.4f} | "
                f"val Dice={val_dice:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}{tag}"
            )

        if improved:
            best_val_dice = val_dice
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in probe.state_dict().items()
            }
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(
                    f"Early stopping at epoch {epoch}. Best val Dice={best_val_dice:.4f} "
                    f"at epoch {best_epoch}."
                )
                break

    if best_state is None:
        msg = "Training finished without producing a checkpoint."
        raise RuntimeError(msg)
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
        best_val_dice,
        ls="--",
        color="gray",
        alpha=0.5,
        label=f"Best={best_val_dice:.4f}",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro Dice (excl. BG)")
    ax2.set_title("Validation Performance")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{model_name}_unetr3d_full_finetune_training_curves.png", dpi=150)
    plt.close()

    test_metrics = evaluate_live_vol(probe, test_loader, device)
    print("Per-class Dice scores (test set):")
    for name, dice_value in test_metrics["per_class_dice"].items():
        print(f"  {name:>3s}: {dice_value:.4f}")
    print(f"\nMacro Dice (excl. BG): {test_metrics['macro_dice']:.4f}")

    n_show = min(6, len(test_ds))
    show_indices = np.linspace(0, len(test_ds) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show), dpi=150)
    if n_show == 1:
        axes = axes[np.newaxis, :]

    probe.eval()
    with torch.no_grad():
        for row, vol_idx in enumerate(show_indices):
            data = test_ds[int(vol_idx)]
            ns = int(data["n_slices"])
            mid_z = ns // 2

            logits = probe({"image": data["image"].unsqueeze(0).to(device)})
            pred_vol = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            pred = pred_vol[:, :, mid_z]
            label = data["label"][0, :, :, mid_z].numpy()

            gt_overlay = overlay_labels(label, IMAGE_SIZE, IMAGE_SIZE)
            pred_overlay = overlay_labels(pred, IMAGE_SIZE, IMAGE_SIZE)

            axes[row, 0].imshow(label, cmap="tab10", vmin=0, vmax=3)
            axes[row, 0].set_title(f"GT ({data['pid']}, z={mid_z}/{ns})", fontsize=9)
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
    plt.savefig(f"{model_name}_unetr3d_full_finetune_test_predictions.png", dpi=150)
    plt.close()

    save_path = Path(f"dense_unetr3d_full_finetune_{model_name}.pt")
    torch.save(
        {
            "model_state_dict": probe.state_dict(),
            "model_name": model_name,
            "embed_dim": embed_dim,
            "layer_indices": LAYER_INDICES,
            "dec_chans": DEC_CHANS,
            "dec_patch_size": DEC_PATCH_SIZE,
            "dec_scale_factor": DEC_SCALE_FACTOR,
            "num_classes": NUM_CLASSES,
            "z_pad": args.z_pad,
            "freeze_backbone": args.freeze_backbone,
            "best_epoch": best_epoch,
            "best_val_dice": best_val_dice,
            "test_metrics": test_metrics,
            "val_pids": val_pids,
            "backbone_lr": args.backbone_lr,
            "decoder_lr": args.decoder_lr,
        },
        save_path,
    )
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
