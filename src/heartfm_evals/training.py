"""Training and evaluation loops for segmentation.

Provides:
- ``train_one_epoch()`` / ``evaluate()``: 2D slice-level training/eval.
- ``train_one_epoch_vol()`` / ``evaluate_vol()``: 3D volume-level training/eval.
- ``evaluate_per_sample()`` / ``evaluate_vol_per_sample()``: test-only
  per-sample Dice reporting.
- ``train_segmentation()``: High-level wrapper with early stopping and LR scheduling.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from heartfm_evals.constants import CLASS_NAMES, NUM_CLASSES
from heartfm_evals.losses import MaskedVolumeLoss
from heartfm_evals.metrics import (
    dice_score,
    macro_dice,
    per_sample_dice_metrics,
)

# ── Helpers ──────────────────────────────────────────────────────────────────
_NON_MODEL_KEYS = {"is_ed", "label", "n_slices", "pid", "z_idx"}


def _batch_to_device(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Move all tensor values (except label/n_slices/pid) to *device*."""
    return {
        k: v.to(device)
        for k, v in batch.items()
        if k not in _NON_MODEL_KEYS and isinstance(v, torch.Tensor)
    }


def _batch_item(batch_value, idx: int):
    """Extract a Python scalar or object from a collated batch value."""
    if isinstance(batch_value, torch.Tensor):
        return batch_value[idx].item()
    return batch_value[idx]


def _frame_name(is_ed: bool) -> str:
    """Return the frame name for a boolean ED/ES indicator."""
    return "ed" if is_ed else "es"


# ── 2D Training ─────────────────────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch on cached 2D features. Returns mean loss."""
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
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> dict:
    """Evaluate on cached 2D features. Returns per-class Dice and macro Dice."""
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

    all_preds_arr = np.concatenate(all_preds)
    all_labels_arr = np.concatenate(all_labels)

    per_class = {
        CLASS_NAMES[c]: dice_score(all_preds_arr, all_labels_arr, c)
        for c in range(NUM_CLASSES)
    }
    m_dice = macro_dice(all_preds_arr, all_labels_arr)

    return {"per_class_dice": per_class, "macro_dice": m_dice}


@torch.inference_mode()
def evaluate_per_sample(
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> list[dict]:
    """Evaluate on cached 2D features and return one metrics row per slice."""
    model.eval()
    rows: list[dict] = []

    for batch in dataloader:
        features = batch["features"].to(device)
        labels = batch["label"]

        logits = model(features)
        preds = logits.argmax(dim=1).cpu()  # (B, H, W)

        for i in range(preds.shape[0]):
            is_ed = bool(_batch_item(batch["is_ed"], i))
            row = {
                "pid": str(_batch_item(batch["pid"], i)),
                "frame": _frame_name(is_ed),
                "is_ed": is_ed,
                "z_idx": int(_batch_item(batch["z_idx"], i)),
            }
            row.update(per_sample_dice_metrics(preds[i].numpy(), labels[i].numpy()))
            rows.append(row)

    return rows


# ── 3D Volume Training ──────────────────────────────────────────────────────
def train_one_epoch_vol(
    model: nn.Module,
    dataloader,
    criterion: MaskedVolumeLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    **_kwargs,
) -> float:
    """Train for one epoch on cached volume features.  Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        batch_gpu = _batch_to_device(batch, device)

        labels = batch["label"].to(device)  # (B, 1, H, W, Z)
        n_slices = batch["n_slices"].to(device)  # (B,)

        logits = model(batch_gpu)  # (B, C, H, W, Z)
        loss = criterion(logits, labels, n_slices)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.inference_mode()
def evaluate_vol(
    model: nn.Module,
    dataloader,
    device: torch.device,
    **_kwargs,
) -> dict:
    """Evaluate on cached volume features.

    Computes per-class Dice and macro Dice on valid (non-padded) slices only.
    """
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in dataloader:
        batch_gpu = _batch_to_device(batch, device)

        labels = batch["label"]  # (B, 1, H, W, Z)
        n_slices = batch["n_slices"]  # (B,)

        logits = model(batch_gpu)
        preds = logits.argmax(dim=1).cpu()  # (B, H, W, Z)

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


@torch.inference_mode()
def evaluate_vol_per_sample(
    model: nn.Module,
    dataloader,
    device: torch.device,
    **_kwargs,
) -> list[dict]:
    """Evaluate on cached volume features and return one row per patient-frame."""
    model.eval()
    rows: list[dict] = []

    for batch in dataloader:
        batch_gpu = _batch_to_device(batch, device)

        labels = batch["label"]  # (B, 1, H, W, Z)
        n_slices = batch["n_slices"]  # (B,)

        logits = model(batch_gpu)
        preds = logits.argmax(dim=1).cpu()  # (B, H, W, Z)

        for i in range(preds.shape[0]):
            ns = int(_batch_item(n_slices, i))
            is_ed = bool(_batch_item(batch["is_ed"], i))
            pred_i = preds[i, :, :, :ns].numpy()
            label_i = labels[i, 0, :, :, :ns].numpy()
            row = {
                "pid": str(_batch_item(batch["pid"], i)),
                "frame": _frame_name(is_ed),
                "is_ed": is_ed,
                "n_slices": ns,
            }
            row.update(per_sample_dice_metrics(pred_i, label_i))
            rows.append(row)

    return rows


# ── High-Level Training Wrapper ──────────────────────────────────────────────
def train_segmentation(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    n_epochs: int = 100,
    patience: int = 20,
    is_volume: bool = False,
) -> dict:
    """Train a segmentation model with early stopping and LR scheduling.

    Args:
        model: Decoder model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: LR scheduler (stepped after each epoch).
        device: Device for training.
        n_epochs: Maximum number of epochs.
        patience: Early stopping patience (epochs without improvement).
        is_volume: If True, use volume training/eval loops.

    Returns:
        Dict with keys: ``best_val_dice``, ``best_epoch``, ``history``.
        The model is restored to the best checkpoint before returning.
    """
    train_fn = train_one_epoch_vol if is_volume else train_one_epoch  # type: ignore[assignment]
    eval_fn = evaluate_vol if is_volume else evaluate  # type: ignore[assignment]

    best_val_dice = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    best_state: dict | None = None
    history: dict[str, list] = {"train_loss": [], "val_macro_dice": [], "lr": []}

    for epoch in range(1, n_epochs + 1):
        train_loss = train_fn(model, train_loader, criterion, optimizer, device)  # type: ignore[operator]
        scheduler.step()

        val_metrics = eval_fn(model, val_loader, device)
        val_dice = val_metrics["macro_dice"]

        history["train_loss"].append(train_loss)
        history["val_macro_dice"].append(val_dice)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        improved = val_dice > best_val_dice
        if epoch == 1 or epoch % 5 == 0 or improved:
            tag = " *" if improved else ""
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{n_epochs} | loss={train_loss:.4f} "
                f"| val Dice={val_dice:.4f} | lr={lr:.2e}{tag}"
            )

        if improved:
            best_val_dice = val_dice
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best val Dice={best_val_dice:.4f} at epoch {best_epoch}."
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"Restored best checkpoint from epoch {best_epoch} "
            f"(val Dice={best_val_dice:.4f})"
        )

    return {
        "best_val_dice": best_val_dice,
        "best_epoch": best_epoch,
        "history": history,
    }
