"""Fine-tuning classification for patient-level pathology prediction.

Trains a linear classification head (optionally with backbone unfrozen) on
ACDC cardiac MRI using the same patient-level pooling as the logistic
regression probe: mean-pool ED slices + mean-pool ES slices → concatenate →
linear head.

Supports three backbones (DINOv3, CineMA, SAM) with a unified interface.
Hyperparameter selection uses stratified k-fold CV over a small LR grid,
matching the logistic regression probe protocol.
"""

from __future__ import annotations

import copy
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from heartfm_evals.classification_probe import (
    CINEMA_SAX_TARGET_DEPTH,
    NUM_PATHOLOGIES,
    PATHOLOGY_CLASSES,
    _extract_cinema_per_slice_tokens,
)
from heartfm_evals.dense_linear_probe import preprocess_slice

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_LR_GRID: tuple[float, ...] = (1e-5, 5e-5, 1e-4, 5e-4, 1e-3)
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_EPOCHS = 50
DEFAULT_PATIENCE = 10
DEFAULT_BATCH_SIZE = 8


# ── Classification Head ──────────────────────────────────────────────────────
class ClassificationHead(nn.Module):
    """Single linear layer for classification. Weights ~ N(0, 0.01), bias = 0."""

    def __init__(self, in_dim: int, num_classes: int = NUM_PATHOLOGIES):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ── Per-patient feature extraction (differentiable) ──────────────────────────
def _extract_patient_feature_dinov3(
    backbone: nn.Module,
    sample: dict,
    device: torch.device,
) -> torch.Tensor:
    """Extract (2*embed_dim,) patient feature from DINOv3 backbone (differentiable)."""
    image_3d = sample["sax_image"]  # (1, H, W, z)
    n_slices = int(sample["n_slices"])
    is_ed = sample["is_ed"]

    slice_feats = []
    for z in range(n_slices):
        image_2d = image_3d[0, :, :, z]
        img = preprocess_slice(image_2d).to(device)
        feats = backbone.get_intermediate_layers(
            img, n=1, return_class_token=True, norm=True
        )
        cls_token = feats[0][1].squeeze(0)  # (embed_dim,)
        slice_feats.append(cls_token)

    return torch.stack(slice_feats).mean(dim=0)  # (embed_dim,)


def _extract_patient_feature_cinema(
    backbone: nn.Module,
    sample: dict,
    device: torch.device,
) -> torch.Tensor:
    """Extract (embed_dim,) per-frame feature from CineMA backbone (differentiable)."""
    image_3d = sample["sax_image"]  # (1, H, W, z)
    n_slices = int(sample["n_slices"])

    vol = image_3d
    z = int(vol.shape[-1])
    used_depth = min(z, CINEMA_SAX_TARGET_DEPTH)

    if z > CINEMA_SAX_TARGET_DEPTH:
        vol = vol[..., :CINEMA_SAX_TARGET_DEPTH]
    elif z < CINEMA_SAX_TARGET_DEPTH:
        vol = F.pad(vol, (0, CINEMA_SAX_TARGET_DEPTH - z), mode="constant", value=0.0)

    batch = {"sax": vol.unsqueeze(0).to(device=device, dtype=torch.float32)}
    tokens = backbone.feature_forward(batch)["sax"]  # (1, n_tokens, C)

    gx, gy, gz = backbone.enc_down_dict["sax"].patch_embed.grid_size
    token_grid = tokens.squeeze(0).reshape(gx, gy, gz, -1)

    per_slice = []
    for z_idx in range(n_slices):
        src_z = min(z_idx, max(used_depth - 1, 0))
        feat_z = int(round(src_z * (gz - 1) / max(used_depth - 1, 1)))
        slice_token = token_grid[:, :, feat_z, :].mean(dim=(0, 1))
        per_slice.append(slice_token)

    return torch.stack(per_slice).mean(dim=0)  # (embed_dim,)


def _extract_patient_feature_sam(
    backbone: nn.Module,
    image_processor,
    sample: dict,
    device: torch.device,
) -> torch.Tensor:
    """Extract (embed_dim,) per-frame feature from SAM backbone (differentiable)."""
    from PIL import Image

    image_3d = sample["sax_image"]
    n_slices = int(sample["n_slices"])

    slice_feats = []
    for z in range(n_slices):
        image_2d = image_3d[0, :, :, z]
        img_np = (image_2d.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        pil = Image.fromarray(img_np, mode="L").convert("RGB")

        proc = image_processor(images=pil, return_tensors="pt")
        pixel_values = proc["pixel_values"].to(device)

        feats = backbone.get_image_embeddings(pixel_values)  # (1, C, h, w)
        cls_token = feats.squeeze(0).mean(dim=(1, 2))  # (C,)
        slice_feats.append(cls_token)

    return torch.stack(slice_feats).mean(dim=0)  # (embed_dim,)


def extract_patient_feature(
    backbone: nn.Module,
    ed_sample: dict,
    es_sample: dict,
    device: torch.device,
    backbone_type: str,
    image_processor=None,
) -> torch.Tensor:
    """Extract (2*embed_dim,) patient feature by pooling ED and ES frames.

    Args:
        backbone: The backbone model.
        ed_sample: CineMA dataset sample for the ED frame.
        es_sample: CineMA dataset sample for the ES frame.
        device: Device for inference.
        backbone_type: One of "dinov3", "cinema", "sam".
        image_processor: Required for SAM backbone.

    Returns:
        Patient feature vector of shape (2 * embed_dim,).
    """
    if backbone_type == "dinov3":
        extract_fn = lambda s: _extract_patient_feature_dinov3(backbone, s, device)
    elif backbone_type == "cinema":
        extract_fn = lambda s: _extract_patient_feature_cinema(backbone, s, device)
    elif backbone_type == "sam":
        extract_fn = lambda s: _extract_patient_feature_sam(
            backbone, image_processor, s, device
        )
    else:
        raise ValueError(f"Unknown backbone_type: {backbone_type}")

    ed_feat = extract_fn(ed_sample)
    es_feat = extract_fn(es_sample)
    return torch.cat([ed_feat, es_feat])  # (2 * embed_dim,)


# ── Dataset helpers ──────────────────────────────────────────────────────────
def _group_samples_by_patient(
    cinema_dataset,
    pathology_map: dict[str, str],
) -> list[dict]:
    """Group ED/ES samples from CineMA dataset by patient.

    Returns:
        List of dicts with keys: pid, label, ed_idx, es_idx.
    """
    patient_samples: dict[str, dict[str, int | None]] = defaultdict(
        lambda: {"ed_idx": None, "es_idx": None}
    )

    for i in range(len(cinema_dataset)):
        sample = cinema_dataset[i]
        pid = sample["pid"]
        if pid not in pathology_map:
            continue
        if sample["is_ed"]:
            patient_samples[pid]["ed_idx"] = i
        else:
            patient_samples[pid]["es_idx"] = i

    patients = []
    for pid in sorted(patient_samples.keys()):
        info = patient_samples[pid]
        if info["ed_idx"] is None or info["es_idx"] is None:
            logger.warning("Skipping %s: missing ED or ES frame", pid)
            continue
        patients.append({
            "pid": pid,
            "label": PATHOLOGY_CLASSES[pathology_map[pid]],
            "ed_idx": info["ed_idx"],
            "es_idx": info["es_idx"],
        })

    return patients


# ── Pre-extraction for frozen backbone ────────────────────────────────────────
@torch.no_grad()
def _preextract_all_features(
    backbone: nn.Module,
    cinema_dataset,
    all_patients: list[dict],
    device: torch.device,
    backbone_type: str,
    image_processor=None,
) -> torch.Tensor:
    """Pre-extract (2*embed_dim,) features for all patients. Returns (N, 2*embed_dim)."""
    backbone.eval()
    feats = []
    for pinfo in tqdm(all_patients, desc="Pre-extracting features"):
        ed_sample = cinema_dataset[pinfo["ed_idx"]]
        es_sample = cinema_dataset[pinfo["es_idx"]]
        feat = extract_patient_feature(
            backbone, ed_sample, es_sample, device,
            backbone_type, image_processor,
        )
        feats.append(feat.cpu())
    return torch.stack(feats)  # (N, 2*embed_dim)


def _train_one_epoch_cached(
    head: ClassificationHead,
    features: torch.Tensor,
    labels: torch.Tensor,
    patient_indices: np.ndarray,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train head for one epoch on pre-extracted features. Returns mean loss."""
    head.train()
    order = np.random.permutation(len(patient_indices))
    total_loss = 0.0

    for idx in order:
        i = patient_indices[idx]
        feat = features[i].to(device).unsqueeze(0)
        label = labels[i].unsqueeze(0).to(device)

        logits = head(feat)
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(patient_indices)


@torch.no_grad()
def _evaluate_patients_cached(
    head: ClassificationHead,
    features: torch.Tensor,
    labels: torch.Tensor,
    patient_indices: np.ndarray,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate head on pre-extracted features."""
    head.eval()
    feat_batch = features[patient_indices].to(device)
    label_batch = labels[patient_indices]

    logits = head(feat_batch)
    preds = logits.argmax(dim=1).cpu().numpy()
    y_true = label_batch.numpy()
    return float(accuracy_score(y_true, preds)), preds, y_true


def _train_with_lr_cached(
    head: ClassificationHead,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    device: torch.device,
    scaler: StandardScaler | None = None,
) -> tuple[float, dict]:
    """Train head on cached features, return best val accuracy and head state."""
    if scaler is not None:
        features = torch.tensor(
            scaler.transform(features.numpy()), dtype=features.dtype,
        )
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
    epochs_without_improvement = 0

    for epoch in range(epochs):
        _train_one_epoch_cached(
            head, features, labels, train_indices,
            optimizer, criterion, device,
        )
        scheduler.step()

        val_acc, _, _ = _evaluate_patients_cached(
            head, features, labels, val_indices, device,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, patience)
            break

    return best_val_acc, best_head_state


# ── Training ─────────────────────────────────────────────────────────────────
def _train_one_epoch(
    backbone: nn.Module,
    head: ClassificationHead,
    cinema_dataset,
    patient_indices: list[dict],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    backbone_type: str,
    freeze_backbone: bool,
    image_processor=None,
) -> float:
    """Train for one epoch over the given patients. Returns mean loss."""
    if freeze_backbone:
        backbone.eval()
    else:
        backbone.train()
    head.train()

    # Shuffle patient order
    order = np.random.permutation(len(patient_indices))
    total_loss = 0.0

    for idx in order:
        pinfo = patient_indices[idx]
        ed_sample = cinema_dataset[pinfo["ed_idx"]]
        es_sample = cinema_dataset[pinfo["es_idx"]]
        label = torch.tensor([pinfo["label"]], dtype=torch.long, device=device)

        if freeze_backbone:
            with torch.no_grad():
                feat = extract_patient_feature(
                    backbone, ed_sample, es_sample, device,
                    backbone_type, image_processor,
                )
        else:
            feat = extract_patient_feature(
                backbone, ed_sample, es_sample, device,
                backbone_type, image_processor,
            )

        logits = head(feat.unsqueeze(0))
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(patient_indices)


@torch.no_grad()
def _evaluate_patients(
    backbone: nn.Module,
    head: ClassificationHead,
    cinema_dataset,
    patient_indices: list[dict],
    device: torch.device,
    backbone_type: str,
    image_processor=None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate accuracy on a set of patients.

    Returns:
        accuracy, predictions (N,), true labels (N,).
    """
    backbone.eval()
    head.eval()

    all_preds = []
    all_labels = []

    for pinfo in patient_indices:
        ed_sample = cinema_dataset[pinfo["ed_idx"]]
        es_sample = cinema_dataset[pinfo["es_idx"]]

        feat = extract_patient_feature(
            backbone, ed_sample, es_sample, device,
            backbone_type, image_processor,
        )
        logits = head(feat.unsqueeze(0))
        pred = logits.argmax(dim=1).item()

        all_preds.append(pred)
        all_labels.append(pinfo["label"])

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)
    return float(accuracy_score(y_true, y_pred)), y_pred, y_true


def _train_with_lr(
    backbone: nn.Module,
    head: ClassificationHead,
    cinema_dataset,
    train_patients: list[dict],
    val_patients: list[dict],
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    device: torch.device,
    backbone_type: str,
    freeze_backbone: bool,
    image_processor=None,
) -> tuple[float, dict, dict]:
    """Train with a specific LR, return best val accuracy and state dicts.

    Returns:
        best_val_acc, best_backbone_state, best_head_state.
    """
    params = list(head.parameters())
    if not freeze_backbone:
        params = list(backbone.parameters()) + params

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=0,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_backbone_state = {k: v.cpu().clone() for k, v in backbone.state_dict().items()}
    best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
    epochs_without_improvement = 0

    pbar = tqdm(range(epochs), desc="Fine-tuning", unit="epoch")
    for epoch in pbar:
        _train_one_epoch(
            backbone, head, cinema_dataset, train_patients,
            optimizer, criterion, device, backbone_type,
            freeze_backbone, image_processor,
        )
        scheduler.step()

        val_acc, _, _ = _evaluate_patients(
            backbone, head, cinema_dataset, val_patients,
            device, backbone_type, image_processor,
        )

        improved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_backbone_state = {
                k: v.cpu().clone() for k, v in backbone.state_dict().items()
            }
            best_head_state = {
                k: v.cpu().clone() for k, v in head.state_dict().items()
            }
            epochs_without_improvement = 0
            improved = " *"
        else:
            epochs_without_improvement += 1

        pbar.set_postfix(
            val_acc=f"{val_acc:.4f}",
            best=f"{best_val_acc:.4f}",
            no_improv=epochs_without_improvement,
        )
        logger.info(
            "Epoch %d/%d — val_acc=%.4f, best=%.4f, lr=%.4g%s",
            epoch + 1, epochs, val_acc, best_val_acc,
            optimizer.param_groups[0]["lr"], improved,
        )

        if epochs_without_improvement >= patience:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, patience)
            break

    return best_val_acc, best_backbone_state, best_head_state


def finetune_sweep_and_train(
    backbone: nn.Module,
    cinema_dataset,
    pathology_map: dict[str, str],
    embed_dim: int,
    device: torch.device,
    backbone_type: str,
    freeze_backbone: bool = True,
    image_processor=None,
    lr_grid: tuple[float, ...] = DEFAULT_LR_GRID,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    epochs: int = DEFAULT_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    n_folds: int = 10,
) -> tuple[float, nn.Module, ClassificationHead, list[dict]]:
    """Sweep LR via stratified k-fold CV, then retrain on all training data.

    The LR sweep is **always performed with a frozen backbone** using
    pre-extracted features, even when ``freeze_backbone=False``.  This is
    orders of magnitude faster (one forward pass per patient instead of one
    per patient × epoch × fold × LR) and is standard practice in transfer
    learning: the frozen sweep reliably identifies the right order of
    magnitude for the learning rate, and the final unfrozen training run
    adapts from there.

    When ``freeze_backbone=False``, the selected LR is then used for a
    single end-to-end fine-tuning run on all training data with the
    backbone unfrozen.

    Args:
        backbone: The backbone model (will be modified in-place if not frozen).
        cinema_dataset: CineMA EndDiastoleEndSystoleDataset (training split).
        pathology_map: {pid: pathology_string} from metadata.
        embed_dim: Embedding dimension of the backbone.
        device: Device for training.
        backbone_type: One of "dinov3", "cinema", "sam".
        freeze_backbone: If True, only train the classification head.
        image_processor: Required for SAM backbone.
        lr_grid: Learning rates to sweep.
        weight_decay: AdamW weight decay (fixed).
        epochs: Max training epochs per run.
        patience: Early stopping patience.
        n_folds: Number of stratified CV folds.

    Returns:
        best_lr: Optimal learning rate from CV.
        backbone: Backbone (potentially fine-tuned).
        head: Trained ClassificationHead.
        sweep_results: List of dicts with keys lr, mean_cv_acc, std_cv_acc.
    """
    all_patients = _group_samples_by_patient(cinema_dataset, pathology_map)
    labels_np = np.array([p["label"] for p in all_patients])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    folds = list(skf.split(np.zeros(len(all_patients)), labels_np))

    in_dim = 2 * embed_dim
    best_mean_acc = -1.0
    best_lr = lr_grid[0]
    sweep_results: list[dict] = []

    # ── LR sweep: always frozen (pre-extracted features) ──────────────────
    # Even when freeze_backbone=False, the sweep runs on cached features for
    # speed. The frozen LR ranking is a reliable proxy for the unfrozen case.
    cached_features = _preextract_all_features(
        backbone, cinema_dataset, all_patients, device,
        backbone_type, image_processor,
    )
    cached_labels = torch.tensor(labels_np, dtype=torch.long)

    for lr in tqdm(lr_grid, desc="LR sweep (CV)"):
        fold_accs = []
        for train_idx, val_idx in folds:
            scaler = StandardScaler()
            scaler.fit(cached_features[train_idx].numpy())
            head = ClassificationHead(in_dim).to(device)
            val_acc, _ = _train_with_lr_cached(
                head, cached_features, cached_labels,
                np.array(train_idx), np.array(val_idx),
                lr=lr, weight_decay=weight_decay,
                epochs=epochs, patience=patience, device=device,
                scaler=scaler,
            )
            fold_accs.append(val_acc)

        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs))
        sweep_results.append({"lr": lr, "mean_cv_acc": mean_acc, "std_cv_acc": std_acc})
        logger.info("LR=%.4g → mean CV acc=%.4f ± %.4f", lr, mean_acc, std_acc)

        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc
            best_lr = lr

    logger.info("Best LR=%.4g (mean CV accuracy=%.4f)", best_lr, best_mean_acc)

    # ── Final training with best LR ───────────────────────────────────────
    # Fit scaler on all training data for final training / evaluation
    final_scaler = StandardScaler()
    final_scaler.fit(cached_features.numpy())

    if freeze_backbone:
        # Frozen: retrain head on cached features (fast)
        all_idx = np.arange(len(all_patients))
        head = ClassificationHead(in_dim).to(device)
        _, best_head_state = _train_with_lr_cached(
            head, cached_features, cached_labels,
            all_idx, all_idx,
            lr=best_lr, weight_decay=weight_decay,
            epochs=epochs, patience=epochs, device=device,
            scaler=final_scaler,
        )
        head.load_state_dict(best_head_state)
    else:
        # Unfrozen: single end-to-end fine-tuning run with backbone + head
        head = ClassificationHead(in_dim).to(device)
        _, best_backbone_state, best_head_state = _train_with_lr(
            backbone, head, cinema_dataset,
            all_patients, all_patients,
            lr=best_lr, weight_decay=weight_decay,
            epochs=epochs, patience=epochs,
            device=device, backbone_type=backbone_type,
            freeze_backbone=False,
            image_processor=image_processor,
        )
        backbone.load_state_dict(best_backbone_state)
        head.load_state_dict(best_head_state)

    return best_lr, backbone, head, sweep_results


@torch.no_grad()
def evaluate_finetune_classification(
    backbone: nn.Module,
    head: ClassificationHead,
    cinema_dataset,
    pathology_map: dict[str, str],
    device: torch.device,
    backbone_type: str,
    image_processor=None,
) -> dict:
    """Evaluate a fine-tuned backbone+head on a dataset.

    Returns the same metric dict as classification_probe.evaluate_classification():
        accuracy, macro_f1, per_class_accuracy, per_class_sensitivity,
        per_class_specificity, macro_sensitivity, macro_specificity,
        confusion_matrix, classification_report, predictions, probabilities.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    backbone.eval()
    head.eval()

    patients = _group_samples_by_patient(cinema_dataset, pathology_map)

    all_probs = []
    all_preds = []
    all_labels = []

    for pinfo in patients:
        ed_sample = cinema_dataset[pinfo["ed_idx"]]
        es_sample = cinema_dataset[pinfo["es_idx"]]

        feat = extract_patient_feature(
            backbone, ed_sample, es_sample, device,
            backbone_type, image_processor,
        )
        logits = head(feat.unsqueeze(0))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = logits.argmax(dim=1).item()

        all_probs.append(probs)
        all_preds.append(pred)
        all_labels.append(pinfo["label"])

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_PATHOLOGIES)))

    per_class_acc: dict[str, float] = {}
    per_class_sensitivity: dict[str, float] = {}
    per_class_specificity: dict[str, float] = {}
    total = cm.sum()

    for cls_name, cls_idx in PATHOLOGY_CLASSES.items():
        tp = cm[cls_idx, cls_idx]
        fn = cm[cls_idx, :].sum() - tp
        fp = cm[:, cls_idx].sum() - tp
        tn = total - tp - fn - fp

        mask = y_true == cls_idx
        if mask.sum() > 0:
            per_class_acc[cls_name] = float(accuracy_score(y_true[mask], y_pred[mask]))

        per_class_sensitivity[cls_name] = (
            float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        )
        per_class_specificity[cls_name] = (
            float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        )

    macro_sensitivity = float(
        sum(per_class_sensitivity.values()) / len(per_class_sensitivity)
    )
    macro_specificity = float(
        sum(per_class_specificity.values()) / len(per_class_specificity)
    )

    report = classification_report(
        y_true, y_pred,
        labels=list(range(NUM_PATHOLOGIES)),
        target_names=list(PATHOLOGY_CLASSES.keys()),
        output_dict=True,
    )
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

    # Also return pids in order for per-patient results
    pids = [p["pid"] for p in patients]

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_sensitivity": macro_sensitivity,
        "macro_specificity": macro_specificity,
        "per_class_accuracy": per_class_acc,
        "per_class_sensitivity": per_class_sensitivity,
        "per_class_specificity": per_class_specificity,
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": y_pred,
        "probabilities": y_prob,
        "pids": pids,
        "labels": y_true,
    }
