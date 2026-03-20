"""SGD linear-probing classification probe for patient-level pathology prediction.

Adapts the DINOv3 paper's linear evaluation protocol (nn.Linear + SGD with
cosine annealing) to classify ACDC cardiac MRI patients into 5 pathology
classes using frozen multi-block features.

Protocol:
    frozen DINOv3 backbone → extract CLS tokens from last *N* blocks +
    mean-pooled patch tokens from the last block per 2D slice →
    mean-pool ED slices + mean-pool ES slices per patient → concatenate →
    nn.Linear trained with SGD + CosineAnnealingLR, sweeping LR x weight decay.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from heartfm_evals.classification_probe import (
    NUM_PATHOLOGIES,
    PATHOLOGY_CLASSES,
    PATHOLOGY_NAMES,
    get_pathology_map,
)
from heartfm_evals.dense_linear_probe import preprocess_slice

logger = logging.getLogger(__name__)

# ── Re-export for convenience ──────────────────────────────────────────────────
__all__ = [
    "NUM_PATHOLOGIES",
    "PATHOLOGY_CLASSES",
    "PATHOLOGY_NAMES",
    "get_pathology_map",
    "DEFAULT_LEARNING_RATES",
    "DEFAULT_WEIGHT_DECAYS",
    "DEFAULT_N_LAST_BLOCKS",
    "DEFAULT_EPOCHS",
    "LinearClassifier",
    "cache_linear_features",
    "load_and_build_patient_features",
    "train_linear_classifier",
    "sweep_lr_wd_and_train",
    "evaluate_linear_classification",
]

# ── Constants ──────────────────────────────────────────────────────────────────
# Same 13 learning rates as DINOv3 linear.py
DEFAULT_LEARNING_RATES: tuple[float, ...] = (
    1e-5,
    2e-5,
    5e-5,
    1e-4,
    2e-4,
    5e-4,
    1e-3,
    2e-3,
    5e-3,
    1e-2,
    2e-2,
    5e-2,
    0.1,
)
DEFAULT_WEIGHT_DECAYS: tuple[float, ...] = (0.0, 1e-5)
DEFAULT_N_LAST_BLOCKS = 1
DEFAULT_EPOCHS = 10


# ── Linear Classifier ─────────────────────────────────────────────────────────
class LinearClassifier(nn.Module):
    """Single linear layer trained on top of frozen features.

    Matches DINOv3 ``linear.py``: weight ~ N(0, 0.01), bias = 0.
    """

    def __init__(self, in_dim: int, num_classes: int = NUM_PATHOLOGIES):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ── Feature Extraction ─────────────────────────────────────────────────────────
def _extract_one_slice(
    backbone: nn.Module,
    image_2d: torch.Tensor,
    n_last_blocks: int,
    use_avgpool: bool,
    device: torch.device | None,
) -> torch.Tensor:
    """Extract the linear-probe feature vector for a single 2D slice.

    Returns:
        Feature tensor of shape ``(feature_dim,)`` on CPU, where
        ``feature_dim = n_last_blocks * embed_dim [+ embed_dim if use_avgpool]``.
    """
    img = preprocess_slice(image_2d)  # (1, 3, H, W)
    if device is not None:
        img = img.to(device)

    # Returns tuple of length n_last_blocks: ((patch_tokens, cls_token), ...)
    feats = backbone.get_intermediate_layers(
        img,
        n=n_last_blocks,
        return_class_token=True,
        reshape=True,
        norm=True,
    )

    # Concatenate CLS tokens from all N blocks → (n_last_blocks * embed_dim,)
    cls_concat = torch.cat([cls_token for (_, cls_token) in feats], dim=-1)

    if use_avgpool:
        # Mean-pool spatial patch tokens from the last block → (embed_dim,)
        last_patches = feats[-1][0]  # (1, embed_dim, h, w)
        patch_mean = last_patches.mean(dim=(2, 3))  # (1, embed_dim)
        output = torch.cat((cls_concat, patch_mean), dim=-1)
    else:
        output = cls_concat

    return output.squeeze(0).cpu()  # (feature_dim,)


def compute_feature_dim(
    embed_dim: int,
    n_last_blocks: int = DEFAULT_N_LAST_BLOCKS,
    use_avgpool: bool = True,
) -> int:
    """Compute the expected feature dimension for the linear probe."""
    dim = n_last_blocks * embed_dim
    if use_avgpool:
        dim += embed_dim
    return dim


# ── Feature Caching ────────────────────────────────────────────────────────────
@torch.inference_mode()
def cache_linear_features(
    backbone: nn.Module,
    cinema_dataset,
    cache_dir: Path,
    n_last_blocks: int = DEFAULT_N_LAST_BLOCKS,
    use_avgpool: bool = True,
    device: torch.device | None = None,
) -> list[dict]:
    """Extract and cache multi-block features to disk, one .pt per slice.

    Saves ``{"features": Tensor(feature_dim,)}`` per file.  Skips files that
    already exist so the function is safe to re-run.

    Returns:
        Manifest — list of dicts with keys: path, pid, is_ed, z_idx.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for sample_idx in tqdm(range(len(cinema_dataset)), desc="Caching linear features"):
        sample = cinema_dataset[sample_idx]
        image_3d = sample["sax_image"]  # (1, H, W, z)
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
            feature = _extract_one_slice(
                backbone,
                image_2d,
                n_last_blocks,
                use_avgpool,
                device,
            )
            torch.save({"features": feature}, fpath)
            manifest.append({"path": fpath, "pid": pid, "is_ed": is_ed, "z_idx": z})

    return manifest


def load_and_build_patient_features(
    manifest: list[dict],
    pathology_map: dict[str, str],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Load cached features from disk, pool per patient, and return tensors.

    For each patient: mean-pool ED slices, mean-pool ES slices, concatenate
    -> ``(2 x feature_dim,)`` vector.

    Returns:
        features: ``(N_patients, 2 * feature_dim)`` float32 tensor.
        labels:   ``(N_patients,)`` long tensor.
        pids:     list of patient IDs in the same order.
    """
    patient_features: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: {"ed": [], "es": []}
    )

    for entry in manifest:
        data = torch.load(entry["path"], weights_only=True)
        frame_key = "ed" if entry["is_ed"] else "es"
        patient_features[entry["pid"]][frame_key].append(data["features"])

    features_list: list[torch.Tensor] = []
    labels_list: list[int] = []
    pids: list[str] = []

    for pid in sorted(patient_features.keys()):
        if pid not in pathology_map:
            logger.warning("Skipping %s: no pathology label found", pid)
            continue

        frames = patient_features[pid]
        ed_feats = torch.stack(frames["ed"])  # (n_ed, feat_dim)
        es_feats = torch.stack(frames["es"])  # (n_es, feat_dim)

        ed_mean = ed_feats.mean(dim=0)  # (feat_dim,)
        es_mean = es_feats.mean(dim=0)  # (feat_dim,)
        patient_feat = torch.cat([ed_mean, es_mean])  # (2 * feat_dim,)

        features_list.append(patient_feat)
        labels_list.append(PATHOLOGY_CLASSES[pathology_map[pid]])
        pids.append(pid)

    features = torch.stack(features_list).float()
    labels = torch.tensor(labels_list, dtype=torch.long)
    return features, labels, pids


# ── Training ───────────────────────────────────────────────────────────────────
def train_linear_classifier(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    lr: float,
    wd: float,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = 128,
    device: torch.device | None = None,
) -> tuple[float, dict]:
    """Train a :class:`LinearClassifier` and return best val accuracy + state.

    Uses SGD with momentum 0.9, cosine annealing, and cross-entropy loss,
    matching the DINOv3 linear evaluation protocol.

    Returns:
        best_val_acc: Best validation accuracy across epochs.
        best_state: State dict of the model at the best epoch.
    """
    if device is None:
        device = torch.device("cpu")

    in_dim = train_features.shape[1]
    model = LinearClassifier(in_dim).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=wd,
    )

    train_ds = TensorDataset(train_features.to(device), train_labels.to(device))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=0,
    )
    criterion = nn.CrossEntropyLoss()

    val_X = val_features.to(device)
    val_y = val_labels.to(device)

    best_val_acc = -1.0
    best_state: dict = {}

    for _epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Evaluate on val
        model.eval()
        with torch.inference_mode():
            val_logits = model(val_X)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_y).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_val_acc, best_state


def sweep_lr_wd_and_train(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    learning_rates: tuple[float, ...] = DEFAULT_LEARNING_RATES,
    weight_decays: tuple[float, ...] = DEFAULT_WEIGHT_DECAYS,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = 128,
    device: torch.device | None = None,
) -> tuple[float, float, LinearClassifier, list[dict]]:
    """Sequential sweep over LR x WD, retrain best on train+val.

    Returns:
        best_lr: Learning rate that achieved highest val accuracy.
        best_wd: Weight decay that achieved highest val accuracy.
        final_model: :class:`LinearClassifier` retrained on train+val.
        sweep_results: List of dicts with keys lr, wd, val_acc.
    """
    sweep_results: list[dict] = []
    best_acc = -1.0
    best_lr = learning_rates[0]
    best_wd = weight_decays[0]

    total_combos = len(learning_rates) * len(weight_decays)
    pbar = tqdm(total=total_combos, desc="LRxWD sweep")

    for lr in learning_rates:
        for wd in weight_decays:
            val_acc, _ = train_linear_classifier(
                train_features,
                train_labels,
                val_features,
                val_labels,
                lr=lr,
                wd=wd,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
            )
            sweep_results.append({"lr": lr, "wd": wd, "val_acc": val_acc})

            if val_acc > best_acc:
                best_acc = val_acc
                best_lr = lr
                best_wd = wd

            pbar.set_postfix(lr=f"{lr:.1e}", wd=f"{wd:.1e}", val=f"{val_acc:.3f}")
            pbar.update(1)

    pbar.close()
    logger.info(
        "Best LR=%.4g, WD=%.4g (val accuracy=%.4f)",
        best_lr,
        best_wd,
        best_acc,
    )

    # Retrain on train + val with best hyperparams
    combined_features = torch.cat([train_features, val_features], dim=0)
    combined_labels = torch.cat([train_labels, val_labels], dim=0)

    # Use a dummy 1-sample val set (we just need the training)
    _, best_state = train_linear_classifier(
        combined_features,
        combined_labels,
        combined_features[:1],
        combined_labels[:1],
        lr=best_lr,
        wd=best_wd,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
    )

    in_dim = train_features.shape[1]
    final_model = LinearClassifier(in_dim)
    final_model.load_state_dict(best_state)

    return best_lr, best_wd, final_model, sweep_results


# ── Evaluation ─────────────────────────────────────────────────────────────────
@torch.inference_mode()
def evaluate_linear_classification(
    model: LinearClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device | None = None,
) -> dict:
    """Evaluate a trained :class:`LinearClassifier`.

    Returns:
        Dict with keys: accuracy, macro_f1, per_class_accuracy,
        per_class_sensitivity, per_class_specificity, macro_sensitivity,
        macro_specificity, confusion_matrix, classification_report,
        predictions, probabilities.
    """
    if device is None:
        device = torch.device("cpu")

    model = model.to(device).eval()
    X = features.to(device)
    logits = model(X)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    y_pred = logits.argmax(dim=1).cpu().numpy()
    y_true = labels.numpy()

    acc = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_PATHOLOGIES)))

    # Per-class accuracy, sensitivity (TPR), and specificity (TNR)
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
        y_true,
        y_pred,
        labels=list(range(NUM_PATHOLOGIES)),
        target_names=list(PATHOLOGY_CLASSES.keys()),
        output_dict=True,
    )
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

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
        "probabilities": probs,
    }
