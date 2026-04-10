"""Fine-tuning classification for patient-level pathology prediction.

Trains a linear classification head on pre-extracted (frozen backbone) features
using the same patient-level pooling as the logistic regression probe: mean-pool
ED slices + mean-pool ES slices → concatenate → linear head.

Features are expected to be pre-cached to disk and loaded before calling
``finetune_sweep_and_train``.  Both the logreg and finetune evaluation modes
share the same disk-cached features.

Hyperparameter selection uses stratified k-fold CV over a small LR grid,
matching the logistic regression probe protocol.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from heartfm_evals.classification_probe import NUM_PATHOLOGIES

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


# ── sklearn-compatible wrapper ────────────────────────────────────────────────
class ClassificationHeadPredictor:
    """Wrap a ``ClassificationHead`` + ``StandardScaler`` with sklearn interface.

    Provides ``.predict(X)`` and ``.predict_proba(X)`` (where *X* is a numpy
    array) so a trained ``ClassificationHead`` can be passed directly to
    ``evaluate_classification()`` from ``classification_probe``.
    """

    def __init__(
        self,
        head: ClassificationHead,
        scaler: StandardScaler,
        device: torch.device,
    ) -> None:
        self.head = head
        self.scaler = scaler
        self.device = device

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.head.eval()
        X_scaled = self.scaler.transform(X)
        t = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        return self.head(t).argmax(dim=1).cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.head.eval()
        X_scaled = self.scaler.transform(X)
        t = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        return torch.softmax(self.head(t), dim=1).cpu().numpy()


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
            scaler.transform(features.numpy()),
            dtype=features.dtype,
        )
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=0
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
    epochs_without_improvement = 0

    for epoch in range(epochs):
        _train_one_epoch_cached(
            head,
            features,
            labels,
            train_indices,
            optimizer,
            criterion,
            device,
        )
        scheduler.step()

        val_acc, _, _ = _evaluate_patients_cached(
            head,
            features,
            labels,
            val_indices,
            device,
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
def finetune_sweep_and_train(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    device: torch.device,
    lr_grid: tuple[float, ...] = DEFAULT_LR_GRID,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    epochs: int = DEFAULT_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    n_folds: int = 10,
    num_classes: int = NUM_PATHOLOGIES,
    val_features: torch.Tensor | None = None,
    val_labels: torch.Tensor | None = None,
) -> tuple[float, ClassificationHead, list[dict], StandardScaler]:
    """Sweep LR via stratified k-fold CV (or val split), then retrain on all training data.

    Operates entirely on pre-extracted patient feature tensors (frozen backbone).
    Both logreg and finetune evaluation modes share the same disk-cached features.

    When ``val_features`` and ``val_labels`` are provided, LR selection uses
    the dedicated validation set instead of k-fold CV.

    Args:
        train_features: Training features, shape ``(N, feature_dim)``.
        train_labels: Training labels, shape ``(N,)``, long tensor.
        device: Device for training.
        lr_grid: Learning rates to sweep.
        weight_decay: AdamW weight decay (fixed).
        epochs: Max training epochs per run.
        patience: Early stopping patience.
        n_folds: Number of stratified CV folds (ignored when val data provided).
        num_classes: Number of output classes for the classification head.
        val_features: Optional validation features, shape ``(M, feature_dim)``.
        val_labels: Optional validation labels, shape ``(M,)``, long tensor.

    Returns:
        best_lr: Optimal learning rate from CV.
        head: Trained ClassificationHead.
        sweep_results: List of dicts with keys lr, mean_cv_acc, std_cv_acc.
        scaler: Fitted StandardScaler used during training (must be applied at eval).
    """
    labels_np = train_labels.numpy()
    use_val_split = val_features is not None and val_labels is not None

    if not use_val_split:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
        folds = list(skf.split(np.zeros(len(train_labels)), labels_np))

    in_dim = train_features.shape[1]
    best_mean_acc = -1.0
    best_lr = lr_grid[0]
    sweep_results: list[dict] = []

    for lr in lr_grid:
        if use_val_split:
            scaler = StandardScaler()
            scaler.fit(train_features.numpy())
            train_idx = np.arange(len(train_features))
            # Combine train+val features for _train_with_lr_cached interface
            combined_features = torch.cat([train_features, val_features], dim=0)
            combined_labels = torch.cat([train_labels, val_labels], dim=0)
            val_idx_shifted = np.arange(
                len(train_features), len(train_features) + len(val_features)
            )
            head = ClassificationHead(in_dim, num_classes=num_classes).to(device)
            val_acc, _ = _train_with_lr_cached(
                head,
                combined_features,
                combined_labels,
                train_idx,
                val_idx_shifted,
                lr=lr,
                weight_decay=weight_decay,
                epochs=epochs,
                patience=patience,
                device=device,
                scaler=scaler,
            )
            mean_acc = float(val_acc)
            std_acc = 0.0
        else:
            fold_accs = []
            for train_idx, val_idx in folds:
                scaler = StandardScaler()
                scaler.fit(train_features[train_idx].numpy())
                head = ClassificationHead(in_dim, num_classes=num_classes).to(device)
                val_acc, _ = _train_with_lr_cached(
                    head,
                    train_features,
                    train_labels,
                    np.array(train_idx),
                    np.array(val_idx),
                    lr=lr,
                    weight_decay=weight_decay,
                    epochs=epochs,
                    patience=patience,
                    device=device,
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
    final_scaler.fit(train_features.numpy())

    all_idx = np.arange(len(train_features))
    head = ClassificationHead(in_dim, num_classes=num_classes).to(device)
    _train_with_lr_cached(
        head,
        train_features,
        train_labels,
        all_idx,
        all_idx,
        lr=best_lr,
        weight_decay=weight_decay,
        epochs=epochs,
        patience=epochs,
        device=device,
        scaler=final_scaler,
    )

    return best_lr, head, sweep_results, final_scaler
