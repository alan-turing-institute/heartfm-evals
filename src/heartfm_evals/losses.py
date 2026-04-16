"""Loss functions for segmentation training.

Provides:
- ``DiceLoss``: Soft Dice loss (multi-class, optionally excluding background).
- ``CombinedLoss``: CrossEntropy + DiceLoss.
- ``WeightedCombinedLoss``: Class-weighted CrossEntropy + DiceLoss.
- ``MaskedVolumeLoss``: Weighted CE + Dice for 3D volumes with z-padding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation."""

    def __init__(self, smooth: float = 1.0, exclude_bg: bool = True):
        super().__init__()
        self.smooth = smooth
        self.exclude_bg = exclude_bg

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw predictions.
            targets: (B, H, W) integer class labels.
        """
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        targets_oh = F.one_hot(targets.long(), num_classes=logits.shape[1])  # (B,H,W,C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        start_c = 1 if self.exclude_bg else 0
        probs = probs[:, start_c:]
        targets_oh = targets_oh[:, start_c:]

        dims = (0, 2, 3)  # reduce over batch and spatial
        intersection = (probs * targets_oh).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets_oh.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """CrossEntropy + DiceLoss."""

    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(
            logits, targets.long()
        ) + self.dice_weight * self.dice(logits, targets)


class WeightedCombinedLoss(nn.Module):
    """Class-weighted CrossEntropy + DiceLoss."""

    def __init__(
        self,
        ce_weight_tensor: torch.Tensor,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight_tensor)
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(
            logits, targets.long()
        ) + self.dice_weight * self.dice(logits, targets)


class MaskedVolumeLoss(nn.Module):
    """Weighted CE + Dice loss, with CE masked to exclude z-padded slices.

    Padded slices have label 0 (background).  Since the Dice component
    excludes the background class, padded slices contribute nothing to Dice.
    The CE component is explicitly masked to avoid biasing toward BG.
    """

    def __init__(
        self,
        ce_weight_tensor: torch.Tensor,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.register_buffer("class_weights", ce_weight_tensor)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        n_slices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked loss.

        Args:
            logits: (B, C, H, W, Z)
            targets: (B, 1, H, W, Z) — padded slices have label 0 (BG).
            n_slices: (B,) tensor of actual slice counts.
        """
        B, C, H, W, Z = logits.shape
        targets = targets.squeeze(1)  # (B, H, W, Z)

        # ── Masked CE: only valid (non-padded) z-slices ──
        z_idx = torch.arange(Z, device=logits.device)
        valid = z_idx.unsqueeze(0) < n_slices.unsqueeze(1)  # (B, Z)
        valid_mask = valid[:, None, None, :].expand(B, H, W, Z)  # (B, H, W, Z)

        logits_bhwzc = logits.permute(0, 2, 3, 4, 1)  # (B, H, W, Z, C)
        class_w: torch.Tensor = self.class_weights  # type: ignore[assignment]
        ce = F.cross_entropy(
            logits_bhwzc[valid_mask],  # (N_valid, C)
            targets[valid_mask].long(),  # (N_valid,)
            weight=class_w,
        )

        # ── Dice: padded slices are all-BG → excluded by foreground-only Dice ──
        probs = F.softmax(logits, dim=1)[:, 1:]  # (B, C-1, H, W, Z)
        targets_oh = (
            F.one_hot(targets.long(), num_classes=C)
            .permute(0, 4, 1, 2, 3)
            .float()[:, 1:]
        )  # (B, C-1, H, W, Z)

        dims = (0, 2, 3, 4)
        intersection = (probs * targets_oh).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets_oh.sum(dim=dims)
        dice = (2.0 * intersection + 1.0) / (cardinality + 1.0)
        dice_loss = 1.0 - dice.mean()

        return self.ce_weight * ce + self.dice_weight * dice_loss
