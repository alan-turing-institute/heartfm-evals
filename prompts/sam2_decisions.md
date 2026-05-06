# SAM v1 vs SAM2 — Decision Record

## Decision

**We use SAM v1 for both classification and segmentation.** SAM2 was explored for both tasks but
ultimately set aside. This document records why.

---

## Context: what we are comparing

The primary goal is to compare **backbone transferability** across three models for cardiac MRI
classification and segmentation:

| Backbone | Architecture | Pretraining | Domain |
|----------|-------------|-------------|--------|
| DINOv3 | Isotropic ViT | Self-supervised | Natural images |
| SAM v1 | Isotropic ViT | Supervised (segmentation) | Natural images |
| CineMA | Conv + ViT | Supervised | Cardiac MRI |

DINOv3 vs SAM v1 isolates **pretraining objective** (self-supervised vs task-supervised) while
keeping architecture fixed. SAM v1 vs CineMA isolates **domain specificity**. This gives a legible
three-way comparison that varies one thing at a time.

---

## Why not SAM2?

SAM2 (`facebook/sam2.1-hiera-*`) uses a Hiera hierarchical ViT — a different architecture class
from the isotropic ViT shared by DINOv3 and SAM v1. Including SAM2 in the comparison would
conflate **architecture** with **pretraining objective**, making it harder to interpret results.

### For segmentation

SAM2's main architectural advantage is genuine multi-scale spatial features from its four Hiera
stages (256×256 → 128×128 → 64×64 → 32×32). However, our pipeline downsamples all features to a
12×12 grid before caching, which discards the native spatial pyramid. The UNetR decoder then
reconstructs spatial hierarchy by upsampling from 12×12 regardless of backbone — so SAM2 gets no
benefit from its multi-scale design.

Additionally:
- SAM2 requires per-stage channel tracking (`stage_embed_dims`) and updated decoders, adding
  complexity without improving the comparison.
- SAM v1 fits the same UNetR framework as DINOv3 with no special treatment (uniform channel
  counts, simple `layer_indices`).

### For classification

SAM2's Hiera encoder has no CLS token. Features must be extracted by global-average-pooling the
final hidden state (Stage 4). While this works, SAM v1 follows the same GAP-of-final-features
approach and keeps the comparison architecturally consistent with DINOv3 and CineMA.

---

## What was explored with SAM2

SAM2 classification and segmentation were implemented and run experimentally:

- **Classification**: `cache_sam2_cls_features` extracts Stage 4 (`hidden_states[-1]`) and GAPs
  over spatial dims. Results are in `results/classification/`.
- **Segmentation**: multi-scale `layer_indices` were fixed to pick one block per Hiera stage
  (rather than all from Stage 3), and `DINOv3UNetRDecoder` was extended to accept per-stage
  channel counts via `embed_dims`. Experiments were run on ACDC/MNM.

The supporting library code (`backbones.py`, `caching.py`, `features.py`) retains SAM2 support.
The primary evaluation pipeline (`run_segmentation.py`, `run_classification.py`) uses SAM v1 only.
