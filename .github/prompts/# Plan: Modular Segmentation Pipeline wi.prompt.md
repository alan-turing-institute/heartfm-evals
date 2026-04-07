# Plan: Modular Segmentation Pipeline with Hydra

## TL;DR
Restructure 15+ standalone ACDC segmentation scripts into a single config-driven pipeline.
One entry-point script (`scripts/segmentation/train.py`) plus per-dataset configs in `configs/segmentation/`.
Backbones (DINOv3, CineMA, SAM2) × decoders (linear_probe, conv_decoder, unetr) × training mode (frozen/finetune) are all selected via Hydra config overrides.
Shared logic moves into `src/heartfm_evals/` submodules. Plotting moves to a dedicated visualization module. Optional W&B logging added alongside CSV/matplotlib.

---

## Phase 1: Refactor `src/heartfm_evals/` into submodules

**Goal**: Break the two monolith modules (`dense_linear_probe.py`, `dense_unetr_probe.py`) into focused submodules while preserving backward compatibility.

### Step 1.1 — Create backbone loader module
**New file**: `src/heartfm_evals/backbones.py`

Designed to serve **both segmentation and classification** — the classification
modules (`classification_probe.py`, `finetune_classification.py`) duplicate the
same backbone loading, preprocessing, and forward-pass logic that the
segmentation scripts use. This module becomes the single source of truth.

- `load_backbone(config) -> nn.Module` — unified backbone factory
  - `_load_dinov3(model_name, weights_path, repo_dir)` — wraps `torch.hub.load()`
  - `_load_cinema(hf_cache_dir, auto_download)` — wraps `CineMA.from_pretrained()`
  - `_load_sam2(model_id, hf_cache_dir, auto_download)` — wraps `Sam2Model.from_pretrained()` + `Sam2Processor.from_pretrained()`
- `freeze_backbone(model)` / `unfreeze_backbone(model)` helpers
- Move `MODEL_CONFIGS` dict here (currently in `dense_linear_probe.py`)
- Add SAM2 variant configs (embed_dim, layer_indices per variant)
- **Shared preprocessing helpers** (currently duplicated across classification + segmentation):
  - `preprocess_slice(image_2d)` — grayscale→RGB + ImageNet norm (move from `dense_linear_probe.py`)
  - `preprocess_sam_slice(image_2d, image_processor)` — grayscale→PIL→SAM processor (extract from `classification_probe.cache_sam_cls_features` and SAM2 scripts)
  - `pad_volume_z(vol, target_depth)` — z-padding/truncation (consolidate from `dense_unetr_probe._pad_volume_z` and `classification_probe._extract_cinema_volume_token`)
- **Backbone-type dispatch helper**: `get_backbone_type(config) -> str` returning `"dinov3"` / `"cinema"` / `"sam2"`, matching the `backbone_type` dispatch pattern already used in `finetune_classification.extract_patient_feature()`

### Step 1.2 — Create decoder module
**New file**: `src/heartfm_evals/decoders.py`
- Move existing decoder classes into this module:
  - `DenseLinearProbe` (from `dense_linear_probe.py`)
  - `DINOdenseDecoderProbe` (currently inline in `acdc_dino_conv_decoder_segmentation.py`)
  - `WeightedCombinedLoss` + conv decoder pattern (currently inline in SAM/CineMA conv_decoder scripts)
  - `DINOv3UNetRDecoder` (from `dense_unetr_probe.py`)
  - `CineMAUNetRDecoder` (from `dense_unetr_probe.py`)
- `build_decoder(config, backbone_info) -> nn.Module` — decoder factory function
  - Takes backbone type, embed_dim, decoder type from config
  - Returns appropriate decoder with correct channel/skip configuration
- `FinetunableDINOv3UNetR` wrapper (from `acdc_dino_unetr_full_finetune.py`)
  - Generalize to `FinetunableModel(backbone, decoder)` that works for any backbone+decoder combo

### Step 1.3 — Create feature extraction & caching module
**New file**: `src/heartfm_evals/feature_cache.py`
- Consolidate all caching functions currently scattered:
  - `cache_features()` (2D DINOv3, from `dense_linear_probe.py`)
  - `cache_dino_volume_features()` (3D DINOv3, from `dense_unetr_probe.py`)
  - `cache_cinema_volume_features()` (3D CineMA, from `dense_unetr_probe.py`)
  - `cache_sam2_features()` (2D SAM2, currently inline in scripts)
  - `cache_sam2_volume_features()` (3D SAM2, currently inline in scripts)
- Unified interface: `cache_features(backbone, dataset, cache_dir, config) -> manifest`
  - Dispatches to correct extraction function based on backbone type + decoder dimensionality (2D vs 3D)
- Move `CachedFeatureDataset`, `CachedVolumeDataset`, `CachedCinemaVolumeDataset` here
- Move helper functions: `_pad_volume_z()`, `extract_multilayer_features()`, `extract_dino_volume_features()`, etc.

### Step 1.4 — Create dataset module
**New file**: `src/heartfm_evals/datasets.py`
- `load_dataset(config) -> (train_dataset, val_dataset, test_dataset)` — dataset factory
- Dataset-specific loaders registered by name:
  - `_load_acdc(data_dir, ...)` — current ACDC loading logic (train/val/test split, pathology-stratified val)
  - `_load_mnm(data_dir, ...)` — stub for M&M dataset (same CineMA format: EndDiastoleEndSystoleDataset)
- Move `ACDCSliceDataset` here (from `dense_linear_probe.py`)
- Move `RawVolumeDataset` here (from `acdc_dino_unetr_full_finetune.py`)
- Common val-split logic: `split_train_val(meta_df, strategy="pathology_stratified", val_frac=0.1)`
- Returns metadata alongside datasets (val_pids, pathology_map, etc.)
- **Metadata helpers shared with classification** (move from `classification_probe.py` and `finetune_classification.py`):
  - `get_pathology_map(meta_df) -> dict[str, str]` — used by both classification and segmentation for val-split stratification
  - `_group_samples_by_patient(cinema_dataset, pathology_map) -> list[dict]` — used by classification; useful for any patient-level grouping

### Step 1.5 — Create losses module
**New file**: `src/heartfm_evals/losses.py`
- Move all loss classes:
  - `DiceLoss` (from `dense_linear_probe.py`)
  - `CombinedLoss` (from `dense_linear_probe.py`)
  - `MaskedVolumeLoss` (from `dense_unetr_probe.py`)
  - `WeightedCombinedLoss` (currently duplicated in conv_decoder scripts)
- `build_loss(config, class_weights) -> nn.Module` — loss factory
- `compute_class_weights(dataset_or_manifest, num_classes) -> Tensor` — shared weight computation

### Step 1.6 — Create training module
**New file**: `src/heartfm_evals/training.py`
- Move and unify training loops:
  - `train_one_epoch()` (2D, from `dense_linear_probe.py`)
  - `train_one_epoch_vol()` (3D, from `dense_unetr_probe.py`)
  - Fine-tune epoch logic (from `acdc_dino_unetr_full_finetune.py`)
- Unified `train_one_epoch(model, dataloader, criterion, optimizer, device, is_volume=False)` that dispatches based on mode
- Move `evaluate()` and `evaluate_vol()` into this module
- `run_training_loop(model, train_loader, val_loader, config) -> (best_model_state, history)` — full loop with early stopping, scheduling, checkpointing. Generalise the pattern from `finetune_classification._train_with_lr()` which already has epoch→train→eval→track-best→early-stop logic.
- Optional W&B integration: `if config.logging.wandb.enabled: wandb.log({...})`

### Step 1.7 — Create visualization module
**New file**: `src/heartfm_evals/visualization.py`
- `plot_training_curves(history, output_dir)` — loss and validation Dice plots
- `plot_test_predictions(model, test_loader, output_dir, config)` — prediction overlay grids
- `overlay_labels()` (move from `dense_linear_probe.py`)
- `save_results_csv(metrics, output_dir, config)` — standardized result saving
- `CLASS_COLORS`, `CLASS_NAMES` constants stay in a shared `constants.py` or here

### Step 1.8 — Slim down original modules
- `dense_linear_probe.py` becomes thin: re-exports from new submodules for backward compat
- `dense_unetr_probe.py` becomes thin: re-exports from new submodules for backward compat
- Existing notebooks/scripts that import from the old modules continue to work

**Relevant files to modify/create**:
- Create: `src/heartfm_evals/backbones.py`, `decoders.py`, `feature_cache.py`, `datasets.py`, `losses.py`, `training.py`, `visualization.py`
- Modify: `src/heartfm_evals/dense_linear_probe.py` — slim to re-exports
- Modify: `src/heartfm_evals/dense_unetr_probe.py` — slim to re-exports
- Modify: `src/heartfm_evals/__init__.py` — register new submodules
- Future consumers (deferred): `src/heartfm_evals/classification_probe.py` and `finetune_classification.py` — will import `preprocess_slice`, `pad_volume_z`, `preprocess_sam_slice`, `get_pathology_map`, `_group_samples_by_patient` from the new modules instead of duplicating them

---

## Phase 2: Hydra Config Structure

**Goal**: Define the config hierarchy so that any backbone × decoder × dataset × training mode combo is expressible.

### Step 2.1 — Create config directory structure
```
configs/
└── segmentation/
    ├── config.yaml                  # Main defaults file
    ├── backbone/
    │   ├── dinov3_vits16.yaml
    │   ├── dinov3_vitb16.yaml
    │   ├── dinov3_vitl16.yaml
    │   ├── cinema.yaml
    │   ├── sam2_hiera_tiny.yaml
    │   ├── sam2_hiera_small.yaml
    │   ├── sam2_hiera_base_plus.yaml
    │   └── sam2_hiera_large.yaml
    ├── decoder/
    │   ├── linear_probe.yaml
    │   ├── conv_decoder.yaml
    │   └── unetr.yaml
    ├── dataset/
    │   ├── acdc.yaml
    │   └── mnm.yaml
    └── training/
        ├── frozen.yaml
        └── finetune.yaml
```

### Step 2.2 — Define main config.yaml (Hydra defaults)
```yaml
defaults:
  - backbone: dinov3_vits16
  - decoder: linear_probe
  - dataset: acdc
  - training: frozen
  - _self_

seed: 42
output_dir: results/segmentation/${backbone.name}_${decoder.name}_${dataset.name}_${training.mode}

logging:
  wandb:
    enabled: false
    project: heartfm-evals-segmentation
    entity: null
```

### Step 2.3 — Define backbone configs
Each YAML specifies: name, type (dinov3/cinema/sam2), embed_dim, n_layers, layer_indices, weights_path, and any model-specific params (e.g. repo_dir for DINOv3, hf_model_id for SAM2).

### Step 2.4 — Define decoder configs
Each YAML specifies: name, type (linear_probe/conv_decoder/unetr), dimensionality (2d/3d), channels, dropout, and decoder-specific params (e.g. dec_chans, patch_size for UNetR).

### Step 2.5 — Define dataset configs
Each YAML specifies: name, data_dir, num_classes, class_names, val_split strategy, and transform config (normalization, augmentation).

### Step 2.6 — Define training configs
`frozen.yaml`: cache_features=true, freeze_backbone=true, lr for head only, batch_size, epochs, patience
`finetune.yaml`: cache_features=false, freeze_backbone=false, backbone_lr, decoder_lr, batch_size, epochs, patience

---

## Phase 3: Entry-Point Script

**Goal**: Single script that reads Hydra config and orchestrates the full pipeline.

### Step 3.1 — Create main training script
**New file**: `scripts/segmentation/train.py`
- Entry point with `@hydra.main(version_base=None, config_path="../../configs/segmentation", config_name="config")`
- Pipeline stages:
  1. Set seed
  2. `backbone = load_backbone(cfg.backbone)` 
  3. `train_ds, val_ds, test_ds, metadata = load_dataset(cfg.dataset)`
  4. If `cfg.training.cache_features`:
     - `manifest = cache_features(backbone, datasets, cfg)` 
     - Build `CachedFeatureDataset` / `CachedVolumeDataset` from manifest
  5. Else (fine-tuning):
     - Build `RawVolumeDataset` / `ACDCSliceDataset` directly
  6. `decoder = build_decoder(cfg.decoder, backbone_info)`
  7. If fine-tuning: wrap in `FinetunableModel(backbone, decoder)`
  8. `criterion = build_loss(cfg, class_weights)`
  9. `history = run_training_loop(model, train_loader, val_loader, cfg)`
  10. `metrics = evaluate(model, test_loader, device)`
  11. `plot_training_curves(history, output_dir)`
  12. `plot_test_predictions(model, test_loader, output_dir, cfg)`
  13. `save_results_csv(metrics, output_dir, cfg)`
  14. Optional W&B summary logging

### Step 3.2 — Example usage patterns
```bash
# Default: DINOv3 ViT-S + linear probe on ACDC (frozen)
python scripts/segmentation/train.py

# DINOv3 ViT-B + UNetR on ACDC (frozen)
python scripts/segmentation/train.py backbone=dinov3_vitb16 decoder=unetr

# CineMA + conv decoder on ACDC (frozen)
python scripts/segmentation/train.py backbone=cinema decoder=conv_decoder

# SAM2 base + UNetR on ACDC (frozen)
python scripts/segmentation/train.py backbone=sam2_hiera_base_plus decoder=unetr

# DINOv3 ViT-S + UNetR on ACDC (full fine-tuning)
python scripts/segmentation/train.py backbone=dinov3_vits16 decoder=unetr training=finetune

# Any combo on M&M dataset
python scripts/segmentation/train.py dataset=mnm backbone=cinema decoder=unetr

# With W&B logging
python scripts/segmentation/train.py logging.wandb.enabled=true

# Hydra multirun for sweeping backbones
python scripts/segmentation/train.py -m backbone=dinov3_vits16,dinov3_vitb16,cinema
```

---

## Phase 4: M&M Dataset Stub

### Step 4.1 — Add M&M dataset loader
- Add `_load_mnm()` in `src/heartfm_evals/datasets.py`
- Uses same `EndDiastoleEndSystoleDataset` from CineMA with M&M data paths
- M&M config defines different class mapping if needed (same 4 ACDC classes for M&M cardiac segmentation)
- Stub raises `NotImplementedError` with instructions if data dir doesn't exist

### Step 4.2 — Add M&M config
**New file**: `configs/segmentation/dataset/mnm.yaml`
- Same structure as acdc.yaml but with M&M-specific defaults (data_dir, num_classes, val_split strategy)

---

## Phase 5: Testing & Verification

### Step 5.1 — Unit tests for new modules
- `tests/test_backbones.py` — test `load_backbone()` returns correct model type/dim for each config
- `tests/test_decoders.py` — test `build_decoder()` output shapes for each backbone+decoder combo
- `tests/test_datasets.py` — test `load_dataset()` returns correct splits
- `tests/test_losses.py` — test loss functions with known inputs

### Step 5.2 — Integration smoke test
- Shell script: `scripts/segmentation/smoke_test.sh`
- Run `train.py` with very small N_EPOCHS=1 and subset of data for each backbone × decoder combo
- Verify: no crashes, outputs saved correctly, metrics computed

### Step 5.3 — Backward compatibility test
- Verify existing notebook imports still work: `from heartfm_evals.dense_linear_probe import DenseLinearProbe, cache_features, ...`
- Verify existing notebook imports from `dense_unetr_probe` still work

---

## Steps & Dependencies Summary

| Step | Description | Depends On | Parallel? |
|------|-------------|------------|-----------|
| 1.1 | backbones.py | — | Yes (parallel with 1.4, 1.5, 1.7) |
| 1.2 | decoders.py | — | Yes (parallel with 1.1) |
| 1.3 | feature_cache.py | 1.1 (backbone interface) | After 1.1 |
| 1.4 | datasets.py | — | Yes (parallel with 1.1) |
| 1.5 | losses.py | — | Yes (parallel with 1.1) |
| 1.6 | training.py | 1.2, 1.3, 1.5 | After Phase 1 core |
| 1.7 | visualization.py | — | Yes (parallel with 1.1) |
| 1.8 | Slim original modules | 1.1–1.7 | After all Phase 1 |
| 2.1–2.6 | Hydra configs | 1.1, 1.2, 1.4 (need to know param names) | After Phase 1 |
| 3.1 | Entry-point train.py | Phase 1 + Phase 2 | After Phase 2 |
| 4.1–4.2 | M&M stub | 1.4, 2.5 | Parallel with Phase 3 |
| 5.1–5.3 | Testing | Phase 3 | After Phase 3 |

---

## Decisions
- **Drop SAM v1 and SAM v3**: Only SAM 2 supported going forward
- **Fine-tuning for all combos**: Architecture supports any backbone+decoder fine-tuning (not just DINOv3+UNetR)
- **Backward compatibility**: Old imports via `dense_linear_probe` / `dense_unetr_probe` preserved through re-exports
- **Old scripts preserved**: Existing scripts in `scripts/segmentation/` left as-is (they still work). New pipeline is additive.
- **Dataset extensibility**: Adding a new dataset = one new YAML config + one `_load_X()` function in `datasets.py`

## Scope Boundaries
- **Included**: Segmentation pipeline restructuring, Hydra configs, backbone/decoder/dataset factories, visualization module, W&B optional logging, M&M stub, unit tests
- **Excluded**: Distributed training, Mask2Former head, data augmentation pipeline (keep existing simple transforms), hyperparameter sweep (use Hydra multirun instead)
- **Deferred (classification follow-up)**: Once the new shared modules (`backbones.py`, `datasets.py`) exist, `classification_probe.py` and `finetune_classification.py` should be refactored to import from them — eliminating their duplicated backbone loading, preprocessing, z-padding, and SAM forward-pass logic. This is out of scope for now but the shared modules must be designed with both tasks in mind.

## Further Considerations
1. **CineMA's multi-view support**: CineMA natively supports SAX + LAX views. The current scripts only use SAX. Should we keep SAX-only scope, or plan the config to support multi-view? Recommend: SAX-only for now, but structure configs so `views: [sax]` is configurable.
2. **Gradient checkpointing**: CineMA's pipeline uses `model.set_grad_ckpt(True)` for memory-efficient fine-tuning. Recommend: add as `training.grad_ckpt: true` in `finetune.yaml`.
3. **Feature cache invalidation**: Currently caches are never invalidated. If backbone weights change, stale caches could cause silent bugs. Recommend: include a hash of (backbone_name + weights_path) in cache directory name.
