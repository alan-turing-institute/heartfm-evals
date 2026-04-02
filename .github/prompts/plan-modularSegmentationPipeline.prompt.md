## Plan: Modular Segmentation Pipeline with Hydra

Restructure 15+ standalone ACDC segmentation scripts into a single config-driven pipeline. One entry-point script (`scripts/segmentation/train.py`) plus per-dataset configs in `configs/segmentation/`. Backbones (DINOv3, CineMA, SAM2) × decoders (linear_probe, conv_decoder, unetr) × training mode (frozen/finetune) are all selected via Hydra config overrides. Shared logic moves into `src/heartfm_evals/` submodules. Plotting moves to a dedicated visualization module. Optional W&B logging added.

---

### Phase 1: Refactor `src/heartfm_evals/` into submodules

**Goal**: Break the two monolith modules (`dense_linear_probe.py`, `dense_unetr_probe.py`) into focused submodules while preserving backward compatibility.

**Step 1.1 — Create backbone loader module** *(parallel with 1.4, 1.5, 1.7)*
- **New file**: `src/heartfm_evals/backbones.py`
- `load_backbone(config) -> nn.Module` — unified backbone factory dispatching to:
  - `_load_dinov3(model_name, weights_path, repo_dir)` — wraps `torch.hub.load()`
  - `_load_cinema(hf_cache_dir, auto_download)` — wraps `CineMA.from_pretrained()`
  - `_load_sam2(model_id, hf_cache_dir, auto_download)` — wraps `Sam2Model.from_pretrained()` + `Sam2Processor`
- `freeze_backbone()` / `unfreeze_backbone()` helpers
- Move `MODEL_CONFIGS` dict here (currently in `dense_linear_probe.py`); add SAM2 variant configs (embed_dim, layer_indices per variant)

**Step 1.2 — Create decoder module** *(parallel with 1.1)*
- **New file**: `src/heartfm_evals/decoders.py`
- Move existing decoder classes into this module:
  - `DenseLinearProbe` (from `dense_linear_probe.py`)
  - `DINOdenseDecoderProbe` / `WeightedCombinedLoss` conv decoder (currently inline in the conv_decoder scripts)
  - `DINOv3UNetRDecoder`, `CineMAUNetRDecoder` (from `dense_unetr_probe.py`)
- `build_decoder(config, backbone_info) -> nn.Module` — factory function
- Generalize `FinetunableDINOv3UNetR` (from `acdc_dino_unetr_full_finetune.py`) into `FinetunableModel(backbone, decoder)` that works for any backbone+decoder combo

**Step 1.3 — Create feature extraction & caching module** *(depends on 1.1)*
- **New file**: `src/heartfm_evals/feature_cache.py`
- Consolidate all caching functions currently scattered across scripts and modules:
  - `cache_features()` (2D DINOv3), `cache_dino_volume_features()` (3D DINOv3), `cache_cinema_volume_features()` (3D CineMA), inline SAM2 caching
- Unified interface: `cache_features(backbone, dataset, cache_dir, config) -> manifest`
- Move `CachedFeatureDataset`, `CachedVolumeDataset`, `CachedCinemaVolumeDataset` here
- Move helpers: `_pad_volume_z()`, `extract_multilayer_features()`, etc.

**Step 1.4 — Create dataset module** *(parallel with 1.1)*
- **New file**: `src/heartfm_evals/datasets.py`
- `load_dataset(config) -> (train_ds, val_ds, test_ds, metadata)`
- Dataset-specific loaders registered by name:
  - `_load_acdc(data_dir)` — current ACDC logic (pathology-stratified val split)
  - `_load_mnm(data_dir)` — stub for M&M (same `EndDiastoleEndSystoleDataset` format)
- Move `ACDCSliceDataset`, `RawVolumeDataset` here
- Common split logic: `split_train_val(meta_df, strategy, val_frac)`

**Step 1.5 — Create losses module** *(parallel with 1.1)*
- **New file**: `src/heartfm_evals/losses.py`
- Move `DiceLoss`, `CombinedLoss`, `MaskedVolumeLoss`, `WeightedCombinedLoss`
- `build_loss(config, class_weights) -> nn.Module`
- `compute_class_weights(manifest, num_classes) -> Tensor`

**Step 1.6 — Create training module** *(depends on 1.2, 1.3, 1.5)*
- **New file**: `src/heartfm_evals/training.py`
- Unify `train_one_epoch()` (2D) and `train_one_epoch_vol()` (3D) into a single dispatcher
- Unify `evaluate()` and `evaluate_vol()` similarly
- `run_training_loop(model, train_loader, val_loader, cfg) -> (best_state, history)` — full loop with early stopping, scheduler, checkpointing
- Optional W&B logging: `if cfg.logging.wandb.enabled: wandb.log({...})`

**Step 1.7 — Create visualization module** *(parallel with 1.1)*
- **New file**: `src/heartfm_evals/visualization.py`
- `plot_training_curves(history, output_dir)`, `plot_test_predictions(model, test_loader, output_dir, cfg)`
- Move `overlay_labels()`, `CLASS_COLORS`, `CLASS_NAMES` here
- `save_results_csv(metrics, output_dir, cfg)` — standardized result saving

**Step 1.8 — Slim down original modules** *(depends on all of 1.1–1.7)*
- `dense_linear_probe.py` becomes thin: re-exports from new submodules for backward compat
- `dense_unetr_probe.py` becomes thin: re-exports similarly
- Existing notebooks/scripts that import from old modules continue to work unchanged

---

### Phase 2: Hydra Config Structure *(depends on Phase 1 interfaces)*

**Step 2.1 — Create config directory**
```
configs/
└── segmentation/
    ├── config.yaml              # Main Hydra defaults file
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

**Step 2.2 — Main config.yaml** (Hydra defaults composition)
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

**Steps 2.3–2.6** — Define individual YAML configs:
- **Backbone configs**: name, type, embed_dim, n_layers, layer_indices, weights_path, repo_dir / hf_model_id
- **Decoder configs**: name, type, dimensionality (2d/3d), channels, dropout, decoder-specific params (dec_chans, patch_size for UNetR)
- **Dataset configs**: name, data_dir, num_classes, class_names, val_split strategy, transform params
- **Training configs**: `frozen.yaml` (cache_features=true, freeze_backbone=true, head-only lr/batch/epochs), `finetune.yaml` (cache_features=false, freeze_backbone=false, backbone_lr, decoder_lr)

---

### Phase 3: Entry-Point Script *(depends on Phase 1 + Phase 2)*

**Step 3.1 — Create `scripts/segmentation/train.py`**
- `@hydra.main(version_base=None, config_path="../../configs/segmentation", config_name="config")`
- Pipeline:
  1. Set seed
  2. `backbone = load_backbone(cfg.backbone)`
  3. `train_ds, val_ds, test_ds, meta = load_dataset(cfg.dataset)`
  4. If `cfg.training.cache_features`: cache features → build cached datasets
  5. Else: build raw datasets (for fine-tuning)
  6. `decoder = build_decoder(cfg.decoder, backbone_info)`
  7. If fine-tuning: wrap in `FinetunableModel(backbone, decoder)`
  8. `criterion = build_loss(cfg, class_weights)`
  9. `history = run_training_loop(model, loaders, cfg)`
  10. `metrics = evaluate(model, test_loader)`
  11. Save plots, results CSV, optional W&B summary

**Example usage**:
```bash
# Default: DINOv3 ViT-S + linear probe on ACDC (frozen)
python scripts/segmentation/train.py

# DINOv3 ViT-B + UNetR (frozen)
python scripts/segmentation/train.py backbone=dinov3_vitb16 decoder=unetr

# CineMA + conv decoder (frozen)
python scripts/segmentation/train.py backbone=cinema decoder=conv_decoder

# Full fine-tuning
python scripts/segmentation/train.py backbone=dinov3_vits16 decoder=unetr training=finetune

# M&M dataset
python scripts/segmentation/train.py dataset=mnm backbone=cinema decoder=unetr

# Hydra multirun sweep
python scripts/segmentation/train.py -m backbone=dinov3_vits16,dinov3_vitb16,cinema
```

---

### Phase 4: M&M Dataset Stub *(parallel with Phase 3)*

- Add `_load_mnm()` in `datasets.py` using same `EndDiastoleEndSystoleDataset` format
- Add `configs/segmentation/dataset/mnm.yaml` with M&M defaults
- Stub raises `NotImplementedError` with instructions if data dir doesn't exist

---

### Phase 5: Testing & Verification *(after Phase 3)*

1. **Unit tests**: `tests/test_backbones.py`, `test_decoders.py`, `test_datasets.py`, `test_losses.py` — verify correct types, shapes, configs
2. **Integration smoke test**: Shell script running `train.py` with `N_EPOCHS=1` for each backbone × decoder combo
3. **Backward compatibility**: Verify existing imports from `dense_linear_probe` and `dense_unetr_probe` still work

---

### Relevant Files

| File | Action |
|------|--------|
| `src/heartfm_evals/backbones.py` | **Create** — backbone factory + MODEL_CONFIGS |
| `src/heartfm_evals/decoders.py` | **Create** — decoder factory + all decoder classes |
| `src/heartfm_evals/feature_cache.py` | **Create** — caching logic + cached datasets |
| `src/heartfm_evals/datasets.py` | **Create** — dataset factory (ACDC + M&M stub) |
| `src/heartfm_evals/losses.py` | **Create** — all loss classes + factories |
| `src/heartfm_evals/training.py` | **Create** — training loops + evaluation |
| `src/heartfm_evals/visualization.py` | **Create** — plotting + result saving |
| `src/heartfm_evals/dense_linear_probe.py` | **Modify** — slim to re-exports |
| `src/heartfm_evals/dense_unetr_probe.py` | **Modify** — slim to re-exports |
| `src/heartfm_evals/__init__.py` | **Modify** — register new submodules |
| `scripts/segmentation/train.py` | **Create** — Hydra entry point |
| `configs/segmentation/**/*.yaml` | **Create** — ~15 YAML config files |

### Verification

1. `pytest tests/test_backbones.py tests/test_decoders.py tests/test_datasets.py tests/test_losses.py` — unit tests pass
2. `python scripts/segmentation/train.py training.n_epochs=1` — smoke test for default config
3. `python scripts/segmentation/train.py -m backbone=dinov3_vits16,cinema decoder=linear_probe,unetr training.n_epochs=1` — multirun smoke test
4. `from heartfm_evals.dense_linear_probe import DenseLinearProbe, cache_features, MODEL_CONFIGS` — backward compat check
5. `pre-commit run -a` — code quality check

### Decisions

- **Drop SAM v1 and SAM v3** — only SAM 2 supported going forward
- **Fine-tuning for all combos** — architecture supports any backbone+decoder fine-tuning
- **Backward compatibility** — old imports preserved through re-exports in slimmed modules
- **Old scripts preserved** — existing scripts left as-is; new pipeline is purely additive
- **Dataset extensibility** — adding a new dataset = one YAML + one `_load_X()` function

### Further Considerations

1. **Multi-view support**: Current scripts only use SAX. CineMA supports SAX + LAX. Recommend: keep SAX-only for now, but make `views: [sax]` configurable in dataset YAML so multi-view can be added later without refactoring.
2. **Gradient checkpointing**: CineMA uses `model.set_grad_ckpt(True)` for memory-efficient fine-tuning. Recommend: add as `training.grad_ckpt: true` in `finetune.yaml`.
3. **Feature cache invalidation**: Stale caches could silently corrupt results if weights change. Recommend: include a hash of (backbone + weights_path) in the cache subdirectory name.
