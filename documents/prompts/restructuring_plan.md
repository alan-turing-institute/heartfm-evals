**Context**: This repository (`heartfm-evals`) evaluates frozen foundation model backbones on cardiac MRI tasks (segmentation and classification). The codebase has grown organically and now has significant duplication. I want to restructure it into a modular framework. **Do not use Hydra for config management** — use argparse + dataclasses/dicts.

### Current State

**`scripts/segmentation/`** contains **13 standalone scripts**, each hard-coding a specific (backbone × decoder) combination on the ACDC dataset only:
- `acdc_dino_conv_decoder_segmentation.py`, `acdc_dino_dense_linear_probe_segmentation.py`, `acdc_dino_unetr_segmentation.py`
- `acdc_cinema_conv_decoder_segmentation.py`, `acdc_cinema_dense_linear_probe_segmentation.py`, `acdc_cinema_unetr_segmentation.py`
- `acdc_sam2_conv_decoder_segmentation.py`, `acdc_sam2_dense_linear_probe_segmentation.py`, `acdc_sam2_unetr_segmentation.py`
- `acdc_sam_conv_decoder_segmentation.py`, `acdc_sam_unetr_segmentation.py`

Each of these scripts independently re-implements the same boilerplate:
1. **Device detection** — identical `torch.backends.mps.is_available()` / `torch.cuda.is_available()` block
2. **ACDC data loading** — reads `train_metadata.csv` / `test_metadata.csv`, creates val split via `groupby("pathology").sample(n=2)`, instantiates `EndDiastoleEndSystoleDataset` with `ScaleIntensityd`
3. **Backbone loading** — DINOv3 via `torch.hub.load()`, CineMA via `CineMA.from_pretrained()`, SAM/SAM2 via HuggingFace `transformers`
4. **Feature caching** — each script has its own caching logic to `feature_cache/{model_name}/`
5. **Training loop** — train_one_epoch + evaluate + early stopping with patience, cosine LR schedule
6. **Visualization** — overlay predictions, training curves, result printing
7. **Decoder definition** — some scripts define their decoders inline (e.g. `DINOdenseDecoderProbe` in `acdc_dino_conv_decoder_segmentation.py`), rather than importing from `src/`

**`scripts/classification/`** is already better structured:
- `run_classification.py` — **reference implementation**: a single unified script with `--backbone {cinema,dinov3,sam}`, `--eval-mode {logreg,finetune}`, `--dataset {acdc,mnm,mnm2}` flags. Uses a `load_backbone()` factory function. **The `finetune` eval mode and `--no-freeze-backbone` option should be removed in the refactor** (see Constraints).
- `run_acdc_classification.py` — older, ACDC-only version (can be deprecated)
- `build_summary.py` — aggregates JSON results into summary CSV

**`src/heartfm_evals/`** has four modules that scripts import from:
- `dense_linear_probe.py` — `MODEL_CONFIGS`, `preprocess_slice()`, `extract_multilayer_features()`, `cache_features()`, `CachedFeatureDataset`, `DenseLinearProbe`, `DiceLoss`, `CombinedLoss`, `train_one_epoch()`, `evaluate()`, `dice_score()`, `macro_dice()`, `overlay_labels()`, plus constants (`IMAGE_SIZE`, `NUM_CLASSES`, `CLASS_NAMES`, etc.)
- `dense_unetr_probe.py` — `DINOv3UNetRDecoder`, `CineMAUNetRDecoder`, `CachedVolumeDataset`, `cache_dino_volume_features()`, `train_one_epoch_vol()`, `evaluate_vol()`
- `classification_probe.py` — `DATASET_PATHOLOGY_CLASSES`, `extract_cls_features()`, `cache_cls_features()`, `cache_cinema_cls_features()`, `cache_sam_cls_features()`, `build_patient_features()`, `sweep_C_and_train()`, `evaluate_classification()`
- `finetune_classification.py` — `ClassificationHead`, `finetune_sweep_and_train()`, `evaluate_finetune_classification()`. **This entire module should be removed** — only frozen-backbone evaluation is needed going forward.

### Key Architectural Differences Between Tasks

- **Feature extraction differs by task**: classification uses CLS token or global average pooling → `(embed_dim,)` per slice/volume; segmentation uses spatial patch tokens from multiple intermediate layers → `(embed_dim * n_layers, h, w)` per slice.
- **2D vs 3D**: The UNetR decoder operates on 3D volumes (features stacked along z-axis), while linear probe and conv decoder operate on 2D slices. CineMA natively processes 3D SAX volumes. Feature caching differs accordingly (`CachedFeatureDataset` for 2D, `CachedVolumeDataset` for 3D).
- **Decoder types for segmentation**: (a) Dense linear probe — 1×1 Conv2d (DINOv3 paper protocol), (b) Convolutional decoder — small CNN (Conv-BN-ReLU blocks), (c) UNetR — CineMA's `UpsampleDecoder` on pseudo-3D features. Each is currently defined in a different place.
- **Dataset scope**: Classification already supports ACDC, M&M, and M&M2 via `DATASET_PATHOLOGY_CLASSES` mapping and dataset-aware loading. Segmentation is currently ACDC-only.

### Inspiration: CineMA Package Structure

Analyze the CineMA package installed at `.venv/lib/python3.14/site-packages/cinema/` for its layered architecture:

```
cinema/
├── segmentation/
│   ├── __init__.py
│   ├── dataset.py          # EndDiastoleEndSystoleDataset (shared across datasets)
│   ├── train.py            # get_segmentation_model(), segmentation_loss(), segmentation_eval() (shared)
│   ├── eval.py             # segmentation_eval_edes_dataset() (shared evaluation logic)
│   ├── convunetr.py        # Model architecture
│   ├── unet.py             # Model architecture
│   ├── acdc/               # Dataset-specific: config.yaml, train.py (load_dataset + @hydra.main), eval.py
│   ├── mnms/               # Same structure
│   └── mnms2/              # Same structure
├── classification/
│   ├── __init__.py
│   ├── dataset.py          # EndDiastoleEndSystoleDataset (classification variant w/ class_col)
│   ├── train.py            # get_classification_or_regression_model(), classification_loss(), classification_eval()
│   ├── eval.py             # classification_eval_dataset()
│   ├── acdc/               # config.yaml, train.py (load_acdc_dataset + @hydra.main), eval.py
│   ├── mnms/
│   └── mnms2/
```

Key CineMA patterns to adopt (without Hydra):
- **Shared task-level modules** for dataset loading, training loops, evaluation, and metrics
- **Dataset-specific sub-packages** only for loading/splitting logic, not for model or training code
- **Factory functions** for models (`get_segmentation_model()`, `get_classification_or_regression_model()`)

### Goals

1. **Unified segmentation entry point**: Create a single `scripts/segmentation/run_segmentation.py` (mirroring the existing `scripts/classification/run_classification.py`) that accepts `--backbone {dinov3,cinema,sam2}`, `--decoder {linear_probe,conv_decoder,unetr}`, `--dataset {acdc,...}` and runs the full pipeline. No `--eval-mode` flag is needed for segmentation since there is no fine-tuning option.

2. **Shared backbone loading**: Extract the `load_backbone()` pattern from `run_classification.py` into `src/heartfm_evals/backbones.py` (or similar), so both classification and segmentation scripts use the same backbone factory. This should handle DINOv3 (multiple variants via `--dinov3-model-name`), CineMA, and SAM2.

3. **Shared dataset loading**: Abstract the repeated ACDC loading logic (metadata CSV parsing, val split creation, `EndDiastoleEndSystoleDataset` instantiation, `ScaleIntensityd` transform) into `src/heartfm_evals/data.py`. Extend to support M&M and M&M2 for segmentation (classification already handles this). Expose a function like `load_datasets(dataset_name, data_dir, split_seed=0) -> (train_ds, val_ds, test_ds)`.

4. **Decoder registry**: Move all segmentation decoders into `src/heartfm_evals/decoders.py` (or a `decoders/` subpackage): `DenseLinearProbe` (already in `dense_linear_probe.py`), `DINOdenseDecoderProbe` (currently inline in scripts), `DINOv3UNetRDecoder` and `CineMAUNetRDecoder` (currently in `dense_unetr_probe.py`). Provide a factory: `get_decoder(decoder_type, in_channels, num_classes, ...)`.

5. **Feature extraction unification**: The segmentation feature extraction (spatial multi-layer features) and classification feature extraction (CLS/GAP tokens) are fundamentally different. Keep them as separate functions, but consolidate the per-backbone extraction logic. Currently there are separate functions for DINOv3 (`extract_multilayer_features`), CineMA (custom per-script), and SAM2 (custom per-script). Unify these under a backbone-aware extraction API.

6. **Feature caching unification**: Consolidate the various caching patterns (`cache_features()`, `cache_cls_features()`, `cache_cinema_cls_features()`, `cache_sam_cls_features()`, `cache_dino_volume_features()`) into a coherent caching module that handles both 2D and 3D cases, both spatial and CLS features.

7. **Shared training utilities**: Extract the common training loop (train_one_epoch + evaluate + early stopping with patience + cosine LR schedule + best-checkpoint restoration) into `src/heartfm_evals/training.py`. Both tasks already use nearly identical patterns.

8. **Shared device detection**: Create a single `detect_device()` utility.

9. **Shared visualization**: The segmentation scripts all repeat overlay/training curve plotting. Consolidate into `src/heartfm_evals/visualization.py`.

10. **Results**: Extend the JSON result saving and summary building pattern from `scripts/classification/build_summary.py` to segmentation.

### Target `src/heartfm_evals/` Structure

Propose the new module layout under `src/heartfm_evals/`, showing which existing code maps to which new module. Account for the fact that `dense_linear_probe.py` is currently a "catch-all" that contains preprocessing, metrics, datasets, models, training loops, and constants — it needs to be decomposed.

### Constraints

- **Frozen backbones only**. Remove all fine-tuning functionality (`--eval-mode finetune`, `--no-freeze-backbone`, `finetune_classification.py`, `finetune_sweep_and_train()`, `evaluate_finetune_classification()`). The framework only evaluates frozen backbone features via downstream heads (logistic regression, linear probe, conv decoder, UNetR). Backbone parameters must always have `requires_grad = False`. The `ClassificationHead` linear layer used in fine-tuning is not needed; classification uses sklearn `LogisticRegression` on cached features.
- **No Hydra**. Use argparse for CLI and plain dicts/dataclasses for configuration.
- **Keep `models/dinov3/hubconf.py` and `model_weights/` unchanged** — they are external artifacts.
- **Preserve all existing functionality** (other than fine-tuning, which is being removed) — this is a refactor, not a rewrite. Existing shell scripts (e.g., `run_dino_variants.sh`) should still work, or have clear replacements.
- **Incremental migration path** — the plan should allow migrating one script at a time rather than requiring a big-bang rewrite.
- **Maintain DINOv3 licensing** — any files derived from DINOv3 code must keep the license header.

### Deliverable

Produce a phased plan with:
1. The proposed new `src/heartfm_evals/` module structure (directory tree)
2. For each new module: what functions/classes move there, and from which existing file
3. The new unified `run_segmentation.py` interface (CLI arguments)
4. Changes needed to `run_classification.py` to use the shared modules **and to remove the `finetune` eval mode** — the script should only support `--eval-mode logreg` (or drop the flag entirely since it's the only mode). Remove the `--freeze-backbone` / `--no-freeze-backbone` flag.
5. Migration order (which scripts to port first, dependencies between phases)
6. A mapping of current shell scripts to their new equivalents
