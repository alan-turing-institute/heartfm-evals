## Plan: CMR Foundation Model Benchmark Framework

Build a structured evaluation framework comparing DINOv3 backbones (ViT-S/B/L, pre-trained on natural images) against CineMA (pre-trained on CMR) across 3 clinical cardiac tasks using 7 adaptation strategies on ACDC + M&Ms datasets. Hydra configs drive experiments; wandb tracks results; CLI scripts automate runs; notebooks handle analysis.

---

### Phase 1: Infrastructure & Data Pipeline

**Step 1.1 — M&Ms Preprocessing**
Create `src/heartfm_evals/data/mnms_preprocessing.py` to convert raw M&Ms NIfTI data to CineMA-compatible format (192×192, 1×1×10mm, intensity [0-255], labels remapped to {0=BG, 1=RV, 2=MYO, 3=LV}). Output: `processed/mnms/{train,val,test}/` + metadata CSVs.

**Step 1.2 — Hydra Config Structure**
Create `configs/` with hierarchical configs: `model/` (dinov3_vits16, vitb16, vitl16, cinema), `task/` (classification, segmentation, ef_prediction), `strategy/` (knn_classify, knn_segment, dense_probe, logreg, finetune_classify, finetune_segment), `dataset/` (acdc, mnms). Each captures task-specific hyperparams (C_range, k_range, lr, epochs).

**Step 1.3 — Backbone Registry & Unified Feature Extraction**
Create `src/heartfm_evals/backbones.py` with `load_backbone(cfg)` (DINOv3 via local hub + CineMA) and `extract_features(backbone, images, mode)` supporting CLS-token, dense-patch, and multi-layer modes. Refactors out of existing `classification_probe.py` and `dense_linear_probe.py`.

**Step 1.4 — Feature Caching**
Extend existing `cache_features()` from `dense_linear_probe.py` to support generalized cache paths (`feature_cache/{model}/{dataset}/{split}/`) and both CLS and dense features. Promote `CachedFeatureDataset` to shared module. Extract once per backbone, reuse across all strategies.

**Step 1.5 — Metrics & Tracking**
Create `src/heartfm_evals/metrics.py` (centralize `dice_score`, `macro_dice`, add `ejection_fraction`, `mae`, `rmse`, `r_squared`, `bland_altman`). Create `src/heartfm_evals/tracking.py` for wandb initialization and structured result logging.

---

### Phase 2: Evaluation Strategies

**Step 2.1 — kNN Classification** (*parallel with 2.2*)
`strategies/knn_classifier.py`: `KNeighborsClassifier` with cosine distance, k-sweep ∈ {1,3,5,10,20,50} on validation. Patient-level features via mean-pool ED ⊕ mean-pool ES → concat (reusing `build_patient_features()`). Dataset: ACDC 5-way pathology.

**Step 2.2 — Logistic Regression Classification** (*parallel with 2.1*)
`strategies/logreg_classifier.py`: Refactor existing `classification_probe.py` `sweep_C_and_train()` into strategy module. L2-regularized LogReg, 45-value C-sweep, L-BFGS solver. Dataset: ACDC.

**Step 2.3 — kNN Segmentation** (*depends on 1.3, 1.4*)
`strategies/knn_segmentation.py`: Per-pixel kNN on dense patch features. Flatten training patches + labels → build index → query test patches → upsample to 192×192. ~144K training vectors (manageable with sklearn; FAISS as optional acceleration). k-sweep on validation. Dataset: ACDC + M&Ms.

**Step 2.4 — Dense Linear Probe for Segmentation** (*parallel with 2.3*)
`strategies/dense_probe_segment.py`: Refactor existing `DenseLinearProbe` + `CombinedLoss` + training loop from `dense_linear_probe.py`. Add LR scheduling (cosine annealing), wandb logging. Multi-layer concat (layers 3,6,9,11) → bilinear upsample → 1×1 Conv. Dataset: ACDC + M&Ms.

**Step 2.5 — Dense Linear Probe for Classification** (*depends on 1.3*)
`strategies/dense_probe_classify.py`: Simple `nn.Linear` on CLS tokens (or global-avg-pooled features). PyTorch training loop with CE loss, LR scheduling. Patient-level aggregation at eval. Dataset: ACDC.

**Step 2.6 — End-to-End Fine-Tuning for Classification** (*depends on 1.3*)
`strategies/finetune_classifier.py`: Unfreeze DINOv3 backbone + linear head. Discriminative LR (backbone 1e-5, head 1e-3), linear warmup (5% steps), cosine annealing, mixed precision, gradient clipping (max_norm=1.0), early stopping. Dataset: ACDC.

**Step 2.7 — End-to-End Fine-Tuning for Segmentation** (*depends on 1.3, 2.4*)
`strategies/finetune_segmenter.py`: Unfreeze backbone + `DenseLinearProbe` head. Same training regime as 2.6. Data augmentation via MONAI (random flips, rotations, intensity jitter — only for fine-tuning, not frozen probes). Combined Dice + CE loss. Dataset: ACDC + M&Ms.

---

### Phase 3: EF Prediction Pipeline

**Step 3.1 — EF Ground Truth** (*depends on 1.1*)
`data/ef_labels.py`: Compute EF = (LVEDV − LVESV) / LVEDV × 100 from ground-truth segmentation masks. Count LV voxels × voxel volume. Add EF column to metadata CSVs.

**Step 3.2 — EF Regression Probe** (*depends on 3.1*)
`strategies/regression_probe.py`: Ridge regression on patient-level features (mean-pool ED ⊕ ES → concat). Alpha-sweep on validation. Evaluate: MAE, RMSE, R², Pearson correlation, Bland-Altman. Also SVR with RBF kernel as alternative.

**Step 3.3 — EF from Predicted Segmentations** (*depends on 2.4 or 2.7*)
After training segmentation, run inference on test volumes → compute EF from predicted LV volumes → compare to ground-truth EF. This evaluates the full pipeline (backbone → seg → clinical metric).

---

### Phase 4: CLI Runner & Benchmarking

**Step 4.1 — Unified CLI**
`src/heartfm_evals/run_eval.py`: Hydra entry point dispatching to strategy based on config. Invocable as:
`python -m heartfm_evals.run_eval model=dinov3_vitb16 task=classification strategy=knn_classify dataset=acdc`

**Step 4.2 — Batch Sweep**
`scripts/run_benchmark.sh`: Hydra multirun across all model × strategy × dataset combinations (~48-64 runs). Separate `scripts/run_finetune.sh` for GPU-intensive fine-tuning.

**Step 4.3 — Results Aggregation**
`src/heartfm_evals/results.py`: Pull wandb results → comparison tables (LaTeX/Markdown). `notebooks/model_comparison.ipynb` for radar charts, bar plots, statistical tests, and side-by-side with CineMA paper numbers.

---

### Phase 5: Testing & Documentation

**Step 5.1 — Unit Tests** (*parallel with Phase 2*)
`tests/test_backbones.py`, `test_metrics.py`, `test_strategies.py`, `test_datasets.py`, `conftest.py` (shared fixtures with tiny synthetic data). Target: >80% coverage on new modules.

**Step 5.2 — Notebooks**
Update existing classification/segmentation notebooks to use new API. Create `ef_analysis.ipynb` (Bland-Altman plots) and `model_comparison.ipynb` (cross-model visualizations).

**Step 5.3 — Documentation**
Update README with project goals & quick start. Add `docs/experiment_matrix.md` and `docs/reproducing_cinema.md`.

---

### Relevant Files to Modify/Reuse

- `src/heartfm_evals/classification_probe.py` — Extract `extract_cls_features()`, `build_patient_features()`, `sweep_C_and_train()`, `evaluate_classification()` into shared/strategy modules
- `src/heartfm_evals/dense_linear_probe.py` — Extract `DenseLinearProbe`, `CachedFeatureDataset`, `extract_multilayer_features()`, `cache_features()`, `CombinedLoss`, `preprocess_slice()`, `dice_score()`, `macro_dice()`
- `models/dinov3/hubconf.py` — Reference only (DINOv3 license; do not modify)
- `pyproject.toml` — Add FAISS (optional), add CLI entry point

---

### New Files to Create

```
src/heartfm_evals/
  backbones.py                         # Unified backbone loading & feature extraction
  metrics.py                           # All metric functions (Dice, accuracy, MAE, R², etc.)
  tracking.py                          # wandb integration
  run_eval.py                          # Hydra CLI entry point
  results.py                           # Results aggregation & comparison tables
  data/
    __init__.py
    datasets.py                        # Shared dataset wrappers (ACDC, M&Ms)
    mnms_preprocessing.py              # M&Ms raw → processed pipeline
    ef_labels.py                       # EF ground truth from segmentation masks
  strategies/
    __init__.py                        # Strategy registry
    knn_classifier.py                  # kNN for classification
    knn_segmentation.py                # kNN for dense segmentation
    logreg_classifier.py               # Logistic regression classification
    dense_probe_segment.py             # Dense linear probe segmentation
    dense_probe_classify.py            # Linear probe classification (PyTorch)
    regression_probe.py                # Ridge regression for EF
    finetune_classifier.py             # End-to-end fine-tuning classification
    finetune_segmenter.py              # End-to-end fine-tuning segmentation
configs/
  config.yaml
  model/*.yaml
  task/*.yaml
  strategy/*.yaml
  dataset/*.yaml
  trainer/*.yaml
scripts/
  run_benchmark.sh
  run_finetune.sh
tests/
  conftest.py
  test_backbones.py
  test_metrics.py
  test_strategies.py
  test_datasets.py
notebooks/
  ef_analysis.ipynb
  model_comparison.ipynb
```

---

### Verification

1. `pytest tests/ --cov=heartfm_evals` — all pass, >80% coverage
2. Smoke test: single CLI run completes on ACDC subset (5 patients)
3. Dense linear probe Dice matches existing notebook results (within 1%) after refactor
4. EF from GT segmentation masks matches known ACDC reference values
5. Feature cache consistency: new `backbones.py` output matches existing `.pt` files in `feature_cache/`
6. CineMA backbone runs through same pipeline and logs to same wandb project
7. Hydra multirun of 4 experiments completes and logs correctly

---

### Decisions

- **EF**: Derived from LV segmentation volumes (primary) + direct Ridge regression (secondary)
- **M&Ms**: Same preprocessing conventions as ACDC (192×192, 4-class, matched labels)
- **Feature caching**: Extract once per backbone, reuse across strategies
- **CineMA comparison**: Run CineMA through same pipeline for fair side-by-side
- **Data augmentation**: Only for end-to-end fine-tuning (2.6, 2.7); frozen probes use unaugmented data
- **DINOv3 license**: All derived files include required header; `hubconf.py` untouched

### Experiment Matrix

| Strategy | Classification (ACDC) | Seg (ACDC) | Seg (M&Ms) | EF (ACDC) |
|---|:---:|:---:|:---:|:---:|
| kNN | ✓ | ✓ | ✓ | — |
| Logistic/Ridge Regression | ✓ | — | — | ✓ (Ridge) |
| Dense Linear Probe (frozen) | ✓ | ✓ | ✓ | — |
| End-to-End Fine-Tuning | ✓ | ✓ | ✓ | — |
| EF from Predicted Segmentation | — | — | — | ✓ |

**Backbones**: DINOv3 ViT-S/16, ViT-B/16, ViT-L/16, CineMA
**Total**: ~4 backbones × 5 strategies × 3–4 task/dataset combos ≈ 48–64 runs
