# Segmentation refresher — 2026-08-09

State of the segmentation task after a period away from the codebase.

| | |
| --- | --- |
| Written | 2026-08-09 |
| `main` at | `3796713` |
| `sam2-classification` at | `c237fc8` |
| Merge base | `7517e7e` |

**One-line summary:** the code is in good shape and unified behind a single driver, but the two
branches have genuinely forked — `sam2-classification` holds the current segmentation results
(SAM v1) and `main` holds an older, higher-scoring SAM2 generation that only exists there.
Reproducibility is off (`set_seed()` is commented out), so cross-branch differences of
~0.005 macro Dice are noise.

---

## 1. Code path

One driver for the whole grid —
[scripts/segmentation/run_segmentation.py](../scripts/segmentation/run_segmentation.py):

```bash
python scripts/segmentation/run_segmentation.py \
    --dataset acdc --backbone dinov3 --dinov3-model-name dinov3_vits16 --decoder unetr
```

Flow: `detect_device()` → `load_backbone()` (frozen: `.eval()` + `requires_grad=False`) →
`load_segmentation_datasets()` → pre-extract features to `.pt` under
`feature_cache/{dataset}/{model}[_unetr3d]/` → train decoder with AdamW + cosine schedule and
early stopping on val macro Dice → evaluate → write results.

Outputs land in `results/segmentation/{dataset}/` as `{model}_{decoder}_{YYYYmmdd_HHMMSS}.json`
(config + per-class and macro Dice + full training history) with companion CSVs:

| Decoder | Companion CSVs |
| --- | --- |
| `linear_probe`, `conv_decoder` (2D) | `_per_slice.csv`, `_per_stack.csv` |
| `unetr` (3D) | `_per_slice.csv`, `_per_sample.csv` |

`_per_stack.csv` and `_per_sample.csv` have **identical schemas**
(`pid,frame,is_ed,n_slices,dice_BG,dice_RV,dice_MYO,dice_LV,macro_dice`); the two names are pure
redundancy from the 2D vs 3D code paths. This matters — see issue 3.

### Library pieces it leans on

| Module | Role |
| --- | --- |
| [backbones.py](../src/heartfm_evals/backbones.py) | `load_backbone()` factory. Per-model configs pin `embed_dim` and `layer_indices` (DINOv3 vits/vitb `(3,6,9,11)`, vitl `(5,11,17,23)`, SAM v1 `(2,5,8,11)`). All params frozen. |
| [features.py](../src/heartfm_evals/features.py) | Multi-layer 2D patch tokens (concat over layers → `(embed_dim*n_layers, 12, 12)`) and 3D volume features with z padded/truncated to `SAX_TARGET_DEPTH=16`. |
| [caching.py](../src/heartfm_evals/caching.py) | Writes/reads the `.pt` cache. Keyed by model **and** a `layers_<sorted-indices>` subdirectory so different layer selections can't collide. Existing files are skipped, so extraction is resumable. CineMA caches carry no layers tag. |
| [decoders.py](../src/heartfm_evals/decoders.py) | `get_decoder()` → `DenseLinearProbe` (dropout→BN→1×1 conv→bilinear to 192²), `ConvDecoderProbe` (upsample→2×conv3×3→1×1 head), `DINOv3UNetRDecoder` (also used for SAM; 4 layers → skip adapters + strided conv bottleneck → CineMA `UpsampleDecoder`), `CineMAUNetRDecoder`. |
| [losses.py](../src/heartfm_evals/losses.py) | `CombinedLoss` (CE+Dice) for linear probe, `WeightedCombinedLoss` for conv, `MaskedVolumeLoss` for 3D (masks padded z-slices). |
| [metrics.py](../src/heartfm_evals/metrics.py) | `macro_dice` = mean over RV/MYO/LV, **BG excluded**. `per_sample_dice_score` returns NaN when a class is absent from GT; `per_sample_macro_dice` uses `nanmean`. |
| [data.py](../src/heartfm_evals/data.py) | `load_segmentation_datasets()`. M&Ms/M&Ms-2 have a `val_metadata.csv`; ACDC does not, so a val split of 2 patients per pathology is carved from train with a fixed `split_seed`. |

### Analysis chain

```
run_segmentation.py  →  results/segmentation/{dataset}/*.json
                          ↓
                     build_summary.py          →  {dataset}/summary.csv
                          ↓
                     aggregate_summary.py      →  summary_aggregated.csv
                                                  macro_dice_by_{backbone,decoder}.png
                          ↓
                     report_macro_dice_ci.py   →  {dataset}/macro_dice_ci_summary.csv
                          ↓
                     plot_dataset_macro_dice_ci.py  →  {dataset}/macro_dice_ci_by_*.png
```

The last two exist only on `sam2-classification`.

SLURM: `batch_run_{cinema,dino,sam}_{linear_probe,conv_decoder,unetr}_segmentation.sh`, array
jobs indexed as `dataset_idx = ID / N_CONFIGS`, `config_idx = ID % N_CONFIGS`. Walltimes 2h /
4h / 8h. `run_all_segmentation.sh` is the serial local equivalent.

### Test coverage

Only [tests/test_segmentation_metrics.py](../tests/test_segmentation_metrics.py): the NaN
semantics of the per-sample metrics, the four `evaluate_*` reporting variants, and two
end-to-end tests that load the driver by path with everything monkeypatched to assert the
JSON/CSV output contract. Nothing covers `backbones.py`, `features.py`, cache-key logic,
decoder shapes, or losses. There is no CI workflow.

---

## 2. Branch situation

Both branches are ahead of the merge base `7517e7e`. This is a real fork, not a fast-forward.

**`sam2-classification`** — 13 commits, 2026-04-28 → 2026-08-09. The name is misleading: this
branch **removes SAM2 entirely** and standardizes on **SAM v1** (`facebook/sam-vit-base`,
`-large`, `-huge`) so the comparison against DINOv3 is architecture-matched (plain ViT vs plain
ViT, rather than DINOv3 vs SAM2's Hiera pyramid). Rationale is recorded in
`prompts/sam2_decisions.md` — **which exists only on that branch**; read it with
`git show sam2-classification:prompts/sam2_decisions.md`.

What it changed:

- `SAM2_CONFIGS` + `_load_sam2` deleted from `backbones.py`; `extract_sam2_2d_features` →
  `extract_sam_2d_features`; `cache_sam2_2d_features` → `cache_sam_2d_features`. `_load_sam`
  gained `"layer_indices": (2, 5, 8, 11)` so SAM v1 can drive segmentation.
- Driver: `--backbone {dinov3,cinema,sam}`, `--sam-model-id` (default `facebook/sam-vit-base`).
- Batch scripts: `batch_run_sam2_*` deleted, `batch_run_sam_*` added; SLURM arrays shrink from
  `0-11` (4 SAM2 models) to `0-8` (3 SAM v1 models).
- **Behavioural fix** (`6ead620`): the resume/skip check now globs for any existing
  `{base_name}_*.json` instead of testing a path built from a freshly generated timestamp — the
  old check was dead code that could never fire. Consequence: re-running a config now requires
  deleting the old JSON first.
- Re-ran the **entire** grid, not just SAM — CineMA and DINOv3 results are all from
  2026-04-29 → 05-05.
- Added the bootstrap-CI outputs and `plot_dataset_macro_dice_ci.py`.
- `decoders.py` changes are cosmetic only (docstrings, two redundant `import torch` removed).

`metrics.py`, `data.py`, `losses.py` and `training.py` are **unchanged** between branches.

**`main`** — 3 commits, 2026-05-25 → 2026-08-09: added `aggregate_summary.py` and regenerated
the summaries/PNGs over the older **SAM2-era** results (runs from 2026-04-18 / 04-21 / 04-22),
plus a `.gitignore` tweak. Both branches added `aggregate_summary.py` independently, so it is a
guaranteed merge conflict; the branch version is a strict superset (colour fallback for unknown
backbones, larger bar labels, legend moved).

**Segmentation source of truth is `sam2-classification`.** `main`'s numbers are a stale
snapshot.

---

## 3. Results

Both branches have a complete 7 backbones × 3 decoders × 3 datasets grid (63 runs). Uniform
hyperparameters throughout: `lr=1e-3, weight_decay=1e-3, batch_size=16, n_epochs=100,
patience=20, seed=0`. Single seed, no repeats.

### Mean macro Dice across ACDC + M&Ms + M&Ms-2

`sam2-classification` — `results/segmentation/summary_aggregated.csv`:

| Backbone | Linear | Conv | UNETR |
| --- | --- | --- | --- |
| CineMA | 0.6781 | 0.7134 | **0.8706** |
| DINOv3 vits16 | 0.7335 | 0.8513 | 0.8533 |
| DINOv3 vitb16 | 0.7604 | 0.8566 | 0.8544 |
| DINOv3 vitl16 | 0.7619 | **0.8586** | 0.8574 |
| SAM v1 vit_base | 0.7338 | 0.8298 | 0.8410 |
| SAM v1 vit_large | 0.6949 | 0.8053 | 0.8348 |
| SAM v1 vit_huge | 0.6867 | 0.8106 | 0.8245 |

`main` (SAM2 rows only, for comparison — CineMA/DINOv3 rows differ from the above by
≤0.005, i.e. rerun noise):

| Backbone | Linear | Conv | UNETR |
| --- | --- | --- | --- |
| SAM2 hiera_small | 0.7583 | 0.8375 | 0.8485 |
| SAM2 hiera_base_plus | 0.7501 | 0.8354 | 0.8465 |
| SAM2 hiera_large | **0.7708** | **0.8445** | **0.8542** |

### Per-dataset, UNETR row (`sam2-classification`)

| Backbone | ACDC | M&Ms | M&Ms-2 |
| --- | --- | --- | --- |
| CineMA | **0.8978** | **0.8529** | **0.8612** |
| DINOv3 vitl16 | 0.8796 | 0.8392 | 0.8534 |
| DINOv3 vitb16 | 0.8801 | 0.8354 | 0.8477 |
| SAM v1 vit_base | 0.8570 | 0.8211 | 0.8449 |
| SAM v1 vit_large | 0.8589 | 0.8150 | 0.8304 |

### Bootstrap CIs (`sam2-classification` only)

`{dataset}/macro_dice_ci_summary.csv`, 10 000-resample percentile bootstrap, 95%, sample-level.
These **are** current with the branch's own runs — every referenced timestamp matches a
committed JSON on that branch (one orphan row excepted, see issue 6). ACDC:

| Model | Mean | 95% CI |
| --- | --- | --- |
| CineMA UNETR | 0.8829 | [0.8761, 0.8892] |
| DINOv3 vitb16 UNETR | 0.8624 | [0.8541, 0.8701] |
| DINOv3 vitl16 UNETR | 0.8611 | [0.8522, 0.8693] |
| SAM vit_large UNETR | 0.8363 | [0.8252, 0.8465] |
| SAM vit_base UNETR | 0.8356 | [0.8244, 0.8461] |
| SAM vit_huge UNETR | 0.8163 | [0.8035, 0.8282] |

Note these means sit ~0.015 below the corresponding `summary.csv` figures. That is expected, not
an inconsistency: the CI table averages per-volume `nanmean` macro Dice, while `summary.csv`
reports pooled/global Dice over all voxels. Don't mix the two in a write-up.

### What the numbers say

1. **Decoder capacity dominates backbone choice.** Linear 0.68–0.78 vs Conv/UNETR 0.81–0.87.
   The spread across decoders is far larger than the spread across backbones.
2. **CineMA is worst with weak decoders and best with UNETR** (0.678 → 0.871). Partly real —
   it is the only cardiac-pretrained, natively 3D model — and partly an artefact, see issue 2.
3. **DINOv3 scales mildly with size** (vits → vitl ≈ +0.005–0.03 depending on decoder), and is
   the strongest general-purpose backbone throughout.
4. **SAM v1 scales inversely**: vit_base > vit_large > vit_huge on nearly every
   decoder × dataset cell. Worth understanding before it goes in a paper.
5. **SAM2 beat SAM v1 across the board** (UNETR 0.854 vs 0.841). The branch traded raw score for
   a cleaner architecture-matched comparison.
6. MYO is consistently the hardest class (0.52–0.86), LV the easiest (0.76–0.94). BG sits at
   0.977–0.993 and is excluded from macro Dice everywhere.

---

## 4. Known problems

1. **Reproducibility is off.** `set_seed(args.seed)` is commented out at
   `scripts/segmentation/run_segmentation.py:177` on **both** branches, while `"seed": 0` is
   still written into every results JSON. Decoder init, dropout and BatchNorm are unseeded; only
   the ACDC val split and the train-loader shuffle generator are deterministic. Identical
   configs re-run across branches differ by 0.001–0.005 macro Dice — the **same magnitude as
   the model-size effects being reported**. With a single seed and no repeats there is currently
   nothing committed that could separate the two.
2. **CineMA Linear/Conv are not apples-to-apples.** They use `Probe Layers = 0` (final layer
   only) while every other backbone's Conv/UNETR uses 4 multi-scale layers, and CineMA-UNETR
   itself uses `3,6,9,11`. This largely explains the "CineMA worst then best" pattern.
3. **CIs cover only UNETR.** `report_macro_dice_ci.py` defaults to `--pattern '*_per_sample.csv'`
   (line 255), which only the 3D decoder emits — linear_probe and conv_decoder write
   `*_per_stack.csv` with an identical schema and are silently excluded. Unifying the two
   filenames, or just changing the default pattern, would fill this in. Relatedly,
   `infer_aggregation_unit` labels anything containing `n_slices` as `"sample"`, so stacks are
   mislabelled.
4. **`--use-layers` is broken for `conv_decoder`.** The driver only passes `cached_layers` for
   `linear_probe`, so `get_decoder` sizes `ConvDecoderProbe` as `embed_dim * len(use_layers)`
   while the cache still holds all four layers → channel mismatch at the first forward. Same
   class of failure for `unetr` if `--use-layers` names layers absent from the cache.
5. **`sam2.1-hiera-tiny` was never run.** It appears in `run_all_segmentation.sh` and the SAM2
   batch scripts (arrays sized `0-11`) on `main`, but has zero committed results — the grid is
   9 runs short of what the scripts claim.
6. **Orphan file on the branch:**
   `results/segmentation/acdc/dinov3_vits16_unetr_20260429_170635_per_sample.csv` has no
   matching JSON or `_per_slice.csv`; it is a leftover from a superseded run (replaced by
   `..._20260430_070132`) and it leaks into the ACDC CI table as a row with blank
   backbone/model/decoder metadata.
7. **`README.md` is stale for segmentation** — it documents scripts that no longer exist
   (`run_sam_variants.sh`, `run_sam3.sh`, `run_dino_unetr_variants.sh`) and claims outputs are
   saved to `scripts/segmentation/`. Unlike classification, there is no
   `scripts/segmentation/README.md`.
8. **M&Ms-2 test set size differs between pipelines**: 3562 slices in the 2D `_per_slice.csv`
   files vs 3560 in the 3D ones. Small, but it means UNETR and Conv are not evaluated on
   precisely the same slices on that dataset.

Minor: `compute_class_weights` re-`torch.load`s every training slice on each run; a debug
`print('Validation PIDs …')` remains in `data.py:101`; `dense_linear_probe.py` and
`dense_unetr_probe.py` are deprecated re-export shims.

---

## 5. Open decisions

**Merge the branches, or keep them forked?** A straight merge of `sam2-classification` into
`main` would delete:

- all SAM2 segmentation results (81 files: 9 combos × 3 files × 3 datasets, plus 15 rows in each
  `summary.csv` and 9 in `summary_aggregated.csv`);
- the three `batch_run_sam2_*_segmentation.sh` scripts;
- SAM2 library support entirely (`SAM2_CONFIGS`, `_load_sam2`, `extract_sam2_2d_features`,
  `cache_sam2_2d_features`) — after which nothing in the repo can run SAM2 without reverting
  `e9dbed9`;
- the 2026-04-18/21 CineMA + DINOv3 baseline runs, superseded by the 04-29/05-05 reruns.

Since SAM2 scored *higher* than SAM v1, that is worth archiving deliberately rather than losing
in a merge. Conflicts to resolve by hand: `aggregate_summary.py` (take the branch version), the
three `summary.csv`, `summary_aggregated.csv`, and the two top-level PNGs.

**Keep SAM v1 only, or carry both?** The branch's architecture-matched argument is sound, but
reporting both would pre-empt the obvious reviewer question about why the stronger SAM variant
was dropped.

**Re-run with seeding before any write-up?** Issue 1 is the one that could actually invalidate a
claim. Turning `set_seed()` back on and running ≥3 seeds for at least the headline
backbone × decoder cells would let the model-size and backbone comparisons be stated with an
error bar rather than as point estimates.
