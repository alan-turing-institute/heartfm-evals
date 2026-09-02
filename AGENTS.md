# AGENTS.md — heartfm-evals

Instructions for coding agents. See also [README.md](README.md) and
[CONTRIBUTING.md](CONTRIBUTING.md) (full setup).

## What this project is

A research codebase (Alan Turing Institute) benchmarking **frozen foundation-model
backbones on cardiac MRI**: do frozen representations of a general-purpose vision model
already encode enough cardiac anatomy and function, or is a cardiac-pretrained model
needed?

Research-oriented — **not** production medical device software.

| Axis | Values |
| --- | --- |
| **Tasks** | segmentation (per-pixel RV/MYO/LV), classification (patient-level pathology) |
| **Backbones** | `dinov3` (self-supervised ViT), `sam` (SAM v1, supervised ViT), `sam2` (SAM 2.1 Hiera), `cinema` (cardiac-MRI MAE, 3D) — all four work for **both** tasks |
| **Datasets** | `acdc`, `mnm` (M&Ms), `mnm2` (M&Ms-2) |
| **Segmentation decoders** | `linear_probe` (1×1 conv, DINOv3 protocol), `conv_decoder` (2D CNN), `unetr` (3D UNetR on stacked features) |
| **Classification modes** | `logreg` (linear probe on frozen features), `finetune` (linear head, backbone frozen by default) |

## Layout

- [src/heartfm_evals/](src/heartfm_evals/) — the reusable library; everything shared lives here.
  Key modules: [backbones.py](src/heartfm_evals/backbones.py) (`load_backbone()` factory → frozen
  model + metadata), [features.py](src/heartfm_evals/features.py) and
  [caching.py](src/heartfm_evals/caching.py) (2D multi-layer patch-token extraction and `.pt`
  caches + the `Dataset` classes that read them — **segmentation only**),
  [decoders.py](src/heartfm_evals/decoders.py) (`get_decoder()` factory). Almost all
  classification logic — feature caching, patient-level assembly, the probe, and its metrics —
  lives in [classification_probe.py](src/heartfm_evals/classification_probe.py), not in
  `features.py`/`caching.py`/`metrics.py`. Plus `data`, `training`, `losses`, `metrics`,
  `constants`, `device`, `reproducibility`, `visualization`.
- [scripts/segmentation/](scripts/segmentation/), [scripts/classification/](scripts/classification/) — thin argparse drivers, SLURM batch scripts, analysis scripts
- [results/{task}/{dataset}/](results/) — committed JSON/CSV/PNG results
- [tests/](tests/) — pytest, all synthetic (no real models or data needed)
- [models/dinov3/](models/dinov3/) — vendored DINOv3 hub code (**DINOv3 license**)
- [model_weights/](model_weights/) — checkpoints; HF downloads to `model_weights/hf`
- [notebooks/](notebooks/) — exploratory, largely superseded by `scripts/`
- [prompts/](prompts/) — design/decision records
- [documents/refreshers/](documents/refreshers/) — dated walkthroughs of how a pipeline
  actually works; more current than prose elsewhere when the two disagree

## Entry points

One unified script per task — extend the library and the `choices=`, never fork a script
per backbone/decoder (that duplication is what `prompts/restructuring_plan.md` removed).

```bash
python scripts/segmentation/run_segmentation.py \
    --dataset acdc --backbone dinov3 --dinov3-model-name dinov3_vits16 --decoder unetr

python scripts/classification/run_classification.py \
    --dataset mnm --backbone cinema --eval-mode logreg --pooling cls
```

Data defaults to `../data/heartfm/processed/{dataset}/` — one level *above* the
repo (`train/`, `test/`, optional `val/`,
plus `*_metadata.csv`; ACDC has no val split so one is carved from train). Results go to
`results/{task}/{dataset}/` as `{model}_{decoder|mode}_{timestamp}.json` with sibling
`_per_slice` / `_per_stack` / `_per_sample` CSVs.

Post-hoc analysis rescans those files: `build_summary.py` (per-dataset `summary.csv`),
`aggregate_summary.py` / `report_macro_dice_ci.py` (cross-dataset Dice + bootstrap CIs),
`mcnemar_test.py` / `bootstrap_test.py` (pairwise significance between classifiers).
`batch_run_*.sh` and `run_all_*.sh` are SLURM array jobs for the full grid.

## Environment & workflow

- Python 3.11–3.13, **uv** only (`uv venv .venv && source .venv/bin/activate && uv sync --all-extras`); no bare pip. `CineMA` and `dinov3` install from git. `pre-commit install` once.
- `detect_device()` picks MPS → CUDA → CPU. Local dev is a Mac; real runs are SLURM jobs on **Isambard-AI** (`scripts/install.sh`).
- `pytest` to test (`pytest --cov=heartfm_evals` for coverage) — note `filterwarnings = ["error"]`, so warnings fail. `pre-commit run -a` before proposing changes (ruff + mypy on `src`).
- Type hints are expected; follow existing patterns in `src/heartfm_evals/`.
- Reproducibility is load-bearing: `set_seed()` fixes seeds and enables deterministic algorithms. Past commits chased float64/determinism bugs — don't quietly change seeding, dtype, or ordering in ways that shift published numbers.

## Licensing — read before touching DINOv3 code

**CRITICAL**: dual licensing.

1. **Original code** — MIT ([LICENSE](LICENSE))
2. **DINOv3 materials** — DINOv3 License ([LICENSE-DINOv3.md](LICENSE-DINOv3.md)), covering
   [models/dinov3/](models/dinov3/) and any DINOv3-derived code

When writing code that uses DINOv3 components:

- Include the DINOv3 license header at the top of the file, and never remove an existing one
- Reference [LICENSE-DINOv3.md](LICENSE-DINOv3.md) in documentation
- Don't make changes that violate DINOv3 terms (see sections 1.b.iv and 5)
- Don't reverse-engineer DINOv3 components

DINOv3 backbones are re-exported by [models/dinov3/hubconf.py](models/dinov3/hubconf.py)
(ViT-S/B/L/H/7B at patch 16, plus ConvNeXt variants); this repo uses `dinov3_vits16`,
`dinov3_vitb16`, `dinov3_vitl16`, and `dinov3_vit7b16`, with weights in
[model_weights/](model_weights/).

## Branches

`main` is default; several topic branches are kept as records, not work to merge.

- `sam2-classification` — **most active, ahead of `main`**: SAM v1 replaces SAM2 in the drivers, plus new plotting/aggregation scripts and regenerated results. Check here before assuming `main` is current.
- `isambard-ai`, `profiling` — HPC batch scripts, feature-extraction profiling
- `sign_tests_classification`, `classification_on_spark` — significance testing, largely landed in `main`
- `add_multiple_datasets`, `fine-tunning`, `10-unet-decoder`, `5-decoder-dense-segmentation`, `9-dino-for-linear-probe-classification`, `35-segmentation---adapt-the-dino-segmentation-head` — pre-refactor issue branches (segmentation was ACDC-only, one script per backbone/decoder)
- `old-code-for-comparison`, `temp1`–`temp3` — frozen snapshots for diffing old vs new numbers; not code to build on
- `levan-overleaf` — paper/write-up

## Gotchas

- Feature caches are keyed by model and layer set — change extraction without invalidating or repointing `--cache-dir` and you silently train on stale features.
- `--cache-only` on either driver extracts features and exits without training; several
  experiments share one cache (both 2D decoders; `logreg` + `finetune`), so extract per
  *cache key*, not per experiment. `--max-patients` limits patients for smoke tests —
  stratified by pathology for classification, plain `head()` for segmentation.
- SAM has two families with separate caches and configs: `sam` (v1 ViT, `SAM_CONFIGS`)
  and `sam2` (2.1 Hiera, `SAM2_CONFIGS`). SAM2 segmentation reads Stage 3 (`embed_dim`);
  SAM2 classification reads Stage 4 (`cls_embed_dim`). Neither family has a CLS token, so
  both are `--pooling gap` only. See [prompts/sam2_decisions.md](prompts/sam2_decisions.md).
- **The two SAM families index `layer_indices` differently.** `hidden_states[0]` is the patch
  embedding, so block *i* is at `hidden_states[i+1]`. `SAM_CONFIGS` indices are **block**
  indices (offset 1); `SAM2_CONFIGS` indices are already **`hidden_states`** indices (offset 0,
  chosen so all four land in Stage 3 — shifting them changes the channel count).
  `extract_sam_volume_features` serves both families and so takes a required
  `hidden_state_offset`, carried in `load_backbone` metadata. Route every read through
  `features.py::_select_hidden_state` rather than subscripting `hidden_states` directly.
- Results files are committed and feed the analysis scripts: regenerate summaries after new runs, keep the `{name}_{timestamp}` convention.
- Prefer cardiac-specific approaches over general-purpose image processing where the repo already has one.
