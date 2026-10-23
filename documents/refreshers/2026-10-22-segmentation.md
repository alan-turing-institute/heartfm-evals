# Segmentation refresher — 2026-10-22

State of the segmentation task on branch `74-reproduce-acdc-segmentation-locally`, six weeks
after [2026-09-11-segmentation.md](2026-09-11-segmentation.md). Supersedes it. That document
remains the reference for the parts that have **not** changed (the PR #67/#68/#69/#73
history, the B1/B2 disk and SAM-time arithmetic, the SAM v1 global-attention rationale); they
are not repeated here. Earlier lineage: [2026-09-01](2026-09-01-segmentation.md) →
[2026-08-09](2026-08-09-segmentation.md).

| | |
| --- | --- |
| Written | 2026-10-22 |
| Branch / HEAD | `74-reproduce-acdc-segmentation-locally` @ `9b7ae05` (clean tree) |
| `main` / `origin/main` | `9acc519` — **unchanged** since the last refresher |
| `sam2-classification` | `a77c030` (2026-09-08) — unchanged, 16 commits not on `main` |
| SAM2 archive tag | `archive/sam2-2026-09-08` — present |
| Tests / lint | **39 pass**; `pre-commit run -a` all green |
| Free disk | **68 GiB** (was 95 GiB on 09-11) |

**One-line summary:** nothing has been committed since 2026-09-11. The branch has deleted
every committed segmentation result and ported one plot script, but **no result has been
regenerated** — `results/segmentation/` is an empty untracked directory. The ACDC caches for
CineMA and DINOv3-S are complete, **except that the DINOv3-S 2D cache is contaminated with a
second train/val split** (§3) and must be repaired before the 6 runs.

---

## 1. What changed since 2026-09-11

Three commits, all 09-11, all on the issue-#74 branch; none merged.

| Commit | Effect |
| --- | --- |
| `56e4408` | Added the 09-11 refresher. |
| `cae19a3` | **Deleted all committed segmentation results** (118 files, ~109 k lines: JSONs, `_per_slice` / `_per_stack` / `_per_sample` CSVs, `summary.csv`, `summary_aggregated.csv`, PNGs, CI CSVs) for acdc / mnm / mnm2. Recoverable from `main`. |
| `9b7ae05` | Ported `plot_dataset_macro_dice_ci.py` (228 lines) from `sam2-classification`; changed `report_macro_dice_ci.py` so its default patterns are `*_per_sample.csv` **and** `*_per_stack.csv`, with a docstring explaining that UNETR emits `per_sample` and linear/conv emit `per_stack`. |

Consequences for the 09-11 document:

- Its §2 ("12 JSONs each", orphan SAM2 rows, "regenerate the summaries") is moot: there are
  no JSONs or summaries on this branch at all. On `main` the stale April generation is
  still there.
- **Issue 3** (bootstrap CIs cover only UNETR) is fixed on this branch.
- **B5** (port the CI plot scripts) is **half done**: `plot_dataset_macro_dice_ci.py` is
  ported; `plot_pooled_backbones.py` (299 lines) is still only on `sam2-classification`.
- Issue #74 (the ACDC re-run) is open; its checklist has "branch off `main`" and "delete old
  results" done, and the 6 runs, summary rebuild and write-up not done.

## 2. Results

`results/segmentation/` exists but is empty (untracked, so git does not show it). No
segmentation JSON exists locally on any branch except through git history and the
`sam2-classification` branch. Classification results (`results/classification/`) are
untouched and were not re-checked here; see [2026-08-09-classification.md](2026-08-09-classification.md).

What the re-run is expected to show (from issue #74, still untested):

- CineMA `linear_probe` / `conv_decoder` improve substantially (they were the z-misaligned
  cells, fixed in `a020ad4`).
- CineMA UNETR and all three DINOv3-S cells reproduce April closely; a large move there
  would indicate run-to-run noise, not a code change.

Neither statement should be written down until the runs exist — and the runs are unseeded
(§5, B4).

## 3. ACDC feature caches — complete, but DINOv3-S 2D has a split problem

`feature_cache_segmentation/acdc/` = **19 GiB**, last written 2026-09-11. Nothing is
running (no `run_segmentation` process).

| Cache | train / val / test files | Size | State |
| --- | --- | --- | --- |
| `cinema_pretrained` (2D) | 1534 / 178 / 916 | 1.9 GiB | complete, consistent |
| `cinema_pretrained_unetr3d` | 180 / 20 / 100 | 7.8 GiB | complete |
| `dinov3_vits16` (2D, `layers_3-6-9-11`) | **1554 / 218** / 916 | 3.0 GiB | **contaminated** |
| `dinov3_vits16_unetr3d` | 180 / 20 / 100 | 6.0 GiB | complete |

The CineMA 2D cache was rebuilt during the 09-11 morning run (the 09-11 refresher caught it
mid-write); the 2D and 3D caches for CineMA and the 3D cache for DINOv3-S all agree on
**90 train / 10 val / 50 test patients**. The DINOv3-S **2D** cache does not:

- patient002 is in `train/` (and also `val/`, where it belongs);
- patients 003 and 004 are in `val/` (and also `train/`, where they belong);
- so **three patients (60 slice files) appear in both train and val** of that cache, and
  those three files' mtimes are 09:21, after the rest of the cache (08:44–08:51).

The extras come from a second extraction run under a different val split (the ACDC val
set is carved from train by `split_seed` and the cache path does not record the seed — the
warning at the top of `cache_all_features.sh` describes exactly this failure). Effect if
left alone: the DINOv3-S `linear_probe` / `conv_decoder` validation set (used for model
selection) shares 3 of 12 patients with training, and DINOv3 and CineMA no longer train on
the same patients, so the two backbones' numbers are not like-for-like.

**Fix (not yet done):** delete `dinov3_vits16/train/layers_3-6-9-11/patient002_*` and
`dinov3_vits16/val/layers_3-6-9-11/patient00{3,4}_*`; the result should be 1534 / 178 / 916
matching CineMA. Verify with a patient-set comparison across the four caches before training.
This is the same hazard as N1 in the 09-11 refresher (a cache is trusted because the file
exists): the cache has no manifest, so nothing flags it.

## 4. Code and repo state

- **Tests:** 39 pass (47 s), three files: `test_classification_multi_dataset.py`,
  `test_feature_layer_indexing.py`, `test_segmentation_metrics.py`. `caching.py` still has
  no tests.
- **Lint/CI:** `pre-commit run -a` passes (ruff, ruff-format, mypy, hygiene). `.github/workflows/ci.yml`
  unchanged since PR #73.
- **Library:** `src/heartfm_evals/` unchanged since 09-11 (no source commit after `9acc519`).
- **Scripts:** `scripts/segmentation/` now has `plot_dataset_macro_dice_ci.py` in addition to
  `aggregate_summary.py`, `build_summary.py`, `report_macro_dice_ci.py`, `run_segmentation.py`,
  `run_all_segmentation.sh` and nine `batch_run_{cinema,dino,sam}_*_segmentation.sh`.
- **Local environment:** use `uv run --no-sync …` (`.venv` has no `activate`, bare `python` and
  `pre-commit` are not on PATH). `torch.jit.interface` warning handling is by
  message-scoped `filterwarnings`.
- **Branches:** `main` is 3 commits behind this branch. Stale local branches
  (`temp1-3`, `10-unet-decoder`, `35-…`, `5-decoder-…`, `9-dino-…`, `isambard-ai`,
  `old-code-for-comparison`, `sign_tests_classification`) are untouched.

## 5. Issue status vs the 2026-09-11 refresher

| Ref | Issue | Status |
| --- | --- | --- |
| N1 | stale SAM cache indistinguishable from a correct one (no version in cache key) | ❌ open; **DINOv3 2D cache above is a live instance of the same class** |
| N2 | summaries disagree with JSONs | ✅ moot on this branch (all deleted); still true on `main` |
| N3 | 3D caches spend ~37 % of encoder budget on zero-padded slices | ❌ open, unchanged; decide deliberately |
| N4 | `caching.py` has no tests | ❌ open |
| B1 | disk (~642 GiB for full grid) | ❌ unchanged problem; **free space dropped to 68 GiB**. `feature_cache_backups/` still 35 GiB (pre-fix misaligned CineMA 2D cache) and still deletable |
| B2 | SAM extraction ~5 days on MPS | ❌ unchanged; extractor not batched |
| B3 (skip) | re-run writes a second JSON | ❌ open — `run_segmentation.py:204-207` builds the path from a fresh timestamp so the `exists()` check cannot fire. Relevant now: re-running any of the 6 configs duplicates it |
| B4 | `set_seed()` disabled | ❌ open — `run_segmentation.py:193` commented out, import kept `# noqa: F401`; `"seed": 0` still written to every JSON; issue #59 open. The 6 runs are unseeded, single-seed |
| B5 | port CI plot scripts | ⚠️ half — `plot_pooled_backbones.py` still branch-only |
| 3 | bootstrap CIs only cover UNETR | ✅ **fixed** on this branch (`9b7ae05`) |
| 4 | `--use-layers` broken for `conv_decoder` | ❌ open, low priority |
| 7 | `README.md` stale | ❌ open — still references `run_sam_variants.sh`, `run_sam3.sh`, `run_sam3_unetr.sh`; no `scripts/segmentation/README.md` |

**New this pass:** the DINOv3-S 2D cache split contamination (§3).

## 6. Suggested order of work (issue #74)

1. **Repair the DINOv3-S 2D cache** (delete the 60 extra files, §3) and confirm all four
   caches hold the same 90 / 10 / 50 patients.
2. **Decide on `set_seed()`** before running (B4); otherwise record that results are unseeded.
3. **Run the 6 ACDC configs** (`{linear_probe, conv_decoder, unetr}` × `{cinema, dinov3_vits16}`),
   from the existing caches — no extraction needed. `results/segmentation/acdc/` is empty, so
   B3 cannot bite on a first pass; do not re-run a config without deleting its JSON.
4. `build_summary.py` → `report_macro_dice_ci.py` → `plot_dataset_macro_dice_ci.py` for ACDC only.
   `aggregate_summary.py` and the pooled plot need all three datasets and are out of scope.
5. Compare against `main` (April) and `sam2-classification`; take code from the latter, not numbers.
6. Housekeeping: free `feature_cache_backups/` (35 GiB), port `plot_pooled_backbones.py`, README,
   `caching.py` alignment test.

Open scientific questions are unchanged from 09-11 and should stay unwritten until the grid
is re-run under a fixed seed: whether cardiac pretraining helps beyond UNETR once the CineMA
cache is fixed, and whether SAM v1's inverse size scaling survives the move to
global-attention taps.
