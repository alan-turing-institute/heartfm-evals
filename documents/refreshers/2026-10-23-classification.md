# Classification refresher — 2026-10-23

State of the classification task eleven weeks after
[2026-08-09-classification.md](2026-08-09-classification.md). Supersedes it. That document
remains the reference for everything that has **not** changed — the code path, the class
registry, the logreg/finetune internals, the full result tables and what the numbers say — and
those are not repeated here. Read it first, then this.

| | |
| --- | --- |
| Written | 2026-10-23 |
| `main` at | `e67d47a` |
| Previous refresher was written at | `3796713` |
| `sam2-classification` at | `a77c030` (2026-09-08) — 16 commits not on `main`; `main` has 48 not on it |
| Tests | **49 pass** (`uv run --no-sync pytest`) |
| Local classification feature cache | **none** — `feature_cache_classification/` does not exist |
| Free disk | 171 GiB |

**One-line summary:** no classification result has changed since 08-09 — every committed JSON
is byte-identical apart from a trailing newline — so all the numbers in the 08-09 document still
stand. The code around them moved (SAM2 removal, `--cache-only`, stratified `--max-patients`,
the SAM pooling guard), but only **one** of the eleven known issues is fixed. The 08-09 advice
"just merge `sam2-classification`" no longer applies: `main` has moved on, so the branch's
classification scripts and outputs now have to be **ported**, not merged.

---

## 1. What changed since 2026-08-09

### Results: nothing real

`git diff 3796713..HEAD -- results/classification` touches 69 files with +69/−69 lines — every
one the end-of-file newline added by the pre-commit pass in `27049c5`. The same holds between
`main` and `sam2-classification`: the 66 JSONs that differ are identical under
`git diff -w --ignore-blank-lines`. No run has been added, removed or re-run on either branch.

### Code

| Change | Commit(s) | Effect on classification |
| --- | --- | --- |
| `--cache-only` | `b901199` | Extracts train/test/**val** features and exits. Val is now cached before training, not inside it. One cache serves both `logreg` and `finetune`. |
| Cache dir renamed | `2062943` | `classification_feature_cache/` → `feature_cache_classification/{dataset}/{model}/{pooling}/{split}/`. |
| Data default | `b901199` | `../data/heartfm/processed/{dataset}` (one level above the repo). |
| Stratified `--max-patients` | `b901199` | `subset_patients_stratified()` instead of `head()` — ACDC and M&Ms-2 metadata are sorted by pathology, so `head(50)` gave 3 of 5 / 2 of 6 classes. |
| SAM `--pooling cls` rejected | `b901199` | `p.error(...)` in `parse_args`. Fixes 08-09 issue 6. |
| SAM v1 off-by-one fix (#66) | `c510bac`, `16b2383` | **None.** See below. |
| SAM2 removed | `93a55e5`, `326233c` | None — classification only ever used SAM v1. |
| `run_acdc_classification.py` deleted | `bb19a44` | Dead script gone; `smoke_test_classification.sh` now targets `run_classification.py` with `--max-patients 50` (stratified, 10 per ACDC class). |
| ruff/mypy pass | `fa07f44` | Type narrowing in `sweep_C_and_train` / `finetune_sweep_and_train`; `set_seed` import kept with a comment pointing at #59. No behavioural change. |
| Split-manifest guard (#75) | `38c22f0` | **Segmentation only.** Classification caches have no `split_manifest.json`. |

**The SAM v1 fix does not touch classification, and the committed SAM classification results
remain valid.** #66 was about resolving a block index through the transformers
`hidden_states` tuple. `cache_sam_cls_features` never uses `hidden_states`: it runs
`patch_embed`, adds `pos_embed`, loops over **every** block in `vision_encoder.layers` and
global-average-pools the final block's output, before the neck
([classification_probe.py:411-418](../../src/heartfm_evals/classification_probe.py)). DINOv3
classification likewise takes only the last block via `get_intermediate_layers(n=1)`. Neither
uses `layer_indices`, so neither has the cache-key trap the segmentation refreshers describe.

`AGENTS.md` now describes `classification_probe.py` correctly (the 08-09 note that `CLAUDE.md`
misdescribed it is resolved).

---

## 2. Committed state on `main`

| | ACDC | M&Ms | M&Ms-2 |
| --- | --- | --- | --- |
| Result JSONs | 22 | 23 (one duplicate) | 21 (one missing) |
| `summary.csv` rows | **5** | **6** | **4** |
| `mcnemar_five_way.csv` methods | 33 (11 phantom) | 23 (1 phantom) | 22 (1 phantom) |
| `mcnemar_binary.csv` / `bootstrap_binary_*` | 33 methods, stale | — | — |

Phantoms are rows with no JSON behind them: on ACDC the 11 `*_ft-full_*` and
`dinov3_vit7b16_*` methods; on M&Ms `dinov3_vit7b16_logreg_cls`; on M&Ms-2
`dinov3_vits16_logreg_cls` (the run that was lost — issue 4). Summaries on `main` therefore
still cover about a quarter of the completed runs, and the significance matrices are
unreproducible. All of this is fixed on `sam2-classification` (§4).

---

## 3. Issue status vs the 2026-08-09 refresher

| # | Issue | Status |
| --- | --- | --- |
| 1 | No seeding; finetune shuffles with unseeded `np.random.permutation`; `--seed` ignored by CV | ❌ open (#59). `set_seed` still commented out at `run_classification.py:225` and `finetune_classification.py:249`; permutation at `finetune_classification.py:103`; both `StratifiedKFold`s hardcode `random_state=0`. Logreg is deterministic in practice; `ft-frozen` rows are not reproducible. |
| 2 | Skip-if-exists is dead code | ❌ open — `run_classification.py:236-242` still builds the path from a fresh timestamp before checking `.exists()`. |
| 3 | Duplicate `mnm/cinema_pretrained_logreg_gap` JSON (`_164436` / `_164804`) | ❌ still byte-identical; still double-counted by `summary.csv` and silently collapsed by `_common.load_classifiers` (keyed by label). |
| 4 | `mnm2/dinov3_vits16_logreg_cls` missing | ❌ still missing; still makes the vits16 logreg·cls cross-dataset mean a 2-dataset average. |
| 5 | M&Ms-2 is 6-class but named `five_way` | ❌ open — 14 occurrences across `scripts/classification/*.py`, plus JSON keys and filenames. Metrics themselves are correct. |
| 6 | SAM silently accepts `--pooling cls` | ✅ **fixed** — the driver rejects it. |
| 7 | Leftover `freeze_backbone` / `ft-full` branches and docs | ❌ open — `build_summary.py:45`, `_common.py:46`; `scripts/classification/README.md` lines 55, 76, 93 still document `--freeze-backbone` / `--no-freeze-backbone`, so its quick-start fails. |
| 8 | Final finetune head is last-epoch, early stopping disabled | ❌ open — `finetune_classification.py:333-345` passes `all_idx` as train *and* val with `patience=epochs` and discards the returned best state. |
| 9 | `restructuring_plan.md` says the finetune path should be deleted | ❌ unresolved — the plan (lines 21, 29, 97, 110) still says remove `finetune_classification.py`; the code still has it. |
| 10 | Debris | ⚠️ half — dead script and smoke test fixed; the three `results/classification/acdc/*.json.bak` remain (one materially different, see 08-09). |
| 11 | `_smoke` JSONs not excluded; `roc_auc_score` raises on a missing class; CineMA truncates past 16 z-slices | ❌ open, but the missing-class crash is now much less reachable thanks to stratified `--max-patients`. |

**New this pass:**

- **N1 — no split manifest on classification caches.** The #75 guard (`verify_cache_splits`,
  `split_manifest.json`) is wired into `run_segmentation.py` only. The risk is lower here — ACDC
  classification uses 10-fold CV on train rather than a carved val split, and M&Ms/M&Ms-2 have
  fixed `val/` directories — but a cache is still trusted because its files exist, and the key
  (`{model}/{pooling}`) records no code version.
- **N2 — no local cache.** Any re-run, including the single missing M&Ms-2 config, needs a fresh
  extraction for that backbone/dataset.

---

## 4. Branch situation — port, don't merge

On 08-09 merging `sam2-classification` cost nothing. That is no longer true: `main` has 48
commits the branch lacks (SAM2 removal, #66, CI restoration, `--cache-only`, the dead-script
deletion). Merging would resurrect `run_acdc_classification.py` and fight the segmentation
history. Draft PR #63 still points at the branch.

What the branch has that `main` does not, for classification:

| Kind | Files |
| --- | --- |
| Scripts | `aggregate_summary.py`, `per_dataset_plots.py`, `plot_pooled_backbones.py`; small edits to `_common.py`, `bootstrap_test.py`, `mcnemar_test.py`, `README.md` |
| Regenerated outputs | `summary.csv` ×3 (22/23/21 rows), `mcnemar_five_way.*` and `bootstrap_five_way_macro_f1*` ×3 (no phantoms) |
| New outputs | `summary_aggregated.csv`, `cinema_finetune_paper_auc.csv`, `pooled_logreg_gap_macro_auc_ci.{csv,png}`, `auc_by_{backbone,method}.png` (top level + per dataset) |

Port the **scripts**, then regenerate the outputs on `main` rather than copying them — they are
cheap to rebuild from the JSONs, and rebuilding is the only way to drop the duplicate and the
phantoms consistently. `cinema_finetune_paper_auc.csv` is hand-transcribed literature data and
has to be copied as-is.

---

## 5. GitHub issues

| Issue | State |
| --- | --- |
| #18 classification (meta) | open, still the umbrella |
| #21 / #22 add M&Ms / M&Ms-2 | **done** — both datasets have full grids; can be closed |
| #55 significance tests | open — scripts exist; outstanding work is multiple-comparison correction and the binary-task decision |
| #59 finetune randomness | open — issue 1 above |
| #28 run `cinema.from_finetuned` | open — the most direct route to an **in-house** fine-tune baseline, replacing the literature value in `cinema_finetune_paper_auc.csv` |

---

## 6. Suggested order of work

1. **Port the three branch-only scripts** from `sam2-classification` (`aggregate_summary.py`,
   `per_dataset_plots.py`, `plot_pooled_backbones.py`) plus `cinema_finetune_paper_auc.csv`, on a
   branch off `main`.
2. **Clean the inputs**: delete the duplicate M&Ms JSON (issue 3) and the three `.json.bak`
   files; make the globs skip `_smoke` (issue 11) and drop the `freeze_backbone` branches (issue 7).
3. **Regenerate** `build_summary.py` → `aggregate_summary.py` → significance tests → plots for all
   three datasets. Add an FDR correction (Benjamini–Hochberg) to both significance scripts before
   quoting any "significant" pair.
4. **Decide on the finetune path and on seeding before any new run** (issues 1, 8, 9): either
   delete `finetune` per the restructuring plan, or keep it, fix the final-head bug, re-enable
   `set_seed`, and amend the plan. Re-running `ft-frozen` without that decision just produces
   another unreproducible generation.
5. **Re-run `mnm2/dinov3_vits16_logreg_cls`** (issue 4) — logreg is deterministic in practice, so
   this one is safe now; it needs a fresh DINOv3-S M&Ms-2 extraction (`--cache-only` first).
6. Decide whether to **regenerate or drop the binary task** (stale, ACDC-only,
   prevalence-dominated on M&Ms/M&Ms-2 — see 08-09 §3).
7. Housekeeping: rename `five_way` → `multiclass` (issue 5), fix the skip-if-exists check
   (issue 2), wire the split manifest into classification (N1), close #21/#22.

The scientific picture is unchanged from 08-09: every frozen probe trails the published CineMA
fine-tune (mean 5-way AUC 0.786 best vs 0.842), CineMA's advantage is ACDC-only, and DINOv3-B is
the strongest frozen backbone. That comparison still rests on a literature number and on single
unseeded runs; #28 is the way to make it controlled.
