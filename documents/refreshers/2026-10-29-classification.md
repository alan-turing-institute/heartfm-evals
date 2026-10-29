# Classification refresher — 2026-10-29

State of the classification task six days after
[2026-10-23-classification.md](2026-10-23-classification.md), following three merged PRs
(#79, #81, #84). Supersedes it. Earlier lineage: [2026-08-09](2026-08-09-classification.md),
which is still the reference for the result tables and what the numbers say — they are not
repeated here.

| | |
| --- | --- |
| Written | 2026-10-29 |
| `main` at | `9c97ed6` |
| Previous refresher was written at | `e67d47a` (committed as `ddb6ab2`) |
| `sam2-classification` at | `a77c030` (2026-09-08) — 16 commits not on `main`; `main` now has 62 not on it |
| Tests / lint | **62 pass** (`uv run --no-sync pytest`); `pre-commit run -a` clean |
| Local classification feature cache | **none** — `feature_cache_classification/` does not exist |
| Free disk | 174 GiB |

**One-line summary:** a cleanup week. The `finetune` mode is now **`--head linear`**, the
restructuring-plan contradiction is resolved, the dead `ft-full` code is gone, and the duplicate
M&Ms JSON is deleted and now rejected. Still no new or re-run result, so every number from 08-09
stands — but the committed `summary.csv` files and significance matrices were **not
regenerated** after the rename and are now out of step with the scripts that produce them. A
rebuild works cleanly (§2). Issue #82 sets out the next real piece of work: a CineMA-paper-style
head on frozen features.

---

## 1. What changed since 2026-10-23

| PR / commit | Effect |
| --- | --- |
| #79 (`a79993a`, `e97553c`) | Deleted `mnm/cinema_pretrained_logreg_gap_20260418_164804.json`. `build_summary.py` and `_common.load_classifiers` now **raise** on two JSONs with the same config, naming both files. Fixes 10-23 issue 3. |
| #81 (`26fcea9`, `b2ef87b`) | Removed every `freeze_backbone` / `ft-full` branch from `build_summary.py`, `_common.py` and `run_classification.py`; README quick-start no longer uses non-existent flags. Fixes issue 7. |
| #84 (merged 10-28) | **Renamed the mode.** `--eval-mode {logreg,finetune}` → **`--head {logreg,linear}`**; `finetune_classification.py` → `src/heartfm_evals/linear_head.py`; `finetune_sweep_and_train` → `linear_head_sweep_and_train`; batch scripts `batch_run_*_finetune_*` → `batch_run_*_linear_*`. New JSONs write `config.head` and filenames `*_linear_*`. Loaders read `head`, fall back to legacy `eval_mode`, and map legacy `finetune` → `linear`. `restructuring_plan.md` now opens with a note recording that the path was kept on purpose. Resolves issue 9. |
| `2ad77cd` | `.claude/skills/fix-issue/` — the skill used to file and fix #78/#80/#83. |
| `75625e1`…`9c97ed6` | CineMA paper PDF added at `documents/papers/2026 Fu et al CineMA Nature.pdf` (regular file, not LFS); `documents/prompts/research_brief.md` added — a brief for a separate literature/novelty review on branch `research/novelty-review`, which **does not exist yet** locally or on origin. |

No library code changed apart from the rename: training, seeding, splits, feature extraction
and metrics are untouched, so no committed number is affected. Tests grew from 49 to 62 —
`tests/test_classification_results.py` (new) covers the duplicate guards and legacy-label
mapping; `test_classification_multi_dataset.py` covers the `--head` CLI.

**Naming to use from now on:** "linear head" for what was called `finetune` / `ft-frozen`.
Committed JSONs still say `"eval_mode": "finetune"` and `*_ftfrozen_*`; that is legacy, not a
different method. The backbone is never trained in this repo — the only full fine-tune in any
comparison is the CineMA paper's.

---

## 2. Committed state on `main`

| | ACDC | M&Ms | M&Ms-2 |
| --- | --- | --- | --- |
| Result JSONs | 22 | 22 | 21 (one missing — issue 4) |
| Committed `summary.csv` rows | **5** | **5** | **4** |
| Rows after a rebuild | 22 | 22 | 21 |
| `mcnemar_five_way.csv` methods | 33 (11 phantom) | 23 (1 phantom) | 22 (1 phantom) |

The committed summaries still predate even the April sweep, still have an `Eval Mode` column
and `ft-frozen` labels, and the significance matrices still score phantom methods (see the
10-23 refresher §2) under the old `ft-frozen` label. I ran `build_summary.py` against a copy
of `results/classification/` (not in place): it exits cleanly, the new duplicate guard does not
fire, and it produces 22 / 22 / 21 rows with a `Head` column. **Nothing has been regenerated in
the repo.**

---

## 3. Issue status

Numbering follows the 08-09 / 10-23 refreshers.

| # | Issue | Status |
| --- | --- | --- |
| 1 | No seeding; unseeded `np.random.permutation`; `--seed` ignored by CV | ❌ open (#59). `run_classification.py:224`, `linear_head.py:249` (commented `set_seed`), `linear_head.py:103` (permutation). Logreg deterministic in practice; linear-head rows not reproducible. |
| 2 | Skip-if-exists can never fire | ❌ open — `run_classification.py:238` |
| 3 | Duplicate M&Ms JSON | ✅ **fixed** (#79), and duplicates are now an error |
| 4 | `mnm2/dinov3_vits16_logreg_cls` missing | ❌ open |
| 5 | M&Ms-2 is 6-class but named `five_way` | ❌ open — 14 occurrences in `scripts/classification/*.py`; `5-way` column labels in `build_summary.py` |
| 6 | SAM accepts `--pooling cls` | ✅ fixed (before 10-23) |
| 7 | Dead `freeze_backbone` / `ft-full` code and docs | ✅ **fixed** (#81) |
| 8 | Final linear head is last-epoch, early stopping disabled | ❌ open — `linear_head.py:333-344` passes `all_idx` as train and val with `patience=epochs` |
| 9 | Plan says delete the finetune path | ✅ **resolved** (#84) — kept as `--head linear`, plan annotated |
| 10 | Debris | ⚠️ the three `results/classification/acdc/*.json.bak` remain (one materially different from its live file) |
| 11 | `_smoke` JSONs not excluded; OVR AUC raises on a missing class; CineMA truncates past 16 z | ❌ open |
| N1 | No split manifest on classification caches | ❌ open — #75's guard is segmentation-only |
| N2 | No local classification cache | ❌ open — any run needs extraction first |

**New this pass:**

- **N3 — committed outputs disagree with the scripts** (§2). Regenerating is now safe and cheap,
  but it relabels `ft-frozen` → `linear` everywhere.
- **N4 — the branch scripts to be ported assume the old labels.** `aggregate_summary.py`,
  `per_dataset_plots.py` and `plot_pooled_backbones.py` on `sam2-classification` reference
  `Eval Mode` / `ft-frozen` / `eval_mode` (13, 7 and 12 occurrences). Porting them is now a
  port-and-adapt job, not a copy.
- **N5 — two different "published CineMA" baselines are in play.**
  `cinema_finetune_paper_auc.csv` on `sam2-classification` (the bar in every branch plot, and the
  0.842 mean in the 08-09 table) says **97.98 / 77.41 / 77.20 %** for ACDC / M&Ms / M&Ms-2.
  Issue #82 cites Suppl. Table 8 as **0.992 / 0.810 / 0.801**. Supplementary Table 8 is not in the
  committed PDF, so I could not check which is right. #82 also points out that the paper's AUROC
  is **macro one-vs-one** while ours is macro one-vs-rest, so neither figure is directly
  comparable to our numbers yet. Settle the reference values before any plot or write-up quotes
  the gap.

---

## 4. Issue #82 — the next piece of real work

#82 (open, filed 10-23) proposes a third head that reproduces the CineMA paper's CVD
classification recipe as closely as a frozen backbone allows, to split the frozen-vs-fine-tuned
gap into "head and recipe" vs "unfreezing". In short:

- **Head:** two `nn.Linear` heads — on the CLS token and on the mean of patch tokens — with
  averaged logits; no `StandardScaler`. SAM v1 falls back to GAP only.
- **Recipe:** label smoothing 0.1, AdamW (0.9, 0.95), lr 1e-3 fixed, wd 0.05, 10-epoch warmup
  then cosine, grad clip 5, batch 64, up to 800 epochs, keep the best head on val MCC.
- **Protocol:** CineMA's ACDC val split (2 per class, `random_state=0`), 3 seeds averaged into an
  ensemble, macro OVO AUROC and MCC added to the JSON.
- **Open decisions** in the issue: early (2-channel, CineMA-only) vs late ED/ES fusion; no
  augmentation vs cached-K augmented passes; fixed hyperparameters vs a sweep.

Two things are now stale in the issue text: it says "new `--eval-mode`" and "in
`finetune_classification.py`" — after #84 that is a new `--head` choice in `linear_head.py` (or a
sibling module). Item 5 (seeds) is blocked on #59.

---

## 5. GitHub issues

| Issue | State |
| --- | --- |
| #78, #80, #83 | **closed** this week (PRs #79, #81, #84) |
| #82 CineMA-paper-style head | open — §4 |
| #59 randomness | open — blocks #82 item 5 and any linear-head re-run |
| #55 significance tests | open — outstanding: FDR correction, binary-task decision |
| #28 `cinema.from_finetuned` | open — the in-house full fine-tune reference; would also settle N5 |
| #18 (meta), #21, #22 | open; #21 / #22 are done and can be closed |

---

## 6. Suggested order of work

1. **Regenerate the summaries** (`build_summary.py`) — verified safe (§2). Delete the three
   `.json.bak` files and make the globs skip `_smoke` in the same change.
2. **Port and adapt** `aggregate_summary.py`, `per_dataset_plots.py`, `plot_pooled_backbones.py`
   to the `head` / `linear` labels (N4); bring over `cinema_finetune_paper_auc.csv` only once N5
   is settled.
3. **Regenerate the significance tests** with a Benjamini–Hochberg correction; decide whether the
   binary task stays.
4. **Fix seeding (#59) and the last-epoch head (issue 8)** before any linear-head run, old or
   #82's.
5. **Re-run `mnm2/dinov3_vits16_logreg_cls`** (logreg is deterministic; needs a DINOv3-S M&Ms-2
   extraction with `--cache-only` first).
6. **#82**: settle its fusion/augmentation decisions, add OVO AUROC + MCC to the JSON first (cheap,
   and needed for N5 either way), then the head.
7. Housekeeping: `five_way` → `multiclass` (issue 5), skip-if-exists (issue 2), split manifest
   for classification caches (N1), close #21 / #22.

The scientific picture is unchanged: frozen probes trail the published CineMA fine-tune on all
three datasets, CineMA's advantage is ACDC-only, and DINOv3-B is the strongest frozen backbone.
How big the gap is depends on N5 and on OVO-vs-OVR, and it still rests on single unseeded runs.
