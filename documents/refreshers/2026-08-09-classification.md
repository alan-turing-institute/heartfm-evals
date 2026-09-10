# Classification refresher — 2026-08-09

State of the classification task after a period away from the codebase. Companion to
[2026-08-09-segmentation.md](2026-08-09-segmentation.md).

| | |
| --- | --- |
| Written | 2026-08-09 |
| `main` at | `3796713` |
| `sam2-classification` at | `c237fc8` |
| Merge base | `7517e7e` |

**One-line summary:** unlike segmentation, the two branches share **byte-identical run
artefacts** — not one result JSON differs. The branch is purely additive: it regenerated the
summary tables (which on `main` cover only ~25% of completed runs), fixed significance matrices
that on `main` still score 11 phantom methods whose JSONs no longer exist, and added
cross-dataset aggregation, plots, and a published-CineMA baseline. Merging it loses nothing.

The bigger issue is scientific, not organisational: every frozen-feature probe here loses
substantially to the published full fine-tune, and on M&Ms/M&Ms-2 accuracy is in the 0.33–0.54
range on 5–6 class problems.

---

## 1. Code path

Driver: [scripts/classification/run_classification.py](../scripts/classification/run_classification.py)

```bash
python scripts/classification/run_classification.py \
    --dataset mnm --backbone cinema --eval-mode logreg --pooling cls
```

| Arg | Default | Choices |
| --- | --- | --- |
| `--dataset` | `acdc` | `acdc, mnm, mnm2` |
| `--backbone` (req.) | — | `cinema, dinov3, sam` |
| `--eval-mode` (req.) | — | `logreg, finetune` |
| `--pooling` | `cls` | `cls, gap` |
| `--dinov3-model-name` | `dinov3_vits16` | — |
| `--sam-model-id` | `facebook/sam-vit-base` | — |
| `--max-patients`, `--seed`, `--device`, cache/output dirs | | |

There are **no** `--epochs`, `--lr`, `--batch-size` or `--C` args — those are hardcoded
constants (see below).

Flow: load metadata → `EndDiastoleEndSystoleDataset` (CineMA package, `views="sax"`) →
`load_backbone()` (frozen) → cache one pooled embedding per slice/frame into
`feature_cache_classification/{dataset}/{model}/{pooling}/{split}/` → `build_patient_features`
(mean-pool ED slices, mean-pool ES slices, concatenate → `(N, 2·embed_dim)`) → fit probe →
evaluate N-way → evaluate a derived binary NOR-vs-rest task (from `1 − P(NOR)`, no retraining) →
write one JSON to `results/classification/{dataset}/{model}_{tag}_{pooling}_{timestamp}.json`
where `tag` ∈ `{logreg, ftfrozen}`.

### Where the logic actually lives

Almost all of it is in
[classification_probe.py](../src/heartfm_evals/classification_probe.py) (807 lines) —
**not** in `features.py`/`caching.py`/`metrics.py`, which are segmentation-only. Note
`CLAUDE.md` currently misdescribes this.

| Piece | Location |
| --- | --- |
| Pathology class registry | `DATASET_PATHOLOGY_CLASSES`, `classification_probe.py:50` |
| Feature caching (DINOv3 / CineMA / SAM) | `cache_cls_features:147`, `cache_cinema_cls_features:300`, `cache_sam_cls_features:352` |
| Patient-level feature assembly | `build_patient_features:423` |
| Logreg probe | `sweep_C_and_train:560` |
| Classification metrics (acc, macro F1, sens/spec, AUC) | `evaluate_classification:675`, `evaluate_binary_detection:762` |

**Logreg**: `StandardScaler` + `LogisticRegression(lbfgs, max_iter=1000, tol=1e-12)`, C swept
over 45 log-spaced values from 1e-6 to 1e5, selected by a val split where one exists
(M&Ms, M&Ms-2) or 10-fold `StratifiedKFold` otherwise (ACDC), then refit on all train data.

**Finetune**: despite the name,
[finetune_classification.py](../src/heartfm_evals/finetune_classification.py) **never unfreezes
the backbone**. It trains a single `nn.Linear` head by SGD on the *same cached features* logreg
uses — AdamW + cosine, LR grid `(1e-5, 5e-5, 1e-4, 5e-4, 1e-3)`, 50 epochs, patience 10. So
`logreg` vs `finetune` is really *sklearn head vs torch head on identical inputs*, which is why
their numbers are so close. The `ft-full` mode that once existed is gone.

### Class registry

| Dataset | Classes | N | Test patients |
| --- | --- | --- | --- |
| ACDC | NOR, DCM, HCM, MINF, RV | 5 | 50 (10 per class) |
| M&Ms | NOR, DCM, HCM, ARV, HHD | 5 | 105 (31/35/23/6/10) |
| M&Ms-2 | NOR, HCM, ARR, CIA, FALL, LV | **6** | 108 (30/23/10/10/10/25) |

`NOR` is index 0 in all three, which is what the derived binary task keys off.

### Supporting scripts

| File | Role |
| --- | --- |
| `build_summary.py` | Globs `*.json` per dataset → `summary.csv` |
| `aggregate_summary.py` **(branch only)** | Pivots 5-way AUC across datasets → `summary_aggregated.csv` + 2 top-level PNGs |
| `per_dataset_plots.py` **(branch only)** | Per-dataset `auc_by_backbone.png` / `auc_by_method.png` |
| `mcnemar_test.py` | All-pairs exact McNemar (`scipy.stats.binomtest`) → CSV + p-value heatmap |
| `bootstrap_test.py` | Paired bootstrap, B=1000, shared resample indices, 95% CI → CSV + `_ci.csv` + heatmap |
| `_common.py` | Shared loader/aligner for the two significance scripts |
| `batch_run_{cinema,dino,sam}_{logreg,finetune}_classification.sh` | SLURM arrays (cinema `0-5`, dino `0-17`, sam `0-8`) |

Tests: [tests/test_classification_multi_dataset.py](../tests/test_classification_multi_dataset.py)
(421 lines, synthetic) covers the class registry, `build_patient_features`,
`evaluate_classification` at both 5 and 6 classes, binary detection, `sweep_C_and_train`, and
the head predictor. Untested: all the `cache_*_cls_features` functions,
`finetune_sweep_and_train`, `build_summary.py`, and both significance scripts.

---

## 2. Branch situation

**The raw results are identical.** `git diff main...sam2-classification` over
`results/classification/*/*.json` is **empty** — all 100 run artefacts match byte-for-byte.
Nothing was re-run. This is the opposite of the segmentation picture.

Code changes are cosmetic: a ruff/mypy pass on `run_classification.py` (PEP 604 isinstance,
dropped an unused import) and two type-narrowing asserts in `classification_probe.py`. Seven
batch scripts changed mode 644→755 with zero content lines. The SAM2 removal touches nothing
here — classification only ever used SAM v1, and SAM has **gap pooling only** (no cls token in
its vision encoder) on both branches.

What the branch adds:

| | `main` | `sam2-classification` |
| --- | --- | --- |
| `acdc/summary.csv` rows | 5 | **22** (= 22 JSONs) |
| `mnm/summary.csv` rows | 6 | **23** (= 23 JSONs) |
| `mnm2/summary.csv` rows | 4 | **21** (= 21 JSONs) |
| `summary_aggregated.csv` | absent | 23 rows |
| `cinema_finetune_paper_auc.csv` | absent | present |
| AUC plots | none | 8 PNGs (2 top-level + 2 per dataset) |
| ACDC 5-way significance matrix | 33 methods (11 phantom) | 22 methods, all real |

**`sam2-classification` is the source of truth for classification too, and merging it loses
nothing.** `main`'s only unique classification content is the stale portion of its significance
matrices — 11 ACDC methods (`*_ft-full_*` and all four `dinov3_vit7b16_*`) whose result JSONs
exist on neither branch, so those p-values are unreproducible.

---

## 3. Results

### Cross-dataset 5-way ROC-AUC (`sam2-classification/results/classification/summary_aggregated.csv`)

| Backbone | logreg·cls | logreg·gap | ft-frozen·cls | ft-frozen·gap |
| --- | --- | --- | --- | --- |
| CineMA | 0.7257 | 0.7628 | 0.7361 | 0.7486 |
| DINOv3 vits16 | 0.7633 \* | 0.7534 | 0.7743 | 0.7607 |
| DINOv3 vitb16 | 0.7714 | **0.7856** | 0.7741 | 0.7698 |
| DINOv3 vitl16 | 0.7635 | 0.7533 | 0.7690 | 0.7549 |
| SAM vit_base | — | 0.7569 | — | 0.7365 |
| SAM vit_large | — | 0.7578 | — | 0.7518 |
| SAM vit_huge | — | 0.7561 | — | 0.7398 |
| **CineMA full fine-tune (published paper value)** | | | | **0.8420** |

\* averages only 2 datasets — the M&Ms-2 run is missing (see issue 4). Not comparable to the
3-dataset means beside it.

### Per-dataset accuracy / macro F1 / 5-way AUC, best configuration per backbone

| Backbone | ACDC | M&Ms | M&Ms-2 |
| --- | --- | --- | --- |
| CineMA | .660 / .658 / .881 | .505 / .320 / .688 | .361 / .337 / .702 |
| DINOv3 (best variant) | .620 / .616 / **.911** | .543 / .442 / .699 | .454 / .408 / **.792** |
| SAM v1 (best variant) | .600 / .603 / .869 | .495 / .360 / **.727** | .435 / .423 / .738 |
| Random baseline | .200 | .200 | .167 |

### What the numbers say

1. **The published full fine-tune beats every frozen probe on all three datasets** (mean AUC
   0.842 vs best-of-ours 0.786; on ACDC 0.980 vs 0.911). For the project's framing question —
   *are frozen representations enough?* — the classification answer is currently **no**, and
   more clearly so than for segmentation.
2. **CineMA's advantage is ACDC-only.** It wins on ACDC (0.660 acc, best in that column) but
   sits at or near the bottom on M&Ms and M&Ms-2, where DINOv3 leads. CineMA's M&Ms logreg·cls
   AUC of 0.581 is barely above chance.
3. **DINOv3-b16 is the strongest frozen backbone overall**; size effects are non-monotonic
   (vitl16 is generally *worse* than vitb16).
4. **SAM v1 unexpectedly leads on M&Ms** (large 0.727, huge 0.722 vs CineMA 0.688) despite
   being weakest on ACDC. Worth understanding before it appears in a write-up.
5. **`logreg` vs `ft-frozen` is a wash**, as it should be — they fit different heads on
   identical cached features.
6. **The binary NOR-vs-rest task is prevalence-dominated and close to uninformative** on
   M&Ms/M&Ms-2, where NOR is ~29% of the test set. Several models are all-positive predictors
   (sensitivity 1.0, specificity 0.0) yet post an F1 of ~0.83 — e.g. `mnm/cinema_pretrained_logreg_cls`
   and `mnm/sam_vit_base_logreg_gap`. Do not quote binary F1 without the specificity beside it.

### Significance testing

Both scripts are byte-identical across branches; only outputs differ.

- **5-way** (`mcnemar_five_way.csv`, `bootstrap_five_way_macro_f1.csv` + `_ci.csv`): current and
  self-consistent on the branch — 22/22/21 methods matching exactly the committed JSONs. On
  `main` the ACDC matrix still scores 33 methods including 11 phantoms.
- **Binary** (`mcnemar_binary.csv`, `bootstrap_binary_macro_f1.csv`): **stale on both
  branches**, ACDC only, and its header still lists the 33 stale methods. Nobody regenerated
  these.
- Neither script applies any multiple-comparison correction across 231–528 pairwise tests. With
  that many comparisons at α=0.05, ~12–26 "significant" hits are expected by chance alone.
  Any claim drawn from these matrices needs an FDR correction first.

`cinema_finetune_paper_auc.csv` (branch only) is a hand-transcribed **external literature
value** — the ROC-AUC the CineMA paper reports for full fine-tuning on these same three
datasets (97.98 / 77.41 / 77.20 %). It is injected as a reference bar in every plot and as the
`fine-tune` row of `summary_aggregated.csv`. Nothing in this repo produced it and there is no
recorded check that the splits and protocol match, so it is an indicative upper bound rather
than a controlled comparison.

---

## 4. Known problems

1. **Reproducibility is off, as in segmentation.** `set_seed` is commented out at both
   `run_classification.py:203` and `finetune_classification.py:243`. Worse here: the finetune
   path shuffles with an unseeded `np.random.permutation`, so finetune runs are genuinely
   non-reproducible. `--seed` also has no effect on CV splits — both `StratifiedKFold`s
   hardcode `random_state=0`. Logreg is deterministic in practice, so this bites only the
   `ft-frozen` rows.
2. **Skip-if-exists is dead code** (`run_classification.py:212–216`) — same bug that was fixed
   on the segmentation side but not here. `json_path` is built with a fresh timestamp, so
   `.exists()` is always False. Consequence: re-runs accumulate duplicate JSONs, and
   `_common.load_classifiers` keys by label, so a duplicate **silently overwrites** its twin in
   the significance tests with no error.
3. **There is already such a duplicate**: `mnm/cinema_pretrained_logreg_gap_20260418_164436.json`
   and `..._164804.json` are byte-identical. This is why `mnm/summary.csv` has 23 rows for 22
   distinct configurations and why McNemar loads 22 classifiers from 23 files.
4. **`mnm2/dinov3_vits16_logreg_cls` is missing** — 21 runs where the grid calls for 22. It
   existed before the 2026-04-21 sweep (it is still a row in the older matrices) and was lost.
   Its absence silently produces the 2-dataset mean flagged in the AUC table above.
5. **M&Ms-2 is 6-class but named `five_way` everywhere** — JSON keys, output filenames, CSV
   names, PNG titles, and the `--task five_way` CLI flag. The metrics themselves are computed
   correctly for 6 classes (`num_classes: 6` is recorded in each JSON); only the naming lies.
   `build_summary.py` also hardcodes "5-way" column labels.
6. **SAM silently ignores `--pooling`.** `cache_sam_cls_features` has no `pooling` parameter and
   always global-average-pools, but the cache directory and the recorded config both use the
   requested pooling name. `--backbone sam --pooling cls` would write GAP features into a
   `.../cls/` directory and label the results `cls`. **No committed result is affected** — every
   SAM run on disk is gap — but nothing in the driver rejects the combination.
7. **`ft-full` is unreachable but still referenced.** The `--freeze-backbone` flag is gone and
   `eval_mode_tag` hardcodes frozen, yet `build_summary.py:45` and `_common.py:44` still branch
   on a `freeze_backbone` key that current JSONs no longer write. Every finetune row is labelled
   `ft-frozen` by default — correct, but by accident.
   `scripts/classification/README.md` still documents the removed flag, so its quick-start
   command fails with `unrecognized arguments`.
8. **Finetune keeps the last-epoch head, not the best one.** `_train_with_lr_cached` returns
   `best_head_state`, but the final retrain at `finetune_classification.py:323` discards the
   return value and passes `all_idx` as both train and val with `patience=epochs` — so
   "validation" accuracy is training accuracy and early stopping is disabled for the final
   model.
9. **`prompts/restructuring_plan.md` says the finetune path should have been deleted** — the
   plan of record settles on a frozen-backbone-only framework. That removal never happened, so
   `finetune_classification.py`, its SLURM scripts, and all `*_ftfrozen_*` results are leftovers
   from a decision recorded as final. Either the decision or the code needs to change.
10. **Debris**: three `.json.bak` files in `results/classification/acdc/`. Two differ from their
    live counterparts only by float noise (~1e-7); `dinov3_vits16_ftfrozen_cls.json.bak` is
    **materially different** (acc .54 vs .52, different confusion matrix and per-sample
    predictions) — a genuine earlier run, consistent with issue 1. Also
    `scripts/classification/run_acdc_classification.py` is dead and raises `ImportError`, and
    `smoke_test_classification.sh` still targets it.
11. Minor: smoke-run JSONs (`_smoke` suffix) are not excluded by any glob in `build_summary.py`
    or the significance scripts; `roc_auc_score` raises if a class is absent from a test split,
    reachable with `--max-patients` on M&Ms-2; CineMA silently truncates volumes past 16 z-slices.

---

## 5. Open decisions

**Merge is unambiguous here.** Unlike segmentation, merging `sam2-classification` costs nothing
and fixes real problems: `main`'s summary tables miss ~75% of completed runs and its ACDC
significance matrix scores 11 methods that do not exist. If the branches are reconciled for
segmentation reasons, classification comes along for free.

**Regenerate the binary significance tests, or drop the binary task?** They are stale on both
branches, exist for ACDC only, and the underlying task is prevalence-dominated on the other two
datasets. Dropping it may be cleaner than fixing it.

**Apply a multiple-comparison correction.** With 231–528 pairwise tests per matrix and no
correction, the current "significant" counts are not defensible as-is.

**Resolve the finetune question** (issue 9) — either delete the path per the restructuring plan,
or update that plan to record that `ft-frozen` was deliberately kept as a second head type.

**Decide what to do about the frozen-vs-fine-tuned gap.** The headline finding is that frozen
probes trail the published fine-tune by ~0.06 mean AUC. That is a legitimate result for the
project's research question, but it currently rests on a literature number rather than an
in-house fine-tune, and on single unseeded runs. Reproducing at least the CineMA fine-tune
locally would make the comparison defensible.
