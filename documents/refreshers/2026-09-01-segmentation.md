# Segmentation refresher — 2026-09-01

State of the segmentation task, written while feature re-extraction is in progress on
`re-cache-all-features`. Supersedes
[2026-08-09-segmentation.md](2026-08-09-segmentation.md) — read this one first; the older
document is still the best reference for the *unchanged* parts of the code path (library
module roles, decoder internals, loss/metric semantics), so those are not repeated here.

| | |
| --- | --- |
| Written | 2026-09-01 |
| `re-cache-all-features` (current) at | `4b0b368` |
| `main` at | `7b1210b` — **ancestor of HEAD**, no longer forked |
| `sam2-classification` at | `c237fc8` (unchanged since 2026-08-09) |
| Merge base HEAD / `sam2-classification` | `7517e7e` |
| Test suite | 33 pass; bare `pytest` **fails at collection** (see new issue N4) |

**One-line summary:** the code is now the best it has been — one driver handles all four
backbone families, a real `--cache-only` path exists, and a genuine correctness bug in the
CineMA 2D cache has been fixed — but **every committed segmentation number is stale**, the
CineMA Linear/Conv numbers are outright invalid, and the two things that will actually block
the re-run are **disk (need ~640 GiB, have 119 GiB free)** and **SAM v1 extraction time on MPS
(~4 days)**. Reproducibility is still off.

---

## 1. What changed since 2026-08-09

Ten commits, all on top of `main`. `main` is now an **ancestor** of HEAD, so the
`main` / HEAD fork described in the old refresher is gone. Only `sam2-classification`
is still divergent, and only in three files.

### The big one: CineMA 2D cache was misaligning features and labels along z

`a020ad4` (PR #65, closes #64 — on `main` **and** HEAD). `cache_cinema_2d_features` was
rescaling the slice index onto the feature volume's depth axis:

```python
feat_z = int(round(src_z * (gz - 1) / max(used_depth - 1, 1)))
feats_2d = feat_vol[..., feat_z]      # features from slice feat_z
label_2d = label_3d[0, :, :, z_idx]   # label from slice z_idx
```

Nothing in the CineMA encoder downsamples z (conv strides `(4,4,1)` and `(2,2,1)`, patch size
`(2,2,1)`), so the correct mapping is the identity. Every cached CineMA 2D slice paired one
slice's label with a *different* slice's features, and later slices were paired with features
computed from zero padding. The commit message records the verification: before the fix each
slice best-matched the stretched `feat_z` (cosine 0.89–0.92) and the identity match fell to
0.31; after, every slice best-matches its own depth.

**This invalidates every committed CineMA `linear_probe` and `conv_decoder` result on every
branch.** I confirmed the magnitude with a 2-epoch smoke run on the freshly re-extracted ACDC
cache:

| CineMA ACDC linear_probe | Macro Dice |
| --- | --- |
| Committed (100 epochs, misaligned cache) | 0.7386 |
| Fresh cache, **2 epochs** | **0.8271** |

Two epochs already beats the published 100-epoch number by 0.09. The old refresher's issue 2
("CineMA is worst with weak decoders") was mostly this bug, not a property of the model. The
"CineMA worst with Linear/Conv, best with UNETR" story in the write-up is very likely wrong.

Two side effects of the fix:

- The slice loop is now bounded by `min(n_slices, SAX_TARGET_DEPTH)`, guarded by
  `assert used_depth == n_cached`. Only M&Ms-2 has >16-slice patients (1 train, 1 test), so
  the CineMA 2D cache for mnm2 holds 3438/3560 slices where DINOv3/SAM hold 3440/3562.
  This *resolves* old issue 8 (the 3562-vs-3560 puzzle was exactly this truncation in the 3D
  path) and replaces it with a smaller CineMA-2D-vs-other-2D gap of 2 slices in 3562 (0.06%).
- `.contiguous()` on the sliced feature and label. Slicing `feat_vol` returned a strided view
  whose whole `(C, gx, gy, Z)` storage was being serialised — **16× the real tensor**. Observed
  on the ACDC CineMA 2D cache: **18 GB before → 3.1 GB after**. Values unaffected.

### SAM v1 now drives segmentation on the mainline

`b901199`. The old refresher's central finding — that `main` had SAM2 and
`sam2-classification` had SAM v1, and the two were mutually exclusive — no longer holds.
HEAD supports **both**:

- `--backbone {dinov3,cinema,sam,sam2}`; `--sam-model-id` (default `facebook/sam-vit-base`).
- New `SAM_CONFIGS` in `backbones.py` with `n_layers` and `layer_indices` for base
  `(2,5,8,11)`, large `(5,11,17,23)`, huge `(7,15,23,31)`; unknown ids fail fast before any
  hub download. `embed_dim` is still read from the checkpoint config, not the table.
- New `extract_sam_2d_features` / `cache_sam_2d_features`, mirroring the SAM2 pair.
- `extract_sam_volume_features` was generalised to serve both families (`hidden_states` shape
  is no longer hardcoded to `(B,64,64,768)`).
- `SAM2_CONFIGS` gained `cls_embed_dim` (Stage 4, classification) alongside `embed_dim`
  (Stage 3, segmentation), and `_load_sam2`'s `model_name` now also maps `-` → `_`.
- `batch_run_sam_{linear_probe,conv_decoder,unetr}_segmentation.sh` added (arrays `0-8`);
  the `batch_run_sam2_*` scripts are **still present**, so the SLURM grid is now 11 models.

### `--cache-only`, `--max-patients`, and the cache orchestrator

- `--cache-only` extracts and exits. It deliberately does **not** create `results/` — the
  `json_path` is only built when training will happen.
- `--max-patients` for smoke tests. For segmentation it is a plain `head()` (the label is
  per-pixel anatomy, so pathology balance is irrelevant) and raises a clear error if the ACDC
  val carve-out would leave no training patients. `subset_patients_stratified()` in `data.py`
  is the classification counterpart.
- Cache root renamed `feature_cache/` → **`feature_cache_segmentation/`** (`2062943`), matching
  `feature_cache_classification/`. Both are gitignored, as is `feature_cache_backups/`.
- **`scripts/cache_all_features.sh`** (`a5c283f`, `4b0b368`) — the main new piece of
  infrastructure. Enumerates *cache keys*, not experiments: `--decoder conv_decoder` covers the
  2D cache that `linear_probe` shares, `--decoder unetr` covers the 3D cache, `--eval-mode
  logreg` covers the cache `finetune` shares. 75 runs instead of 141. Per-config logs, failures
  recorded and skipped over, non-zero exit if anything failed, `DRY_RUN=1`, and env overrides
  (`DATASETS`, `TASKS`, `BACKBONES`, `DINO_MODELS`, `SAM_MODELS`). **SAM2 is deliberately
  excluded** — passing `BACKBONES=sam2` exits 2. The header also warns not to pass a non-default
  `--seed`, since the ACDC val split is seed-derived but the cache path does not record it.
- The debug `print('Validation PIDs …')` in `data.py` is still there (it is genuinely useful for
  confirming the ACDC split is stable across models, so arguably fine).

### Not on HEAD, still only on `sam2-classification`

The branch is otherwise a strict subset of HEAD now. Three things are missing:

| Missing on HEAD | Size | Why it matters |
| --- | --- | --- |
| `scripts/segmentation/plot_dataset_macro_dice_ci.py` | 227 lines | The only per-dataset CI plot script. |
| `aggregate_summary.py` improvements | 18 lines | Fallback palette + "keep groups not in `*_ORDER`". **Without it, SAM v1 rows silently vanish from the plots** — see new issue N1. |
| `decoders.py` cosmetics | 14 lines | Two dead `import torch` inside `forward()`, docstring fixes. |

`report_macro_dice_ci.py` and `build_summary.py` are byte-identical between the branches.

---

## 2. Results: everything committed is stale

`results/segmentation/` on HEAD is **`main`'s 2026-04-18/21/22 SAM2-era generation** — 21 JSONs
per dataset (7 models × 3 decoders), no SAM v1, no bootstrap-CI outputs. It is the table the old
refresher labelled "a stale snapshot", and it is now stale in three separate ways:

1. **CineMA Linear/Conv rows are invalid** (z-misalignment). Empirically ~+0.09 too low.
2. **SAM rows are SAM2**, but the mainline code and `cache_all_features.sh` are SAM v1.
3. No CIs — `sam2-classification` has `macro_dice_ci_summary.csv` and the CI PNGs; HEAD does not.

`sam2-classification`'s results (2026-04-29 → 05-05, SAM v1, with CIs) are internally consistent
and were produced by essentially HEAD's SAM code — but its CineMA Linear/Conv rows carry the
**same** z-misalignment bug, since the fix postdates that branch by four months.

**Net: there is currently no branch whose segmentation results are trustworthy end-to-end.**
Treat the tables in the 2026-08-09 refresher as historical. The one claim I would still expect
to survive the re-run is that decoder capacity dominates backbone choice (Linear ≈ 0.68–0.78 vs
Conv/UNETR ≈ 0.81–0.87); the CineMA story and the DINOv3-vs-SAM ordering are both in play.

---

## 3. Where the re-extraction actually stands

Mid-run, ACDC only, 2D only:

| Cache | Files | Size | Status |
| --- | --- | --- | --- |
| `acdc/dinov3_vits16/{train,val,test}/layers_3-6-9-11` | 1534 / 178 / 916 | 1.7 G train | complete |
| `acdc/cinema_pretrained/{train,val,test}` | 1534 / 178 / 916 | 1.1 G train | complete (post-fix) |
| `acdc/sam_vit_base/train/layers_2-5-8-11` | 896 of 1534 | 2.2 G | **in progress** |

Total so far 6.4 GiB. Nothing yet for `*_unetr3d` (3D), `dinov3_vitb16/vitl16`,
`sam_vit_large/huge`, or mnm / mnm2. Cached tensors are `(C, 12, 12)` float32 features plus a
`(192, 192)` int64 label.

Observed throughput on MPS, per (patient, frame) sample:

| Backbone | s/sample | Relative |
| --- | --- | --- |
| DINOv3 vits16 | 0.25 | 1× |
| CineMA | 0.85–1.0 | ~4× |
| **SAM v1 vit_base** | **9.3** | **~37×** |

SAM is the outlier because its ViT runs at 1024×1024 (64×64 tokens) and is then interpolated
down to 12×12, one slice at a time.

---

## 4. Blockers to clear before re-running segmentation

Ordered by how likely each is to stop you.

### B1 — Disk. The full grid needs ~640 GiB; 119 GiB is free.

Derived from the observed per-file sizes (`576·C + 294912` bytes per 2D slice;
`~252 MiB` summed per 3D patient-frame across all seven models), over 17 548 2D slices and
1 636 patient-frames:

| | ACDC | M&Ms | M&Ms-2 | **Total** |
| --- | --- | --- | --- | --- |
| 2D caches (7 models) | 36 GiB | 96 GiB | 107 GiB | **239 GiB** |
| 3D caches (7 models) | 74 GiB | 156 GiB | 173 GiB | **403 GiB** |
| **Both** | **110 GiB** | **252 GiB** | **280 GiB** | **~642 GiB** |

ACDC alone (110 GiB) only just fits in what is free, and that is before anything else on the
machine grows. Options, cheapest first:

- **Delete `feature_cache_backups/` (35 GiB).** It is gitignored and its CineMA 2D cache is the
  *misaligned* one — keeping it is a footgun as well as a cost. Verify nothing else needs it,
  then remove.
- **Store labels as `int8`, not `int64`.** With 4 classes this is free accuracy-wise and saves
  0.28 MiB per 2D slice (**40% of a CineMA slice**, 25% of a DINOv3-S slice) and 4.1 MiB per 3D
  file. Worth ~60 GiB across the grid. Requires a `.long()` at load time in the `Dataset`.
- **Stop duplicating `image` + `label` in every backbone's 3D cache.** 6.75 MiB × 1636 × 7
  models ≈ 75 GiB of identical data.
- **Extract per dataset, train, then delete that dataset's cache** before moving on. The driver
  needs the cache only for the duration of the run, and `cache_all_features.sh` takes
  `DATASETS=`.
- **Do the SAM and large-DINOv3 grid on Isambard-AI instead**, which is where B2 pushes you
  anyway.

### B2 — SAM v1 extraction on MPS is ~4 days of wall-clock.

Extrapolating 9.3 s/sample and assuming roughly FLOP-proportional scaling for large/huge:

| Model | 2D, all 3 datasets | 3D, all 3 datasets |
| --- | --- | --- |
| sam_vit_base | ~5 h | ~5 h |
| sam_vit_large | ~15 h | ~15 h |
| sam_vit_huge | ~28 h | ~28 h |

≈ 96 h for the SAM family alone. DINOv3 and CineMA together are a few hours. Two things would
make a large difference and are worth doing before committing days of compute:

- **Batch the slices.** `extract_sam_2d_features` and `extract_sam_volume_features` both run the
  vision encoder on one 1024×1024 slice at a time. Batching over z is a near-trivial change and
  should be a large win on a GPU.
- **Write both caches from one pass.** The 2D and 3D SAM caches do the *same* per-slice encoder
  forwards for the same (model, dataset) and then store them differently — the entire SAM budget
  is being paid twice. A single extraction writing both would halve it.

Either way, run the SAM grid on Isambard-AI (`batch_run_sam_*_segmentation.sh`), not locally.

### B3 — Delete the old result JSONs first, or summaries will silently blend old and new.

HEAD does **not** have `sam2-classification`'s skip fix (`6ead620`). On HEAD the check is:

```python
timestamp = datetime.now().strftime(...)
json_path = args.output_dir / f"{base_name}_{timestamp}.json"
if json_path.exists(): ...   # dead code: a fresh timestamp never collides
```

So re-running writes a *second* JSON alongside the old one. `build_summary.py` emits one row per
JSON with no dedup, and `aggregate_summary.py` pivots with `pivot_table` (default
`aggfunc='mean'`), so a duplicated config is **silently averaged with its stale predecessor**.
Nothing errors and nothing warns.

Decide deliberately: either `git rm` the old per-dataset JSONs/CSVs before re-running (and keep
the SAM2 generation on an archive branch or tag), or port `6ead620` and delete per config as you
go. Do not leave both generations in the directory.

### B4 — Turn `set_seed()` back on.

Still commented out — `run_segmentation.py:194`, and the same in
`scripts/classification/run_classification.py:226` and
`src/heartfm_evals/finetune_classification.py:243` — while `"seed": 0` is still written into
every results JSON. Only the ACDC val split and the train-loader shuffle generator are
deterministic; decoder init, dropout and BatchNorm are not. Cross-branch reruns of identical
configs differed by 0.001–0.005 macro Dice, which is the same magnitude as the model-size effects
being reported. You are about to regenerate the entire grid — this is the cheapest moment there
will ever be to fix it, and to run ≥3 seeds on the headline backbone × decoder cells.

### B5 — Port `aggregate_summary.py` from `sam2-classification` before plotting.

See new issue N1. Without it the SAM v1 bars you are about to generate do not appear.

---

## 5. Issue status vs the 2026-08-09 refresher

| # | Issue | Status |
| --- | --- | --- |
| 1 | `set_seed()` commented out | **open** — see B4 |
| 2 | CineMA Linear/Conv not apples-to-apples | **mostly a bug, now fixed.** The z-misalignment was the dominant cause. The residual asymmetry is real but small: CineMA 2D features are single-layer by construction (`feature_forward` returns only the final encoder output), so `use_layers=(0,)` and Conv sees 768 channels where DINOv3-S/B see 1536/3072. Not fixable without extracting CineMA intermediate layers. |
| 3 | CIs cover only UNETR | **open.** `report_macro_dice_ci.py:255` still defaults to `--pattern '*_per_sample.csv'`, which only the 3D decoder emits; `*_per_stack.csv` has an identical schema and is silently excluded. `infer_aggregation_unit` still labels anything with `n_slices` as `"sample"`. One-line fix (`'*_per_{sample,stack}.csv'` or unify the filenames) that would triple the CI table's coverage. |
| 4 | `--use-layers` broken for `conv_decoder` | **open.** `run_segmentation.py:441` still sets `cached_layers` only when `args.decoder == "linear_probe"`, so `ConvDecoderProbe` is sized `embed_dim * len(use_layers)` while the cache holds all four layers → channel mismatch on the first forward. Same class of failure for `unetr`. Low priority: nothing in the batch scripts passes `--use-layers`. |
| 5 | `sam2.1-hiera-tiny` never run | **open** and now larger. `run_all_segmentation.sh` runs **11** models × 3 decoders × 3 datasets = 99 configs (SAM v1 *and* SAM2, tiny included), while `cache_all_features.sh` pre-caches only the 7 SAM-v1-and-friends keys. Running the full script would extract SAM2 features on the fly, inside the training job. Pick one grid and make the two scripts agree. |
| 6 | Orphan `dinov3_vits16_unetr_20260429_170635_per_sample.csv` | **open on `sam2-classification`** (leaks a metadata-less row into the ACDC CI table). Not present on HEAD, which never had that generation. |
| 7 | `README.md` stale for segmentation | **open.** Still documents `run_sam_variants.sh` / `run_sam3.sh` / `run_dino_unetr_variants.sh` (all gone) and claims outputs go to `scripts/segmentation/`. Still no `scripts/segmentation/README.md`, unlike classification. |
| 8 | M&Ms-2 test set 3562 (2D) vs 3560 (3D) | **explained and mostly resolved** — see §1. Now a 2-slice CineMA-2D-vs-other-2D gap instead. |
| minor | `compute_class_weights` re-`torch.load`s every training slice per run | **open.** Now more expensive than it was, since it runs once per config over a bigger grid. |
| minor | debug `print` of validation PIDs in `data.py` | still present; arguably useful — it is how you confirm the ACDC split is identical across models. |

---

## 6. New issues found this pass

**N1 — `aggregate_summary.py` on HEAD silently drops SAM v1.**
`BACKBONE_ORDER = ["CineMA", "Dino", "SAM2"]` and `BACKBONE_COLORS` have no `"SAM"` entry.
`_grouped_bar_plot` filters groups to `[g for g in group_order if g in present]`, so in
`macro_dice_by_backbone.png` the SAM group is **omitted entirely**; in
`macro_dice_by_decoder.png` the colour lookup is `color_map.get(key, "#888")`, so SAM bars come
out uniformly grey with no legend entry. No error, no warning. `summary_aggregated.csv` keeps the
rows (they sort to the end with a NaN order key), so the CSV and the PNGs disagree. Fix by taking
`sam2-classification`'s version, which adds a fallback palette and appends present-but-unordered
groups — or minimally add `"SAM"` to both dicts.

**N2 — SAM layer indices are off by one relative to DINOv3, so SAM never sees its final block.**
`backbones.py` documents that `hidden_states[i+1]` is the output of block `i`, but
`extract_sam_2d_features` / `extract_sam_volume_features` index `hidden_states[idx]` with the raw
`layer_indices`. Verified empirically on `facebook/sam-vit-base` (transformers 5.5.3): 13 hidden
states for 12 blocks, `hs[0]` is the patch embedding and `hs[12]` is the final block. So
`layer_indices=(2,5,8,11)` reads blocks **(1,4,7,10)** — the last block is never read.
DINOv3 goes through `get_intermediate_layers(n=list(layer_indices))`, which *is* block-indexed,
so `(3,6,9,11)` does include block 11. The comparison is therefore slightly unfair to SAM, in
the direction that matters (the final block is usually the most linearly separable). Same shift
for large `(5,11,17,23)` → blocks (4,10,16,22) and huge `(7,15,23,31)` → (6,14,22,30). Cheap to
fix (`hidden_states[idx + 1]`) but it **invalidates any existing SAM cache**, so decide now,
before spending the 96 h in B2. Worth checking whether `extract_sam2_2d_features` has the same
issue against the Stage 3 ranges in `documents/prompts/sam2_decisions.md`.

**N3 — `cache_all_features.sh` and `run_all_segmentation.sh` describe different grids.**
The former covers 7 cache keys and rejects SAM2; the latter runs 11 models including SAM2 and
`hiera-tiny`. Also worth noting `cache_all_features.sh` covers *both* tasks, so
`TASKS=seg` is what you want if you are only re-running segmentation.

**N4 — bare `pytest` fails at collection.**
`pyproject.toml` sets `filterwarnings = ["error"]`, and importing
`tests/test_classification_multi_dataset.py` pulls in `monai.networks.nets.dints`, which trips
`DeprecationWarning: torch.jit.interface is deprecated`. Collection aborts, so **zero** tests
run. All 33 pass with `-W ignore::DeprecationWarning`, and
`pytest tests/test_segmentation_metrics.py` alone passes (7 tests) because it never imports
`dints`. Fix with a targeted `filterwarnings` ignore entry for that one warning rather than
loosening `"error"`. This is dependency drift, not a code regression — but it means CI and
`pre-commit` habits are currently giving you no signal.

**N5 — test coverage is still segmentation-metrics-only.** `tests/test_segmentation_metrics.py`
gained `sam_model_id` / `cache_only` / `max_patients` to its two monkeypatched end-to-end
namespaces, and nothing else. Nothing covers `backbones.py`, cache-key logic, the new
`cache_sam_2d_features`, decoder shapes, or losses — and in particular **nothing would have
caught the CineMA z-misalignment**. A test asserting that a cached 2D slice's features match the
corresponding depth of the 3D feature volume would be cheap and would pin the fix.

---

## 7. Suggested order of work

1. Port `aggregate_summary.py` + `plot_dataset_macro_dice_ci.py` from `sam2-classification`
   (**B5/N1**) and un-comment `set_seed()` (**B4**). Both are minutes and both affect everything
   downstream.
2. Decide **N2** (the SAM off-by-one) *before* extracting any more SAM features — it is the only
   open question that would force a re-extraction.
3. Fix **N4** so `pytest` runs, and add the cache-alignment test from **N5**.
4. Free disk (**B1**): drop `feature_cache_backups/`, switch labels to `int8`.
5. Archive the SAM2-era results on a tag, then `git rm` them (**B3**).
6. Batch the SAM slice loop and merge the 2D/3D extraction pass (**B2**), then run the SAM and
   large-DINOv3 grids on Isambard-AI. Keep the local machine for CineMA + DINOv3-S/B.
7. Extract → train → delete, one dataset at a time. Re-run `build_summary.py`,
   `aggregate_summary.py`, `report_macro_dice_ci.py` (with the pattern fixed, **issue 3**) and
   `plot_dataset_macro_dice_ci.py` at the end.
8. Reconcile `run_all_segmentation.sh` with `cache_all_features.sh` (**N3**) and update
   `README.md` (**issue 7**).

The headline expectation to hold loosely: with the CineMA cache fixed, **CineMA may now be
competitive or best at Linear and Conv as well as UNETR**, which would change the paper's story
from "cardiac pretraining only helps with a strong 3D decoder" to something closer to "cardiac
pretraining helps throughout". Do not write either sentence until the grid is re-run under a
fixed seed.
