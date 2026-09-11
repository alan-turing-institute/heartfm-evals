# Segmentation refresher — 2026-09-11

State of the segmentation task after five PRs landed on `main` between 2026-09-08 and
2026-09-11. Supersedes [2026-09-01-segmentation.md](2026-09-01-segmentation.md), which
in turn supersedes [2026-08-09-segmentation.md](2026-08-09-segmentation.md). Read this
one first; the 09-01 document remains the reference for the parts that have **not**
changed (library module roles, decoder internals, loss/metric semantics, the CineMA
z-misalignment story), and those are not repeated here.

| | |
| --- | --- |
| Written | 2026-09-11 |
| `main` at | `9acc519` |
| Previous refresher was written at | `4b0b368` (branch `re-cache-all-features`, now merged) |
| `sam2-classification` at | `a77c030` — **moved** since 09-01, three new commits |
| SAM2 archive tag | `archive/sam2-2026-09-08` → `4728304` (pushed to origin) |
| Test suite | **39 pass**, `pre-commit run -a` clean, CI restored |

**One-line summary:** the codebase is now in genuinely good shape — SAM2 is gone and
properly archived, the SAM v1 off-by-one is fixed *and* its taps moved to the
global-attention blocks, tests/lint/CI are all green, and `re-cache-all-features` has
landed on `main`. Nothing blocks a clean re-run except the two physical constraints that
have not moved: **disk (~642 GiB needed, 95 GiB free)** and **SAM extraction time (~5
days on MPS)**. The committed segmentation results are still the April generation minus
the deleted SAM2 rows, and `summary.csv` / `summary_aggregated.csv` / the two PNGs were
never regenerated after that deletion, so they currently disagree with the JSONs on disk.

---

## 1. What landed since 2026-09-01

Five merged PRs. All of `re-cache-all-features` is now on `main`, so the branch-vs-branch
confusion in the last two refreshers is over.

### PR #67 — SAM v1 off-by-one fixed, and taps moved to global-attention blocks (`4728304`, 09-08)

This closes issue #66, which was **N2** in the 09-01 refresher, and does more than the
minimal fix. Three parts:

1. **One indexing convention, one place.** New `features.py::_block_hidden_state(hidden_states,
   block_idx)` is now the *only* place the `+1` is applied. Every `layer_indices` in the
   codebase is a **block** index; SAM resolves it through the transformers `hidden_states`
   tuple (index 0 is the patch embedding) while DINOv3's `get_intermediate_layers` is
   block-indexed already and needs no shift. It raises `IndexError` naming both the block
   index and the resolved position.
2. **The values are `global_attn_indexes`, not quartiles.** `SAM_CONFIGS` keeps
   `(2,5,8,11)` / `(5,11,17,23)` / `(7,15,23,31)` — but these are now documented as each
   model's global-attention blocks: SAM's encoder runs at 1024×1024 with patch 16
   (64×64 = 4096 tokens), so full self-attention is paid for only four times
   (`window_size=0`); every other block is restricted to 14×14 = 196-token windows.
   Tapping a windowed block gives window-limited features. I verified this against the
   real checkpoint — `[i for i,l in enumerate(vision_encoder.layers) if l.window_size == 0]`
   is exactly `[2, 5, 8, 11]` for `sam-vit-base`. `layer_indices[-1]` is both the final
   block and a global one, so `linear_probe` (which uses only the last entry) now gets a
   globally-attended tap.
3. **SAM2's indices were renumbered down by one** to the same block convention (blocks
   read unchanged) — work that PR #68 then deleted two days later.

**Net behavioural change:** SAM v1 features now come from blocks (2,5,8,11) where they
previously came from (1,4,7,10). Any SAM segmentation result or cache predating 09-08 is
obsolete. See **N1** below for the cache-key trap this creates.

New `tests/test_feature_layer_indexing.py` (6 tests, all synthetic stubs, no weights
downloaded) pins the convention: the `+1`, the out-of-range message, that
`SAM_CONFIGS` *are* the global-attention blocks ("guards against the indices drifting
back to plain even quartiles"), that `max(layer_indices) == n_layers - 1`, and that the
2D and volume extractors both read `hs[i+1]`.

### PR #68 — SAM2 removed outright (`63cd767`, 09-10)

83 391 deletions. `SAM2_CONFIGS`, `_load_sam2`, `extract_sam2_2d_features`,
`cache_sam2_2d_features`, `cache_sam2_cls_features`, `cls_embed_dim`, the five
`batch_run_sam2_*` SLURM scripts, the `--backbone sam2` / `--sam2-model-id` flags, and
the **81 committed SAM2 segmentation result files** are all gone. `grep -ri sam2 src/
scripts/ tests/` returns nothing but a stale `egg-info/PKG-INFO` artifact.

Archived properly: tag **`archive/sam2-2026-09-08`** exists locally *and* on origin, and
holds the full implementation plus all 81 result files.
`documents/prompts/sam2_decisions.md` was trimmed 427 → 154 lines and rewritten as a
removal record with the recovery commands at the top. The stated rationale — Hiera's
non-uniformity, the shared-extractor complication the #66 fix hit, the Stage 3 / Stage 4
split, and the fact that the genuinely multi-scale option was analysed but never applied —
is sound and worth re-reading before anyone proposes bringing SAM2 back.

`AGENTS.md` is updated throughout: backbone table down to three, `documents/prompts/`
paths corrected, the `sam2-classification` entry rewritten from "most active, ahead of
`main`" to "a historical record, now **behind** this line of work", and a new gotcha
spelling out the block-index convention in full.

### PR #69 — `re-cache-all-features` merged (`c74fc81`, 09-10)

Everything from the 09-01 refresher's §1 is now on `main`: `--cache-only`,
`--max-patients`, `feature_cache_segmentation/`, `scripts/cache_all_features.sh`, the
CineMA z-alignment fix, and SAM v1 segmentation support.

### PR #73 — tests, pre-commit and CI restored (`9acc519`, 09-11)

Closes **N4** from the 09-01 refresher. Verified all three locally:

- **`pytest` → 39 passed.** The fix is a message-scoped `filterwarnings` entry
  (`"ignore:.*torch\\.jit\\.interface.* is deprecated"`) rather than loosening `"error"`,
  with a comment explaining the category is left unspecified on purpose because torch
  raises it as `DeprecationWarning` on the pinned version and `FutureWarning` on newer
  ones. Coverage config (`run.source`) was also fixed — it had been inert.
- **`pre-commit run -a` → all 14 hooks pass**, including ruff, ruff-format and mypy.
  20 ruff and 3 mypy errors were fixed in `fa07f44`.
- **`.github/workflows/ci.yml` restored**: pre-commit job, then tests on Python 3.11 and
  3.13. Installs with `uv pip install --torch-backend=cpu` (documented: `uv sync` rejects
  the flag, and the default Linux wheels pull 18 CUDA packages). Both git dependencies
  (CineMA, dinov3) are now pinned, and CineMA was repointed at upstream.

### PR #71 — `run_acdc_classification.py` deleted (`848e0e1`, 09-10). Classification only.

---

## 2. Results: correct but stale, and the summaries now disagree with the JSONs

`results/segmentation/{acdc,mnm,mnm2}/` holds **12 JSONs each** (4 models × 3 decoders:
CineMA + DINOv3 S/B/L), all from 2026-04-18/21. The 9 SAM2 configs per dataset were
deleted by PR #68.

**But the summaries were not regenerated.** `summary.csv` (all three) and
`summary_aggregated.csv` still carry the 9 SAM2 rows whose source JSONs no longer exist,
and both PNGs still plot them. I reran `build_summary.py` + `aggregate_summary.py` against
a scratch copy of the current tree: the regenerated output is **identical except that the
9 SAM2 rows are gone**. So this is a one-command fix, not a recomputation:

```bash
python scripts/segmentation/build_summary.py
python scripts/segmentation/aggregate_summary.py
```

Do it now rather than later — an orphan row in a committed summary is exactly the kind of
thing that ends up in a paper table.

Beyond the SAM2 orphans, the 12 surviving rows are still the April generation and remain
subject to the CineMA verdict from the 09-01 refresher: **CineMA `linear_probe` and
`conv_decoder` are invalid** (z-misalignment, fixed in `a020ad4`; a 2-epoch run on the
fixed cache beat the published 100-epoch number by 0.09). There are **no SAM v1
segmentation results anywhere on `main`**, and none that are valid anywhere else — see §3.

| What | State |
| --- | --- |
| CineMA UNETR, DINOv3 all decoders | April runs; unseeded, single seed, but methodologically fine |
| CineMA Linear / Conv | **invalid** — misaligned cache |
| SAM v1, all decoders | **absent on `main`** |
| SAM2, all decoders | deleted; recoverable from `archive/sam2-2026-09-08` |
| Bootstrap CIs | absent on `main` |

---

## 3. `sam2-classification` has moved — and now holds two scripts worth taking

The branch is **not** an ancestor of `main` and has gained three commits since the 09-01
refresher (`c237fc8` → `a77c030`, all 09-08). Two segmentation scripts exist only there:

| Script | Lines | Why it matters |
| --- | --- | --- |
| `plot_dataset_macro_dice_ci.py` | 227 | Per-dataset CI plots. Flagged as missing on 09-01; still missing. |
| **`plot_pooled_backbones.py`** | 299 | **New.** Pools per-stack macro Dice across all three datasets into one sample and bootstraps *that*, one panel per decoder, shared y-axis. |

`plot_pooled_backbones.py` is the more interesting of the two and its docstring is worth
reading: it discovers result files from the run JSON sidecars **specifically because**
`macro_dice_ci_summary.csv` only indexes `*_per_sample` and so misses the conv and linear
runs — i.e. it routes around old issue 3 rather than fixing it. It is also explicit that
pooling weights datasets by stack count (ACDC 100, M&Ms 268, M&Ms-2 314) and is *not* an
unweighted mean of three dataset means. Port both scripts to `main`.

The branch's own results (`summary_aggregated.csv`, `pooled_*_macro_dice_ci.csv`) are
**doubly obsolete** and should not be quoted: its CineMA rows carry the z-misalignment,
and its SAM v1 rows were produced with the pre-#67 off-by-one, i.e. from windowed blocks
(1,4,7,10) rather than the global-attention blocks. Its `Conv / CineMA = 0.664` pooled
figure is a pure artefact of the misalignment. Treat the branch as a source of *code*, not
numbers.

`aggregate_summary.py` on `main` has since diverged from the branch's version in its own
direction — **N1 from the 09-01 refresher is fixed**: `BACKBONE_ORDER` and
`BACKBONE_COLORS` now both contain `"SAM"`, so SAM v1 bars will render and be coloured.

---

## 4. Where re-extraction stands (as of 09:50, 09-11)

A run is **in progress right now** —
`run_segmentation.py --backbone cinema --decoder unetr --dataset acdc --cache-only`,
having just finished the CineMA 2D cache. The tree was evidently wiped and restarted
after the #67 SAM fix; the `sam_vit_base` 2D cache noted on 09-01 is gone.

| Cache | Files | Size | State |
| --- | --- | --- | --- |
| `acdc/dinov3_vits16/{train,val,test}` | 1534 / 178 / 916 | 2.9 GiB | complete |
| `acdc/dinov3_vits16_unetr3d/{train,val,test}` | 180 / 20 / 100 | 5.9 GiB | complete |
| `acdc/cinema_pretrained/train` | 617 → growing | 506 MiB | **in progress** |
| everything else (DINOv3 B/L, all SAM, mnm, mnm2) | — | — | not started |

Total 9.2 GiB. `feature_cache_backups/` is still 35 GiB and still holds the **pre-fix,
misaligned** CineMA 2D cache.

Measured throughput on MPS, per (patient, frame) sample:

| Backbone | 2D | 3D volume |
| --- | --- | --- |
| DINOv3 vits16 | ~0.25 s | ~0.35 s |
| CineMA | ~0.85 s | ~0.86 s |
| **SAM v1 vit_base** | **9.2 s** | **~17.5 s** |

The 3D SAM figure is ~1.9× the 2D one for a reason worth knowing: the volume extractor
loops `for z in range(target_depth)` over all **16** padded slices, while the 2D path runs
only the real `n_slices` (ACDC averages ~8.8). See **N3**.

---

## 5. The two blockers that have not moved

### B1 — Disk: ~642 GiB needed, **95 GiB free** (was 119 GiB on 09-01)

Derived from measured per-file sizes (ACDC DINOv3-S: 2.9 GiB / 2628 slices = 1.13 MiB per
2D slice; 5.9 GiB / 300 files = 20.1 MiB per 3D file) extrapolated over the 7-model grid,
17 548 2D slices and 1 636 patient-frames:

| | ACDC | M&Ms | M&Ms-2 | **Total** |
| --- | --- | --- | --- | --- |
| 2D caches (7 models) | 36 GiB | 96 GiB | 107 GiB | **239 GiB** |
| 3D caches (7 models) | 74 GiB | 156 GiB | 173 GiB | **403 GiB** |
| **Both** | **110 GiB** | **252 GiB** | **280 GiB** | **~642 GiB** |

ACDC alone no longer fits. Options, cheapest first — unchanged from 09-01 and none of them
done yet:

- **Delete `feature_cache_backups/` (35 GiB).** Gitignored, and its CineMA 2D cache is the
  misaligned one, so it is a footgun as well as a cost.
- **Store labels as `int8`.** Four classes; `int64` costs 0.28 MiB per 2D slice (40% of a
  CineMA slice) and 4.1 MiB per 3D file. Worth ~60 GiB. Needs a `.long()` at load time.
- **Stop duplicating `image` + `label` in all seven 3D caches** — ~75 GiB of identical data.
- **Extract → train → delete, one dataset at a time.** `cache_all_features.sh` takes
  `DATASETS=`.

### B2 — SAM extraction: ~5 days on MPS

From the measured rates, per SAM model across all three datasets: 2D ≈ 5 h, 3D ≈ 7.6 h.
Scaling large ≈ 3× and huge ≈ 5× off base gives **~115 h (~4.8 days)** for the SAM family
alone. DINOv3 and CineMA together are a few hours.

Neither optimisation suggested on 09-01 has been made: `extract_sam_2d_features` still
runs **one 1024×1024 slice per forward**, and the 2D and 3D caches still repeat the *same*
per-slice encoder forwards for the same (model, dataset), so the whole SAM budget is paid
twice. Batching over z and writing both caches from one pass remain the two changes with
the largest return. Otherwise: run the SAM grid on Isambard-AI via
`batch_run_sam_*_segmentation.sh` and keep the local machine for CineMA + DINOv3.

---

## 6. Issue status vs the 2026-09-01 refresher

| Ref | Issue | Status |
| --- | --- | --- |
| N2 | SAM v1 off-by-one, final block never read | ✅ **fixed** — PR #67, better than asked (global-attention taps + 6 regression tests) |
| N1 | `aggregate_summary.py` silently drops SAM v1 | ✅ **fixed** — `"SAM"` in `BACKBONE_ORDER` and `BACKBONE_COLORS` |
| N3 | `cache_all_features.sh` vs `run_all_segmentation.sh` grids disagree | ✅ **fixed** — SAM2 gone from both; both now enumerate the same 7 models |
| N4 | bare `pytest` fails at collection | ✅ **fixed** — PR #73; 39 pass, pre-commit clean, CI restored |
| B3 | archive + delete the SAM2 results before re-running | ✅ **done** — tag `archive/sam2-2026-09-08`, pushed |
| B5 | port the CI plot scripts from `sam2-classification` | ⚠️ **half** — `aggregate_summary.py` fixed independently, but `plot_dataset_macro_dice_ci.py` and the new `plot_pooled_backbones.py` are still branch-only (§3) |
| B4 | `set_seed()` disabled | ❌ **open, but now documented.** Three call sites still commented out, each with a new comment: *"deliberately not called: the call sites were disabled in `0b5d350` while the non-determinism in issue #59 is investigated"*, and the import kept with `# noqa: F401` so re-enabling is a one-liner. Issue #59 is open. `"seed": 0` is still written into every results JSON. |
| B3 (skip) | re-running a config writes a second JSON | ❌ **open** — the check still builds `json_path` from a fresh timestamp, so it can never fire. `build_summary.py` emits one row per JSON with no dedup, and `aggregate_summary.py` pivots with the default `aggfunc='mean'`, so a duplicate is silently averaged with its predecessor. Delete old JSONs before re-running, or port `6ead620` from `sam2-classification`. |
| 3 | bootstrap CIs cover only UNETR | ❌ **open** — `report_macro_dice_ci.py:254` still defaults to `*_per_sample.csv`; `*_per_stack.csv` has an identical schema and is silently excluded. `plot_pooled_backbones.py` on the branch works around it rather than fixing it. |
| 4 | `--use-layers` broken for `conv_decoder` | ❌ **open** — `run_segmentation.py:409` still sets `cached_layers` only for `linear_probe`. Low priority: no batch script passes `--use-layers`. |
| 5 | `sam2.1-hiera-tiny` never run | ✅ **moot** — SAM2 removed |
| 6 | orphan `_per_sample.csv` on `sam2-classification` | ❌ open on that branch; absent from `main` |
| 7 | `README.md` stale for segmentation | ❌ **open** — still documents `run_sam_variants.sh`, `run_sam3.sh`, `run_dino_unetr_variants.sh` (none exist) and claims outputs go to `scripts/segmentation/`. Still no `scripts/segmentation/README.md`. |
| 8 | M&Ms-2 3562 vs 3560 slices | ✅ explained (09-01): 3D truncates at 16; CineMA 2D now does too, leaving a 2-slice CineMA-2D-vs-other-2D gap (0.06%) |
| N5 | thin test coverage | ⚠️ **half** — 6 new layer-indexing tests, but `caching.py` still has none, so the exact class of bug that was issue #64 (CineMA z-misalignment) would still not be caught |

---

## 7. New issues found this pass

**N1 — a stale SAM cache is indistinguishable from a correct one.**
The cache path is `feature_cache_segmentation/{dataset}/{model}[_unetr3d]/{split}/layers_<sorted-indices>/`,
and PR #67 changed *which blocks are read* without changing `SAM_CONFIGS`'s values. So a
pre-#67 SAM cache and a post-#67 one both live at `layers_2-5-8-11` — and every cache
function skips files that already exist. Any SAM segmentation cache on Isambard-AI (or in
a backup) created before 2026-09-08 will be **silently reused and is wrong**. The local
`sam_vit_base` cache was already deleted; check the cluster before the next SLURM run. The
durable fix is to include something version-like in the cache key, or to write a small
manifest next to each cache recording the code revision.

**N2 — `summary.csv` and `summary_aggregated.csv` are out of sync with the JSONs.**
PR #68 deleted the SAM2 result files but did not regenerate the summaries or the two
top-level PNGs. All three `summary.csv` and `summary_aggregated.csv` still carry 9 SAM2
rows each. Verified the fix is exactly `build_summary.py` + `aggregate_summary.py` with no
other change (§2).

**N3 — the 3D caches spend ~37% of their encoder budget on zero padding.**
`extract_dino_volume_features` and `extract_sam_volume_features` both loop
`for z in range(target_depth)` — all 16 slices — while real stacks average ~8.8 (ACDC),
~10.5 (M&Ms) and ~11 (M&Ms-2). For SAM that is ~2.8 h of the ~7.6 h 3D budget per model.
Two caveats before "fixing" it: the convention is **uniform across DINOv3 and SAM**, so the
comparison is currently fair; and the padded features are not zero (the encoder has biases
and positional embeddings), so skipping them would feed genuinely different tensors into
`DINOv3UNetRDecoder` and shift the numbers. `MaskedVolumeLoss` and the `evaluate_vol_*`
functions already mask padded z, so the effect is limited to the 3D convolutions' receptive
field near the stack boundary — small, but not nil. Decide deliberately; do not treat it as
a free win.

**N4 — `caching.py` still has no tests.** PR #67 added good coverage of layer indexing, but
the module that produced both of the real correctness bugs found this year (#64
z-misalignment, and the cache-key aliasing in N1) is untested. A test asserting that a
cached 2D CineMA slice's features equal the corresponding depth of the 3D feature volume
would be cheap and would pin `a020ad4` permanently.

---

## 8. Suggested order of work

1. **Regenerate the summaries** (`build_summary.py`, `aggregate_summary.py`) and commit.
   One command; removes the orphan SAM2 rows. (**N2**)
2. **Check Isambard-AI for pre-09-08 SAM caches and delete them** before any SLURM run.
   (**N1**)
3. **Port `plot_dataset_macro_dice_ci.py` and `plot_pooled_backbones.py`** from
   `sam2-classification`; take the code, ignore its numbers. (**B5**)
4. **Decide on `set_seed()`.** Issue #59 is about *classification* fine-tuning
   non-determinism, but the segmentation driver is disabled by the same blanket change.
   Re-enabling it for segmentation is independent of #59 and is the last cheap moment
   before the grid is regenerated — ideally with ≥3 seeds on the headline
   backbone × decoder cells. (**B4**)
5. **Free disk** — drop `feature_cache_backups/`, switch labels to `int8`. (**B1**)
6. **Delete old result JSONs per config as you re-run**, or port the skip fix. (**B3**)
7. **Batch the SAM slice loop and merge the 2D/3D extraction pass**, then run SAM and the
   larger DINOv3 models on Isambard-AI. (**B2**)
8. Extract → train → delete, one dataset at a time. Regenerate all summaries, CI tables
   and plots at the end.
9. Housekeeping: fix `report_macro_dice_ci.py`'s default pattern (issue 3), add the
   `caching.py` alignment test (**N4**), update `README.md` (issue 7).

The open scientific question is unchanged from 09-01 and now has a second component:

- With the CineMA cache fixed, **CineMA may be competitive or best at Linear and Conv as
  well as UNETR**, which would move the story from "cardiac pretraining only helps with a
  strong 3D decoder" toward "cardiac pretraining helps throughout".
- With SAM v1 now tapping its **global-attention** blocks rather than windowed ones, the
  inverse size scaling reported on `sam2-classification` (base > large > huge) may not
  reproduce. That finding was always suspicious; it now has a plausible mechanical cause
  that has been removed.

Neither sentence should be written down until the grid is re-run under a fixed seed.
