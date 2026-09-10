# SAM2 Removal Record

> **SAM2 was removed from this repo on 2026-09-08.** The last state containing it is the
> tag **`archive/sam2-2026-09-08`**, which holds the full implementation
> (`SAM2_CONFIGS`, `_load_sam2`, `extract_sam2_2d_features`, `cache_sam2_2d_features`,
> `cache_sam2_cls_features`, the five `batch_run_sam2_*` SLURM scripts) and the 81
> committed SAM2 segmentation result files.
>
> ```bash
> git show archive/sam2-2026-09-08:src/heartfm_evals/backbones.py   # SAM2_CONFIGS, _load_sam2
> git show archive/sam2-2026-09-08:documents/prompts/sam2_decisions.md   # the full design record
> ```
>
> This file has been trimmed to the parts that are still load-bearing: the SAM v1
> segmentation design (§1b) and the cross-backbone comparison (§8). The original
> 427-line document — the Hiera stage analysis, the Stage 3 vs Stage 4 classification
> decision, the `embed_dim` / `cls_embed_dim` split, and the unadopted multi-scale
> proposal — is at the tag above.

## Why SAM2 was removed

SAM2 is architecturally distinct enough to be dealt with separately, and later. In its
final state it was stuck at a half-measure that cost more to maintain than it earned:

- **Hiera is non-uniform.** Four stages with different channel counts and resolutions,
  unlike SAM v1's isotropic ViT. All four SAM2 taps were therefore pinned inside
  Stage 3, and its `layer_indices` were originally written as `hidden_states` positions
  rather than block indices — the root of the dual convention untangled in issue #66.
- **It complicated shared code.** `extract_sam_volume_features` serves both SAM
  families, so the #66 off-by-one fix could not be a one-liner; it briefly needed a
  `hidden_state_offset` parameter that existed solely to keep SAM2 working.
- **Two more special cases.** Segmentation read Stage 3 (`embed_dim`) while
  classification read Stage 4 (`cls_embed_dim`) — the only use of `cls_embed_dim`
  anywhere, and the only place `hidden_states[-1]` was read instead of a `layer_indices`
  entry.
- **It was never finished.** The genuinely multi-scale (one-block-per-stage) option was
  analysed in the original §7 and deliberately **not** applied, because it needs
  non-uniform-channel support in `DINOv3UNetRDecoder` and `DenseLinearProbe`. So SAM2's
  four "multi-scale" skips all sat at one resolution — the limitation that section
  identified was real and never addressed.
- **The grid scripts disagreed.** `run_all_segmentation.sh` ran SAM2 while
  `cache_all_features.sh` deliberately excluded it, so a full run would have extracted
  SAM2 features inside the training job. `sam2.1-hiera-tiny` was never run at all.

Standardising on SAM v1 also makes the headline comparison architecture-matched: plain
ViT (DINOv3) against plain ViT (SAM v1), rather than against Hiera's pyramid. The
`sam2-classification` branch (draft PR #63) was the first attempt at this and is kept as
a record.

---

## 1b. SAM v1 for segmentation

SAM v1 was originally classification-only. Two gaps blocked segmentation:

- **No per-slice extractor.** `extract_sam_volume_features` (3D) already worked for SAM
  v1 — it is in fact written for it — but the 2D decoders had no equivalent.
  `extract_sam_2d_features` runs the vision encoder with `output_hidden_states=True`,
  permutes channels-last to channels-first, bilinearly downsamples each layer to
  `grid_size=12`, and concatenates along channels.
- **No `layer_indices`.** `_load_sam` returned none, so `run_segmentation.py` silently
  fell back to its `(3, 6, 9, 11)` default — fine for ViT-B's 12 blocks but sampling only
  the first third of ViT-L (24) and ViT-H (32).

`SAM_CONFIGS` records each model's **global-attention blocks** (`global_attn_indexes`):

| model id | embed_dim | blocks | layer_indices |
|----------|-----------|--------|---------------|
| `facebook/sam-vit-base` | 768 | 12 | `(2, 5, 8, 11)` |
| `facebook/sam-vit-large` | 1024 | 24 | `(5, 11, 17, 23)` |
| `facebook/sam-vit-huge` | 1280 | 32 | `(7, 15, 23, 31)` |

These are **not** evenly spaced quartiles, even though they look close to it for these
three depths — do not "tidy" them. SAM's encoder runs at 1024×1024 with patch 16, i.e.
64×64 = 4096 tokens, so full self-attention is paid for only four times: those blocks
have `window_size=0` and attend across the whole grid, while every other block is
restricted to 14×14 = 196-token windows. Tapping a windowed block yields window-limited
features. `layer_indices[-1]` is both the final block and a global one, so
`linear_probe` (which uses only the last entry) gets a globally-attended tap.

Issue #66 corrected ViT-B from `(3, 6, 9, 11)` to `(2, 5, 8, 11)`; ViT-L and ViT-H
already matched their `global_attn_indexes`.

The SAM v1 ViT is isotropic — every block emits the same channel count at 64×64 — so no
per-stage channel bookkeeping is needed and `DINOv3UNetRDecoder` works unmodified.
`embed_dim` is still read from `vision_config.hidden_size` so the loaded weights stay
the source of truth.

`layer_indices` are **block** indices. SAM v1 goes through
`vision_encoder(..., output_hidden_states=True)`, whose tuple starts with the patch
embedding, so block *i* sits at `hidden_states[i+1]`; `features.py::_block_hidden_state`
is the only place that `+1` is applied. Never subscript `hidden_states` directly — that
is what made SAM v1 skip its final block (issue #66).

---

## 8. Comparison with DINOv3 and CineMA segmentation

> The SAM2 rows are retained for the historical record only. "SAM2 (old)" is how SAM2
> actually behaved; "SAM2 (new)" describes the multi-scale proposal that was never
> applied.

### Feature extraction

| Backbone | Architecture | Layers extracted | Channel count per layer | Spatial res (native) | Caching format |
|----------|-------------|-----------------|------------------------|---------------------|----------------|
| DINOv3 | Isotropic ViT | 4 layers at different depths | Uniform (`embed_dim`) | Same patch grid at all layers | Per-layer `(C, g, g, Z)` stacked → UNetR; concat `(C×4, g, g)` → 2D |
| SAM v1 | Isotropic ViT | 4 global-attention blocks | Uniform (`embed_dim`) | All 64×64 | Same as DINOv3 |
| CineMA | 3D conv encoder + ViT | Conv skips × 2 + ViT output | Varying (conv encoder natural) | Genuine multi-res (conv) + one patch grid (ViT) | Per-skip + ViT tensor separately |
| ~~SAM2 (old)~~ | Hiera ViT | 4 layers, all Stage 3 | Uniform (Stage 3 `embed_dim`) | All 64×64 (one stage) | Same as DINOv3 |
| ~~SAM2 (new)~~ | Hiera ViT | 4 layers, one per stage | Varying per stage | 256×256 / 128×128 / 64×64 / 32×32 | Same as DINOv3 |

After extraction, all feature maps are bilinearly downsampled to `grid_size=12` before
caching, so the spatial dimension in the cache is uniform at `12×12` across all
backbones. The meaningful difference is in the **channel counts** and **semantic depth**
of the features.

### Decoder architecture

| Backbone | Decoder class | Skip connection sources | Per-skip `in_chans` |
|----------|--------------|------------------------|---------------------|
| DINOv3 | `DINOv3UNetRDecoder` | 4 ViT layer outputs | Uniform `embed_dim` |
| SAM v1 | `DINOv3UNetRDecoder` | 4 ViT layer outputs | Uniform `embed_dim` |
| CineMA | `CineMAUNetRDecoder` | 2 conv skips + ViT output + image | Varying (natural from conv encoder) |

SAM v1 shares `DINOv3UNetRDecoder` with DINOv3 unmodified, because both are isotropic and
so all skip adapters use the same `in_chans=embed_dim`. CineMA uses a purpose-built
decoder that mirrors its `ConvUNetR` training architecture. Supporting a non-uniform
backbone would require per-stage channel counts in `DINOv3UNetRDecoder` and
`DenseLinearProbe` — the work that the unadopted SAM2 multi-scale proposal needed, and
the reason it was never applied.

### Multi-scale character

DINOv3 and SAM v1 features are **depth-wise pseudo multi-scale**: all 4 layers come from
the same isotropic ViT, so they share spatial resolution and channel count. The
multi-scale signal comes only from different semantic depths within a homogeneous
architecture.

CineMA features are **natively multi-scale**: the conv encoder produces genuinely
hierarchical spatial pyramids (48×48 → 24×24), and the ViT operates on these
pre-downsampled tokens. This matches the original CineMA design.

### Classification approach

| Backbone | Token used | Why |
|----------|-----------|-----|
| DINOv3 | CLS token (or GAP) | ViT CLS token is trained for global representation |
| CineMA | CLS token (or GAP) | Same — ViT CLS token |
| SAM v1 | GAP of pre-neck features | No CLS token; neck is a task-specific projection, bypassed |
| ~~SAM2~~ | ~~GAP of `hidden_states[-1]` (Stage 4)~~ | ~~Hiera has no CLS token; Stage 4 is the most semantically processed output~~ |

All backbones use their true final output for classification. SAM v1 has no CLS token, so
it is `--pooling gap` only.
