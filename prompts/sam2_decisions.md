# SAM2 Design Decisions

## Context

SAM2 (Segment Anything Model 2, `facebook/sam2.1-hiera-*`) was already used as a backbone for
**segmentation**. This document records the decisions made when extending it to **classification**
and when fixing the segmentation multi-scale feature extraction, including the analysis that
motivated those decisions.

---

## 1. What was added

Four new files / changes:

| File | Change |
|------|--------|
| `src/heartfm_evals/backbones.py` | Added `cls_embed_dim` to `SAM2_CONFIGS`; exposed it from `_load_sam2` |
| `src/heartfm_evals/classification_probe.py` | Added `cache_sam2_cls_features` |
| `scripts/classification/run_classification.py` | Added `sam2` backbone choice, `--sam2-model-id` arg, routing to new cache fn |
| `scripts/classification/batch_run_sam2_{logreg,finetune}_classification.sh` | New SLURM array scripts |
| `scripts/submit_classification_sam2.sh` | Submit script for all SAM2 classification variants |

---

## 2. Hiera architecture — block counts verified empirically

The SAM2 Hiera encoder has 4 stages. The `hidden_states` tuple returned by
`vision_encoder(..., output_hidden_states=True)` has shape:

```
hidden_states[0]     — patch embedding output
hidden_states[i+1]   — output of block i
```

Block counts and spatial shapes were verified by running all four variants through the encoder
with a dummy input:

| Model | Total blocks | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|-------|-------------|---------|---------|---------|---------|
| hiera-tiny | 12 | [1] 256×256 C=96 | [2–3] 128×128 C=192 | [4–10] 64×64 C=384 | [11–12] 32×32 C=768 |
| hiera-small | 16 | [1] 256×256 C=96 | [2–3] 128×128 C=192 | [4–14] 64×64 C=384 | [15–16] 32×32 C=768 |
| hiera-base-plus | 24 | [1–2] 256×256 C=112 | [3–5] 128×128 C=224 | [6–21] 64×64 C=448 | [22–24] 32×32 C=896 |
| hiera-large | 48 | [1–2] 256×256 C=144 | [3–8] 128×128 C=288 | [9–44] 64×64 C=576 | [45–48] 32×32 C=1152 |

The channel count doubles and spatial resolution halves at each stage transition.

---

## 3. Key decision: use Stage 4 (`hidden_states[-1]`) for classification

### What the existing `SAM2_CONFIGS.layer_indices` actually point to

`layer_indices` was defined for **segmentation** multi-scale feature extraction. All four
indices for every variant land in **Stage 3** (64×64 resolution):

| Model | `layer_indices` | Stage at each index |
|-------|----------------|---------------------|
| tiny | (4, 6, 8, 10) | all Stage 3 |
| small | (4, 7, 11, 14) | all Stage 3 |
| base-plus | (6, 11, 16, 21) | all Stage 3 |
| large | (9, 21, 33, 44) | all Stage 3 |

`layer_indices[-1]` is the **last block of Stage 3**, not the model's true final output.
Stage 4 follows with 2–4 more blocks at lower resolution.

### Why this matters for classification

For classification, a single global embedding is needed (via GAP). Comparing the options:

| Option | hidden state | spatial res | channels | semantic level |
|--------|-------------|-------------|----------|---------------|
| Stage 3 final (`layer_indices[-1]`) | mid-model | 64×64 | 384–576 | intermediate |
| Stage 4 final (`hidden_states[-1]`) | true final | 32×32 | 768–1152 | most abstract |

Stage 4 is the right choice for classification because:
- It is the model's genuine final output — analogous to what every other backbone uses
- Lower spatial resolution with higher channel count is better for a global pooled embedding
- More semantically processed features, which is what downstream classifiers benefit from

### Consistency with other backbones

| Backbone | Layer used for classification | True final? |
|----------|------------------------------|-------------|
| DINOv3 | `get_intermediate_layers(n=1)` — last layer | Yes |
| CineMA | `feature_forward()` — full model output | Yes |
| SAM v1 | manually iterates all `ve.layers`, pre-neck | Yes |
| SAM2 (original draft) | `hidden_states[layer_indices[-1]]` — Stage 3 | **No** |
| SAM2 (final) | `hidden_states[-1]` — Stage 4 | **Yes** |

Using `layer_indices[-1]` would have been the only backbone that does not use its true
final output, which would make comparisons across backbones less meaningful.

---

## 4. `embed_dim` vs `cls_embed_dim`

Because Stage 3 and Stage 4 have different channel counts, `SAM2_CONFIGS` now carries
two distinct dimension fields:

```python
SAM2_CONFIGS = {
    "facebook/sam2.1-hiera-tiny": {
        "embed_dim":       384,              # Stage 3 — kept for backward compat
        "cls_embed_dim":   768,              # Stage 4 — used by classification
        "layer_indices":   (1, 3, 10, 12),  # One block per stage (updated — see §7)
        "stage_embed_dims": (96, 192, 384, 768),  # Per-stage C for segmentation
    },
    ...
}
```

| Field | Used by | Value source |
|-------|---------|-------------|
| `embed_dim` | Backward compat only (Stage 3 channels) | Stage 3 channel count |
| `cls_embed_dim` | Classification (`cache_sam2_cls_features`) | Stage 4 channel count |
| `stage_embed_dims` | Segmentation decoders (`stage_embed_dims[i]` per skip level) | Per-stage channel counts |

`run_classification.py` reads `cls_embed_dim` when the backbone is `sam2`, so the logged
embed_dim, results JSON, and the logistic regression input dimension all reflect the actual
feature size (768/768/896/1152).

---

## 5. Feature extraction implementation

`cache_sam2_cls_features` mirrors `cache_sam_cls_features` (SAM v1) in structure:

- Iterates over 2D slices per patient (ED and ES frames separately)
- Converts each slice to RGB PIL image via the SAM2 processor
- Runs `vision_encoder(..., output_hidden_states=True)`
- Takes `hidden_states[-1]` — Stage 4 final block, shape `(1, H', W', C)`
- GAPs over `(H', W')` → `(C,)` vector
- Saves as `{"cls_token": Tensor(C,)}` — same format as all other backbones

Patient-level features are then built by `build_patient_features`, which mean-pools
ED slices, mean-pools ES slices, and concatenates → `(2 * cls_embed_dim,)`.

---

## 6. Batch script layout

Both `batch_run_sam2_logreg_classification.sh` and `batch_run_sam2_finetune_classification.sh`
use a 12-task array:

```
3 datasets × 4 models = 12 tasks (indices 0–11)

Within-dataset layout (4 configs each):
  0  sam2.1-hiera-tiny
  1  sam2.1-hiera-small
  2  sam2.1-hiera-base-plus
  3  sam2.1-hiera-large

acdc → 0–3   mnm → 4–7   mnm2 → 8–11
```

This matches the layout used by the SAM2 segmentation batch scripts. For a smoke test
(tiny only): `--array=0,4,8`.

All scripts use `--pooling gap` (hardcoded, not exposed as a flag) since SAM2 Hiera
has no CLS token.

---

## 7. Segmentation `layer_indices` fix — truly multi-scale features

### The original problem

The original `layer_indices` for segmentation all fell within Stage 3:

| Model | Old `layer_indices` | Stage at each index |
|-------|---------------------|---------------------|
| tiny | (4, 6, 8, 10) | all Stage 3 (C=384) |
| small | (4, 7, 11, 14) | all Stage 3 (C=384) |
| base-plus | (6, 11, 16, 21) | all Stage 3 (C=448) |
| large | (9, 21, 33, 44) | all Stage 3 (C=576) |

All four features were at 64×64 resolution with identical channel counts — not
multi-scale in any meaningful sense. Stages 1, 2, and 4 were never used.

### Fix: one index per stage

`layer_indices` now picks the last block of each stage, giving true multi-scale coverage:

| Model | Stage 1 | Stage 2 | Stage 3 end | Stage 4 end | New `layer_indices` |
|-------|---------|---------|-------------|-------------|---------------------|
| tiny | 1 | 3 | 10 | 12 | (1, 3, 10, 12) |
| small | 1 | 3 | 14 | 16 | (1, 3, 14, 16) |
| base-plus | 2 | 5 | 21 | 24 | (2, 5, 21, 24) |
| large | 2 | 8 | 44 | 48 | (2, 8, 44, 48) |

A new `stage_embed_dims` field records the channel count at each selected index:

| Model | Stage 1 C | Stage 2 C | Stage 3 C | Stage 4 C | `stage_embed_dims` |
|-------|-----------|-----------|-----------|-----------|--------------------|
| tiny | 96 | 192 | 384 | 768 | (96, 192, 384, 768) |
| small | 96 | 192 | 384 | 768 | (96, 192, 384, 768) |
| base-plus | 112 | 224 | 448 | 896 | (112, 224, 448, 896) |
| large | 144 | 288 | 576 | 1152 | (144, 288, 576, 1152) |

### Why the spatial resolution concern was not a blocker

The existing feature extractors (`extract_sam2_2d_features` and
`extract_sam_volume_features`) already call `F.interpolate` to downsample every
layer's feature map to `grid_size × grid_size` (default 12) before
concatenating or stacking. So the mixed native resolutions (256×256, 128×128,
64×64, 32×32) are handled transparently — only the channel counts `C_i` differ.

### What required decoder changes

Since each stage has a different channel count, the decoder needed to be updated
to handle non-uniform `in_chans` per layer. Three components were updated:

#### `DINOv3UNetRDecoder`

Added an `embed_dims: tuple[int, ...] | None` parameter. When provided, each
skip adapter and the bottleneck adapter uses its own `in_chans`:

```python
_dims = embed_dims if embed_dims is not None else (embed_dim,) * 4

self.skip_adapters = nn.ModuleDict({
    f"layer_{layer_indices[0]}": ConvResBlock(in_chans=_dims[0], ...),  # Stage 1
    f"layer_{layer_indices[1]}": ConvResBlock(in_chans=_dims[1], ...),  # Stage 2
    f"layer_{layer_indices[2]}": ConvResBlock(in_chans=_dims[2], ...),  # Stage 3
})
self.bottleneck_adapter = ConvResBlock(in_chans=_dims[3], ...)           # Stage 4
```

Backward compatible: when `embed_dims=None`, `embed_dim` is broadcast to all layers
(DINOv3 and SAM v1 are unaffected).

#### `DenseLinearProbe`

Added `cached_embed_dims: tuple[int, ...] | None` parameter. The linear probe
caches all 4 layers as a single concatenated tensor; when probing only the last
layer (Stage 4), it must select the right channel slice. The existing logic
`start = pos * embed_dim` assumed uniform channels — wrong for multi-scale.
Fixed to:

```python
if cached_embed_dims is not None:
    start = sum(cached_embed_dims[:pos])
    width = cached_embed_dims[pos]
else:
    start = pos * embed_dim
    width = embed_dim
```

For SAM2 tiny with `use_layers=(12,)` and `cached_layers=(1,3,10,12)`:
- Old: `start = 3 × 384 = 1152`, `width = 384` ← wrong (Stage 4 has 768 channels)
- New: `start = 96+192+384 = 672`, `width = 768` ← correct

The `embed_dim` passed to `DenseLinearProbe` for SAM2 linear probe is now
`stage_embed_dims[-1]` (Stage 4 channels = `cls_embed_dim`), not the Stage 3
`embed_dim` from config.

#### `get_decoder` and `conv_decoder`

`get_decoder` now accepts `embed_dims` and `cached_embed_dims` and threads them
through. For `conv_decoder`, total input channels are now computed as
`sum(embed_dims)` rather than `embed_dim × n_layers`:

| Model | Old `in_channels` | New `in_channels` |
|-------|-------------------|--------------------|
| tiny | 384 × 4 = 1536 | 96+192+384+768 = 1440 |
| small | 384 × 4 = 1536 | 96+192+384+768 = 1440 |
| base-plus | 448 × 4 = 1792 | 112+224+448+896 = 1680 |
| large | 576 × 4 = 2304 | 144+288+576+1152 = 2160 |

### `run_segmentation.py` routing

`stage_embed_dims` is read from the backbone info dict and routed differently
per decoder type:

| Decoder | `embed_dim` passed | `embed_dims` passed | `cached_embed_dims` |
|---------|--------------------|---------------------|---------------------|
| `linear_probe` | `stage_embed_dims[-1]` (Stage 4) | `None` | `stage_embed_dims` |
| `conv_decoder` | `embed_dim` (Stage 3, unused) | `stage_embed_dims` | — |
| `unetr` | `embed_dim` (Stage 3, unused) | `stage_embed_dims` | — |

DINOv3 and CineMA are unaffected — `stage_embed_dims` is `None` for those
backbones and all existing code paths are unchanged.

### Cache compatibility

The new `layer_indices` produce different cache subdirectory names
(e.g. `layers_1-3-10-12` for tiny vs. the old `layers_4-6-8-10`), so new
caches are written to separate directories with no collision with any existing
cached data.

---

## 8. Comparison with DINOv3 and CineMA segmentation

### Feature extraction

| Backbone | Architecture | Layers extracted | Channel count per layer | Spatial res (native) | Caching format |
|----------|-------------|-----------------|------------------------|---------------------|----------------|
| DINOv3 | Isotropic ViT | 4 layers at different depths | Uniform (`embed_dim`) | Same patch grid at all layers | Per-layer `(C, g, g, Z)` stacked → UNetR; concat `(C×4, g, g)` → 2D |
| CineMA | 3D conv encoder + ViT | Conv skips × 2 + ViT output | Varying (conv encoder natural) | Genuine multi-res (conv) + one patch grid (ViT) | Per-skip + ViT tensor separately |
| SAM2 (old) | Hiera ViT | 4 layers, all Stage 3 | Uniform (Stage 3 `embed_dim`) | All 64×64 (one stage) | Same as DINOv3 |
| SAM2 (new) | Hiera ViT | 4 layers, one per stage | Varying per stage | 256×256 / 128×128 / 64×64 / 32×32 | Same as DINOv3 |

After extraction, all feature maps are bilinearly downsampled to `grid_size=12`
before caching, so the spatial dimension in the cache is uniform at `12×12` across
all backbones. The meaningful difference is in the **channel counts** and **semantic
depth** of the features.

### Decoder architecture

| Backbone | Decoder class | Skip connection sources | Per-skip `in_chans` |
|----------|--------------|------------------------|---------------------|
| DINOv3 | `DINOv3UNetRDecoder` | 4 ViT layer outputs | Uniform `embed_dim` |
| CineMA | `CineMAUNetRDecoder` | 2 conv skips + ViT output + image | Varying (natural from conv encoder) |
| SAM2 | `DINOv3UNetRDecoder` | 4 Hiera stage outputs | Varying per stage (`stage_embed_dims`) |

SAM2 shares `DINOv3UNetRDecoder` with DINOv3. The key structural difference is that
DINOv3 skip adapters all use the same `in_chans=embed_dim`, while SAM2 now uses per-stage
channel counts via `embed_dims`. CineMA uses a purpose-built decoder that mirrors its
`ConvUNetR` training architecture.

### Multi-scale character

DINOv3 features are **depth-wise pseudo multi-scale**: all 4 layers are extracted from
the same isotropic ViT, so they share the same spatial resolution and channel count.
The multi-scale signal comes only from different semantic depths within a homogeneous
architecture.

CineMA features are **natively multi-scale**: the conv encoder produces genuinely
hierarchical spatial pyramids (48×48 → 24×24), and the ViT operates on these
pre-downsampled tokens. This matches the original CineMA design.

SAM2 (new) features are **architecture-faithful multi-scale**: Hiera was designed with
four stages at progressively lower resolution and higher channel count, analogous to a
ResNet or FPN backbone. Extracting one representative block per stage now exposes the
feature hierarchy the model actually learned, rather than sampling redundantly within
one stage.

### Classification approach

| Backbone | Token used | Why |
|----------|-----------|-----|
| DINOv3 | CLS token (or GAP) | ViT CLS token is trained for global representation |
| CineMA | CLS token (or GAP) | Same — ViT CLS token |
| SAM2 | GAP of `hidden_states[-1]` (Stage 4) | Hiera has no CLS token; Stage 4 is the most semantically processed output |
| SAM v1 | GAP of pre-neck features | No CLS token; neck is task-specific projection, bypassed |

All backbones use their true final output for classification. SAM2 uses Stage 4
(`hidden_states[-1]`), which is the model's genuine last block — analogous to the CLS
token in DINOv3/CineMA. The `cls_embed_dim` field (768/768/896/1152) records the Stage 4
channel count used for this embedding.

---

## 9. SAM v1 vs SAM2 for segmentation — which belongs in the comparison

### Report context

The primary objective is to compare **backbone transferability** across CineMA, DINOv3,
and SAM for cardiac MRI classification and segmentation. This makes the choice of SAM
variant non-trivial.

### Why SAM v1 is the right choice for segmentation

DINOv3 and SAM v1 are both plain isotropic ViTs. The meaningful axis of variation between
them is pretraining objective — self-supervised (DINOv3) vs task-supervised for
segmentation (SAM v1). Same architecture, different training signal. CineMA is then the
domain-specific baseline. Together the three backbones vary one thing at a time:

| Backbone | Architecture | Pretraining | Domain |
|----------|-------------|-------------|--------|
| DINOv3 | Isotropic ViT | Self-supervised | Natural images |
| SAM v1 | Isotropic ViT | Supervised (segmentation) | Natural images |
| CineMA | Conv + ViT | Supervised | Cardiac MRI |

This is a legible story for a transferability report. DINOv3 vs SAM v1 isolates
pretraining objective; SAM v1 vs CineMA isolates domain specificity.

### Why SAM2 muddies the segmentation comparison

SAM2 is a different architecture class (Hiera, hierarchical ViT), so substituting it for
SAM v1 conflates pretraining objective with architecture. More critically, SAM2's main
advantage — genuine multi-scale spatial features from its Hiera stages — is neutralised
by the `grid_size=12` caching step: all stage outputs are downsampled to 12×12 before
saving, discarding the native spatial pyramid.

The UNetR decoder then reconstructs a spatial hierarchy by upsampling from 12×12,
regardless of which backbone was used. Only CineMA escapes this because its conv-encoder
skips are cached at native resolution (48×48, 24×24).

So SAM2 in the current segmentation pipeline:
- Adds architectural complexity (per-stage channel tracking, updated decoders)
- Does not benefit from its genuine multi-scale advantage (lost at caching)
- Introduces a confound that makes the three-way comparison harder to interpret

### Conclusion

**Both tasks: use SAM v1.** It keeps the three-way comparison (CineMA, DINOv3, SAM)
architecturally consistent and interpretable. For segmentation it fits the same UNetR
framework as DINOv3 with no special treatment (uniform channels, simple `layer_indices`).
For classification it uses the same pre-neck GAP approach as before.

SAM2 classification and segmentation code has been removed from the pipeline. The SAM2
segmentation scripts and multi-scale decoder changes remain in the codebase but are not
part of the primary evaluation.
