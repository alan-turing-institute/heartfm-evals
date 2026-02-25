# Plan: Adapt `foreground_segmentation.ipynb` for ACDC Multi-Class Cardiac Segmentation

## Context

The current `foreground_segmentation.ipynb` trains a binary foreground/background classifier on natural (cat) images using DINOv3 patch features + logistic regression. This plan describes how to adapt it for the **ACDC cardiac MRI dataset** with **4-class segmentation** (BG, RV, Myo, LV).

### Preprocessing Already Done (by `cinema.data.acdc.preprocess`)

The on-disk data at `/Users/lbokeria/projects/health_gc/data/heartfm/processed/acdc/` has already undergone:

1. **Label unification** → {0=BG, 1=RV, 2=MYO, 3=LV}
2. **Resampling** → 1×1×10 mm spacing
3. **Center-cropping** → 192×192 (centered on LV)
4. **Intensity normalization** → percentile-clipped, rescaled to [0, 1], then cast to uint8 (×255)
5. **Metadata CSVs** → `train_metadata.csv`, `test_metadata.csv` with columns `pid`, `pathology`, `n_slices`, etc.

On-disk images are **uint8 [0–255]**, shape **(192, 192, z)** with z≈10 slices. Labels are **uint8 {0,1,2,3}**.

### Data Structure

```
processed/acdc/
├── train/                    # patients 001–100
│   ├── patient001/
│   │   ├── patient001_sax_ed.nii.gz
│   │   ├── patient001_sax_ed_gt.nii.gz
│   │   ├── patient001_sax_es.nii.gz
│   │   ├── patient001_sax_es_gt.nii.gz
│   │   └── patient001_sax_t.nii.gz
│   ├── patient002/
│   └── ...
├── test/                     # patients 101+
│   └── ...
├── train_metadata.csv
└── test_metadata.csv
```

### CineMA Dataset API

`cinema.segmentation.dataset.EndDiastoleEndSystoleDataset` returns per sample:
- `sax_image`: tensor `(1, 192, 192, z)` — single-channel ED or ES volume
- `sax_label`: tensor `(1, 192, 192, z)` — label volume, values {0,1,2,3}
- `pid`: patient ID string
- `is_ed`: bool (True=end-diastole, False=end-systole)
- `n_slices`: number of z-slices
- Dataset length = `len(meta_df) * 2` (one ED + one ES per patient)

Constructor:
```python
EndDiastoleEndSystoleDataset(
    data_dir: Path,
    meta_df: pd.DataFrame,   # must have columns: pid, n_slices
    views: str | list[str],  # use "sax"
    transform: Transform | None = None,
    dtype: np.dtype = np.float32,
)
```

---

## Change 1: Imports

**Current:** `PIL`, `urllib`, `tarfile`, `scipy.signal`, `torchvision.transforms.functional`.

**Replace with:**
- `cinema.segmentation.dataset.EndDiastoleEndSystoleDataset` — the ACDC dataloader.
- `cinema` constants — `RV_LABEL`, `MYO_LABEL`, `LV_LABEL`, `LABEL_TO_NAME`.
- `monai.transforms.ScaleIntensityd` — to rescale uint8 → [0, 1], passed as the `transform` argument to the dataset.
- `pandas` — to read metadata CSVs.
- Keep: `sklearn`, `numpy`, `torch`, `matplotlib`, `tqdm`, and the DINOv3 model loading from `models/dinov3/hubconf.py`.
- Drop: `PIL`, `urllib`, `tarfile`, `scipy.signal`, `torchvision.transforms.functional`.

---

## Change 2: Data Loading — `EndDiastoleEndSystoleDataset`

**Current:** `load_images_from_local_tar()` extracts cat images/masks from `.tar.gz`.

**Replace with:**

```python
import pandas as pd
from pathlib import Path
from cinema.segmentation.dataset import EndDiastoleEndSystoleDataset
from monai.transforms import ScaleIntensityd

ACDC_DATA_DIR = Path("/Users/lbokeria/projects/health_gc/data/heartfm/processed/acdc")
N_TRAIN_PATIENTS = 5
N_TEST_PATIENTS = 3

train_meta_df = pd.read_csv(ACDC_DATA_DIR / "train_metadata.csv").head(N_TRAIN_PATIENTS)
test_meta_df = pd.read_csv(ACDC_DATA_DIR / "test_metadata.csv").head(N_TEST_PATIENTS)

transform = ScaleIntensityd(keys="sax_image", factor=1/255, channel_wise=False)

train_dataset = EndDiastoleEndSystoleDataset(
    data_dir=ACDC_DATA_DIR / "train",
    meta_df=train_meta_df,
    views="sax",
    transform=transform,
)

test_dataset = EndDiastoleEndSystoleDataset(
    data_dir=ACDC_DATA_DIR / "test",
    meta_df=test_meta_df,
    views="sax",
    transform=transform,
)
```

After loading each sample, **extract individual 2D slices along the z-axis**. Each `(192, 192)` slice becomes one input for DINOv3, paired with its label slice:

```python
sample = train_dataset[i]
image_3d = sample["sax_image"]   # (1, 192, 192, z)
label_3d = sample["sax_label"]   # (1, 192, 192, z)
n_slices = sample["n_slices"]

for z in range(n_slices):
    image_2d = image_3d[0, :, :, z]   # (192, 192)
    label_2d = label_3d[0, :, :, z]   # (192, 192)
    # process each 2D slice...
```

---

## Change 3: Image Preprocessing (Minimal)

**Current:** `TF.resize` to 768×768, `TF.to_tensor`, `TF.normalize` with ImageNet mean/std.

**Replace with a lightweight pipeline applied to each extracted 2D slice:**

1. **uint8 → [0, 1]** — handled by the `ScaleIntensityd` transform passed to the dataset constructor. No additional intensity normalization needed.
2. **Repeat to 3 channels** — `(1, 192, 192)` → `(3, 192, 192)` via `tensor.repeat(3, 1, 1)`, since DINOv3 expects 3-channel input.
3. **ImageNet normalization** — apply `torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)` on the [0, 1] tensor. Required because DINOv3 was pretrained with this normalization (the original notebook uses it too).
4. **No resizing** — keep native 192×192. With patch_size=16 this gives a 12×12 patch grid (144 patches). 192 is evenly divisible by 16, so no padding needed.

**What we do NOT redo** (already applied by cinema preprocessing script):
- ~~Percentile clipping~~
- ~~Spatial resampling~~
- ~~Center cropping~~
- ~~Intensity rescaling to [0,1]~~

---

## Change 4: Train/Test Split

**Current:** Leave-one-out cross-validation on 9 images.

**Replace with:**
- Use the existing `train/` and `test/` directories directly — construct one `EndDiastoleEndSystoleDataset` for each.
- For C hyperparameter search, do **patient-level leave-one-out CV within the small training subset** (practical with 5 patients, mirrors the original notebook's approach at this small scale).
- Final evaluation on the `test/` patients.

---

## Change 5: Patch Quantization — Majority Vote

**Current:** `Conv2d` box-blur averaging the binary mask, threshold at 0.01/0.99.

**Replace with:**
- Reshape each label slice `(192, 192)` into patch blocks `(144, 256)` — i.e., 144 patches of 16×16=256 pixels each.
- Take `torch.mode` along the pixel dimension → `(144,)` patch labels in {0, 1, 2, 3}.
- Optional purity threshold: discard patches where the dominant class covers <50% of pixels, to reduce noisy boundary labels.

Implementation sketch:
```python
patch_size = 16
# label_2d shape: (192, 192)
patches = label_2d.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
# patches shape: (12, 12, 16, 16)
patches = patches.reshape(-1, patch_size * patch_size)  # (144, 256)
patch_labels = torch.mode(patches, dim=1).values         # (144,)
```

---

## Change 6: Multi-Class Logistic Regression + Metrics

**Current:** Binary `LogisticRegression`; binary AP and PR curves.

**Replace with:**
- `LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)`.
- **Metrics:**
  - Per-class Dice score (from patch-level predictions), reported using `cinema.LABEL_TO_NAME`.
  - Macro-averaged Dice (excluding background).
  - Confusion matrix.
  - Balanced accuracy.
- C-selection loop optimizes for **macro-averaged Dice** or **balanced accuracy**.

Dice per class:
```python
def dice_score(pred, true, class_idx):
    pred_c = (pred == class_idx)
    true_c = (true == class_idx)
    intersection = (pred_c & true_c).sum()
    return 2 * intersection / (pred_c.sum() + true_c.sum() + 1e-8)
```

---

## Change 7: Visualization — Grayscale + Color-Coded Overlays

**Current:** RGB images, binary foreground heatmaps, foreground/background composites.

**Replace with:**
- MRI slices displayed with `cmap='gray'`.
- **Ground truth overlay:** semi-transparent color-coded by class (RV=blue, Myo=green, LV=red).
- **Predicted overlay:** same color scheme, shown side-by-side with ground truth.
- Patch-level label grid visualization (12×12 quantized labels overlaid on downsampled slices).
- Legend mapping colors → `cinema.LABEL_TO_NAME`.
- Drop all binary foreground/background composite images and heatmaps.

Suggested color map:
```python
CLASS_COLORS = {
    0: (0, 0, 0, 0),        # BG — transparent
    1: (0, 0, 1, 0.4),      # RV — blue
    2: (0, 1, 0, 0.4),      # Myo — green
    3: (1, 0, 0, 0.4),      # LV — red
}
```

---

## Change 8: Scale — Small Subset, CPU-Only

**Current:** 9 images, sequential CPU processing.

**Replace with:**
- `N_TRAIN_PATIENTS = 5`, `N_TEST_PATIENTS = 3` defined as constants at the top of the notebook.
- 5 train patients × ~10 slices × 2 frames ≈ 100 slices → 14,400 patch feature vectors.
- 3 test patients ≈ 60 slices.
- Sequential per-slice feature extraction on CPU.
- No batching, GPU, or caching needed at this scale.

---

## Change 9: Test Inference Section

**Current:** Downloads a single cat image from a URL; applies binary foreground scores + median filter.

**Replace with:**
- Load `EndDiastoleEndSystoleDataset` for test patients.
- Extract features, predict multi-class patch labels.
- Reshape predictions to patch grid (12×12).
- Visualize: grayscale slice + predicted overlay + GT overlay.
- Report per-class and macro Dice on the test set.
- Drop the median filter (designed for continuous binary scores; not applicable to categorical predictions).

---

## Change 10: Save Model + Metadata

**Current:** Saves `fg_classifier.pkl`.

**Replace with:** Save a dict containing:
- Trained `LogisticRegression` model.
- Class mapping: `{0: "BG", 1: "RV", 2: "MYO", 3: "LV"}`.
- Optimal C value.
- DINOv3 model name used.
- Image size (192×192), patch size (16).
- Train/test patient IDs.

---

## End-to-End Data Flow

```
                      Already on disk                              In notebook
                ┌─────────────────────────┐    ┌──────────────────────────────────────────┐
Raw ACDC NIfTI  │ preprocess.py:          │    │                                          │
  ──────────►   │  resample, crop, clip,  │    │  EndDiastoleEndSystoleDataset             │
                │  normalize, cast uint8  │    │    + ScaleIntensityd (÷255)               │
                │  → 192×192×z uint8      │    │         │                                 │
                └─────────────────────────┘    │         ▼                                 │
                                               │  Extract 2D slices along z                │
                                               │         │                                 │
                                               │         ▼                                 │
                                               │  Repeat 1ch→3ch + ImageNet norm           │
                                               │         │                                 │
                                               │         ▼                                 │
                                               │  DINOv3 feature extraction (144 patches)  │
                                               │         │                                 │
                                               │         ▼                                 │
                                               │  Majority-vote patch labels from GT       │
                                               │         │                                 │
                                               │         ▼                                 │
                                               │  Multinomial LogReg (train / CV for C)    │
                                               │         │                                 │
                                               │         ▼                                 │
                                               │  Per-class Dice + visualization           │
                                               └──────────────────────────────────────────┘
```

## Notebook Cell Structure (Suggested)

1. **Markdown:** Title + description
2. **Code:** Imports and constants (`N_TRAIN_PATIENTS`, `N_TEST_PATIENTS`, `ACDC_DATA_DIR`, `CLASS_COLORS`)
3. **Code:** Load DINOv3 model (keep existing `hubconf.py` loading)
4. **Code:** Load ACDC data using `EndDiastoleEndSystoleDataset` + metadata CSVs
5. **Code:** Helper functions: preprocessing (channel repeat + ImageNet norm), majority-vote patch quantization, Dice score
6. **Code:** Feature extraction loop — iterate over dataset, extract 2D slices, preprocess, run through DINOv3, collect features + patch labels
7. **Markdown:** Training section header
8. **Code:** Patient-level leave-one-out CV for C selection with multinomial LogReg
9. **Code:** Train final model with optimal C on all training data
10. **Markdown:** Evaluation section header
11. **Code:** Extract features for test set
12. **Code:** Predict + compute per-class Dice, confusion matrix, balanced accuracy
13. **Code:** Visualization — GT vs. predicted overlays on sample test slices
14. **Code:** Save model + metadata
