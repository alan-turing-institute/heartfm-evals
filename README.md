# heartfm-evals

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

A modular package for evaluating foundation model performance on various clinical tasks involving cardiac MRI and CT images.

## Installation (uv)

Install uv (https://astral.sh/uv) and then:

```bash
git clone https://github.com/alan-turing-institute/heartfm-evals
cd heartfm-evals
uv venv .venv
source .venv/bin/activate
uv sync --all-extras

# Verify
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

## Post-install

**Only needed for training the DINOv3 M2F segmentation head.**

Training the DINOv3 Mask2Former segmentation head requires a CUDA extension that is not compiled automatically during install.
You will need to do this yourself as a one time step.

1. Clone the Deformable-DETR repo:

```bash
git clone https://github.com/fundamentalvision/Deformable-DETR
cd Deformable-DETR/models/ops
```

2. Fix deprecated PyTorch API (required for PyTorch >= 2.0):
```bash
sed -i 's/AT_DISPATCH_FLOATING_TYPES(value\.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/g' src/cuda/ms_deform_attn_cuda.cu
```

3. Source your venv and build the extension:
```bash
source /path/to/heartfm-evals/.venv/bin/activate
python setup.py build_ext --inplace
```

4. Copy the compiled extension into the dinov3 package:
```bash
cp MultiScaleDeformableAttention*.so \
  "$(python -c "import dinov3, os; print(os.path.join(os.path.dirname(dinov3.__file__), 'eval/segmentation/models/utils/ops'))")"
```

5. Verify it worked:

```bash
python -c "import torch; import MultiScaleDeformableAttention; print('OK')"
```

## Usage

### Running the scripts

Scripts are in `scripts/segmentation/`. Each script runs one or more model size variants.

Redirect output to a `.out` file to capture logs:

```bash
cd scripts/segmentation/

# --- 2D dense probe (bilinear upsample + conv decoder) ---

# SAM (vit-base, vit-large, vit-huge)
bash run_sam_variants.sh > sam_variants.out

# SAM 2.1 (hiera-tiny, hiera-small, hiera-base-plus, hiera-large)
bash run_sam2_variants.sh > sam2_variants.out

# SAM 3
bash run_sam3.sh > sam3.out

# DINOv3 (vits16, vitb16, vitl16)
bash run_dino_variants.sh > dino_variants.out

# CineMA
bash run_cinema.sh > cinema.out

# --- 3D UNetR decoder probe ---

# SAM + 3D UNetR (vit-base, vit-large, vit-huge)
bash run_sam_unetr_variants.sh > sam_unetr_variants.out

# SAM 2.1 + 3D UNetR (hiera-tiny, hiera-small, hiera-base-plus, hiera-large)
bash run_sam2_unetr_variants.sh > sam2_unetr_variants.out

# SAM 3 + 3D UNetR
bash run_sam3_unetr.sh > sam3_unetr.out

# DINOv3 + 3D UNetR (vits16, vitb16, vitl16)
bash run_dino_unetr_variants.sh > dino_unetr_variants.out

# CineMA + 3D UNetR
bash run_cinema_unetr.sh > cinema_unetr.out
```

Each script runs the corresponding Python segmentation script and saves model checkpoints and result plots to `scripts/segmentation/`.

## Results

### Classification

ACDC 5-way pathology classification and binary disease detection results are available in the [results spreadsheet](https://docs.google.com/spreadsheets/d/1yh8o8nqLrVV9fR_JaKjF-luf-cshnsQdGn1Awjl3Txw/edit?usp=sharing). Per-run details (per-class metrics, confusion matrices) are in the individual JSON files under `results/classification/`.

### Segmentation

Results are available in the [results spreadsheet](https://docs.google.com/spreadsheets/d/1SLpZTtmpmklWUBFqNTCLt8tfWfdyJW51YpjH2ch7xb0/edit?usp=sharing)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

## Licensing

This repository contains code under multiple licenses:

- Original code: MIT License (see LICENSE)
- DINOv3 models and materials: DINOv3 License (see LICENSE-DINOv3.md)


<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/alan-turing-institute/heartfm-evals/workflows/CI/badge.svg
[actions-link]:             https://github.com/alan-turing-institute/heartfm-evals/actions
[pypi-link]:                https://pypi.org/project/heartfm-evals/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/heartfm-evals
[pypi-version]:             https://img.shields.io/pypi/v/heartfm-evals
<!-- prettier-ignore-end -->
