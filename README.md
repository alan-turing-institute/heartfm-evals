# heartfm-evals

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

A modular package for evaluating foundation model performance on various clinical tasks involving cardiac MRI and CT images.

## Installation (uv)

Install uv (https://astral.sh/uv) and then:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

From source:
```bash
git clone https://github.com/alan-turing-institute/heartfm-evals
cd heartfm-evals
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

### Running the scripts

Scripts are in the `scripts/` directory. Each script runs one or more model size variants.

Redirect output to a `.out` file to capture logs:

```bash
cd scripts/

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
```

Each script runs the corresponding Python segmentation script and saves model checkpoints and result plots to `scripts/`.

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
