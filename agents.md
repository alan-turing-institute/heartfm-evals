# Agent Instructions for heartfm-evals

## Project Overview
This repository (`heartfm-evals`) is a modular Python package for evaluating foundation model performance on clinical tasks involving cardiac MRI and CT images. The primary goal is to develop a framework to efficiently evaluate various different models (such as DINOv2, DINOv3, vanilla ViTs, MAEs, etc) which are pre-trained or fine-tuned elsewhere.

## Key Information for Coding Agents

### Architecture
- **Source code**: Located in [`src/heartfm_evals/`](src/heartfm_evals/)
- **Model implementations**: Located in [`models/`](models/)
- **Model weights**: Stored in [`model_weights/`](model_weights/) directory
- **Notebooks**: Example usage in [`notebooks/`](notebooks/)

### Dependencies & Setup
- Uses **uv** for dependency management (preferred over pip)
- Python package managed via [`pyproject.toml`](pyproject.toml)
- Uses pre-commit hooks for code quality (see [`.pre-commit-config.yaml`](.pre-commit-config.yaml))

### Code Style & Quality
- Run `pre-commit run -a` before suggesting code changes
- Follow existing patterns in [`src/heartfm_evals/`](src/heartfm_evals/)
- Type hints are expected (mypy configured)
- Code formatting with ruff

### Testing
- Use pytest for tests
- Run tests with: `pytest`
- Generate coverage with: `pytest --cov=heartfm_evals`

### Licensing Requirements
**CRITICAL**: This repository uses dual licensing:
1. **Original code**: MIT License (see [`LICENSE`](LICENSE))
2. **DINOv3 materials**: DINOv3 License (see [`LICENSE-DINOv3.md`](LICENSE-DINOv3.md))

When suggesting code that uses DINOv3 components (from [`models/dinov3/`](models/dinov3/)), always:
- Include the DINOv3 license header at the top of files
- Reference [`LICENSE-DINOv3.md`](LICENSE-DINOv3.md) in documentation
- Do not suggest modifications that violate DINOv3 terms (see sections 1.b.iv and 5 in [`LICENSE-DINOv3.md`](LICENSE-DINOv3.md))

### DINOv3 Model Access
Available models via [`models/dinov3/hubconf.py`](models/dinov3/hubconf.py):
- `dinov3_vits16`, `dinov3_vitb16`, `dinov3_vitl16`
- Pre-trained weights in [`model_weights/`](model_weights/)

### Development Workflow
1. Create virtual environment: `uv venv .venv`
2. Activate: `source .venv/bin/activate`
3. Install package: `uv pip install -e ".[dev]"`
4. Install pre-commit: `pre-commit install`
5. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for full details

### Medical Imaging Context
- Focus on **cardiac MRI and CT images**
- Clinical evaluation tasks for foundation models
- Research-oriented, not production medical device software

## What NOT to Suggest
- Do not suggest breaking changes to DINOv3 license terms
- Do not recommend reverse engineering DINOv3 components
- Do not suggest removing license headers from DINOv3-derived code
- Avoid general-purpose image processing when cardiac-specific approaches exist
