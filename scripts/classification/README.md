# Pathology Classification

Evaluates how well different pretrained backbones capture clinically relevant
information from cardiac cine MRI, following the protocol introduced for
CineMA (Fu et al., 2025).

> Do the frozen representations of a given foundation model already encode
> enough about cardiac anatomy and function to distinguish between pathologies,
> or is task-specific fine-tuning necessary to unlock that performance?

## Datasets

Three cardiac MRI datasets are supported:

| Dataset  | Patients | Classes |
| -------- | -------- | ------- |
| **ACDC** | 150 (100 train / 50 test) | NOR, DCM, HCM, MINF, RV |
| **M&Ms** | 317 (150 train / 33 val / 134 test) | NOR, DCM, HCM, ARV, HHD |
| **M&Ms-2** | 351 (156 train / 38 val / 157 test) | NOR, HCM, ARR, CIA, FALL, LV |

All datasets are expected under `data/heartfm/processed/{dataset}/` and must
provide `train_metadata.csv` and `test_metadata.csv`. If a
`val_metadata.csv` is also present, it is used as a dedicated validation
split instead of K-fold CV.

## Task

**N-way patient-level pathology classification** (5 classes for ACDC and
M&Ms, 6 for M&Ms-2). A secondary **binary disease detection** task (NOR vs
any disease) is derived from the N-way classifier output without additional
training.

## Backbones

| Backbone   | Input         | Pretraining                              |
| ---------- | ------------- | ---------------------------------------- |
| **CineMA** | 3D SAX volume | Masked autoencoder on 15M cine images    |
| **DINOv3** | 2D slices     | Self-supervised (DINO) on natural images |
| **SAM**    | 2D slices     | Segment Anything on natural images       |

All backbones produce one embedding vector per 2D slice (CineMA: per 3D
volume). Patient-level features are obtained by mean-pooling ED and ES
embeddings separately, then concatenating into a single vector. Two pooling
modes are compared: CLS token (`cls`) and global average pooling (`gap`).
SAM only supports `gap` (no CLS token).

## Evaluation Modes

- **`logreg`** — Logistic regression linear probe over frozen features.
  Regularisation strength C is selected via 10-fold stratified CV (or a
  dedicated val split when available).
- **`finetune`** — Linear head on top of the backbone, trained with AdamW +
  cosine annealing. Learning rate is selected via 10-fold stratified CV (or
  a dedicated val split). The backbone can be kept frozen (`--freeze-backbone`,
  default) or fine-tuned end-to-end (`--no-freeze-backbone`).

## Scripts

| Script                        | Purpose                                                                          |
| ----------------------------- | -------------------------------------------------------------------------------- |
| `run_classification.py`       | Run a single experiment (dataset + backbone + eval mode + pooling)               |
| `run_all_classification.sh`   | Run the full experiment grid (all dataset/backbone/mode/pooling combos)          |
| `build_summary.py`            | Aggregate all result JSONs into per-dataset `summary.csv` files                  |
| `smoke_test_classification.sh`| Quick sanity check using a small patient subset on ACDC                          |

### Quick start

```bash
# Single run
python scripts/classification/run_classification.py \
    --dataset acdc --backbone cinema --eval-mode logreg --pooling cls

# Run on M&Ms with DINOv3, full fine-tune
python scripts/classification/run_classification.py \
    --dataset mnm --backbone dinov3 --eval-mode finetune --no-freeze-backbone

# Full grid (all datasets, all backbones)
bash scripts/classification/run_all_classification.sh

# Rebuild per-dataset summary CSVs
python scripts/classification/build_summary.py
```

### Key CLI options

| Option | Default | Description |
| ------ | ------- | ----------- |
| `--dataset` | `acdc` | Dataset: `acdc`, `mnm`, or `mnm2` |
| `--backbone` | — | `cinema`, `dinov3`, or `sam` |
| `--eval-mode` | — | `logreg` or `finetune` |
| `--pooling` | `cls` | `cls` or `gap` (SAM: `gap` only) |
| `--freeze-backbone` / `--no-freeze-backbone` | frozen | Fine-tune mode only |
| `--data-dir` | `data/heartfm/processed/{dataset}` | Override data directory |
| `--output-dir` | `results/classification/{dataset}` | Override output directory |
| `--max-patients` | — | Limit patients (for debugging/smoke tests) |

## Metrics

Each run reports:

**N-way classification** — accuracy, macro F1, per-class and macro
sensitivity/specificity, per-class and macro ROC AUC, confusion matrix, and
full classification report.

**Binary disease detection** — accuracy, F1, sensitivity, specificity, ROC AUC.

Results are saved as JSON files in `results/classification/{dataset}/`. Use
`build_summary.py` to aggregate them into a `summary.csv` per dataset.
