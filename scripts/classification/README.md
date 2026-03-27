# ACDC Pathology Classification

Evaluates how well different pretrained backbones capture clinically relevant
information from cardiac cine MRI, following the protocol introduced for
CineMA (Fu et al., 2025).

> Do the frozen representations of a given foundation model already encode
> enough about cardiac anatomy and function to distinguish between pathologies,
> or is task-specific fine-tuning necessary to unlock that performance?

## Task

**5-way patient-level pathology classification** on the
[ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/) (150 patients,
100 train / 50 test). Classes: NOR (normal), DCM (dilated cardiomyopathy),
HCM (hypertrophic cardiomyopathy), MINF (myocardial infarction), and
RV (right ventricular cardiomyopathy).

A secondary **binary disease detection** task (NOR vs any disease) is derived
from the 5-way classifier output without additional training.

## Backbones

| Backbone   | Input         | Pretraining                              |
| ---------- | ------------- | ---------------------------------------- |
| **CineMA** | 3D SAX volume | Masked autoencoder on 15M cine images    |
| **DINOv3** | 2D slices     | Self-supervised (DINO) on natural images |
| **SAM**    | 2D slices     | Segment Anything on natural images       |

All backbones produce one embedding vector per 2D slice. Patient-level features
are obtained by mean-pooling ED and ES slices separately, then concatenating
into a single vector. Two pooling modes are compared: CLS token (`cls`) and
global average pooling (`gap`). SAM only supports `gap` (no CLS token).

## Evaluation Modes

- **`logreg`** — Logistic regression linear probe over frozen features.
  Regularisation strength C is selected via 10-fold stratified CV.
- **`finetune`** — Linear head on top of the backbone, trained with AdamW +
  cosine annealing. Learning rate is selected via 10-fold stratified CV over
  frozen features. The backbone can be kept frozen (`--freeze-backbone`, default)
  or fine-tuned end-to-end (`--no-freeze-backbone`).

## Scripts

| Script                        | Purpose                                                            |
| ----------------------------- | ------------------------------------------------------------------ |
| `run_acdc_classification.py`  | Run a single experiment (backbone + eval mode + pooling)           |
| `run_all_classification.sh`   | Run the full experiment grid (all backbone/mode/pooling combos)    |
| `build_summary.py`            | Aggregate all result JSONs into `results/classification/summary.md`|

### Quick start

```bash
# Single run
python scripts/classification/run_acdc_classification.py \
    --backbone cinema --eval-mode logreg --pooling cls

# Full grid
bash scripts/classification/run_all_classification.sh

# Rebuild summary table
python scripts/classification/build_summary.py
```

## Metrics

Each run reports:

**5-way classification** — accuracy, macro F1, per-class and macro
sensitivity/specificity, per-class and macro ROC AUC, confusion matrix, and
full classification report.

**Binary disease detection** — accuracy, F1, sensitivity, specificity, ROC AUC.

Results are saved as JSON files in `results/classification/`. Use the plotting
notebook at `notebooks/classification/plot_classification_results.ipynb` to
visualise individual runs.
