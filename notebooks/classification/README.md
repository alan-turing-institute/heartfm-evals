# ACDC Pathology Classification

Evaluates how well different pretrained backbones capture clinically relevant
information from cardiac cine MRI.

> Do the frozen representations of a given foundation model already encode
> enough about cardiac anatomy and function to distinguish between pathologies,
> or is task-specific fine-tuning necessary to unlock that performance?

The evaluation follows the protocol introduced for CineMA (Fu et al., "A
versatile foundation model for cine cardiac magnetic resonance image analysis
tasks", 2025) and extends it with a binary disease detection report.

## Task

**5-way patient-level pathology classification** on the
[ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/) (150 patients,
100 train / 50 test). Each patient is assigned one of:

| Label | Pathology                        |
| ----- | -------------------------------- |
| NOR   | Normal cardiac function          |
| DCM   | Dilated Cardiomyopathy           |
| HCM   | Hypertrophic Cardiomyopathy      |
| MINF  | Myocardial Infarction            |
| RV    | Right Ventricular Cardiomyopathy |

A secondary **binary disease detection** task (NOR vs. any disease) is derived
from the 5-way classifier output without additional training.

### Backbones

Three backbone families are supported, spanning different pretraining
strategies and input modalities:

| Backbone   | Input         | Pretraining                              | Embedding                                      |
| ---------- | ------------- | ---------------------------------------- | ---------------------------------------------- |
| **CineMA** | 3D SAX volume | Masked autoencoder on 15M cine images    | Per-slice spatial mean-pool from 3D token grid |
| **DINOv3** | 2D slices     | Self-supervised (DINO) on natural images | CLS token per slice                            |
| **SAM**    | 2D slices     | Segment Anything on natural images       | Global-avg-pool of encoder feature map         |

All backbones produce one embedding vector per 2D slice. Patient-level features
are obtained by mean-pooling ED slices and ES slices separately, then
concatenating into a single `(2 x embed_dim)` vector.

### Evaluation Modes

The notebook supports two evaluation modes, controlled by `EVAL_MODE` in the
config cell:

#### `"logreg"` — Logistic Regression Linear Probe

A standard linear probe that measures representation quality without modifying
the backbone:

1. Extract and cache per-slice embeddings from the **frozen** backbone
2. Pool to patient level (mean ED + mean ES)
3. Fit `StandardScaler` + `LogisticRegression` (L2, L-BFGS)
4. Select regularisation strength C via 10-fold stratified CV (45 values)
5. Retrain on all training data, evaluate on held-out test set

#### `"finetune"` — Fine-Tuning with Linear Head

Measures how well the backbone adapts when allowed to update its weights:

1. Attach a linear classification head to the backbone
2. Train with AdamW + cosine annealing (standard ViT fine-tuning recipe)
3. Select learning rate via 10-fold stratified CV over pre-extracted frozen
   features (5 values: 1e-5 to 1e-3). The sweep always runs on frozen
   features for speed — the frozen LR ranking is a reliable proxy for the
   unfrozen case.
4. Retrain on all training data with best LR, evaluate on test set

Set `FREEZE_BACKBONE = True` to train only the linear head (backbone frozen),
or `False` to fine-tune all parameters end-to-end.

### Metrics

The notebook reports the following for each run:

**5-way classification**

- Top-1 accuracy, macro F1
- Per-class and macro sensitivity (TPR) and specificity (TNR)
- Per-class and macro ROC AUC (one-vs-rest)
- Confusion matrix and full classification report

**Binary disease detection** (derived from 5-way output)

- Accuracy, F1, sensitivity, specificity, ROC AUC
- ROC curve

### Quick Start

Open `notebooks/acdc_classification.ipynb` and set the config variables:

```python
BACKBONE = "cinema"     # "cinema", "dinov3", or "sam"
EVAL_MODE = "logreg"    # "logreg" or "finetune"
FREEZE_BACKBONE = True  # only used when EVAL_MODE == "finetune"
```

Then run all cells. Results appear in sections 8–12.

---

## Source Modules

| Module                           | Role                                                                     |
| -------------------------------- | ------------------------------------------------------------------------ |
| `classification_probe.py`        | Feature extraction, caching, logistic regression probe, binary detection |
| `finetune_classification.py`     | Fine-tuning training loop, LR sweep, evaluation                          |
| `dense_linear_probe.py`          | Pixel-level segmentation probe                                           |
| `linear_classification_probe.py` | SGD-based linear probe (DINOv3 protocol variant)                         |
