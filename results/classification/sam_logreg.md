============================================================
ACDC Pathology Classification — Summary
============================================================
Backbone: sam (sam_vit_base)
Feature type: Global-average-pooled image encoder features
embed_dim: 256
Pooling: ED-mean + ES-mean → (512,)
Eval mode: Logistic Regression (frozen backbone)
Normalisation: StandardScaler (zero mean, unit variance)
Classifier: sklearn LogisticRegression (L-BFGS, L2)
Model selection: 10-fold stratified CV
Best C: 3.162e+04
Train patients: 100
Test patients: 50
────────────────────────────────────────────────────────────
5-Way Classification:
Test Accuracy: 0.5000
Test Macro F1: 0.5056
Macro ROC AUC: 0.8225
Binary Disease Detection:
Accuracy: 0.8000
F1 Score: 0.8780
Sensitivity: 0.9000
Specificity: 0.4000
ROC AUC: 0.8300
============================================================
