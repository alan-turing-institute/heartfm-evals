============================================================
ACDC Pathology Classification — Summary
============================================================
Backbone: cinema (cinema_pretrained)
Feature type: Per-slice mean-pooled spatial tokens
embed_dim: 768
Pooling: ED-mean + ES-mean → (1536,)
Eval mode: Logistic Regression (frozen backbone)
Normalisation: StandardScaler (zero mean, unit variance)
Classifier: sklearn LogisticRegression (L-BFGS, L2)
Model selection: 10-fold stratified CV
Best C: 31.62
Train patients: 100
Test patients: 50
────────────────────────────────────────────────────────────
5-Way Classification:
Test Accuracy: 0.5800
Test Macro F1: 0.5768
Macro ROC AUC: 0.8130
Binary Disease Detection:
Accuracy: 0.8400
F1 Score: 0.9024
Sensitivity: 0.9250
Specificity: 0.5000
ROC AUC: 0.8675
============================================================
