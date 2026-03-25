============================================================
ACDC Pathology Classification — Summary
============================================================
Backbone: dinov3 (dinov3_vits16)
Feature type: Final-layer CLS token
embed_dim: 384
Pooling: ED-mean + ES-mean → (768,)
Eval mode: Logistic Regression (frozen backbone)
Normalisation: StandardScaler (zero mean, unit variance)
Classifier: sklearn LogisticRegression (L-BFGS, L2)
Model selection: 10-fold stratified CV
Best C: 562.3
Train patients: 100
Test patients: 50
────────────────────────────────────────────────────────────
5-Way Classification:
Test Accuracy: 0.5000
Test Macro F1: 0.4976
Macro ROC AUC: 0.8275
Binary Disease Detection:
Accuracy: 0.7600
F1 Score: 0.8537
Sensitivity: 0.8750
Specificity: 0.3000
ROC AUC: 0.7500
============================================================
