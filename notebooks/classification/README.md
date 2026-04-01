# Classification Plots

`plot_classification_results.ipynb` generates diagnostic plots from a JSON
results file produced by `scripts/classification/run_acdc_classification.py`.

Point `RESULTS_PATH` at any result JSON and run all cells to get:

- **Hyperparameter sweep curve** — CV accuracy vs regularisation C (logreg) or
  learning rate (finetune), with the selected value highlighted.
- **Per-class ROC curves** — one subplot per pathology class (one-vs-rest).
- **Per-class AUC bar chart** — with macro AUC reference line.
- **Confusion matrix** — annotated heatmap of 5-way predictions.
- **Binary disease detection ROC** — Normal vs any disease.
- **Summary table** — per-class precision, recall, F1, and support.
