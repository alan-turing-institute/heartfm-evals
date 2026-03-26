# ACDC Pathology Classification — Summary

5-way patient-level classification (100 train / 50 test) + binary disease detection (NOR vs disease).
Per-run details (per-class metrics, confusion matrices, plots) are in the individual result files.

| Backbone | Embed Dim | Eval Mode | 5-way Acc | 5-way F1 | 5-way AUC | Binary Acc | Binary F1 | Binary Sens | Binary Spec | Binary AUC |
| -------- | --------- | --------- | --------- | -------- | --------- | ---------- | --------- | ----------- | ----------- | ---------- |
| CineMA   | 768       | logreg    | 0.6400    | 0.6392   | 0.9055    | 0.8600     | 0.9114    | 0.9000      | 0.7000      | 0.9350     |
| Dino     | 384       | logreg    | 0.5000    | 0.4976   | 0.8275    | 0.7600     | 0.8537    | 0.8750      | 0.3000      | 0.7500     |
| SAM      | 256       | logreg    | 0.5000    | 0.5056   | 0.8225    | 0.8000     | 0.8780    | 0.9000      | 0.4000      | 0.8300     |
