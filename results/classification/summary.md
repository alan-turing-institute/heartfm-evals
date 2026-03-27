# ACDC Pathology Classification — Summary

5-way patient-level classification (100 train / 50 test) + binary disease detection (NOR vs disease).
Per-run details (per-class metrics, confusion matrices, plots) are in the individual result files.

| Backbone | Embed Dim | Eval Mode | Pooling | 5-way Acc | 5-way F1 | 5-way AUC | Binary Acc | Binary F1 | Binary Sens | Binary Spec | Binary AUC |
| -------- | --------- | --------- | ------- | --------- | -------- | --------- | ---------- | --------- | ----------- | ----------- | ---------- |
| CineMA   | 768       | logreg    | cls     | 0.6400    | 0.6399   | 0.8935    | 0.8000     | 0.8864    | 0.9750      | 0.1000      | 0.8700     |
| CineMA   | 768       | logreg    | gap     | 0.6400    | 0.6392   | 0.9055    | 0.8600     | 0.9114    | 0.9000      | 0.7000      | 0.9350     |
| CineMA   | 768       | ft-frozen | cls     | 0.6400    | 0.6370   | 0.8830    | 0.8400     | 0.9024    | 0.9250      | 0.5000      | 0.8650     |
| CineMA   | 768       | ft-frozen | gap     | 0.6600    | 0.6569   | 0.9010    | 0.8600     | 0.9114    | 0.9000      | 0.7000      | 0.9250     |
| Dino     | 384       | logreg    | cls     | 0.5000    | 0.4976   | 0.8275    | 0.7600     | 0.8537    | 0.8750      | 0.3000      | 0.7500     |
| Dino     | 384       | logreg    | gap     | 0.4800    | 0.4760   | 0.8280    | 0.7600     | 0.8500    | 0.8500      | 0.4000      | 0.7375     |
| Dino     | 384       | ft-frozen | cls     | 0.5200    | 0.5124   | 0.8400    | 0.7600     | 0.8537    | 0.8750      | 0.3000      | 0.7650     |
| Dino     | 384       | ft-frozen | gap     | 0.4800    | 0.4755   | 0.8350    | 0.7600     | 0.8500    | 0.8500      | 0.4000      | 0.7400     |
| SAM      | 256       | logreg    | gap     | 0.5000    | 0.5056   | 0.8225    | 0.8000     | 0.8780    | 0.9000      | 0.4000      | 0.8300     |
| SAM      | 256       | ft-frozen | gap     | 0.5000    | 0.4890   | 0.8480    | 0.7800     | 0.8675    | 0.9000      | 0.3000      | 0.8475     |
