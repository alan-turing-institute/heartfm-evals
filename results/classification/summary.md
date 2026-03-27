# ACDC Pathology Classification — Summary

5-way patient-level classification (100 train / 50 test) + binary disease detection (NOR vs disease).
Per-run details (per-class metrics, confusion matrices, plots) are in the individual result files.

| Backbone | Embed Dim | Eval Mode | Pooling | 5-way Acc | 5-way F1 | 5-way AUC | Binary Acc | Binary F1 | Binary Sens | Binary Spec | Binary AUC |
| -------- | --------- | --------- | ------- | --------- | -------- | --------- | ---------- | --------- | ----------- | ----------- | ---------- |
| CineMA   | 768       | logreg    | cls     | 0.6400    | 0.6399   | 0.8935    | 0.8000     | 0.8864    | 0.9750      | 0.1000      | 0.8700     |
| CineMA   | 768       | logreg    | gap     | 0.6400    | 0.6392   | 0.9055    | 0.8600     | 0.9114    | 0.9000      | 0.7000      | 0.9350     |
| CineMA   | 768       | ft-frozen | cls     | 0.6400    | 0.6338   | 0.8840    | 0.8400     | 0.9024    | 0.9250      | 0.5000      | 0.8650     |
| CineMA   | 768       | ft-frozen | gap     | 0.6600    | 0.6527   | 0.9000    | 0.8600     | 0.9114    | 0.9000      | 0.7000      | 0.9250     |
| Dino     | 768       | logreg    | cls     | 0.5800    | 0.5686   | 0.9020    | 0.7600     | 0.8571    | 0.9000      | 0.2000      | 0.8450     |
| Dino     | 384       | logreg    | cls     | 0.5000    | 0.4976   | 0.8275    | 0.7600     | 0.8537    | 0.8750      | 0.3000      | 0.7500     |
| Dino     | 768       | logreg    | gap     | 0.6200    | 0.6158   | 0.9105    | 0.8000     | 0.8780    | 0.9000      | 0.4000      | 0.8475     |
| Dino     | 384       | logreg    | gap     | 0.4800    | 0.4760   | 0.8280    | 0.7600     | 0.8500    | 0.8500      | 0.4000      | 0.7375     |
| Dino     | 768       | ft-frozen | cls     | 0.5800    | 0.5686   | 0.9090    | 0.7800     | 0.8675    | 0.9000      | 0.3000      | 0.8475     |
| Dino     | 384       | ft-frozen | cls     | 0.5400    | 0.5346   | 0.8420    | 0.7800     | 0.8642    | 0.8750      | 0.4000      | 0.7750     |
| Dino     | 768       | ft-frozen | gap     | 0.6400    | 0.6321   | 0.9080    | 0.8000     | 0.8810    | 0.9250      | 0.3000      | 0.8450     |
| Dino     | 384       | ft-frozen | gap     | 0.4800    | 0.4735   | 0.8360    | 0.7800     | 0.8642    | 0.8750      | 0.4000      | 0.7425     |
| SAM      | 256       | logreg    | gap     | 0.4600    | 0.4469   | 0.8465    | 0.8200     | 0.8966    | 0.9750      | 0.2000      | 0.8900     |
| SAM      | 256       | ft-frozen | gap     | 0.5200    | 0.5130   | 0.8505    | 0.8000     | 0.8810    | 0.9250      | 0.3000      | 0.8500     |
