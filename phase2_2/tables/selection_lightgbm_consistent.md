**Simulated DES selection on PLAsTiCC training data: LightGBM, consistent preprocessing (5 split seeds)**

| Training selection | N train | PLAsTiCC F1 | Transfer F1 | abs mag KS | abs mag median gap (DES − train) |
| --- | --- | --- | --- | --- | --- |
| None | 3625 | 0.782 ± 0.006 | 0.506 ± 0.020 | 0.35 ± 0.00 | -1.10 ± 0.02 |
| Logistic (DES-matched) | 1181 | 0.724 ± 0.014 | 0.561 ± 0.026 | 0.22 ± 0.01 | -0.44 ± 0.02 |
| Random, same size | 1181 | 0.753 ± 0.011 | 0.469 ± 0.032 | 0.35 ± 0.01 | -1.11 ± 0.08 |
