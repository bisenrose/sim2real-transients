**Simulated DES selection on PLAsTiCC training data: XGBoost, consistent preprocessing (5 split seeds)**

| Training selection | N train | PLAsTiCC F1 | Transfer F1 | abs mag KS | abs mag median gap (DES − train) |
| --- | --- | --- | --- | --- | --- |
| None | 3625 | 0.785 ± 0.007 | 0.488 ± 0.023 | 0.35 ± 0.00 | -1.10 ± 0.02 |
| Logistic (DES-matched) | 1181 | 0.727 ± 0.013 | 0.571 ± 0.009 | 0.22 ± 0.01 | -0.44 ± 0.02 |
| Random, same size | 1181 | 0.750 ± 0.017 | 0.455 ± 0.033 | 0.35 ± 0.01 | -1.11 ± 0.08 |
| Hard cut (DES 99th pct) | 2325 | 0.738 ± 0.003 | 0.546 ± 0.014 | 0.23 ± 0.00 | -0.47 ± 0.01 |
| Random, same size | 2325 | 0.768 ± 0.009 | 0.484 ± 0.007 | 0.35 ± 0.01 | -1.05 ± 0.04 |
