**Simulated DES selection on PLAsTiCC training data: XGBoost, hybrid rule (5 split seeds)**

| Training selection | N train | PLAsTiCC F1 | Transfer F1 | abs mag KS | abs mag median gap (DES − train) |
| --- | --- | --- | --- | --- | --- |
| None | 3625 | 0.785 ± 0.007 | 0.602 ± 0.012 | 0.35 ± 0.00 | -1.10 ± 0.02 |
| Logistic (DES-matched) | 1181 | 0.727 ± 0.013 | 0.570 ± 0.036 | 0.22 ± 0.01 | -0.44 ± 0.02 |
| Random, same size | 1181 | 0.750 ± 0.017 | 0.562 ± 0.028 | 0.35 ± 0.01 | -1.11 ± 0.08 |
| Hard cut (DES 99th pct) | 2325 | 0.738 ± 0.003 | 0.585 ± 0.015 | 0.23 ± 0.00 | -0.47 ± 0.01 |
| Random, same size | 2325 | 0.768 ± 0.009 | 0.613 ± 0.012 | 0.35 ± 0.01 | -1.05 ± 0.04 |
