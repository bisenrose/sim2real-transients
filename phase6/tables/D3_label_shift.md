**Label-shift (prior) correction by EM on saved probabilities (5 split seeds)**

| Target | Model | Preprocessing | Macro-F1 | After EM prior correction | EM-estimated SNIa prior | After true-prior correction (exploratory) |
| --- | --- | --- | --- | --- | --- | --- |
| ZTF | xgboost | Consistent | 0.503 ± 0.043 | 0.420 ± 0.020 | 0.51 (true 0.76) | 0.567 ± 0.017 |
| ZTF | xgboost | Per-survey | 0.510 ± 0.015 | 0.415 ± 0.043 | 0.56 (true 0.76) | 0.558 ± 0.012 |
| ZTF | xgboost | Hybrid rule | 0.442 ± 0.023 | 0.164 ± 0.049 | 0.14 (true 0.76) | 0.422 ± 0.026 |
| ZTF | xgboost | Hybrid, abs mag only | 0.529 ± 0.021 | 0.319 ± 0.034 | 0.63 (true 0.76) | 0.566 ± 0.012 |
| DES | xgboost | Consistent | 0.488 ± 0.023 | 0.238 ± 0.015 | 0.40 (true 0.74) | 0.458 ± 0.025 |
| DES | xgboost | Per-survey | 0.528 ± 0.005 | 0.466 ± 0.022 | 0.19 (true 0.74) | 0.558 ± 0.010 |
| DES | xgboost | Hybrid rule | 0.602 ± 0.012 | 0.565 ± 0.009 | 0.69 (true 0.74) | 0.582 ± 0.015 |
| ZTF | lightgbm | Consistent | 0.517 ± 0.020 | 0.444 ± 0.032 | 0.51 (true 0.76) | 0.561 ± 0.016 |
| ZTF | lightgbm | Per-survey | 0.526 ± 0.005 | 0.500 ± 0.017 | 0.57 (true 0.76) | 0.567 ± 0.019 |
| ZTF | lightgbm | Hybrid rule | 0.454 ± 0.026 | 0.252 ± 0.065 | 0.23 (true 0.76) | 0.485 ± 0.025 |
| ZTF | lightgbm | Hybrid, abs mag only | 0.530 ± 0.023 | 0.437 ± 0.030 | 0.63 (true 0.76) | 0.585 ± 0.008 |
| DES | lightgbm | Consistent | 0.506 ± 0.020 | 0.387 ± 0.012 | 0.50 (true 0.74) | 0.475 ± 0.024 |
| DES | lightgbm | Per-survey | 0.518 ± 0.009 | 0.465 ± 0.019 | 0.23 (true 0.74) | 0.549 ± 0.008 |
| DES | lightgbm | Hybrid rule | 0.603 ± 0.016 | 0.565 ± 0.012 | 0.68 (true 0.74) | 0.585 ± 0.006 |
| ZTF | catboost | Consistent | 0.500 ± 0.034 | 0.429 ± 0.020 | 0.55 (true 0.76) | 0.528 ± 0.030 |
| ZTF | catboost | Per-survey | 0.506 ± 0.015 | 0.433 ± 0.070 | 0.48 (true 0.76) | 0.546 ± 0.015 |
| ZTF | catboost | Hybrid rule | 0.399 ± 0.043 | 0.046 ± 0.022 | 0.00 (true 0.76) | 0.390 ± 0.027 |
| ZTF | catboost | Hybrid, abs mag only | 0.552 ± 0.020 | 0.388 ± 0.034 | 0.58 (true 0.76) | 0.545 ± 0.019 |
| DES | catboost | Consistent | 0.499 ± 0.027 | 0.108 ± 0.014 | 0.23 (true 0.74) | 0.427 ± 0.014 |
| DES | catboost | Per-survey | 0.517 ± 0.010 | 0.402 ± 0.016 | 0.18 (true 0.74) | 0.548 ± 0.012 |
| DES | catboost | Hybrid rule | 0.589 ± 0.012 | 0.533 ± 0.012 | 0.68 (true 0.74) | 0.554 ± 0.006 |
| ZTF | mlp | Consistent | 0.192 ± 0.031 | 0.083 ± 0.048 | 0.22 (true 0.76) | 0.291 ± 0.035 |
| ZTF | mlp | Per-survey | 0.511 ± 0.016 | 0.509 ± 0.033 | 0.60 (true 0.76) | 0.540 ± 0.006 |
| ZTF | mlp | Hybrid rule | 0.495 ± 0.010 | 0.413 ± 0.018 | 0.68 (true 0.76) | 0.479 ± 0.011 |
| ZTF | mlp | Hybrid, abs mag only | 0.452 ± 0.019 | 0.411 ± 0.009 | 0.67 (true 0.76) | 0.554 ± 0.009 |
| DES | mlp | Consistent | 0.466 ± 0.033 | 0.274 ± 0.048 | 0.53 (true 0.74) | 0.441 ± 0.048 |
| DES | mlp | Per-survey | 0.564 ± 0.019 | 0.549 ± 0.020 | 0.38 (true 0.74) | 0.572 ± 0.012 |
| DES | mlp | Hybrid rule | 0.617 ± 0.022 | 0.542 ± 0.022 | 0.83 (true 0.74) | 0.577 ± 0.018 |
