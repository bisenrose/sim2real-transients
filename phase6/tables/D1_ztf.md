**Which features keep the shared scale: transfer macro-F1 on ZTF (5 split seeds)**

| Model | Nothing shared (per-survey) | Redshift only shared | M_peak only shared | Both shared (hybrid rule) | Both + brightness shared |
| --- | --- | --- | --- | --- | --- |
| XGBoost | 0.510 ± 0.015 | 0.449 ± 0.015 | 0.529 ± 0.021 | 0.442 ± 0.023 | 0.552 ± 0.010 |
| LightGBM | 0.526 ± 0.005 | 0.492 ± 0.011 | 0.530 ± 0.023 | 0.454 ± 0.026 | 0.538 ± 0.010 |
| CatBoost | 0.506 ± 0.015 | 0.400 ± 0.042 | 0.553 ± 0.020 | 0.399 ± 0.043 | 0.516 ± 0.015 |
| MLP | 0.511 ± 0.016 | 0.525 ± 0.012 | 0.452 ± 0.019 | 0.495 ± 0.010 | 0.244 ± 0.037 |
