**Which features keep the shared scale: transfer macro-F1 on DES (5 split seeds)**

| Model | Nothing shared (per-survey) | Redshift only shared | M_peak only shared | Both shared (hybrid rule) | Both + brightness shared |
| --- | --- | --- | --- | --- | --- |
| XGBoost | 0.528 ± 0.005 | 0.527 ± 0.010 | 0.607 ± 0.012 | 0.602 ± 0.012 | 0.597 ± 0.019 |
| LightGBM | 0.518 ± 0.009 | 0.530 ± 0.007 | -- | 0.603 ± 0.016 | 0.598 ± 0.018 |
| CatBoost | 0.517 ± 0.010 | 0.526 ± 0.008 | -- | 0.589 ± 0.012 | 0.617 ± 0.020 |
| MLP | 0.564 ± 0.019 | 0.566 ± 0.023 | -- | 0.617 ± 0.022 | 0.661 ± 0.015 |
