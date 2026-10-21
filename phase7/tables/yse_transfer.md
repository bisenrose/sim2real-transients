**PLAsTiCC → YSE transfer macro-F1 over classes with >= 10 objects, all spectroscopic YSE objects; class counts {'SNII': 102, 'SNIbc': 27, 'SNIa': 248, 'SLSN-I': 2} (5 split seeds)**

| Model | Consistent | Per-survey | Per-survey − abs mag | Redshift only shared | M_peak only shared | Hybrid rule | Hybrid + brightness shared | YSE-only CV (pooled out-of-fold) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.586 ± 0.008 | 0.518 ± 0.020 | 0.478 ± 0.015 | 0.472 ± 0.024 | 0.521 ± 0.014 | 0.463 ± 0.029 | 0.546 ± 0.011 | 0.599 ± 0.015 |
| LightGBM | 0.571 ± 0.012 | 0.511 ± 0.020 | 0.475 ± 0.009 | 0.481 ± 0.016 | 0.520 ± 0.013 | 0.471 ± 0.024 | 0.535 ± 0.006 | 0.596 ± 0.026 |
| CatBoost | 0.587 ± 0.007 | 0.519 ± 0.008 | 0.497 ± 0.010 | 0.455 ± 0.034 | 0.515 ± 0.002 | 0.449 ± 0.027 | 0.520 ± 0.014 | 0.550 ± 0.013 |
| MLP | 0.273 ± 0.016 | 0.522 ± 0.013 | 0.474 ± 0.016 | 0.549 ± 0.016 | 0.507 ± 0.010 | 0.517 ± 0.011 | 0.352 ± 0.052 | 0.574 ± 0.017 |
