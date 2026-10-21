**PLAsTiCC → YSE transfer macro-F1 over classes with >= 10 objects, excluding objects also in ZTF BTS (robustness); class counts {'SNII': 75, 'SNIbc': 16, 'SNIa': 136, 'SLSN-I': 1} (5 split seeds)**

| Model | Consistent | Per-survey | Per-survey − abs mag | Redshift only shared | M_peak only shared | Hybrid rule | Hybrid + brightness shared | YSE-only CV (pooled out-of-fold) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.599 ± 0.016 | 0.533 ± 0.023 | -- | -- | -- | 0.448 ± 0.020 | 0.541 ± 0.007 | 0.544 ± 0.013 |
| LightGBM | 0.584 ± 0.010 | 0.519 ± 0.020 | -- | -- | -- | 0.468 ± 0.012 | 0.527 ± 0.010 | 0.552 ± 0.012 |
