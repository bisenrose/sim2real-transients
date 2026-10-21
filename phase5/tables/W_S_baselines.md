**Magnitude-ratio weights and StratLearn (consistent preprocessing, 5 split seeds)**

| Target | Model | Method | Transfer F1 | In-domain F1 | Diagnostics |
| --- | --- | --- | --- | --- | --- |
| DES | XGBoost | Consistent (baseline) | 0.488 ± 0.023 | 0.785 ± 0.007 |  |
| DES | XGBoost | + magnitude weights | 0.570 ± 0.015 | 0.770 ± 0.011 | ESS 1892 |
| DES | XGBoost | + StratLearn, all features | 0.341 ± 0.024 | 0.785 ± 0.007 | AUC 1.000; fallback 0.00 |
| DES | XGBoost | + StratLearn, survey features hidden | 0.560 ± 0.017 | 0.785 ± 0.007 | AUC 0.859; fallback 0.00 |
| DES | LightGBM | Consistent (baseline) | 0.506 ± 0.020 | 0.782 ± 0.006 |  |
| DES | LightGBM | + magnitude weights | 0.565 ± 0.016 | 0.774 ± 0.010 | ESS 1892 |
| DES | LightGBM | + StratLearn, all features | 0.371 ± 0.025 | 0.782 ± 0.006 | AUC 1.000; fallback 0.00 |
| DES | LightGBM | + StratLearn, survey features hidden | 0.575 ± 0.007 | 0.782 ± 0.006 | AUC 0.859; fallback 0.00 |
| ZTF | XGBoost | Consistent (baseline) | 0.503 ± 0.043 | 0.785 ± 0.007 |  |
| ZTF | XGBoost | + magnitude weights | 0.516 ± 0.033 | 0.682 ± 0.016 | ESS 83 |
| ZTF | XGBoost | + StratLearn, all features | 0.491 ± 0.021 | 0.785 ± 0.007 | AUC 1.000; fallback 0.95 |
| ZTF | XGBoost | + StratLearn, survey features hidden | 0.491 ± 0.028 | 0.785 ± 0.007 | AUC 0.998; fallback 0.95 |
| ZTF | LightGBM | Consistent (baseline) | 0.517 ± 0.020 | 0.782 ± 0.006 |  |
| ZTF | LightGBM | + magnitude weights | 0.474 ± 0.020 | 0.775 ± 0.009 | ESS 83 |
| ZTF | LightGBM | + StratLearn, all features | 0.496 ± 0.018 | 0.782 ± 0.006 | AUC 1.000; fallback 0.95 |
| ZTF | LightGBM | + StratLearn, survey features hidden | 0.496 ± 0.014 | 0.782 ± 0.006 | AUC 0.998; fallback 0.95 |
