**SHAP rank correlations over 27 features, corrected pipeline (Spearman; mean ± SD over 5 split seeds; ranks are medians)**

| model | mode | rho label-free (P model: PLAsTiCC vs DES) | max p | rho labelled (P model vs DES model) | max p  | abs_peak_mag rank (P / P-on-DES / DES) |
| --- | --- | --- | --- | --- | --- | --- |
| catboost | consistent | 0.964 ± 0.015 | 1.2e-13 | 0.726 ± 0.046 | 1.4e-04 | 1 / 2 / 1 |
| catboost | hybrid | 0.974 ± 0.003 | 1.1e-16 | 0.726 ± 0.046 | 1.4e-04 | 1 / 1 / 1 |
| lightgbm | consistent | 0.901 ± 0.012 | 9.8e-10 | 0.741 ± 0.030 | 7.3e-05 | 1 / 2 / 1 |
| lightgbm | hybrid | 0.863 ± 0.026 | 1.9e-07 | 0.741 ± 0.030 | 7.3e-05 | 1 / 1 / 1 |
| xgboost | consistent | 0.955 ± 0.008 | 1.6e-13 | 0.735 ± 0.017 | 3.4e-05 | 1 / 1 / 1 |
| xgboost | hybrid | 0.935 ± 0.009 | 5.4e-12 | 0.735 ± 0.017 | 3.4e-05 | 1 / 1 / 1 |
