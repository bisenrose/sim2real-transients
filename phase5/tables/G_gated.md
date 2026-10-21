**EXPLORATORY overlap-gated rule: sensitivity to the threshold τ (5 split seeds)**

| Target | Model | τ | Kept on shared scale | Overlap | Transfer F1 |
| --- | --- | --- | --- | --- | --- |
| DES | XGBoost | 0.30 | redshift, abs_peak_mag | redshift 0.92, abs_peak_mag 0.97 | 0.602 ± 0.012 |
| DES | XGBoost | 0.50 | redshift, abs_peak_mag | redshift 0.92, abs_peak_mag 0.97 | 0.602 ± 0.012 |
| DES | XGBoost | 0.70 | redshift, abs_peak_mag | redshift 0.92, abs_peak_mag 0.97 | 0.602 ± 0.012 |
| DES | XGBoost | 0.90 | redshift, abs_peak_mag | redshift 0.92, abs_peak_mag 0.97 | 0.602 ± 0.012 |
| DES | LightGBM | 0.30 | redshift, abs_peak_mag | redshift 0.92, abs_peak_mag 0.97 | 0.603 ± 0.016 |
| DES | LightGBM | 0.50 | redshift, abs_peak_mag | redshift 0.92, abs_peak_mag 0.97 | 0.603 ± 0.016 |
| DES | LightGBM | 0.70 | redshift, abs_peak_mag | redshift 0.92, abs_peak_mag 0.97 | 0.603 ± 0.016 |
| DES | LightGBM | 0.90 | redshift, abs_peak_mag | redshift 0.92, abs_peak_mag 0.97 | 0.603 ± 0.016 |
| ZTF | XGBoost | 0.30 | abs_peak_mag | redshift 0.17, abs_peak_mag 0.99 | 0.529 ± 0.021 |
| ZTF | XGBoost | 0.50 | abs_peak_mag | redshift 0.17, abs_peak_mag 0.99 | 0.529 ± 0.021 |
| ZTF | XGBoost | 0.70 | abs_peak_mag | redshift 0.17, abs_peak_mag 0.99 | 0.529 ± 0.021 |
| ZTF | XGBoost | 0.90 | abs_peak_mag | redshift 0.17, abs_peak_mag 0.99 | 0.529 ± 0.021 |
| ZTF | LightGBM | 0.30 | abs_peak_mag | redshift 0.17, abs_peak_mag 0.99 | 0.530 ± 0.023 |
| ZTF | LightGBM | 0.50 | abs_peak_mag | redshift 0.17, abs_peak_mag 0.99 | 0.530 ± 0.023 |
| ZTF | LightGBM | 0.70 | abs_peak_mag | redshift 0.17, abs_peak_mag 0.99 | 0.530 ± 0.023 |
| ZTF | LightGBM | 0.90 | abs_peak_mag | redshift 0.17, abs_peak_mag 0.99 | 0.530 ± 0.023 |
| PLAsTiCC low-z | XGBoost | 0.30 | abs_peak_mag | redshift 0.00, abs_peak_mag 0.60 | 0.434 ± 0.019 |
| PLAsTiCC low-z | XGBoost | 0.50 | abs_peak_mag | redshift 0.00, abs_peak_mag 0.60 | 0.434 ± 0.019 |
| PLAsTiCC low-z | XGBoost | 0.70 | none | redshift 0.00, abs_peak_mag 0.60 | 0.254 ± 0.013 |
| PLAsTiCC low-z | XGBoost | 0.90 | none | redshift 0.00, abs_peak_mag 0.60 | 0.254 ± 0.013 |
| PLAsTiCC low-z | LightGBM | 0.30 | abs_peak_mag | redshift 0.00, abs_peak_mag 0.60 | 0.393 ± 0.042 |
| PLAsTiCC low-z | LightGBM | 0.50 | abs_peak_mag | redshift 0.00, abs_peak_mag 0.60 | 0.393 ± 0.042 |
| PLAsTiCC low-z | LightGBM | 0.70 | none | redshift 0.00, abs_peak_mag 0.60 | 0.236 ± 0.014 |
| PLAsTiCC low-z | LightGBM | 0.90 | none | redshift 0.00, abs_peak_mag 0.60 | 0.236 ± 0.014 |
| PLAsTiCC random | XGBoost | 0.30 | redshift, abs_peak_mag | redshift 0.87, abs_peak_mag 0.88 | 0.743 ± 0.016 |
| PLAsTiCC random | XGBoost | 0.50 | redshift, abs_peak_mag | redshift 0.87, abs_peak_mag 0.88 | 0.743 ± 0.016 |
| PLAsTiCC random | XGBoost | 0.70 | redshift, abs_peak_mag | redshift 0.87, abs_peak_mag 0.88 | 0.743 ± 0.016 |
| PLAsTiCC random | XGBoost | 0.90 | none | redshift 0.87, abs_peak_mag 0.88 | 0.744 ± 0.012 |
| PLAsTiCC random | LightGBM | 0.30 | redshift, abs_peak_mag | redshift 0.87, abs_peak_mag 0.88 | 0.733 ± 0.008 |
| PLAsTiCC random | LightGBM | 0.50 | redshift, abs_peak_mag | redshift 0.87, abs_peak_mag 0.88 | 0.733 ± 0.008 |
| PLAsTiCC random | LightGBM | 0.70 | redshift, abs_peak_mag | redshift 0.87, abs_peak_mag 0.88 | 0.733 ± 0.008 |
| PLAsTiCC random | LightGBM | 0.90 | none | redshift 0.87, abs_peak_mag 0.88 | 0.730 ± 0.002 |
