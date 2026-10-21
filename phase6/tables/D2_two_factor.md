**Two-factor controlled test inside PLAsTiCC (ZTF-like class priors; 5 split seeds)**

| Population | Instrument shift | Model | Consistent | Per-survey | Hybrid rule | Hybrid, abs mag only |
| --- | --- | --- | --- | --- | --- | --- |
| Ia-rich random (control) | none | XGBoost | 0.685 ± 0.018 | 0.557 ± 0.028 | 0.654 ± 0.019 | 0.644 ± 0.026 |
| Ia-rich random (control) | none | LightGBM | 0.689 ± 0.011 | 0.537 ± 0.013 | 0.612 ± 0.016 | 0.617 ± 0.009 |
| Ia-rich random (control) | I1 cadence + noise | XGBoost | 0.679 ± 0.029 | 0.557 ± 0.028 | 0.654 ± 0.019 | 0.644 ± 0.026 |
| Ia-rich random (control) | I1 cadence + noise | LightGBM | 0.666 ± 0.031 | 0.537 ± 0.013 | 0.612 ± 0.016 | 0.617 ± 0.009 |
| Ia-rich random (control) | I2 flux calibration | XGBoost | 0.683 ± 0.022 | 0.557 ± 0.028 | 0.647 ± 0.023 | 0.635 ± 0.027 |
| Ia-rich random (control) | I2 flux calibration | LightGBM | 0.665 ± 0.027 | 0.537 ± 0.013 | 0.605 ± 0.020 | 0.603 ± 0.010 |
| Ia-rich low-z | none | XGBoost | 0.548 ± 0.026 | 0.326 ± 0.013 | 0.536 ± 0.025 | 0.533 ± 0.012 |
| Ia-rich low-z | none | LightGBM | 0.558 ± 0.008 | 0.355 ± 0.025 | 0.517 ± 0.018 | 0.552 ± 0.021 |
| Ia-rich low-z | I1 cadence + noise | XGBoost | 0.508 ± 0.024 | 0.326 ± 0.013 | 0.536 ± 0.025 | 0.533 ± 0.012 |
| Ia-rich low-z | I1 cadence + noise | LightGBM | 0.531 ± 0.032 | 0.355 ± 0.025 | 0.517 ± 0.018 | 0.552 ± 0.021 |
| Ia-rich low-z | I2 flux calibration | XGBoost | 0.589 ± 0.005 | 0.326 ± 0.013 | 0.514 ± 0.029 | 0.530 ± 0.010 |
| Ia-rich low-z | I2 flux calibration | LightGBM | 0.594 ± 0.014 | 0.355 ± 0.025 | 0.490 ± 0.025 | 0.551 ± 0.020 |
