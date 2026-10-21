**PLAsTiCC → ZTF transfer macro-F1 split by light-curve source**

| Model / preprocessing | detections (n=4362) | forced (n=1902) |
| --- | --- | --- |
| xgboost / consistent | 0.508 ± 0.042 | 0.495 ± 0.044 |
| xgboost / per_survey | 0.527 ± 0.020 | 0.473 ± 0.013 |
| xgboost / per_survey_drop_absmag | 0.497 ± 0.015 | 0.481 ± 0.017 |
| xgboost / hybrid | 0.443 ± 0.017 | 0.439 ± 0.036 |
| lightgbm / consistent | 0.518 ± 0.022 | 0.516 ± 0.027 |
| lightgbm / per_survey | 0.538 ± 0.006 | 0.498 ± 0.019 |
| lightgbm / per_survey_drop_absmag | 0.503 ± 0.007 | 0.482 ± 0.018 |
| lightgbm / hybrid | 0.449 ± 0.027 | 0.463 ± 0.029 |
| catboost / consistent | 0.494 ± 0.033 | 0.499 ± 0.041 |
| catboost / per_survey | 0.531 ± 0.019 | 0.460 ± 0.015 |
| catboost / per_survey_drop_absmag | 0.543 ± 0.007 | 0.484 ± 0.011 |
| catboost / hybrid | 0.392 ± 0.048 | 0.420 ± 0.033 |
| mlp / consistent | 0.196 ± 0.030 | 0.184 ± 0.038 |
| mlp / per_survey | 0.511 ± 0.018 | 0.506 ± 0.015 |
| mlp / per_survey_drop_absmag | 0.477 ± 0.019 | 0.456 ± 0.018 |
| mlp / hybrid | 0.496 ± 0.013 | 0.492 ± 0.008 |
