**PLAsTiCC → ZTF transfer macro-F1 by ZTF redshift tercile**

| Model / preprocessing | z 0.002–0.045 | z 0.045–0.070 | z 0.070–0.390 |
| --- | --- | --- | --- |
| xgboost / consistent | 0.448 ± 0.007 | 0.383 ± 0.010 | 0.382 ± 0.042 |
| xgboost / per_survey | 0.393 ± 0.005 | 0.343 ± 0.040 | 0.328 ± 0.019 |
| xgboost / hybrid | 0.401 ± 0.009 | 0.276 ± 0.013 | 0.332 ± 0.031 |
| xgboost / explore_hybrid_absmag_only | 0.408 ± 0.009 | 0.307 ± 0.006 | 0.331 ± 0.027 |
| lightgbm / consistent | 0.444 ± 0.013 | 0.369 ± 0.013 | 0.413 ± 0.031 |
| lightgbm / per_survey | 0.399 ± 0.010 | 0.354 ± 0.030 | 0.340 ± 0.017 |
| lightgbm / hybrid | 0.404 ± 0.009 | 0.281 ± 0.017 | 0.346 ± 0.034 |
| lightgbm / explore_hybrid_absmag_only | 0.415 ± 0.009 | 0.308 ± 0.014 | 0.335 ± 0.029 |
| catboost / consistent | 0.464 ± 0.010 | 0.400 ± 0.009 | 0.364 ± 0.036 |
| catboost / per_survey | 0.405 ± 0.007 | 0.317 ± 0.005 | 0.320 ± 0.014 |
| catboost / hybrid | 0.420 ± 0.009 | 0.273 ± 0.008 | 0.270 ± 0.053 |
| catboost / explore_hybrid_absmag_only | 0.424 ± 0.007 | 0.309 ± 0.006 | 0.357 ± 0.020 |
| mlp / consistent | 0.187 ± 0.029 | 0.220 ± 0.090 | 0.169 ± 0.037 |
| mlp / per_survey | 0.397 ± 0.007 | 0.341 ± 0.031 | 0.344 ± 0.032 |
| mlp / hybrid | 0.408 ± 0.005 | 0.307 ± 0.020 | 0.352 ± 0.012 |
| mlp / explore_hybrid_absmag_only | 0.415 ± 0.005 | 0.298 ± 0.003 | 0.268 ± 0.028 |
