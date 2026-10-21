# Phase 8: per-feature shift comparison and a stronger controlled test

Commit `PREREGISTRATION_8.md` (with its amendment), then `bash run_phase8.sh` from the `xai` folder.

| Block | Files | Question |
| --- | --- | --- |
| E1 | `features_compare.py` | Are YSE's noise and cadence features closer to PLAsTiCC than DES's (why consistent preprocessing wins on YSE)? |
| E2 (confirmatory) | `configs/E2_*` (96) | Does a ZTF-like brightness offset (objects moved nearer x3 or x6), with or without larger calibration offsets or a strong season shift, reproduce the held-out failure? |

Outputs: `tables/E1_groups.md`, `tables/E1_per_feature_ks.csv`, `figures/E1_ks_heatmap.png`, `tables/E2_controlled.md|tex`, `results/verdicts_8.json`.
