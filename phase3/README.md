# Phase 3: rigour checks

Uses the engine in `../phase1`. Run everything with `bash run_phase3.sh` from the `xai`
folder (in `des_env`). Read `PREREGISTRATION_3.md` first: it records the predictions.

| Step | What | Files |
| --- | --- | --- |
| 3.1 | Nested tuning on PLAsTiCC only, equal budget per model | `tune.py` -> `tuning/<model>.json`; configs `t_*` |
| 3.2 | Importance reweighting, CORAL, DES-matched selection, each alone and with the hybrid rule | configs `b_*` |
| 3.3 | Paired bootstrap, log-loss/Brier/ECE, reliability diagram, per-class F1, verdicts | `stats.py` |

Outputs: `tables/*.md|tex`, `figures/reliability.png`, `results/verdicts_3.json`.

Configs named `*_iw_no_survey_features` are exploratory (added after the pre-registration):
plain importance reweighting gives every training object the same weight here, because the
domain classifier separates the surveys perfectly using `time_span`; this variant hides the
survey-design features from the domain classifier.

`stats.py` reads the matched (untuned) results from `../phase2/results/`, so run Phase 2
first (or the "matched" columns of the tuning table stay empty).
