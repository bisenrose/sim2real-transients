# Phase 9: survey-artefact ablation (reviewer-requested)

Removes `time_span`, `t_0_ZTF_g` and `t_0_ZTF_r` from the classifier for every target, model and
preprocessing mode, and checks whether the paper's conclusions still hold (predictions A1-A6 in
`PREREGISTRATION_9.md`, written before running).

Run from the `xai` folder: `bash run_phase9.sh` (about 1 hour; ZTF is the slowest).
Outputs: `tables/ablation.md` and `.tex`, `results/verdicts_9.json`, `results/domain_auc.json`.
