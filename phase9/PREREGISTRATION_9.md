# Phase 9: removing the survey-artefact features (reviewer-requested ablation)

Written before any Phase 9 run. Requested by two independent reviews of the submitted paper; it is
reported in the paper as an additional analysis whose predictions were recorded before it was run.

## What changes
The three survey-artefact features are removed from the classifier entirely, for every target and
preprocessing mode: `time_span`, `t_0_ZTF_g`, `t_0_ZTF_r`. Everything else is as in the paper:
PLAsTiCC training set, five split seeds (0-4), matched model settings, macro-F1 over classes with
at least ten target objects. Targets: DES (all types), ZTF BTS, YSE DR1. Models: XGBoost, LightGBM,
CatBoost, MLP. Modes: consistent, per-survey, hybrid rule, hybrid + brightness.

## Predictions ("supported" = mean difference larger than one pooled SD across seeds)
- **A1 (DES):** the hybrid rule still beats consistent preprocessing, for all four models.
- **A2 (ZTF):** the hybrid rule is still below per-survey rescaling, for the three tree models.
- **A3 (YSE):** the hybrid rule is still below per-survey rescaling, for the three tree models.
- **A4 (YSE):** consistent preprocessing is still the best of the four modes, for the three tree models.
- **A5 (ZTF):** hybrid + brightness still beats the hybrid rule, for the three tree models.
- **A6 (label-free):** without the artefact features, a domain classifier still separates PLAsTiCC
  from each target with AUC >= 0.95 (the near-absence of overlap is not caused by the artefacts alone).

If A1-A5 hold, the paper's conclusions do not depend on the survey-artefact features.
