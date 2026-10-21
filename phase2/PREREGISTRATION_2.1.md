# Phase 2.1 pre-registration: hybrid scaling rule

Written: 19 October 2026, before any of the confirmatory runs listed below.
Commit this file to git before running anything in `phase2/configs/`.

## Disclosure: how the rule was found

The rule below was developed on the development set: XGBoost, PLAsTiCC train
-> DES (corrected labels, all spectroscopic types), 5 split seeds. On that set it
scored 0.605 ± 0.010 transfer macro-F1, against 0.486 for consistent preprocessing.
That number is **exploratory** and must be reported as such. Everything in the
"Confirmatory tests" section has not been run at the time of writing.

## The rule (fixed)

Preprocessing statistics (median fill, ±20σ clip, standardisation) are fitted **per survey**:
PLAsTiCC statistics from the PLAsTiCC training set, applied to PLAsTiCC train and test;
DES statistics from the unlabelled DES feature table, applied to DES.

**Exception:** a feature is scaled with the PLAsTiCC training statistics in both surveys
if and only if it is a *source-frame physical quantity*: a property of the astrophysical
source that does not depend on the telescope, filter system, cadence or observer frame.

Applying the criterion to the 27 features:

| Feature | Source-frame physical? | Why |
| --- | --- | --- |
| redshift | Yes | Property of the source |
| abs_peak_mag | Yes | Corrected for distance, so independent of the survey's depth |
| A_*, amplitude, std_flux, extra_sigma_*, log_amplitude, g_r_peak_color | No | Observed fluxes: depend on distance and instrument |
| tau_rise_*, tau_fall_*, gamma_*, beta_*, tau_ratio_*, *_diff_gr, is_fast_transient | No | Observer-frame times (stretched by 1+z) |
| t_0_*, time_span | No | Survey cadence and window |
| g_r_amp_ratio | No | Observer-frame colour; depends on bandpasses and redshift |

So `CALIBRATED_FEATURES = ["redshift", "abs_peak_mag"]`. Fixed; not tuned on DES labels.

## Confirmatory tests (not yet run)

Model families, matched settings, 5 split seeds each:

1. LightGBM, CatBoost, MLP: DES corrected labels.
2. All four (XGBoost, LightGBM, CatBoost, MLP): DES confident labels only (no '?' SNTYPEs).

Each model is run under four preprocessings: `consistent`, `per_survey`, `per_survey`
with abs_peak_mag removed, and `hybrid` (the rule).

**Predictions**, for every model family and both label sets:

- P1: hybrid > consistent.
- P2: hybrid > per_survey.
- P3: hybrid ≥ per_survey with abs_peak_mag removed (keeping the feature on a shared
  scale is at least as good as discarding it).

A prediction counts as supported when the mean over seeds is higher and the
difference exceeds one pooled standard deviation. All outcomes are reported,
including failures.

## XAI diagnostic (label-free, not yet run)

For the PLAsTiCC-trained XGBoost model, compute TreeSHAP values on DES twice: with
consistent and with per-survey preprocessing. For each feature j:

    damage_j = mean over DES objects of |SHAP_j(per_survey) − SHAP_j(consistent)|

This uses no DES labels. It measures how much re-centring changes each feature's
contribution to the predictions.

- P4: abs_peak_mag has the largest damage score of all 27 features.
- P5: redshift ranks in the top 5.

## Later held-out test

The same rule, unchanged, will be applied to ZTF Bright Transient Survey data in Phase 4.
