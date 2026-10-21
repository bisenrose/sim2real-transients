# Phase 2.2 pre-registration: does simulating DES's selection explain the shift?

Written: 19 October 2026, before any 2.2 run.

## Question

DES's spectroscopic sample is flux-limited: it lacks faint objects (Phase 1 showed the faint
tail of absolute magnitude is missing for every supernova class). If that selection explains
part of the sim-to-real gap, then applying a DES-like selection to the PLAsTiCC training set
should make the training set more DES-like and improve transfer.

## Method (fixed)

- Selection variable: apparent peak magnitude m = 27.5 − 2.5 log10(amplitude), the same
  quantity that enters abs_peak_mag.
- **Logistic selection (primary):** estimate the DES/PLAsTiCC density ratio in m with a
  logistic regression on (m, m²), domain as target. Uses unlabelled DES features only. Keep
  each PLAsTiCC training object with probability proportional to that ratio (capped at 1 at
  the 99th percentile).
- **Hard cut (secondary):** keep PLAsTiCC objects brighter than the 99th percentile of DES m.
- **Sample-size controls:** for each, a random PLAsTiCC subset of the same size.
- The PLAsTiCC test set is never selected. Model: XGBoost, same settings as 2.1. 5 split seeds.
  Run under consistent preprocessing (primary) and the 2.1 hybrid rule (secondary).
  LightGBM repeats the primary comparison as a robustness check.

## Predictions (consistent preprocessing, logistic selection, XGBoost)

- Q1: transfer F1 with selection > without selection.
- Q2: transfer F1 with selection > random same-size control.
- Q3: the abs_peak_mag shift between PLAsTiCC training data and DES shrinks (KS statistic
  and median gap both smaller than without selection).
- Q4: among SNIa, SNII, AGN and SLSN-I, SNIa has the largest per-class transfer F1 gain from
  selection. (SNIbc, with only 16 DES objects, is excluded as too noisy.)

"Supported" = mean difference larger than one pooled SD over seeds (Q1, Q2); Q3 and Q4
compare means. Under the hybrid rule, results are reported without a prediction.
