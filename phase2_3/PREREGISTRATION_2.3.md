# Phase 2.3 pre-registration: which shift measure predicts harmful features?

Written: 19 October 2026, before any 2.3 run.

## Measures (label-free, computed per feature on consistent-scaled data)

- KS statistic and Wasserstein distance between PLAsTiCC training data and DES (all objects).
- shift_j: one-feature domain classifier (gradient boosting, 5-fold CV), (AUC − 0.5) × 2.
- imp_j: mean |SHAP| of the PLAsTiCC-trained XGBoost on PLAsTiCC training data, normalised.
- risk_j = imp_j × shift_j.
- damage_j: SHAP damage from re-centring (from 2.1, `phase2/results/diagnostic_damage.csv`).

## Diagnostic (uses DES labels; not part of any method)

Remove each of the 27 features in turn and record the change in transfer macro-F1 (XGBoost,
5 split seeds), under consistent preprocessing and under the hybrid rule.

## Prediction

- R1: under consistent preprocessing, the Spearman correlation between risk_j and the F1
  gain from removing feature j is higher than the correlation between shift_j alone and
  that gain.

Everything else (per-class KS table, hybrid-mode removals, correlations with damage_j) is
reported descriptively, without a prediction.
