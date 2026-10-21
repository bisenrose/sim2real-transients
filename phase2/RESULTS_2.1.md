# Phase 2.1 results: the hybrid scaling rule

Run on 19 October 2026, after writing `PREREGISTRATION_2.1.md`. Every number comes from
`results/*.json` via `make_tables.py`; tables are in `tables/`.

## Headline

The pre-registered rule (re-centre each survey's features with its own statistics, except
source-frame physical quantities: redshift and absolute magnitude) improved transfer for
**all four model families and both label sets**. All three pre-registered predictions
(P1–P3) held in all 7 confirmatory settings. Both predictions about the SHAP diagnostic
(P4, P5) failed, and the failure is informative.

| Model | Consistent | Hybrid (rule) | DES-only CV |
| --- | --- | --- | --- |
| XGBoost (development set) | 0.486 ± 0.019 | 0.605 ± 0.010 | 0.572 ± 0.008 |
| LightGBM | 0.502 ± 0.026 | 0.600 ± 0.012 | 0.589 ± 0.017 |
| CatBoost | 0.498 ± 0.028 | 0.589 ± 0.012 | 0.565 ± 0.009 |
| MLP | 0.406 ± 0.022 | 0.575 ± 0.030 | 0.575 ± 0.021 |

DES all spectroscopic types; transfer macro-F1, mean ± SD over 5 split seeds. Full tables,
including the confident-labels-only set and per-survey variants: `tables/transfer_*.md`.

For every model, the label-free hybrid rule matches or beats a model trained and tested on
real DES labels with cross-validation.

## Pre-registered predictions

| | Prediction | Outcome |
| --- | --- | --- |
| P1 | Hybrid > consistent | Supported in 7/7 confirmatory settings |
| P2 | Hybrid > per-survey | Supported in 7/7 |
| P3 | Hybrid ≥ per-survey with abs mag removed | Supported in 7/7 (hybrid higher in every case) |
| P4 | abs_peak_mag has the largest SHAP damage | **Not supported**: rank 2 in all 5 seeds; `extra_sigma_g` is rank 1 |
| P5 | Redshift in the top 5 for SHAP damage | **Not supported**: rank 6 (5 or 6 across seeds) |

## What the P4/P5 failure tells us

The damage score measures how much re-centring *changes* a feature's contribution, not
whether the change helps. An exploratory follow-up (uses DES labels; not part of the rule)
separates the two:

| Variant (XGBoost, all types) | Transfer macro-F1 |
| --- | --- |
| Hybrid (rule: redshift + abs mag on shared scale) | 0.605 ± 0.010 |
| Hybrid, abs mag only on shared scale | 0.609 ± 0.009 |
| Hybrid, plus extra_sigma_g/r on shared scale | 0.371 ± 0.020 |

So re-centring `extra_sigma` (residual scatter in flux units, which depends on each survey's
noise) is strongly *beneficial*, while re-centring absolute magnitude is harmful. Both rank
high on SHAP damage. **SHAP shows where re-centring acts; physics decides whether it should.**
That is a cleaner and more honest version of the paper's XAI claim than "SHAP finds the bad
feature", and it explains why a physics-based criterion, not a SHAP threshold, is the rule.

Redshift barely matters either way (0.605 vs 0.609). Keep the pre-registered rule as it is;
report the abs-mag-only variant as a sensitivity check.

## Caveats to state in the paper

- The rule was found on XGBoost with all DES types (marked "development" in the verdicts
  table). The confirmatory evidence is the other three models and the confident-only set.
  The decisive held-out test is ZTF in Phase 4.
- The MLP here used scikit-learn's MLP because PyTorch was not available in my environment.
  Re-run the `mlp_*` configs on your Mac (des_env has PyTorch) to get the exp12 architecture.
- The DES statistics are fitted on the whole unlabelled DES sample (transductive). This is
  realistic for a survey but must be stated.
- LightGBM now scores 0.783 in-domain, not 0.53 as in exp12. The difference is row and column
  subsampling (0.8), now matched across all tree models. The old "LightGBM asymmetry" was a
  settings artefact; drop that paragraph from the paper.
- The neural model in exp12 was called "SuperNNova" but is a plain MLP on features. Call it MLP.

## Files

- `PREREGISTRATION_2.1.md`: the rule and predictions, written before the runs
- `configs/`: 32 confirmatory/development configs + 2 exploratory ones (`explore_*`)
- `diagnose_damage.py`: label-free SHAP damage diagnostic (P4, P5)
- `make_tables.py`: tables and `results/verdicts.json`
