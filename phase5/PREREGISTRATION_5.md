# Phase 5 pre-registration: reviewer-requested checks

Written: 19 October 2026, before any Phase 5 run. Commit to git before running `run_phase5.sh`.
XGBoost and LightGBM use the matched settings of Phases 2-4; the controlled test (C) also uses
CatBoost and the MLP. 5 PLAsTiCC split seeds. "Supported" = mean difference larger than one pooled
SD across seeds (an "equal" prediction = difference no larger than one pooled SD).

Everything here is added after the ZTF result. Only test C uses data on which the support
hypothesis has not been looked at; it is the one confirmatory test in this phase.

## C. Controlled test of the support condition (confirmatory)

A pseudo-target is cut out of PLAsTiCC itself, so the only shift is the one we create:
- **low-z**: every five-class PLAsTiCC object with z < 0.12 (524 objects: 315 SNII, 123 SNIbc, 85 SNIa, 1 SLSN-I).
  Training and in-domain test keep only z >= 0.12, so the target's redshifts lie entirely outside
  the training range, as for ZTF.
- **random** (control): a random set of the same size (seed 0), with full redshift overlap.
Macro-F1 over pseudo-target classes with at least 10 objects. Modes: consistent, per-survey,
hybrid rule, hybrid with abs_peak_mag only.

- C1: low-z: hybrid with abs_peak_mag only > hybrid rule, for at least 3 of 4 models.
- C2: random: hybrid with abs_peak_mag only = hybrid rule, for at least 3 of 4 models.
- C3: low-z: hybrid rule < per-survey, for at least 3 of 4 models.

## W. Magnitude density ratio as sample weights (reviewer request)

The Phase 2.2 apparent-magnitude density ratio, used as training weights (capped at the 99th
percentile, mean 1) instead of sub-sampling. Consistent preprocessing.

- W1: DES: weighted > unweighted, for XGBoost and LightGBM.
- W2: DES: in-domain PLAsTiCC macro-F1 with weights >= 0.76 (selection gave 0.727).
- W3: ZTF: weighted is not better than unweighted (difference <= 1 pooled SD, or worse).

## S. StratLearn (reviewer request)

Propensity scores from logistic regression on the scaled features; 5 strata at the quantiles of the
combined source + target scores; one learner per stratum; strata with < 30 source objects or a single
class fall back to the global model. Consistent preprocessing.

- S1: DES, all features: StratLearn is not better than consistent (difference <= 1 pooled SD).
- S2: DES, survey-design features (t_0_g, t_0_r, time_span) hidden from the propensity model:
  StratLearn > consistent, for XGBoost and LightGBM.
- ZTF: reported without a prediction.

## K. Robustness of peak absolute magnitude (reviewer request)

Two alternative definitions: `bandwidth` (adds 2.5 log10(1+z), the bandwidth part of the
K-correction) and `rband` (apparent magnitude from the fitted r-band peak A_r).

- K1: DES: hybrid > consistent, for both variants and both models.
- K2: ZTF: hybrid < per-survey, for both variants and both models (the ZTF failure is not caused by
  the M_peak definition).

## N. Photometric-redshift robustness (reviewer request)

Target redshifts replaced by z + N(0, sigma (1+z)), sigma = 0.02 and 0.05, one fixed realisation.
XGBoost.

- N1: DES, sigma = 0.02: hybrid > consistent.
- N2: DES, sigma = 0.05: hybrid > consistent.
- N3: the DES hybrid minus per-survey difference is smaller at sigma = 0.05 than at 0.02.

## G. Overlap-gated rule (EXPLORATORY, no predictions)

A source-frame feature keeps the shared scale only if at least a fraction tau of the target lies
inside the training set's 5th-95th percentile range; tau in {0.3, 0.5, 0.7, 0.9}. Run on DES, ZTF
and both pseudo-targets with XGBoost and LightGBM. Reported as a sensitivity analysis of an idea
formulated after the ZTF result; it is not a validated method.

## Quick checks from saved results (descriptive)

ZTF confusion matrices; label-free support-overlap table; per-class bootstrap intervals on DES.
