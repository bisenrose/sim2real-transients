# Phase 3 pre-registration: rigour checks

Written: 19 October 2026, before any Phase 3 run.

## 3.1 Fair tuning

Each model family gets the same random-search budget (default 20 settings here, 50 on the
laptop), chosen by 3-fold CV **on the PLAsTiCC training set only**, separately for each of
the 5 split seeds (nested). DES is never used for tuning. The tuned models are then run
under consistent preprocessing and the hybrid rule.

- T1: for every model family, hybrid > consistent after tuning (one pooled SD).
- T2: tuning changes in-domain PLAsTiCC F1 by less than 0.03 for the tree models (they were
  already reasonably set up after Phase 2's matched subsampling).

## 3.2 Standard domain-adaptation baselines (XGBoost and LightGBM)

- Importance reweighting: a domain classifier (PLAsTiCC train vs unlabelled DES, out-of-fold
  probabilities) gives each training object weight p/(1−p), clipped at the 99th percentile
  and normalised to mean 1.
- CORAL: PLAsTiCC features are linearly transformed so their covariance matches DES's.
- Each applied on top of consistent preprocessing and on top of the hybrid rule.

Predictions:

- B1: hybrid > importance reweighting (on consistent preprocessing).
- B2: hybrid > CORAL (on consistent preprocessing).
- No prediction for combinations (hybrid + reweighting, hybrid + CORAL); reported as found.

## 3.3 Statistics and calibration

- Paired bootstrap (same DES objects resampled for both methods, 1000 draws, per seed) for
  hybrid vs consistent and hybrid vs each baseline. Report the 95% interval of the difference.
- Multi-class log-loss, Brier score and expected calibration error (ECE, 10 bins) on DES.
- C1: the hybrid rule also improves log-loss over consistent preprocessing (XGBoost).
- SLSN-I policy, fixed now: SLSN-I stays in every headline number; per-class tables show it
  separately, and a 4-class macro-F1 without SLSN-I is reported as a sensitivity check.
