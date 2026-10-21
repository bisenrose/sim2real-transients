# Phase 8 pre-registration: per-feature shift comparison and a stronger controlled test

Written: 20 October 2026, before any Phase 8 run. Commit to git before running `run_phase8.sh`.
5 PLAsTiCC split seeds; matched XGBoost and LightGBM settings; "supported" = mean difference larger
than one pooled SD across seeds.

Before writing this file I computed, label-free, how bright the candidate pseudo-targets would be
(no model was run): moving the Ia-rich control pseudo-target nearer by a factor 3 gives a median
flux amplitude 11.7x PLAsTiCC's (median z 0.10, redshift overlap 0.65); by a factor 6, 50.9x
(median z 0.051, overlap 0.17), close to ZTF (45.1x, 0.059, 0.17). The Phase 5 and 6 low-z
pseudo-targets were only about 1.5x brighter than PLAsTiCC, which may be why they never reproduced
the held-out failure.

## E1. Per-feature shift: DES vs ZTF vs YSE (label-free, descriptive with three predictions)

For each target, KS statistic of every feature against the PLAsTiCC training set (seed-0 split,
five classes), grouped as: source-frame (redshift, abs_peak_mag), brightness (A_g, A_r, amplitude,
std_flux, log_amplitude), noise (extra_sigma_g, extra_sigma_r), cadence (time_span, t_0_g, t_0_r),
other (shape, timescale, colour). Also the domain-classifier AUC (gradient boosting, 5-fold CV)
using each group alone, and the KS of relative noise (extra_sigma / amplitude, g and r).

- E1a: mean KS over the noise and cadence features is lower for YSE than for DES.
- E1b: domain-classifier AUC on the noise and cadence features is lower for YSE than for DES.
- E1c: mean KS of relative noise (g, r) is lower for YSE than for DES.

## E2. Stronger controlled test (confirmatory)

Pseudo-targets: the Phase 6 Ia-rich control (458 objects with ZTF-like class proportions, drawn
from all redshifts), unchanged or **moved nearer** by a factor 3 or 6: each object placed at
z' = z / factor, flux features scaled by (d_L(z) / d_L(z'))^2, observer-frame timescales by
(1 + z') / (1 + z) (so abs_peak_mag is unchanged, as for a real nearer object), cadence features
unchanged. Instrument shifts applied after that: none; 0.5 mag and 1.0 mag flux-calibration
offsets; a strong season shift (time_span x 0.068, t_0 x 0.015 / 0.014, residual scatter x 3).
Modes: consistent, per-survey, hybrid rule, hybrid + brightness. XGBoost and LightGBM.

"ZTF pattern" = hybrid < per-survey AND hybrid + brightness > hybrid.

- R1: moved nearer x6, no instrument shift: ZTF pattern, for both models.
- R2: moved nearer x3, no instrument shift: ZTF pattern, for both models.
- R3: control, 1.0 mag calibration offset: hybrid < per-survey, for both models.
- R4: control, strong season shift: per-survey > consistent and hybrid > consistent, for both models.
- R5: moved nearer x6 with the 1.0 mag offset: ZTF pattern, for both models.

All other cells are reported descriptively.

## Amendment (20 October 2026, before the laptop run; recorded for transparency)

While testing the code I ran E1 with a placeholder YSE table, so the DES and ZTF columns of E1 were
computed on the real data (they need no YSE data), and I ran one split seed of five E2 configs. No
prediction above was changed. One observation that affects how E1b should be read: the domain
classifier separates PLAsTiCC from DES almost perfectly on the noise and cadence features alone
(AUC about 1.00), so E1b can only be met if YSE is clearly separable less well; a tie counts as
"not supported".
