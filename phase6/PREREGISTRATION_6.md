# Phase 6 pre-registration: probing the ZTF failure (reviewer round 2)

Written: 20 October 2026, before any Phase 6 run. Commit to git before running `run_phase6.sh`.
Matched model settings as before; 5 PLAsTiCC split seeds. "Supported" = mean difference larger than
one pooled SD across seeds; ">= / =" predictions use the same tolerance. All of this is designed after
the ZTF result; D2 is the confirmatory test, because it uses controlled data.

## Candidate mechanism being tested

Per-survey rescaling maps ZTF's bright, nearby objects onto PLAsTiCC's typical flux values while the
hybrid rule keeps their true low redshift (and their absolute magnitude). If brightness is measured
differently in ZTF (a calibration-like offset), the shared-scale quantities are no longer comparable,
and SNe Ia look like fainter classes. Alternatives: class-prior (label) shift; an effect of very low
redshift alone (already rejected for a pure population shift in Phase 5).

## D1. Completing the decomposition on ZTF (and DES)

New hybrid variants: (a) redshift only on the shared scale; (b) redshift, abs_peak_mag and the
brightness features (A_g, A_r, amplitude, std_flux, log_amplitude) on the shared scale.
XGBoost, LightGBM, CatBoost, MLP; ZTF and DES.

- F1: ZTF: redshift-only < abs-mag-only (Phase 4 exploratory variant), for >= 3 of 4 models.
- F2: ZTF: variant (b) > hybrid rule, for >= 3 of 4 models.
- DES: reported without a prediction.

## D2. Two-factor controlled test (confirmatory)

Pseudo-targets from PLAsTiCC with ZTF-like class priors:
- **Ia-rich low-z**: all five-class objects with z < 0.2 that are SNIa (348), plus a random set of
  SNII and SNIbc (no AGN or SLSN-I, as in ZTF) so that SNe Ia are 76% (seed 0): 458 objects
  (348 SNIa, 83 SNII, 27 SNIbc). Training and in-domain test keep z >= 0.2.
- **Ia-rich random** (control): the same class counts drawn from all redshifts (seed 0).

Instrument shifts, applied to the pseudo-target's raw features only (fixed now, derived label-free
from medians of the five-class PLAsTiCC and ZTF feature tables):
- **none**
- **I1, cadence and noise**: time_span x 0.068, t_0_g x 0.015, t_0_r x 0.014 (ZTF/PLAsTiCC median
  ratios); extra_sigma_g x 1.8, extra_sigma_r x 1.3 (ZTF/PLAsTiCC residual-scatter ratio divided by
  the amplitude ratio 45.1, i.e. noise relative to brightness).
- **I2, flux calibration**: every flux feature (A_g, A_r, amplitude, std_flux, extra_sigma_g/r)
  multiplied by 10^(-0.4 x 0.3), a 0.3 mag calibration offset, so abs_peak_mag shifts by +0.3 mag.

Modes: consistent, per-survey, hybrid rule, hybrid abs-mag-only. XGBoost and LightGBM.

- T1: control + I1: per-survey > consistent and hybrid > consistent (both models).
- T2: control + I2: hybrid < per-survey (both models).
- T3: low-z + I2: hybrid < per-survey (both models). (Reproduces the ZTF pattern.)
- T4: low-z + I1: hybrid > per-survey (both models).
- T5: low-z, no instrument shift: hybrid > per-survey (both models).

## D3. Label-shift correction (from saved probabilities)

EM prior adjustment (Saerens et al. 2002) of the saved target probabilities, with source priors from
each seed's PLAsTiCC training set. No labels used. ZTF (four models x four preprocessings) and DES.

- L1: ZTF: after correction, the hybrid rule is still below per-survey for >= 3 of 4 models.

## D4. Per-class recall by redshift on ZTF (descriptive, saved predictions)

Bins z < 0.03, 0.03-0.06, 0.06-0.10, >= 0.10. No prediction.

## D5. Target-sample size for the scaling statistics (deployability)

Per-survey and hybrid statistics fitted on n = 25, 50, 100, 200 random unlabelled target objects
(re-drawn per seed), applied to the whole target. XGBoost, DES and ZTF.

- SS1: DES hybrid with n = 100 is within one pooled SD of the full-sample result.
- SS2: DES hybrid with n = 25 is lower than the full-sample result by more than one pooled SD.

## Amendment (20 October 2026, before the laptop run; recorded for transparency)

While testing the code, I ran the D3 script on the existing saved predictions (it needs no training,
so this is the real D3 data) and a one-seed smoke test of a few D1, D2 and D5 configs. No prediction
above was changed. Two things were added because of what the D3 test showed:

- The EM prior estimates are unstable on these targets (e.g. CatBoost hybrid on ZTF estimates an SNIa
  prior of 0.00 against a true 0.76), because the transferred probabilities are badly calibrated. L1
  is kept as written, but it should be read with that caveat.
- An EXPLORATORY diagnostic that corrects with the TRUE target class proportions (uses target labels
  for the prior only) asks whether known priors would close the hybrid rule's deficit at all.
