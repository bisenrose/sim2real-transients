# Phase 7 pre-registration: YSE DR1 as a third, held-out survey

Written: 20 October 2026. At this point only the folder listing and the header of one light-curve
file (2020nxt) have been seen; no YSE features have been computed and no model has been run on YSE.
Commit this file to git before running `run_phase7.sh`.

## Why YSE

YSE DR1 (Aleo et al. 2023, ApJS 266, 9; Zenodo record 7317476, file `yse_dr1_zenodo.tar.gz`, no
S/N cut) is a third real survey whose objects are, on average, brighter and nearer than DES but
fainter and farther than ZTF BTS. It tests (i) the hybrid rule, and (ii) the variant found in
Phase 6 on ZTF (redshift, abs_peak_mag and calibrated brightness features all on the shared scale),
which has not been tested on any data other than the data that suggested it.

## Data (fixed)

- Objects: every `*.snana.dat` file whose `SPEC_CLASS` maps to one of the shared classes with the
  same TNS-type mapping as ZTF (`BTS_TYPE_TO_PLASTICC` in `phase1/common.py`; e.g. SN Ibn and other
  peculiar types are excluded). Photometric-only objects are not used.
- Redshift: `REDSHIFT_FINAL`, used only if its label marks it as spectroscopic (contains "SPEC" or
  "HOST", not "PHOT"); objects without a positive spectroscopic redshift are dropped.
- Photometry: Pan-STARRS1 `g` and `r` only (`FLT` = g, r); ZTF points and PS1 i, z are not used, so
  YSE measurements are independent of the ZTF photometry. FLUXCAL on zero point 27.5.
- Window: 100 days before to 300 days after the brightest g/r point (as for ZTF).
- Fit and cuts: the DES fitting code (via `phase4/ztf_extract.py`), >= 10 points in total, >= 4 per band.
- Overlap with ZTF BTS: objects whose IAU name is in `phase4/bts_catalogue.csv` are flagged
  `in_bts`. Primary analysis: all YSE objects (the photometry is independent of ZTF's). Robustness:
  the same comparisons excluding `in_bts` objects (`yse_no_bts`).
- Macro-F1 over classes with at least 10 objects in the evaluated set (YSE has no AGN; SLSN-I counted
  only if >= 10 objects).

## Methods (unchanged from earlier phases)

Consistent, per-survey, per-survey without abs_peak_mag, hybrid rule (redshift + abs_peak_mag shared),
abs_peak_mag only shared, redshift only shared, and hybrid + brightness shared (redshift, abs_peak_mag,
A_g, A_r, amplitude, std_flux, log_amplitude). XGBoost, LightGBM, CatBoost, MLP; matched settings;
5 PLAsTiCC split seeds; YSE-only 5-fold CV as reference.

## Predictions

"Supported": mean difference larger than one pooled SD across seeds; ">=" means not worse by more
than one pooled SD. Tree models = XGBoost, LightGBM, CatBoost.

- Y1: consistent transfer < YSE-only CV, for all 4 models.
- Y2: hybrid + brightness > hybrid rule, for >= 2 of 3 tree models.
- Y3: hybrid + brightness >= per-survey, for all 3 tree models.
- Within-YSE split (label-free, at the median YSE redshift; evaluated on saved predictions):
  - Y4: the gain of hybrid + brightness over the hybrid rule is larger in the near half than in the
    far half, for >= 2 of 3 tree models (difference of seed-mean gains).
  - Y5: near half: hybrid rule < per-survey, for >= 2 of 3 tree models (ZTF-like).
  - Y6: far half: hybrid rule >= per-survey, for >= 2 of 3 tree models (DES-like).
- MLP: no prediction for the brightness-shared variant (on ZTF it failed through extrapolation).

Everything else (brightness and redshift offsets relative to PLAsTiCC, magnitude check against YSE's
own PEAK_ABS_MAG, the no-BTS robustness run) is reported descriptively.
