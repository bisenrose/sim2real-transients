# Phase 1 Audit: Results

Audit date: 19 October 2026 (updated with verified DES SNTYPE codes). All numbers below come from `results/*.json` and the tables in `tables/`, produced by `run_experiment.py` and `make_tables.py` in this folder.

## Summary

Phase 1 found two bugs that change every headline number in the abstract. Neither is a problem with the idea; both are pipeline errors, and finding them now is exactly what this phase is for.

1. **The "SLSN-I" class in PLAsTiCC is actually M-dwarf stellar flares.** In PLAsTiCC, class 65 is M-dwarf flares (981 objects, all at redshift 0) and class 95 is SLSN-I (175 objects). The pipeline used 65. This is why "SLSN-I" scored a perfect F1 = 1.0 in PLAsTiCC (M-dwarfs at z = 0 are trivial to separate) and 0.0 on DES (no real SLSN-I in training).
2. **Each dataset was standardised with its own statistics.** `build_xai_engineered_features(..., is_train=False)` fits a fresh scaler on the DES set (and on the PLAsTiCC test set). This re-centres every DES feature onto PLAsTiCC's scale. For a physically calibrated feature like peak absolute magnitude, that destroys its meaning: a DES SN Ia at M = −19.8 is mapped to wherever the average PLAsTiCC object sits. This, not the Malmquist bias alone, is what made abs_peak_mag look "corrupted".

The original pipeline reproduces exactly in this audit (PLAsTiCC 0.7899, transfer 0.2193, drop-abs_peak_mag 0.3296), so the audit code is faithful to yours. With both bugs fixed:

| Benchmark | Original (abstract) | Corrected (5 split seeds) |
| --- | --- | --- |
| PLAsTiCC in-domain macro-F1 | 0.79 | 0.783 ± 0.008 |
| DES-only 5-fold CV macro-F1 | 0.59 ± 0.10 | 0.572 ± 0.008 |
| PLAsTiCC → DES transfer macro-F1 | 0.22 | 0.486 ± 0.019 |
| Transfer, abs_peak_mag removed | 0.33 | 0.488 ± 0.010 |
| Transfer, redshift removed | 0.19 | 0.516 ± 0.013 |
| Per-domain scaling, abs_peak_mag removed | — | 0.561 ± 0.010 |

**What this means for the paper.** The sim-to-real gap is real but much smaller: transfer reaches 0.49 against a DES-only score of 0.57, not 0.22 against 0.59. With consistent preprocessing, removing abs_peak_mag does not help, and the redshift control no longer shows the asymmetry. The current abstract's central claims cannot be submitted as they stand. There is still a paper here, and you have chosen its framing (see "Chosen framing" at the end).

## 1.1 Number reconciliation

Every conflicting number traced to its source:

| Number(s) | Where it appears | Cause | Correct value now |
| --- | --- | --- | --- |
| ρ = 0.97 vs 0.71 | Pretalx vs PDF | Both exist. 0.968 = PLAsTiCC model's SHAP ranking on PLAsTiCC vs the same model on DES (label-free). 0.711 = PLAsTiCC model on PLAsTiCC vs a DES-trained model on DES (needs DES labels). **The Pretalx abstract is right; the PDF mislabels 0.71 as label-free.** | Must be recomputed: both were computed with the wrong SLSN-I class and per-domain scaling |
| 0.7899 vs 0.7889 | Benchmark table vs robustness table | 0.7899 is the point estimate (exp9); 0.7889 is the **bootstrap mean** from exp14 | Report point estimate with its CI, never the bootstrap mean |
| 0.2193 vs 0.2181 | Same | Same cause | Same rule |
| DES CV 0.5937 vs 0.6220 | exp9 vs exp14 | exp9 averages per-fold F1 with per-fold scaling; exp14 pools out-of-fold predictions and scales DES on the full set before splitting | Pick one estimator (mean of folds) and use it everywhere |
| 0.3296 vs 0.3274 | exp13 vs exp22 CSV | exp13 uses 250 trees; exp22 uses 200. The stored exp22 CSV is also stale: re-running exp22 today gives 0.3262 | Superseded |
| SLSN-I robustness: 0.7889→0.7361, 0.2181→0.2726 | Abstract Section 7 | Mixes bootstrap means (exp14) with a newer run. The stored exp22 CSV instead shows "PLAsTiCC in-domain 0.9452", which is the model scored **on its own training data** (see the comment in exp22) | Superseded: SLSN-I is now the real class |
| N = 6278 / 546 | dataset_manifest.txt | Stale; written before the scipy re-extraction | 6273 train / 1570 test PLAsTiCC; 532 DES fitted, 530 in shared classes after the corrected SNTYPE mapping |

**Rule going forward:** every number in the paper comes from `results/*.json` via `make_tables.py`. The ρ values belong in that pipeline too (add a config for the SHAP comparison in Phase 2).

**Caveat on ρ = 0.97.** Ranking a model's SHAP importances on two different inputs will correlate highly almost by construction, because the same trees are being explained. Present it as weak evidence, not a key result.

## Critical bug 1: SLSN-I class code

PLAsTiCC class codes (Kessler et al. 2019): 6 µLens-single, 15 TDE, 16 eclipsing binary, 42 SNII, 52 SNIax, 53 Mira, 62 SNIbc, 64 kilonova, **65 M-dwarf flare**, 67 SNIa-91bg, 88 AGN, 90 SNIa, 92 RR Lyrae, **95 SLSN-I**. The counts in your raw file confirm it: class 65 has 981 objects, class 95 has 175.

Evidence in your own data: the "SLSN-I" class in PLAsTiCC has median redshift 0.00 and median abs_peak_mag −0.36. Real SLSN-I (class 95) have median z = 1.19 and M = −22.6.

Fixed in this update (see `CHANGES.md` in the project root):

- `src/data_loader.py`, `CLASS_NAMES`: 65 is now `'M-dwarf'`, 95 `'SLSN-I'`, 6 `'uLens-Single'`, 64 `'KN'`.
- Every experiment's `code_map` (8 files in `src/experiments/` plus `run_final_sanity_audit.py`, `exp_bootstrap_rank.py`, `exp22_sensitivity_and_bootstrap.py`): `65: 'SLSN-I'` is now `95: 'SLSN-I'`.
- `des_scipy_extraction.py` and `des_superphot_local.py`, `SNTYPE_TO_PLASTICC`: verified mapping (SLSN-I to 95, SNIIb to SNII, SNIa-pec removed).
- Both DES feature files have their `plasticc_class` column rebuilt from `sntype` with that mapping. No re-fitting was needed. `phase1/` always rebuilds DES labels from `sntype`, so it does not depend on this column.

## Critical bug 2: per-domain standardisation

`feature_engineering.py` fits a new median, clip range and `StandardScaler` on whatever data it is given when `is_train=False`. The docstring explains why: an earlier version of the PLAsTiCC features was in different units. After the scipy re-extraction, both datasets are in the same flux units (FLUXCAL, zero point 27.5), so this is no longer needed.

Effects:

- It is a **transductive alignment step** using the whole DES set. If kept, it must be described as a domain-adaptation method, not as preprocessing.
- It re-centres physically calibrated features. Peak absolute magnitude has the same units in both surveys; shifting DES's distribution onto PLAsTiCC's makes bright DES SNe Ia look like average PLAsTiCC objects. In the original run, 250 of 392 DES SNe Ia were predicted as SNIbc; with consistent preprocessing that falls to about 14.
- It also rescales the PLAsTiCC **test** set with its own statistics, which lowers in-domain F1 (0.724 vs 0.783 with consistent preprocessing).

Tree models are unaffected by any per-feature monotonic transform applied identically to train and test, so with consistent preprocessing the scaler choice no longer matters.

**A finding worth keeping:** with correct classes, per-domain scaling plus removing abs_peak_mag gives the best label-free transfer (0.561 ± 0.010), on par with the DES-only score (0.572). So re-centring is a useful cheap alignment for most features, but harmful for a physically calibrated one, and SHAP flags exactly that feature. See "Options for the paper".

## 1.2 Feature extraction audit

| Check | Status | Finding |
| --- | --- | --- |
| Filters | OK | Both pipelines fit only g and r (LSST passbands 1, 2; DES g, r). The `_ZTF_g/_ZTF_r` names are a naming convention only; rename to `_g/_r` for the paper. LSST and DECam g, r bandpasses differ slightly; mention it. |
| Redshift source | OK | PLAsTiCC uses `hostgal_specz` (spectroscopic for 99.9% of the five classes, photo-z fallback otherwise). DES uses `REDSHIFT_FINAL` (spectroscopic), photo-z fallback. One DES object has no redshift and is median-imputed. |
| Absolute magnitude | Same code both sides, but crude | M = 27.5 − 2.5 log10(amplitude) − μ(z), Planck18 cosmology. **amplitude = max − min of all g and r flux points combined**, not the peak in one band. No K-correction, no Milky Way extinction correction. State all of this in the paper; consider using the fitted peak flux in r instead of max − min. |
| Cadence / survey design | **Problem** | PLAsTiCC training light curves span ~850 days (whole survey); DES light curves span ~180 days (one season). `time_span` and `t_0` (measured from the first observation) encode survey design, not physics. In DES, time_span also varies by class (AGN median 322 d, SNe ~170 d), which can inflate the DES-only ceiling. Remove both, or redefine t_0 relative to peak and trim PLAsTiCC light curves to a window around peak. Removing them raises transfer to 0.501 ± 0.017. |
| Fitting procedure | **Mismatch** | DES fits normalised flux with `maxfev=2000`; PLAsTiCC fits raw flux with `maxfev=5000`. Same model and bounds, but different optimiser behaviour can shift fitted parameters and which objects fail. Use one shared fitting function for both. |
| Feature naming | Fix | `g_r_peak_color` is a flux **difference** (A_g − A_r), which scales with distance; it is not a colour. A true colour is −2.5 log10(A_g / A_r). `log_amplitude` is a monotonic copy of `amplitude` and adds nothing for tree models. |
| Non-detections | Consistent | Neither pipeline filters non-detections (DES `photflag_detected` is defined but unused). Consistent, but say so. |

### Malmquist signature (descriptive, still holds)

Peak absolute magnitude percentiles, corrected classes:

| Class | PLAsTiCC p10 / p50 / p90 | DES p10 / p50 / p90 |
| --- | --- | --- |
| SNIa | −19.92 / −19.19 / −16.62 | −20.43 / −19.77 / −19.03 |
| SNII | −19.52 / −17.34 / −15.23 | −18.95 / −17.61 / −16.55 |
| SNIbc | −18.69 / −17.12 / −14.89 | −18.84 / −18.16 / −17.48 |
| AGN | −23.72 / −21.38 / −18.47 | −23.44 / −21.32 / −19.24 |
| SLSN-I | −23.98 / −22.64 / −20.99 | −23.12 / −22.25 / −20.52 |

The faint tail (p90) is cut off in DES for every supernova class, most strongly for SNIa (−16.6 → −19.0) and SNIbc (−14.9 → −17.5). That is the flux-limited selection you described. It just isn't what drove the 0.22 collapse.

### Class mapping (verified)

DES SNTYPE meanings are from the DES-SN5YR data release documentation (0_DATA, "SNTYPE nomenclature"). A trailing "?" means the spectroscopic classification is uncertain.

| Class | PLAsTiCC code | DES SNTYPE codes | DES count |
| --- | --- | --- | --- |
| SNIa | 90 | 1 SNIa (336), 101 SNIa? (55) | 391 |
| SNII | 42 | 29 SNII (44), 129 SNII? (16), 23 SNIIb (4) | 64 |
| SNIbc | 62 | 39 SNIbc (7), 139 SNIbc? (4), 33 SNIc (4), 32 SNIb (1) | 16 |
| AGN | 88 | 80 AGN (40), 180 AGN? (1) | 41 |
| SLSN-I | 95 | 41 SLSN-I (15), 141 SLSN-I? (3) | 18 |
| Excluded | — | 0 no spectrum, 4 SNIa-pec (1), 5 SNI (Ia vs Ib unclear), 122 SNIIn?, 66 SLSN-II?, 82 M-Star, 81 TDE (1) | — |

Two corrections to the original mapping:

- **SNIIb (23) moved from SNIbc to SNII.** DES lists it as a type II subtype. This is a judgement call (SNe IIb are transitional), so state it in the paper.
- **SNIa-pec (4) removed.** PLAsTiCC's SNIa class is normal SNe Ia; one peculiar object should not be counted.

A sensitivity check keeps only confident types (drops 101, 129, 139, 141, 180; 451 objects). Transfer rises from 0.486 ± 0.019 to 0.523 ± 0.025, and the best label-free setting (per-domain scaling, abs_peak_mag removed) from 0.561 to 0.574 ± 0.016. Uncertain labels cost a little, but do not change any conclusion. Report both.

## 1.3 Sample construction

PLAsTiCC:

1. Start from the PLAsTiCC **training set** (7,848 objects, the spectroscopically confirmed sample; not the 3.5 million test set). Note this sample is itself biased toward bright, low-redshift objects.
2. Keep g and r only; require ≥ 10 observations total and ≥ 4 per band.
3. Fit the Villar model per band with `scipy.optimize.curve_fit`; 7,843 objects succeed (5 fail).
4. Stratified split by class, test fraction 1570/7843, on all 14 classes.
5. Keep the five shared classes: 3,625 train / 908 test (corrected classes). The old 4,270 / 1,069 included 981 M-dwarfs.

DES-SN 5-year:

1. Start from the HEAD file; keep spectroscopically confirmed, unambiguous types.
2. Keep g and r; ≥ 10 observations total, ≥ 4 per band.
3. Fit the same model: 532 objects. Map SNTYPE to class (table above) and keep the five shared classes: 530 (one TDE and one SNIa-pec dropped). Restricting to confident types (no '?') leaves 451.

**Leakage check:** passed. The raw PLAsTiCC file keeps `object_id`, and no object appears in both train and test for any split seed (asserted in `run_experiment.py`). Keep `object_id` in the split files from now on.

**Split robustness:** the corrected baseline was re-run with 5 different split seeds (0–4). Transfer macro-F1 ranges 0.456–0.506; mean 0.486 ± 0.019. Results hold across splits.

## 1.4 Feature table

Per-band parameters come from the Villar et al. (2019) model fitted separately to g and r (12 features + 2 scatter terms):

| Feature | How it is computed | Group |
| --- | --- | --- |
| redshift | Spectroscopic redshift (photo-z fallback) | Redshift |
| A_g, A_r | Fitted flux scale of the light curve | Brightness |
| beta_g, beta_r | Fitted plateau slope | Shape |
| gamma_g, gamma_r | Fitted plateau duration (days) | Timescale |
| t_0_g, t_0_r | Fitted reference time, days since first observation | Survey artefact |
| tau_rise_g, tau_rise_r | Fitted rise timescale (days) | Timescale |
| tau_fall_g, tau_fall_r | Fitted decline timescale (days) | Timescale |
| extra_sigma_g, extra_sigma_r | Std of fit residuals (flux units) | Shape / noise |
| amplitude | max − min of all g and r flux points | Brightness |
| std_flux | Std of all g and r flux points | Brightness |
| time_span | Last − first observation (days) | Survey artefact |
| abs_peak_mag | 27.5 − 2.5 log10(amplitude) − μ(z), Planck18 | Brightness (derived) |
| tau_ratio_g, tau_ratio_r | tau_rise / tau_fall | Timescale (derived) |
| is_fast_transient | 1 if tau_rise_g < 5 d and tau_fall_g < 10 d | Timescale (derived) |
| log_amplitude | log(1 + amplitude) | Brightness (derived, redundant) |
| g_r_peak_color | A_g − A_r (flux difference, not a colour) | Colour proxy (derived) |
| g_r_amp_ratio | A_g / A_r | Colour (derived) |
| beta_diff_gr | beta_g − beta_r | Shape (derived) |
| tau_fall_diff_gr | tau_fall_g − tau_fall_r | Timescale (derived) |

That is 27 features.

**How abs_peak_mag, amplitude and redshift relate.** abs_peak_mag is computed from two other features in the set: apparent magnitude comes from amplitude, and distance comes from redshift. So when abs_peak_mag is removed, a model can partly rebuild it from amplitude plus redshift; and when redshift is removed, abs_peak_mag still carries distance information. Ablations on these three features are not independent, and the paper should say so.

## Remaining transfer failure (input for Phase 2)

With corrected classes and consistent preprocessing, the main error is **DES SNe Ia predicted as AGN** (107 of 391 on average across seeds), which drags down AGN precision (transfer AGN F1 0.37 vs 0.96 in-domain). DES SLSN-I are also mostly predicted as AGN. SNIbc is poor everywhere. With only 16 DES SNIbc, the DES-only model scores 0.02 on it, lower than the transferred model (0.20), so for this class the DES-only score is not a ceiling. Say so in the paper rather than calling 0.57 an upper bound. Understanding the SNIa → AGN confusion is the most promising Phase 2 target.

## Chosen framing

Option 3: per-domain re-centring plus XAI-guided removal of the physically calibrated feature it breaks is the **method**; the corrected three-way benchmark is the **context**.

What that means for Phase 2:

- The baseline everything is compared against is consistent preprocessing (0.486).
- The method is per-domain scaling plus a feature-removal rule. Today's result (0.561) used a feature chosen by hand; Phase 2.1 must replace that with a rule fixed in advance that uses no DES labels.
- The rule's target changes. It is no longer "which feature shifts most"; it is "which features does re-centring damage". Features in physical units that should agree across surveys (absolute magnitude, redshift, timescales in rest-frame days) are the candidates; SHAP importance combined with how much re-centring moves each feature is a natural score.
- The Malmquist signature becomes supporting evidence for why absolute magnitude in particular is damaged, not the headline.

## Files in this folder

- `common.py`: data loading, the corrected class codes and DES SNTYPE mapping, feature derivation, both preprocessing modes
- `run_experiment.py`: runs one JSON config (or `--all`), writes `results/<name>.json`
- `configs/`: the 12 Phase 1 configurations, including an exact reproduction of the original pipeline
- `make_tables.py`: builds every table in `tables/` (Markdown and LaTeX) from the results files
- `results/`, `tables/`: outputs from this audit
