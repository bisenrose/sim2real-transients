# Phase 4 pre-registration: ZTF held-out test and honest label acquisition

Written: 19 October 2026, **before any ZTF data was downloaded** and before any 4.2 run.
Commit this file to git before running `run_phase4.sh`.

## 4.1 ZTF Bright Transient Survey (held-out test of the hybrid rule)

### Data (fixed)

- Sample: the BTS Sample Explorer table, exported as CSV, all classified transients.
- Light curves: ALeRCE ZTF forced photometry (difference-image fluxes, including epochs with no
  detection, as for DES), g and r only; alert detections are used only if an object has no
  forced photometry. The source used is recorded per object.
- Window: epochs from 100 days before to 300 days after the brightest point, so ZTF light curves
  cover one event like a DES season does.
- Fit: the same Villar model, bounds, starting guesses and cuts as `des_scipy_extraction.py`
  (≥ 10 points in total, ≥ 4 per band), fluxes on the FLUXCAL scale (zero point 27.5).
- Redshift: the BTS redshift; objects without a positive redshift are dropped.
- Classes (TNS types): SNIa = Ia, Ia-norm, Ia-91T, Ia-99aa; SNII = II, IIP, IIL, IIn, IIb;
  SNIbc = Ib, Ic, Ib/c, Ic-BL; SLSN-I = SLSN-I; AGN = AGN. Excluded: peculiar subtypes
  (Ia-pec, Ia-91bg, Iax, Ia-CSM, Ia-SC, Ia-03fg, II-pec, Ib-pec, Ic-pec), Ibn, Icn, SLSN-II,
  any type with "?", and non-SN types.
- Macro-F1 averages over the classes present in ZTF (AGN is expected to be rare or absent).

### The rule (unchanged from Phase 2.1)

`CALIBRATED_FEATURES = ["redshift", "abs_peak_mag"]`: every other feature is scaled with each
survey's own statistics. No ZTF labels are used for anything except evaluation.

### Predictions (four model families, matched settings, 5 PLAsTiCC split seeds)

- Z1: PLAsTiCC → ZTF transfer with consistent preprocessing is below ZTF-only 5-fold CV.
- Z2: hybrid > consistent, for every model family.
- Z3: hybrid > per-survey, for every model family.
- Z4: hybrid ≥ per-survey with abs_peak_mag removed, for every model family.
- Z5: (XGBoost, consistent) ZTF-matched brightness selection of the PLAsTiCC training set beats
  both no selection and a random subset of the same size.

"Supported" = mean difference over seeds larger than one pooled SD (Z4: hybrid not worse by more
than one pooled SD). Each model family counts separately. The label-free SHAP damage ranking on
ZTF is reported descriptively, without a prediction.

## 4.2 Label acquisition on DES (honest version)

### Design (fixed)

- Held-out DES test set: 30%, class-stratified, seed 0, the same for every run. Acquisition draws
  only from the remaining 70% pool.
- Base model: XGBoost (matched settings) trained on PLAsTiCC with the hybrid rule.
- Budgets: 5%, 10%, 20%, 40% of the pool; 10 repeats each (PLAsTiCC split seed = repeat mod 5,
  model seed = repeat, acquisition seed = repeat).
- Strategies:
  - **Random**: uniform from the pool.
  - **Predicted-class stratified**: equal numbers per class *as predicted by the base model*;
    classes with too few candidates give their share to the others.
  - **Uncertainty**: highest entropy of the base model's predicted probabilities.
  - **True-class stratified**: equal numbers per *true* class. Not usable in practice; reported
    only as an upper bound.
- Acquired DES objects (scaled exactly like the DES test set) are added to the PLAsTiCC training
  set with weight 5 and the model is retrained.

### Predictions

- A1: predicted-class stratified > random in macro-F1 at the 10% and 20% budgets.
- A2: true-class stratified ≥ every usable strategy at every budget (sanity check of the bound).
- No prediction is made for predicted-class stratified vs uncertainty sampling; reported as found.
