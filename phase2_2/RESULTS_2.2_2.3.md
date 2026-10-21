# Phase 2.2 and 2.3 results

Run on 19 October 2026, after writing `PREREGISTRATION_2.2.md` and
`../phase2_3/PREREGISTRATION_2.3.md`. XGBoost unless stated, DES corrected labels, 5 split seeds.

## 2.2 Simulating DES's selection on PLAsTiCC

Keeping only PLAsTiCC training objects whose apparent brightness matches DES's (estimated
without DES labels) lifts transfer from 0.486 to 0.563, while a random subset of the same
size does worse than no selection. The Malmquist explanation now has direct support.

| Training selection (consistent preprocessing) | N train | Transfer F1 | abs mag KS vs DES |
| --- | --- | --- | --- |
| None | 3625 | 0.486 ± 0.019 | 0.35 |
| DES-matched (logistic) | 1181 | **0.563 ± 0.022** | 0.22 |
| Random, same size | 1181 | 0.453 ± 0.031 | 0.35 |
| Hard cut at DES 99th percentile | 2325 | 0.552 ± 0.020 | 0.23 |
| Random, same size | 2325 | 0.480 ± 0.019 | 0.35 |

LightGBM gives the same picture: 0.502 → 0.561, random control 0.469.

| | Prediction | Outcome |
| --- | --- | --- |
| Q1 | Selection beats no selection | Supported (XGBoost and LightGBM) |
| Q2 | Selection beats a random same-size subset | Supported (both) |
| Q3 | abs mag shift shrinks | Supported: KS 0.35 → 0.22; median gap −1.10 → −0.44 mag |
| Q4 | SNIa gains the most | **Not supported**: SLSN-I gains most (+0.24), then AGN (+0.08), SNIa third (+0.06) |

**Q4 makes physical sense in hindsight.** SLSN-I are seen to the highest redshifts, so
PLAsTiCC's SLSN-I sample contains the most objects too faint for DES's spectroscopy;
selection changes them most. Report it as a failed prediction with this explanation.

**Selection and the hybrid rule fix the same problem.** Under the hybrid rule, adding
selection does not help (0.605 without, 0.570 with logistic selection, 0.593 with the hard
cut, all within noise of their random controls). Both repair the same mismatch in absolute
magnitude; the hybrid rule does it without throwing away two thirds of the training set.
That is evidence the hybrid rule works for the reason we claim.

Figure: `figures/abs_mag_by_class.png` (absolute magnitude per class: PLAsTiCC all,
PLAsTiCC after DES-matched selection, DES).

## 2.3 Which shift measure predicts harmful features?

None of them, reliably. That is the result.

- The most shifted features are survey artefacts and noise terms: time_span (KS 1.00),
  extra_sigma_g/r, t_0. abs_peak_mag ranks 10th by KS.
- Removing one feature at a time under consistent preprocessing, the biggest gain comes
  from extra_sigma_g (+0.073), then redshift (+0.030). Removing abs_peak_mag changes
  almost nothing (+0.002).
- Correlation of each shift measure with that gain (Spearman, 27 features): shift 0.22,
  KS 0.07, risk (importance × shift) 0.20, SHAP damage 0.32. None significant (all p > 0.09).
- **R1 not supported**: risk does not predict harm better than shift alone.
- Under the hybrid rule, no single removal changes transfer by more than ±0.017; the
  feature-level problems are largely already handled.

What this means for the paper: generic shift scores, with or without SHAP weighting, cannot
tell you which features to fix. Large shifts are often harmless (time_span) or best handled
by re-centring (extra_sigma), and the one feature that re-centring breaks (abs_peak_mag)
is not the most shifted. That supports a physics-based rule over a statistical one, which
is the paper's argument. Present the original plan's risk-score idea as tested and rejected.

Figure: `../phase2_3/figures/removal_vs_shift.png`. Tables: `tables/` in each folder.

## Overall Phase 2 story

1. Consistent preprocessing: transfer 0.49 (DES-only 0.57).
2. The DES sample is flux-limited; simulating that selection on PLAsTiCC recovers transfer
   to 0.56 (2.2), and shrinks the absolute-magnitude mismatch.
3. A label-free hybrid scaling rule (re-centre survey-dependent features; keep source-frame
   physical quantities on a shared scale) reaches 0.60, across four model families (2.1),
   and makes selection redundant (2.2), showing it fixes the same mismatch.
4. SHAP locates where re-centring acts, but neither SHAP nor shift statistics can say whether
   that is good or bad (2.1 P4/P5, 2.3 R1); physics decides.
