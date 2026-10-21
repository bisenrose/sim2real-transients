# Results: Phases 2 and 3

All numbers in this report come from the laptop runs (des_env: xgboost 3.2.0, scikit-learn 1.9.0,
pandas 2.2.2, PyTorch MLP), which are now the canonical results. They supersede the numbers in
`RESULTS_2.1.md` and `RESULTS_2.2_2.3.md`, which came from my sandbox (scikit-learn MLP,
different library versions); the conclusions are the same. Every number comes from a
`results/*.json` file and the tables in each folder's `tables/`.

Setup throughout: PLAsTiCC training set → DES-SN 5-year spectroscopic sample, five classes,
DES labels with the corrected SNTYPE mapping ("all types", 530 objects) unless stated, macro-F1,
mean ± SD over 5 PLAsTiCC split seeds.

## Prediction scorecard

Every prediction was written down before its run (`PREREGISTRATION_*.md`).

| | Prediction | Outcome |
| --- | --- | --- |
| P1 | Hybrid > consistent | Supported, 8/8 (4 models × 2 label sets) |
| P2 | Hybrid > per-survey | Supported, 8/8 |
| P3 | Hybrid ≥ per-survey with abs mag removed | Supported, 8/8 (MLP: a tie, 0.617 vs 0.616) |
| P4 | abs_peak_mag has the largest SHAP damage | **Not supported** (rank 2) |
| P5 | Redshift in the top 5 for SHAP damage | **Not supported** (rank 6) |
| Q1 | DES-matched selection > none | Supported (XGBoost and LightGBM) |
| Q2 | Selection > random same-size subset | Supported (both) |
| Q3 | Selection shrinks the abs mag shift | Supported |
| Q4 | SNIa gains most from selection | **Not supported** (SLSN-I gains most) |
| R1 | Risk score predicts harmful features better than shift | Technically met, but a null result (see 2.3) |
| T1 | Hybrid > consistent after tuning | Supported, 4/4 |
| T2 | Tuning moves tree models' in-domain F1 by < 0.03 | Supported, 3/3 |
| B1 | Hybrid > importance reweighting | Supported, but reweighting does nothing here (see 3.2) |
| B2 | Hybrid > CORAL | Supported (XGBoost and LightGBM) |
| C1 | Hybrid improves log-loss | Supported |

## Phase 2.1: the hybrid scaling rule

Rule (fixed before the confirmatory runs): rescale each survey's features with that survey's
own statistics, except source-frame physical quantities (redshift, absolute magnitude),
which keep PLAsTiCC's scaling in both surveys. Uses no DES labels.

| Model | Consistent | Per-survey | Per-survey − abs mag | Hybrid rule | DES-only CV |
| --- | --- | --- | --- | --- | --- |
| XGBoost (development) | 0.488 ± 0.023 | 0.528 ± 0.005 | 0.565 ± 0.010 | **0.602 ± 0.012** | 0.575 ± 0.011 |
| LightGBM | 0.506 ± 0.020 | 0.518 ± 0.009 | 0.543 ± 0.035 | **0.603 ± 0.016** | 0.586 ± 0.014 |
| CatBoost | 0.498 ± 0.028 | 0.517 ± 0.010 | 0.577 ± 0.015 | **0.589 ± 0.012** | 0.565 ± 0.009 |
| MLP | 0.466 ± 0.033 | 0.564 ± 0.019 | 0.616 ± 0.018 | **0.617 ± 0.022** | 0.596 ± 0.016 |

With confident DES labels only (no "?" types), the hybrid rule gives 0.593–0.628, again level
with or above DES-only CV (0.590–0.616).

The label-free hybrid rule matches or beats a model trained on real DES labels, for all four
model families. The rule was found on XGBoost with all types, so that row is development
evidence; the other seven settings are confirmatory. The MLP is the one model where simply
dropping abs_peak_mag works as well as the rule.

**SHAP diagnostic (P4, P5 failed).** Ranking features by how much re-centring changes their
SHAP contribution puts extra_sigma_g first, abs_peak_mag second (all 5 seeds), redshift sixth.
An exploratory check explains why this is informative rather than a problem: keeping
extra_sigma on the shared scale collapses transfer to 0.357, while keeping abs mag only on the
shared scale gives 0.607. Re-centring the survey-dependent noise term helps a lot; re-centring
absolute magnitude hurts. **SHAP shows where re-centring acts; physics decides whether it should.**

## Phase 2.2: simulating DES's selection on PLAsTiCC

| XGBoost, consistent preprocessing | N train | Transfer F1 | abs mag KS vs DES |
| --- | --- | --- | --- |
| No selection | 3625 | 0.488 ± 0.023 | 0.35 |
| DES-matched selection (label-free) | 1181 | **0.571 ± 0.009** | 0.22 |
| Random subset, same size | 1181 | 0.455 ± 0.033 | 0.35 |
| Hard cut at DES 99th percentile | 2325 | 0.546 ± 0.014 | 0.23 |
| Random subset, same size | 2325 | 0.484 ± 0.007 | 0.35 |

LightGBM: 0.506 → 0.561 with selection, 0.469 for the random control.

Making PLAsTiCC's training set as bright as DES's recovers most of the gap, shrinks the
absolute-magnitude mismatch (median gap −1.10 → −0.44 mag), and a random subset of the same size
does worse. That is direct evidence for the flux-limited (Malmquist) selection explanation.
SLSN-I gains most (+0.26), not SNIa (+0.06): they are seen to the highest redshifts, so the
faintest PLAsTiCC SLSN-I are the ones DES could never have confirmed.

With the hybrid rule, selection adds nothing (0.602 without; 0.562–0.613 with selection or its
random controls, none clearly better). Both fix the same mismatch.

## Phase 2.3: does any shift score identify harmful features?

No. Removing one feature at a time (diagnostic, uses DES labels), the largest gains under
consistent preprocessing come from extra_sigma_g (+0.075) and redshift (+0.028); removing
abs_peak_mag changes nothing (−0.004). Under the hybrid rule no single removal moves transfer
by more than ±0.013.

Spearman correlation between each label-free score and the gain from removal (27 features):

| Score | Consistent | Hybrid |
| --- | --- | --- |
| Shift (domain classifier) | 0.15 (p = 0.44) | 0.30 (p = 0.13) |
| KS statistic | 0.19 (p = 0.33) | 0.21 (p = 0.29) |
| Risk (importance × shift) | 0.17 (p = 0.40) | 0.17 (p = 0.41) |
| SHAP damage | 0.34 (p = 0.08) | 0.07 (p = 0.74) |

R1 asked whether risk beats shift: 0.17 vs 0.15 technically meets it, but neither is
significant, and in my earlier run the order was reversed. Report it as a null result. The
most-shifted features are survey artefacts (time_span, KS = 1.00) or noise terms, and the one
feature re-centring breaks (abs_peak_mag) ranks tenth by KS. Generic shift statistics cannot
pick out what to fix, which is the argument for a physics-based rule.

## Phase 3.1: tuning

Equal random-search budget (50 settings) per model and split seed, nested 3-fold CV on PLAsTiCC
only.

| Model | PLAsTiCC F1 matched → tuned | Consistent transfer matched → tuned | Hybrid transfer matched → tuned |
| --- | --- | --- | --- |
| XGBoost | 0.785 → 0.785 | 0.488 → 0.488 | 0.602 → 0.596 |
| LightGBM | 0.782 → 0.776 | 0.506 → 0.512 | 0.603 → 0.593 |
| CatBoost | 0.753 → 0.772 | 0.498 → 0.496 | 0.589 → 0.579 |
| MLP | 0.738 → 0.739 | 0.466 → 0.465 ± 0.077 | 0.617 → 0.619 |

Conclusions do not depend on hyperparameters. The old "LightGBM scores 0.53" was a settings
artefact; drop that paragraph. Tuning on the simulation slightly lowers hybrid transfer for the
tree models, a small sign of fitting simulation-specific structure. The tuned MLP is erratic
under consistent preprocessing (one seed 0.33) and stable under the hybrid rule.

## Phase 3.2: standard domain-adaptation baselines

| Method (no DES labels) | XGBoost | LightGBM |
| --- | --- | --- |
| Consistent (baseline) | 0.488 ± 0.023 | 0.506 ± 0.020 |
| + importance reweighting | 0.488 ± 0.023 | 0.506 ± 0.020 |
| + reweighting, survey features hidden (exploratory) | 0.469 ± 0.033 | 0.504 ± 0.024 |
| + CORAL | 0.474 ± 0.016 | 0.475 ± 0.020 |
| + DES-matched selection (2.2) | 0.571 ± 0.009 | -- |
| **Hybrid rule** | **0.602 ± 0.012** | **0.603 ± 0.016** |
| Hybrid + reweighting | 0.602 ± 0.012 | 0.603 ± 0.016 |
| Hybrid + CORAL | 0.507 ± 0.025 | 0.523 ± 0.013 |

- **Importance reweighting is a no-op.** The domain classifier separates the surveys perfectly
  (AUC 1.000, mainly through time_span), so every training object gets the same weight. Hiding
  the survey-design features still leaves AUC 0.99 and does not help. This is the known failure of
  reweighting when the two datasets barely overlap. B1 holds, but say why.
- **CORAL hurts,** most of all on top of the hybrid rule (0.602 → 0.507). CORAL mixes all
  features together, including absolute magnitude, which re-creates the problem the rule avoids.
- **Selection and the hybrid rule are statistically tied** (next section). The rule keeps the
  whole training set and costs nothing in-domain (0.785 vs 0.727 with selection).

## Phase 3.3: statistics and calibration

Paired bootstrap on DES objects, XGBoost (1000 resamples × 5 seeds):

| Comparison | Mean ΔF1 | 95% interval | P(Δ > 0) |
| --- | --- | --- | --- |
| Hybrid − consistent | +0.115 | [+0.027, +0.209] | 0.996 |
| Hybrid − CORAL | +0.129 | [+0.041, +0.214] | 0.997 |
| Hybrid − reweighting (survey features hidden) | +0.133 | [+0.040, +0.227] | 0.998 |
| Hybrid − DES-matched selection | +0.032 | [−0.031, +0.095] | 0.849 |
| Tuned: hybrid − consistent | +0.109 | [+0.034, +0.192] | 0.998 |

Calibration on DES, XGBoost:

| | Log-loss | Brier | ECE | Macro-F1 | Macro-F1 without SLSN-I |
| --- | --- | --- | --- | --- | --- |
| Consistent | 1.030 ± 0.055 | 0.505 ± 0.033 | 0.157 ± 0.024 | 0.488 | 0.529 |
| Hybrid rule | 0.581 ± 0.027 | 0.285 ± 0.015 | 0.045 ± 0.011 | 0.602 | 0.597 |

The hybrid rule nearly halves log-loss and cuts calibration error by more than two thirds, which
matters for follow-up prioritisation. In-domain (PLAsTiCC) calibration is identical for both
(ECE 0.022), as it should be: the rule only changes how DES is scaled.
Figure: `figures/reliability_both.png`.

## The per-class trade-off (must be discussed in the paper)

| Class (DES n) | F1 consistent → hybrid | Precision | Recall |
| --- | --- | --- | --- |
| SNIa (391) | 0.792 → 0.898 | 0.95 → 0.93 | 0.68 → 0.86 |
| AGN (41) | 0.373 → 0.675 | 0.23 → 0.59 | 0.95 → 0.79 |
| SLSN-I (18) | 0.326 → 0.622 | 0.95 → 0.62 | 0.20 → 0.62 |
| SNII (64) | 0.732 → **0.638** | 0.82 → **0.59** | 0.66 → 0.69 |
| SNIbc (16) | 0.219 → 0.177 | 0.19 → 0.16 | 0.28 → 0.20 |

What the confusion matrices (`tables/confusion_transfer.md`, `figures/confusion_transfer.png`)
show:

- Under consistent preprocessing, **108 of 391 DES SNe Ia are called AGN** (and 13 of 18 SLSN-I).
  That single error dominates the failure. The hybrid rule cuts it to 16.
- **SNII are not lost under the hybrid rule**: SNII recall is unchanged (0.66 → 0.69). The SNII
  drop is a precision drop: about 20 DES SNe Ia are now called SNII, against 2 before. Because
  SNe Ia outnumber SNe II six to one, a 5% leak from SNIa is a large contamination of SNII.
- SNIbc stays poor everywhere, including DES-only (16 objects); treat it as unresolved.

## What this means for the paper

1. With correct classes and consistent preprocessing, sim-to-real transfer reaches about 0.49
   macro-F1, against about 0.58 for a model trained on DES itself.
2. The dominant failure is DES SNe Ia predicted as AGN. Flux-limited spectroscopic selection makes
   DES's absolute-magnitude distribution unlike PLAsTiCC's; simulating that selection on PLAsTiCC
   recovers transfer to 0.57 (2.2).
3. A label-free hybrid scaling rule reaches 0.59–0.62 across four model families and both label
   sets, at or above the DES-only level, and nearly halves log-loss. It is significantly better than
   consistent preprocessing, reweighting and CORAL, and statistically tied with simulated selection,
   which it makes redundant.
4. SHAP locates where re-centring acts, but neither SHAP nor any shift statistic can tell good
   re-centring from harmful; physical units decide (2.1 P4/P5, 2.3).
5. Honest limits: the rule was developed on DES (XGBoost, all types); SNII precision falls; SNIbc
   is unresolved; DES statistics are fitted on the whole unlabelled DES sample (transductive).

The decisive remaining test is Phase 4: apply the rule, unchanged, to the ZTF Bright Transient
Survey.
