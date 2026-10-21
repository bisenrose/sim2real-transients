# Phase 5: reviewer-requested checks

Read and commit `PREREGISTRATION_5.md` first. Run everything with `bash run_phase5.sh` from the
`xai` folder in `des_env`. Needs the Phase 3 and Phase 4 results in place (the quick checks read them)
and the ZTF features from Phase 4.

| Block | Configs | What it answers |
| --- | --- | --- |
| Quick checks | `quick_checks.py` | ZTF confusion matrices; label-free support overlap; DES per-class 95% intervals |
| C (confirmatory) | `C_*` (32) | Does keeping redshift on the shared scale hurt when the target lies outside the training redshift range? A PLAsTiCC pseudo-target (z < 0.12) vs a random control |
| W | `W_*` (4) | Magnitude density ratio as weights instead of sub-sampling |
| S | `S_*` (8) | StratLearn (propensity-score strata), with and without survey-design features |
| K | `K_*` (24) | Two alternative peak absolute magnitude definitions |
| N | `N_*` (12) | Noisy photometric-like target redshifts |
| G (exploratory) | `G_*` (32) | Overlap-gated rule, threshold sensitivity |
| baseline | `B_*` (4) | Consistent-preprocessing baselines for W and S |

Outputs: `tables/*.md|tex`, `figures/ztf_confusion.png`, `results/verdicts_5.json`.
