# Phase 2

Uses the engine in `../phase1` (common.py, models.py, run_experiment.py). Run in `des_env`.

```bash
cd <repo>/phase1
git add ../phase2/PREREGISTRATION_2.1.md && git commit -m "Pre-register 2.1 rule"   # if using git
python run_experiment.py --dir ../phase2          # all 2.1 configs (~15-20 min)
python run_experiment.py ../phase2/configs/mlp_*.json   # re-run MLPs with PyTorch
cd ../phase2
python diagnose_damage.py                          # label-free SHAP damage (P4, P5)
python make_tables.py                              # tables/ + results/verdicts.json
```

Read `RESULTS_2.1.md` for findings.
