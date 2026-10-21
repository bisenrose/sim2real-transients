# Phase 1 audit code

This folder lives inside the project root (`<repo>/phase1`).
It finds the data automatically: it looks for `superphot features/` in the folder above it,
then in its own folder, then at `<repo>`. To point it
elsewhere, set `XAI_ROOT=/path/to/xai`.

```bash
cd <repo>/phase1
conda activate des_env           # or any env with the packages below
pip install xgboost==3.2.0 scikit-learn pandas numpy astropy
python run_experiment.py --all     # ~5 minutes; writes results/*.json
python make_tables.py              # writes tables/*.md and tables/*.tex
```

To add an experiment, copy a file in `configs/`, change its `name` and settings,
and run `python run_experiment.py configs/<new>.json`. Config keys:

- `preprocessing`: `"consistent"` (fit on PLAsTiCC train, apply everywhere) or `"per_domain"`
- `split_seeds`: list of PLAsTiCC train/test split seeds (model seed follows the split seed)
- `drop_features`: feature names to remove
- `des_label_policy`: `"corrected"` (default) or `"confirmed_only"` (drops uncertain '?' SNTYPEs)
- `legacy_class_codes`: `true` only to reproduce the original (buggy) pipeline exactly
- `des_cv`: `true` to also compute the DES-only 5-fold CV score (needs `des_model`)

Read `PHASE1_AUDIT.md` for the findings.
