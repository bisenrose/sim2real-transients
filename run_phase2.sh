#!/usr/bin/env bash
# Run all of Phase 2 (2.1, 2.2, 2.3) on your laptop. ~45-60 minutes in total.
# Usage, from the xai folder:   conda activate des_env && bash run_phase2.sh
# Each step skips nothing: it re-runs every config and overwrites results.
set -e
cd "$(dirname "$0")"
XAI="$(pwd)"
for f in common.py models.py adaptation.py run_experiment.py; do
  [ -f "phase1/$f" ] || { echo "phase1/$f is missing. Copy the four engine files from phase1_engine.zip into phase1/ first."; exit 1; }
done
grep -q "import models as M" phase1/run_experiment.py || { echo "phase1/run_experiment.py is an old version. Replace it with the one from phase1_engine.zip."; exit 1; }
echo "== Removing old Phase 2 results (they were produced by the old engine) =="
rm -f phase2/results/*.json phase2/results/*.csv phase2_2/results/*.json phase2_3/results/*.json phase2_3/results/*.csv
python -c "import xgboost, lightgbm, catboost, shap, torch" || {
  echo "Missing packages. Run: conda activate des_env && pip install lightgbm catboost shap torch"; exit 1; }

echo "== Check: original pipeline still reproduces (expect 0.7899 / 0.2193) =="
python phase1/run_experiment.py phase1/configs/00_legacy_reproduction.json

echo "== 2.1: hybrid scaling rule, 4 model families x 4 preprocessings x 2 label sets =="
python phase1/run_experiment.py --dir "$XAI/phase2"
(cd phase2 && python diagnose_damage.py && python make_tables.py)

echo "== 2.2: simulated DES selection on PLAsTiCC =="
python phase1/run_experiment.py --dir "$XAI/phase2_2"
(cd phase2_2 && python make_tables.py && python plot_abs_mag.py)

echo "== 2.3: per-feature shift and removal diagnostics =="
python phase1/run_experiment.py --dir "$XAI/phase2_3"
(cd phase2_3 && python analysis_2_3.py)

echo "Done. Verdicts: phase2/results/verdicts.json, phase2_2/results/verdicts_2.2.json, phase2_3/results/verdicts_2.3.json"
