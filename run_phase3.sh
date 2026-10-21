#!/usr/bin/env bash
# Run all of Phase 3 (3.1 tuning, 3.2 baselines, 3.3 statistics) on your laptop.
# Usage, from the xai folder:   conda activate des_env && bash run_phase3.sh [TRIALS]
# TRIALS = random-search settings per model and split seed (default 50).
# Tuning is resumable: re-running skips model/seed pairs already in phase3/tuning/.
set -e
cd "$(dirname "$0")"
XAI="$(pwd)"
TRIALS="${1:-50}"
python -c "import xgboost, lightgbm, catboost, shap, torch, scipy" || {
  echo "Missing packages. Run: conda activate des_env && pip install lightgbm catboost shap torch scipy"; exit 1; }
[ -f phase2/results/xgboost_corrected_hybrid.json ] || echo "Note: Phase 2 results not found; run run_phase2.sh first for the full tuning table."

echo "== Check: original pipeline still reproduces (expect 0.7899 / 0.2193) =="
python phase1/run_experiment.py phase1/configs/00_legacy_reproduction.json

echo "== 3.1: nested tuning on PLAsTiCC only ($TRIALS settings per model and seed) =="
for m in xgboost lightgbm catboost mlp; do
  (cd phase3 && python -u tune.py "$m" --trials "$TRIALS")
done

echo "== 3.1 + 3.2: tuned models and domain-adaptation baselines =="
python phase1/run_experiment.py --dir "$XAI/phase3"

echo "== 3.3: statistics, calibration, verdicts =="
(cd phase3 && python stats.py)

echo "Done. Tables: phase3/tables/  Figure: phase3/figures/reliability.png  Verdicts: phase3/results/verdicts_3.json"
