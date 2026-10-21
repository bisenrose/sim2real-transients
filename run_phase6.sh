#!/usr/bin/env bash
# Phase 6: probing the ZTF failure (reviewer round 2). Usage, from the xai folder:
#   conda activate des_env && bash run_phase6.sh
# ~1-2 hours (80 configs x 5 seeds, one thread). Commit phase6/PREREGISTRATION_6.md to git first.
set -e
cd "$(dirname "$0")"
XAI="$(pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
grep -q "def instrument_shift" phase1/common.py || { echo "phase1/common.py is older than Phase 6: unzip the Phase 6 zip over xai."; exit 1; }
grep -q "target_stats_n" phase1/run_experiment.py || { echo "phase1/run_experiment.py is older than Phase 6: unzip the Phase 6 zip over xai."; exit 1; }
[ -f phase4/results/ztf_xgboost_hybrid.json ] || { echo "Phase 4 ZTF results are needed (phase4/results)."; exit 1; }

echo "== Check: original pipeline still reproduces (expect 0.7899 / 0.2193) =="
python phase1/run_experiment.py phase1/configs/00_legacy_reproduction.json

echo "== D1, D2, D5: new configs =="
python phase1/run_experiment.py --dir "$XAI/phase6"

echo "== D3, D4: analyses of saved predictions =="
(cd phase6 && python analysis_saved.py)

echo "== Tables and verdicts =="
(cd phase6 && python make_tables.py)
echo "Done. Verdicts: phase6/results/verdicts_6.json and phase6/results/D3_label_shift.json  Tables: phase6/tables/"
