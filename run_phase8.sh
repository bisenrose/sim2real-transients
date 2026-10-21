#!/usr/bin/env bash
# Phase 8: per-feature shift comparison (E1) and a stronger controlled test (E2).
# Usage, from the xai folder:  conda activate des_env && bash run_phase8.sh
# Needs the DES, ZTF and YSE feature tables. ~1 hour. Commit phase8/PREREGISTRATION_8.md first.
set -e
cd "$(dirname "$0")"
XAI="$(pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
grep -q "def population_shift" phase1/common.py || { echo "phase1/common.py is older than Phase 8: unzip the Phase 8 zip over xai."; exit 1; }
grep -q "apply_target_shifts" phase1/run_experiment.py || { echo "phase1/run_experiment.py is older than Phase 8: unzip the Phase 8 zip over xai."; exit 1; }

echo "== Check: original pipeline still reproduces (expect 0.7899 / 0.2193) =="
python phase1/run_experiment.py phase1/configs/00_legacy_reproduction.json

echo "== E1: per-feature shift of DES, ZTF and YSE =="
(cd phase8 && python features_compare.py)

echo "== E2: stronger controlled test (96 configs) =="
python phase1/run_experiment.py --dir "$XAI/phase8"

echo "== Tables and verdicts =="
(cd phase8 && python make_tables.py)
echo "Done. Verdicts: phase8/results/verdicts_8.json  Tables: phase8/tables/  Figure: phase8/figures/E1_ks_heatmap.png"
