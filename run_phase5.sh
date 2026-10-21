#!/usr/bin/env bash
# Phase 5: reviewer-requested checks. Usage, from the xai folder:
#   conda activate des_env && bash run_phase5.sh
# ~1.5-3 hours (116 configs x 5 seeds, one thread). Commit phase5/PREREGISTRATION_5.md to git first.
set -e
cd "$(dirname "$0")"
XAI="$(pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"   # avoids the macOS OpenMP freeze
grep -q "def stratlearn" phase1/adaptation.py || { echo "phase1/adaptation.py is older than Phase 5: unzip the Phase 5 zip over xai."; exit 1; }
grep -q "plasticc_pseudo" phase1/run_experiment.py || { echo "phase1/run_experiment.py is older than Phase 5: unzip the Phase 5 zip over xai."; exit 1; }
grep -q "def support_overlap" phase1/common.py || { echo "phase1/common.py is older than Phase 5: unzip the Phase 5 zip over xai."; exit 1; }

echo "== Check: original pipeline still reproduces (expect 0.7899 / 0.2193) =="
python phase1/run_experiment.py phase1/configs/00_legacy_reproduction.json

echo "== Quick checks from saved results (ZTF confusion, support overlap, DES per-class intervals) =="
(cd phase5 && python quick_checks.py)

echo "== Phase 5 experiments: controlled support test, weights, StratLearn, M_peak variants, photo-z, gated rule =="
python phase1/run_experiment.py --dir "$XAI/phase5"

echo "== Tables and verdicts =="
(cd phase5 && python make_tables.py)
echo "Done. Verdicts: phase5/results/verdicts_5.json  Tables: phase5/tables/  Figure: phase5/figures/ztf_confusion.png"
