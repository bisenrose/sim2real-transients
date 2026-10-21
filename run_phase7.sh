#!/usr/bin/env bash
# Phase 7: YSE DR1 as a third, held-out survey. Usage, from the xai folder:
#   conda activate des_env && bash run_phase7.sh
# Needs: DES dataset/YSE dataset/*.snana.dat (from yse_dr1_zenodo.tar.gz), phase4/ztf_extract.py,
# phase4/bts_catalogue.csv and the ZTF features (for the descriptive comparison).
# Commit phase7/PREREGISTRATION_7.md to git first. ~1-1.5 hours.
set -e
cd "$(dirname "$0")"
XAI="$(pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
grep -q "YSE_FEATURES" phase1/common.py || { echo "phase1/common.py is older than Phase 7: unzip the Phase 7 zip over xai."; exit 1; }
grep -q "TARGET_PATHS" phase1/run_experiment.py || { echo "phase1/run_experiment.py is older than Phase 7: unzip the Phase 7 zip over xai."; exit 1; }
[ -f phase4/ztf_extract.py ] || { echo "phase4/ztf_extract.py is needed (it holds the shared fitting code)."; exit 1; }

echo "== Check: original pipeline still reproduces (expect 0.7899 / 0.2193) =="
python phase1/run_experiment.py phase1/configs/00_legacy_reproduction.json

echo "== Fit YSE light curves (Pan-STARRS g and r only) =="
(cd phase7 && python -u yse_extract.py)

echo "== PLAsTiCC -> YSE: 4 models x 7 preprocessings, plus the no-BTS robustness runs =="
python phase1/run_experiment.py --dir "$XAI/phase7"

echo "== Tables and verdicts =="
(cd phase7 && python analysis_7.py)
echo "Done. Verdicts: phase7/results/verdicts_7.json  Tables: phase7/tables/"
