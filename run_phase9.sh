#!/usr/bin/env bash
# Phase 9: survey-artefact ablation. Usage, from the xai folder:
#   conda activate des_env && bash run_phase9.sh
set -e
cd "$(dirname "$0")"
XAI="$(pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
echo "== Check: original pipeline still reproduces (expect 0.7899 / 0.2193) =="
python phase1/run_experiment.py phase1/configs/00_legacy_reproduction.json
echo "== A6: domain-classifier AUC with and without the artefact features =="
(cd phase9 && python domain_auc.py)
echo "== A1-A5: 48 configs (3 targets x 4 models x 4 modes) without the artefact features =="
python phase1/run_experiment.py --dir "$XAI/phase9"
echo "== Tables and verdicts =="
(cd phase9 && python make_tables.py)
echo "Done. Send phase9/tables/ablation.md and phase9/results/verdicts_9.json."
