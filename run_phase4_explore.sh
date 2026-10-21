#!/usr/bin/env bash
# EXPLORATORY follow-up to the ZTF result (not pre-registered). ~10-20 minutes.
# Usage, from the xai folder:   conda activate des_env && bash run_phase4_explore.sh
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
echo "== Hybrid with only abs_peak_mag on the shared scale (redshift re-centred), 4 models on ZTF =="
python phase1/run_experiment.py phase4/configs/ztf_*_explore_hybrid_absmag_only.json
echo "== Redshift coverage and F1 by redshift =="
(cd phase4 && python diagnose_ztf_redshift.py)
echo "Done. Send phase4/tables/ztf_diag_redshift*.md, ztf_diag_by_redshift.md and phase4/results/ztf_*explore*.json"
