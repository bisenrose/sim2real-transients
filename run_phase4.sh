#!/usr/bin/env bash
# Phase 4: 4.1 ZTF held-out test of the hybrid rule, 4.2 honest label acquisition on DES.
# Usage, from the xai folder:   conda activate des_env && bash run_phase4.sh
# 4.1 needs phase4/bts_catalogue.csv (see phase4/README.md); without it, only 4.2 runs.
set -e
cd "$(dirname "$0")"
XAI="$(pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"   # avoids the macOS OpenMP freeze (see Phase 2.3)
for f in common.py models.py adaptation.py run_experiment.py; do
  [ -f "phase1/$f" ] || { echo "phase1/$f missing: unzip the Phase 4 zip over the xai folder."; exit 1; }
done
grep -q "load_target" phase1/common.py || { echo "phase1/common.py is older than Phase 4: replace it from the Phase 4 zip."; exit 1; }

echo "== Check: original pipeline still reproduces (expect 0.7899 / 0.2193) =="
python phase1/run_experiment.py phase1/configs/00_legacy_reproduction.json

if [ -f phase4/bts_catalogue.csv ]; then
  python -c "import alerce" 2>/dev/null || { echo "Install the ALeRCE client first: pip install alerce"; exit 1; }
  echo "== 4.1a: download ZTF light curves (resumable; can take a while) =="
  (cd phase4 && python -u ztf_download.py --workers 4)
  echo "== 4.1b: fit ZTF light curves (same fit as DES) =="
  (cd phase4 && python -u ztf_extract.py)
  echo "== 4.1c: PLAsTiCC -> ZTF, 4 models x 4 preprocessings + selection =="
  python phase1/run_experiment.py --dir "$XAI/phase4"
  (cd phase4 && python diagnose_damage_target.py ztf)
  echo "== 4.1d: exploratory ZTF diagnostics (light-curve source, magnitude check) =="
  (cd phase4 && python diagnose_ztf.py)
else
  echo "== 4.1 skipped: phase4/bts_catalogue.csv not found (see phase4/README.md) =="
fi

echo "== 4.2: label acquisition on DES (10 repeats x 4 budgets x 4 strategies) =="
(cd phase4 && python -u acquisition.py)

echo "== Tables, figures, verdicts =="
(cd phase4 && python make_tables.py)
echo "Done. Verdicts: phase4/results/verdicts_4.json  Tables: phase4/tables/  Figures: phase4/figures/"
