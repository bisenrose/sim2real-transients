#!/usr/bin/env bash
# Reproduce every phase in order (several hours). Requires the data described in data/README.md.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
python phase1/run_experiment.py --dir phase1
for p in 2 3 4 5 6 7 8; do bash "run_phase$p.sh"; done
