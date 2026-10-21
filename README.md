# When do label-free fixes for simulation-to-real shift transfer?

Code, configurations, pre-registrations, tables and figures for the paper *When Do Label-Free Fixes
for Simulation-to-Real Shift Transfer? A Three-Survey Study of Photometric Transient Classification*.

Classifiers are trained on the PLAsTiCC simulation and evaluated on three real spectroscopic samples
(DES-SN 5-year, ZTF Bright Transient Survey, YSE DR1). Every experiment is a JSON config run by one
engine; every table and figure is generated from saved results.

## Layout

| Path | Contents |
| --- | --- |
| `phase1/` | Engine (`common.py`, `models.py`, `adaptation.py`, `run_experiment.py`), pipeline audit, reproduction of the preliminary pipeline |
| `phase2/` … `phase8/` | One folder per study phase: `PREREGISTRATION_*.md`, `configs/`, analysis scripts, `tables/`, `figures/`, summary `results/verdicts*.json` |
| `run_phase2.sh` … `run_phase8.sh` | Run each phase end to end (each first checks that the preliminary pipeline reproduces) |
| `des_superphot_local.py`, `des_scipy_extraction.py`, `extraction/` | Light-curve fitting and feature extraction for PLAsTiCC and DES (ZTF: `phase4/`, YSE: `phase7/`) |
| `data/` | Download instructions and scripts; `export_splits.py` |
| `splits/` | Exact object identifiers in every split and target sample |

| Paper section | Phase |
| --- | --- |
| Pipeline audit (Appendix) | 1 |
| Hybrid rule on DES, simulated selection, shift diagnostics | 2, 2.2, 2.3 |
| Tuning, baselines, calibration, bootstrap | 3 |
| ZTF held-out test, label acquisition | 4 |
| Controlled test, StratLearn, weights, robustness, ρ | 5 |
| Decomposition, two-factor test, label shift, sample size | 6 |
| YSE held-out test | 7 |
| Per-feature shift, stronger controlled test | 8 |

## Setup

```bash
conda env create -f environment.yml
conda activate des_env
```
On macOS run with `OMP_NUM_THREADS=1` (the run scripts set it).

## Data

The datasets are not redistributed. See `data/README.md` for where to obtain each one and how to
build the feature tables. The engine looks for a data root containing `superphot features/` and
`DES dataset/`, or set `XAI_ROOT=/path/to/data/root`.

DES-SN 5-year: https://github.com/des-science/DES-SN5YR, folder `0_DATA/DES-SN5YR_DES/`
(`DES-SN5YR_DES_HEAD.FITS.gz`, `DES-SN5YR_DES_PHOT.FITS.gz`, decompressed), main branch at
commit `48de863` (11 December 2025); light curves from Sánchez et al. (2024). The tagged releases
contain an older HEAD file, so check out this commit rather than a tag.

## Reproducing

```bash
python phase1/run_experiment.py phase1/configs/00_legacy_reproduction.json   # expect 0.7899 / 0.2193
bash run_phase2.sh && bash run_phase3.sh && bash run_phase4.sh && bash run_phase5.sh
bash run_phase6.sh && bash run_phase7.sh && bash run_phase8.sh
```
Full results JSONs (one per config and split seed) are archived separately (see the paper's
reproducibility statement); the summary verdicts and all tables are in this repository.

## Pre-registration

Each phase's predictions were written in its `PREREGISTRATION_*.md` before that phase was run. Amendments (Phases 6 and 8) are recorded at the end of those files.

## License

MIT (see `LICENSE`).
