# Phase 4

Read `PREREGISTRATION_4.md` first and commit it to git before running anything: it was written
before any ZTF data was downloaded, which is what makes 4.1 a genuine held-out test.

## One-time setup for 4.1 (ZTF)

1. `conda activate des_env && pip install alerce`
2. Export the BTS catalogue: open https://sites.astro.caltech.edu/ztf/bts/explorer.php, choose the
   sample with all classified transients, set **Display as** to **CSV**, submit, and save the file as
   `phase4/bts_catalogue.csv`. The scripts find the ZTF ID, type and redshift columns by name.
3. Make sure you have a few GB of free disk space (each light curve is a small CSV, but there are
   thousands).

## Run

From the `xai` folder: `bash run_phase4.sh`. It runs every step below; each is resumable.

| Step | Script | Output |
| --- | --- | --- |
| 4.1a | `ztf_download.py` | `<data root>/ZTF dataset/lightcurves/*.csv` (ALeRCE forced photometry; alert detections as fallback) |
| 4.1b | `ztf_extract.py` | `<data root>/ZTF dataset/ztf_superphot_features.csv` (same Villar fit as DES) |
| 4.1c | `../phase1/run_experiment.py --dir .` | `results/ztf_*.json` (18 configs) |
| 4.1d | `diagnose_damage_target.py ztf` | label-free SHAP damage ranking on ZTF (descriptive) |
| 4.2 | `acquisition.py` | `results/acquisition.json` |
| all | `make_tables.py` | `tables/`, `figures/acquisition_curves.png`, `results/verdicts_4.json` |

`<data root>` is the folder the scripts already use for your data (on your laptop,
`xai/DES dataset`).

## Expected time

- Download: roughly 1–3 hours for a few thousand objects at 4 parallel requests (ALeRCE's speed
  varies). Interrupting and rerunning skips light curves already saved.
- Fitting: 10–30 minutes. Configs: 30–60 minutes. Acquisition: 15–30 minutes.

## Things to check after the download

`download_status.json` records, per object, whether forced photometry or alert detections were
used. If most objects fell back to detections, say so in the paper: alert detections contain only
significant positive fluxes, unlike DES's forced photometry.
