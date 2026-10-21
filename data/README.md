# Data

None of the datasets are redistributed here. Place them under one data root (default: the
repository root, or set `XAI_ROOT`) with this layout:

```
<data root>/
  superphot features/plasticc_raw_superphot_features.csv
  DES dataset/des_superphot_features.csv
  DES dataset/ZTF dataset/ztf_superphot_features.csv
  DES dataset/YSE dataset/yse_superphot_features.csv
```

| Sample | Source | How to build the feature table |
| --- | --- | --- |
| PLAsTiCC training set | Kaggle "PLAsTiCC Astronomical Classification" (https://www.kaggle.com/c/PLAsTiCC-2018/data): `training_set.csv`, `training_set_metadata.csv` (Kessler et al. 2019; The PLAsTiCC team 2018). Requires a Kaggle account and accepting the competition rules. | `python extraction/plasticc_scipy_extraction.py --data <kaggle folder> --out "<data root>/superphot features/plasticc_raw_superphot_features.csv"` (the original script; see below) |
| DES-SN 5-year | https://github.com/des-science/DES-SN5YR, folder `0_DATA/DES-SN5YR_DES/` (`DES-SN5YR_DES_HEAD.FITS.gz`, `DES-SN5YR_DES_PHOT.FITS.gz`), main branch at commit `48de863` (11 December 2025); light curves from Sánchez et al. (2024). The tagged releases contain an older HEAD file, so check out this commit rather than a tag. | Decompress both files into the repository root (next to `des_scipy_extraction.py`), run `python des_scipy_extraction.py`, then copy the output `des_superphot_features_SCIPY.csv` to `<data root>/DES dataset/des_superphot_features.csv` |
| ZTF BTS | Export the catalogue from the BTS Sample Explorer (https://sites.astro.caltech.edu/ztf/bts/explorer.php) as CSV to `phase4/bts_catalogue.csv`; light curves from ALeRCE (Förster et al. 2021) | `python phase4/ztf_download.py` then `python phase4/ztf_extract.py` |
| YSE DR1 | Zenodo record 7317476, `yse_dr1_zenodo.tar.gz` (Aleo et al. 2023) | `bash data/download_yse.sh` then `python phase7/yse_extract.py` |

The exact object identifiers used in every split and target sample are in `splits/`
(regenerate with `python data/export_splits.py`).

## Note on the PLAsTiCC extraction script

`extraction/plasticc_scipy_extraction.py` is the original code used for the paper, recovered from the
Kaggle notebook `extraction/plasticc_feature_extraction.ipynb` (kept with its outputs: 7,848 objects
read, 7,843 fitted, 5 failed). Only the input/output paths were made configurable. The features are
written in object_id order, which fixes the train/test splits; if a run is interrupted and resumed from
its checkpoint, sort the output by object_id. `extraction/compare_plasticc_features.py` compares a
re-run with the table used for the paper.