# Phase 7: YSE DR1 as a third, held-out survey

1. Commit `PREREGISTRATION_7.md` to git.
2. Make sure the YSE light curves are extracted in `DES dataset/YSE dataset/` (the `*.snana.dat` files).
3. From the `xai` folder: `bash run_phase7.sh`.

| Step | File | Output |
| --- | --- | --- |
| Fit | `yse_extract.py` | `DES dataset/YSE dataset/yse_superphot_features.csv` (Pan-STARRS g/r, DES fit code, BTS overlap flagged) |
| Runs | `configs/` (36) | `results/yse_*.json`: 4 models x 7 preprocessings on all YSE; 2 models x 4 on YSE without BTS objects |
| Analysis | `analysis_7.py` | `tables/yse_transfer.md`, `yse_no_bts_transfer.md`, `yse_near_far.md`, `yse_position.md`; `results/verdicts_7.json` |
