# Phase 6: probing the ZTF failure

Read and commit `PREREGISTRATION_6.md` first (including its amendment). Run `bash run_phase6.sh`
from the `xai` folder in `des_env`. Needs the Phase 2 and Phase 4 results in place.

| Block | Files | Question |
| --- | --- | --- |
| D1 | `D1_*` configs (16) | Which feature on the shared scale breaks ZTF: redshift alone? Does also sharing brightness features fix it? |
| D2 (confirmatory) | `D2_*` configs (48) | Two-factor controlled test: population shift x synthetic instrument shift (cadence + noise, or flux calibration), with ZTF-like class priors |
| D3 | `analysis_saved.py` | Does correcting class priors (EM; and, exploratory, the true priors) close the hybrid rule's ZTF deficit? |
| D4 | `analysis_saved.py` | Per-class recall by redshift bin on ZTF |
| D5 | `D5_*` configs (16) | How many unlabelled target objects do the scaling statistics need? |

Outputs: `tables/*.md|tex`, `results/verdicts_6.json`, `results/D3_label_shift.json`.
