import warnings, os, sys, shutil, subprocess
warnings.filterwarnings("ignore")

from unittest.mock import MagicMock
for mod in [
    "healpy", "_healpy_pixel_lib", "_healpy_sph_transform_lib",
    "dustmaps", "dustmaps.config", "dustmaps.sfd",
    "antares_client", "antares_client.models",
    "antares_client.search", "antares_client.exceptions",
    "tensorflow_probability", "tensorflow_probability.substrates",
]:
    sys.modules[mod] = MagicMock()

import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

# =============================================================================
# CONFIGURATION  ← only section you need to edit
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # same folder as this script

CONFIG = {
    "head_fits":         os.path.join(BASE_DIR, "DES-SN5YR_DES_HEAD.FITS"),
    "phot_fits":         os.path.join(BASE_DIR, "DES-SN5YR_DES_PHOT.FITS"),
    "output_labelled":   os.path.join(BASE_DIR, "des_superphot_features.csv"),
    "output_unlabelled": os.path.join(BASE_DIR, "des_unlabelled_features.csv"),
    "checkpoint_path":   os.path.join(BASE_DIR, "des_checkpoint_features.csv"),
    "flat_df_path":      os.path.join(BASE_DIR, "des_flat_pipeline_df.csv"),
    "save_every":         50,
    "min_observations":   10,
    "min_bands":           2,
    "flux_err_threshold":  0.0,
    "flux_clip_min":      -1e5,
    "flux_clip_max":       1e7,
    "n_workers":           os.cpu_count(),  # uses all your cores automatically
    "photflag_detected":   4096,
}

# =============================================================================
# CONSTANTS
# =============================================================================
# DES SNTYPE meanings: DES-SN5YR data release docs, 0_DATA "SNTYPE nomenclature".
# FIXED: SLSN-I (41, 141) -> 95 (PLAsTiCC SLSN-I; 65 is M-dwarf); SNIIb (23) -> SNII (42);
# SNIa-pec (4) removed (PLAsTiCC SNIa are normal Ia). Codes ending in '?' (101, 129, 139,
# 141, 180) are uncertain classifications; phase1/ can exclude them (des_label_policy).
SNTYPE_TO_PLASTICC = {
    1: 90, 101: 90, 20: 42, 21: 42, 22: 42, 23: 42, 29: 42, 129: 42,
    32: 62, 33: 62, 39: 62, 139: 62, 41: 95, 141: 95, 80: 88, 180: 88, 81: 15,
}
AMBIGUOUS_SNTYPES = {5, 122, 66, 82}
PLASTICC_CLASS_NAMES = {
    6: "mu-lens Single", 15: "TDE", 16: "EB", 42: "SNII", 52: "SNIax",
    53: "Mira", 62: "SNIbc", 64: "KN", 65: "M-dwarf", 67: "SNIa-91bg",
    88: "AGN", 90: "SNIa", 92: "RRL", 95: "SLSN-I",
}
DES_BAND_TO_ZTF = {"g": "ZTF_g", "r": "ZTF_r"}
DES_BAND_TO_INT = {"g": 1, "r": 2}

# =============================================================================
# SETUP — dustmaps + superphot + priors
# =============================================================================
def setup():
    try:
        from superphot_plus.samplers.dynesty_sampler import DynestySampler
        from superphot_plus.priors import generate_priors
        print("superphot-plus ready ✓")
    except Exception as e:
        print(f"[ERROR] superphot-plus failed to import: {e}")
        print("Make sure you ran the pip install commands in the docstring above.")
        sys.exit(1)

    # find priors dir (works for any Python/venv path)
    import superphot_plus
    priors_dir = os.path.join(os.path.dirname(superphot_plus.__file__), "priors")
    os.makedirs(priors_dir, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/VTDA-Group/superphot-plus/main/src/superphot_plus/priors"
    for fname in ["priors_ZTF_g.csv", "priors_ZTF_r.csv"]:
        fpath = os.path.join(priors_dir, fname)
        if not os.path.exists(fpath):
            print(f"Downloading {fname}...")
            subprocess.run(["wget", "-q", "-O", fpath, f"{base_url}/{fname}"], check=True)
            print(f"  {fname} ✓")
        else:
            print(f"  {fname} already exists ✓")

    print(f"Using {CONFIG['n_workers']} CPU workers")
    print("Setup complete ✓\n")


# =============================================================================
# STEP 1 — Load FITS
# =============================================================================
def load_fits(head_path, phot_path):
    from astropy.io import fits
    print("Step 1: Loading FITS files...")
    with fits.open(head_path) as hdul:
        head_data = hdul[1].data
        head_df = pd.DataFrame({
            "snid":            np.array([int(s.strip()) for s in head_data["SNID"]]),
            "sntype":          head_data["SNTYPE"].astype(int),
            "ptrobs_min":      head_data["PTROBS_MIN"].astype(int),
            "ptrobs_max":      head_data["PTROBS_MAX"].astype(int),
            "hostgal_photoz":  head_data["HOSTGAL_PHOTOZ"].astype(float),
            "hostgal_logmass": head_data["HOSTGAL_LOGMASS"].astype(float),
            "peakmjd":         head_data["PEAKMJD"].astype(float),
            "redshift_final":  head_data["REDSHIFT_FINAL"].astype(float),
            "nobs":            head_data["NOBS"].astype(int),
        })
    with fits.open(phot_path) as hdul:
        phot_data = hdul[1].data
        phot_arr = {
            "mjd":        np.array(phot_data["MJD"],        dtype=np.float64),
            "band":       np.array(phot_data["BAND"],        dtype=str),
            "fluxcal":    np.array(phot_data["FLUXCAL"],    dtype=np.float32),
            "fluxcalerr": np.array(phot_data["FLUXCALERR"], dtype=np.float32),
            "photflag":   np.array(phot_data["PHOTFLAG"],   dtype=np.int32),
        }
    labelled_mask = (head_df["sntype"] != 0) & ~head_df["sntype"].isin(AMBIGUOUS_SNTYPES)
    print(f"  HEAD: {len(head_df):,} SNe | Spec-confirmed: {labelled_mask.sum():,} | Unlabelled: {(head_df['sntype']==0).sum():,}")
    return head_df, phot_arr


# =============================================================================
# STEP 2 — Build light curves
# =============================================================================
def build_light_curves(head_df, phot_arr, cfg):
    print("Step 2: Building per-object light curves...")
    light_curves, skipped_short, skipped_bands = {}, 0, 0
    for _, row in tqdm(head_df.iterrows(), total=len(head_df), desc="  Slicing", unit="SN"):
        snid    = int(row["snid"])
        ptr_min = int(row["ptrobs_min"]) - 1
        ptr_max = int(row["ptrobs_max"])
        mjd      = phot_arr["mjd"][ptr_min:ptr_max]
        band     = np.array([b.strip() for b in phot_arr["band"][ptr_min:ptr_max]])
        flux     = phot_arr["fluxcal"][ptr_min:ptr_max].astype(np.float64)
        flux_err = phot_arr["fluxcalerr"][ptr_min:ptr_max].astype(np.float64)
        photflag = phot_arr["photflag"][ptr_min:ptr_max]
        valid = mjd != -777.0
        mjd, band, flux, flux_err, photflag = mjd[valid], band[valid], flux[valid], flux_err[valid], photflag[valid]
        good_err = flux_err > cfg["flux_err_threshold"]
        mjd, band, flux, flux_err, photflag = mjd[good_err], band[good_err], flux[good_err], flux_err[good_err], photflag[good_err]
        flux     = np.clip(flux,     cfg["flux_clip_min"], cfg["flux_clip_max"])
        flux_err = np.clip(flux_err, 1e-6, None)
        known = np.isin(band, list(DES_BAND_TO_ZTF.keys()))
        mjd, band, flux, flux_err, photflag = mjd[known], band[known], flux[known], flux_err[known], photflag[known]
        if len(mjd) < cfg["min_observations"]:
            skipped_short += 1
            continue
        if np.unique(band).size < cfg["min_bands"]:
            skipped_bands += 1
            continue
        idx = np.argsort(mjd)
        mjd, band, flux, flux_err, photflag = mjd[idx], band[idx], flux[idx], flux_err[idx], photflag[idx]
        light_curves[snid] = {
            "snid": snid, "sntype": int(row["sntype"]),
            "plasticc_class": SNTYPE_TO_PLASTICC.get(int(row["sntype"]), -1),
            "redshift": float(row["redshift_final"]) if not np.isnan(row["redshift_final"]) else float(row["hostgal_photoz"]),
            "hostgal_logmass": float(row["hostgal_logmass"]),
            "time": mjd - mjd[0], "mjd": mjd, "flux": flux, "flux_err": flux_err, "band": band,
            "detected": ((photflag & cfg["photflag_detected"]) > 0).astype(int),
            "passband_int": np.array([DES_BAND_TO_INT.get(b, 0) for b in band], dtype=int),
        }
    print(f"  Built: {len(light_curves):,} | Skipped short: {skipped_short:,} | Skipped bands: {skipped_bands:,}")
    return light_curves


# =============================================================================
# STEP 3 — Superphot fitting (worker function — must be at module level for multiprocessing)
# =============================================================================
def _fit_single_curve_des(lc):
    try:
        from superphot_plus.samplers.dynesty_sampler import DynestySampler
        from superphot_plus.priors import generate_priors

        class SafeDynestySampler(DynestySampler):
            def score(self, X, y, **kwargs): return np.nan

        valid_indices = [i for i, b in enumerate(lc["band"]) if b in DES_BAND_TO_ZTF]
        if len(valid_indices) == 0:
            return None

        lc_time = lc["time"][valid_indices]
        lc_flux = lc["flux"][valid_indices]
        lc_err  = lc["flux_err"][valid_indices]
        mapped_bands = np.array([DES_BAND_TO_ZTF[b] for b in lc["band"][valid_indices]])

        MAX_OBS = 100
        if len(lc_time) > MAX_OBS:
            idx = np.linspace(0, len(lc_time) - 1, MAX_OBS, dtype=int)
            time, flux, flux_err, bands = lc_time[idx], lc_flux[idx], lc_err[idx], mapped_bands[idx]
        else:
            time, flux, flux_err, bands = lc_time, lc_flux, lc_err, mapped_bands

        band_counts = {b: int(np.sum(bands == b)) for b in np.unique(bands)}
        valid_bands = [b for b, c in band_counts.items() if c >= 3]
        if len(valid_bands) < 2:
            return None

        mask = np.isin(bands, valid_bands)
        time, flux, flux_err, bands = time[mask], flux[mask], flux_err[mask], bands[mask]
        priors  = generate_priors(list(np.unique(bands)))
        sampler = SafeDynestySampler(priors=priors, verbose=False, nlive=100, max_iter=500)
        sampler.fit(np.column_stack([time, bands, flux_err]), flux)
        params = sampler.result.fit_parameters.median().to_dict()

        return {
            "snid": lc["snid"], "sntype": lc["sntype"], "plasticc_class": lc["plasticc_class"],
            "redshift": lc["redshift"], "hostgal_logmass": lc["hostgal_logmass"],
            **{k: float(v) for k, v in params.items()},
            "amplitude": float(np.max(flux) - np.min(flux)),
            "std_flux":  float(np.std(flux)),
            "time_span": float(np.max(time) - np.min(time)),
        }
    except Exception as e:
        print(f"[FIT ERROR] snid={lc['snid']}: {e}")
        return None


def _append_checkpoint(buffer, path):
    pd.DataFrame(buffer).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def run_superphot_features(light_curves, cfg):
    print("Step 3: Superphot+ fitting...")
    checkpoint_path = cfg["checkpoint_path"]
    done_ids = set()
    if os.path.exists(checkpoint_path):
        done_ids = set(pd.read_csv(checkpoint_path)["snid"].tolist())
        print(f"  Resuming — {len(done_ids):,} already done")
    lc_list = sorted([lc for lc in light_curves.values() if lc["snid"] not in done_ids], key=lambda lc: 0 if lc['sntype'] != 0 else 1)
    print(f"  Remaining: {len(lc_list):,}")
    if not lc_list:
        print("  All done — loading from checkpoint.")
        return pd.read_csv(checkpoint_path).to_dict("records")
    buffer, n_failed = [], 0
    with Pool(cfg.get("n_workers", 4)) as pool:
        for i, result in enumerate(tqdm(
            pool.imap(_fit_single_curve_des, lc_list),
            total=len(lc_list), desc="  Fitting", unit="lc"
        )):
            if result is not None:
                buffer.append(result)
            else:
                n_failed += 1
            if (i + 1) % cfg["save_every"] == 0 and buffer:
                _append_checkpoint(buffer, checkpoint_path)
                buffer = []
    if buffer:
        _append_checkpoint(buffer, checkpoint_path)
    all_records = pd.read_csv(checkpoint_path).to_dict("records") if os.path.exists(checkpoint_path) else []
    print(f"  Done — {len(all_records):,} successful, {n_failed:,} failed")
    return all_records


# =============================================================================
# STEP 4 — Build and split
# =============================================================================
def build_and_split(records):
    print("Step 4: Building feature table...")
    df = pd.DataFrame(records)
    id_cols = ["snid", "sntype", "plasticc_class", "redshift", "hostgal_logmass"]
    df = df[id_cols + [c for c in df.columns if c not in id_cols]]
    labelled_df   = df[df["sntype"] != 0].copy()
    unlabelled_df = df[df["sntype"] == 0].copy()
    print(f"  Total: {len(df):,} | Labelled: {len(labelled_df):,} | Unlabelled: {len(unlabelled_df):,} | Features: {len(df.columns)-5}")
    if len(labelled_df) > 0:
        print("  PLAsTiCC class breakdown:")
        for cls, cnt in labelled_df["plasticc_class"].value_counts().items():
            print(f"    class {int(cls):2d} ({PLASTICC_CLASS_NAMES.get(int(cls), 'unknown'):15s}): {cnt}")
    return labelled_df, unlabelled_df


def build_pipeline_dataframe(light_curves, cfg):
    print("Building flat pipeline DataFrame...")
    rows = []
    for snid, lc in tqdm(light_curves.items(), desc="  Flattening", unit="SN"):
        rows.append(pd.DataFrame({
            "object_id": snid, "mjd": lc["mjd"], "flux": lc["flux"], "flux_err": lc["flux_err"],
            "passband": lc["passband_int"], "band_str": lc["band"], "detected": lc["detected"],
            "target": lc["plasticc_class"], "sntype": lc["sntype"],
            "hostgal_photoz": lc["redshift"], "hostgal_logmass": lc["hostgal_logmass"],
        }))
    df = pd.concat(rows, ignore_index=True)
    print(f"  Shape: {df.shape}")
    return df


# =============================================================================
# MAIN
# =============================================================================
def run_pipeline(cfg=CONFIG):
    print("=" * 65)
    print("  DES-SN5YR Superphot Feature Extraction Pipeline")
    print("=" * 65)

    head_df, phot_arr          = load_fits(cfg["head_fits"], cfg["phot_fits"])
    light_curves               = build_light_curves(head_df, phot_arr, cfg)

    if not light_curves:
        raise ValueError("No light curves survived filtering.")

    records                    = run_superphot_features(light_curves, cfg)
    labelled_df, unlabelled_df = build_and_split(records)

    labelled_df.to_csv(cfg["output_labelled"],   index=False)
    unlabelled_df.to_csv(cfg["output_unlabelled"], index=False)
    flat_df = build_pipeline_dataframe(light_curves, cfg)
    flat_df.to_csv(cfg["flat_df_path"], index=False)

    print(f"\n  Saved: {cfg['output_labelled']}")
    print(f"  Saved: {cfg['output_unlabelled']}")
    print(f"  Saved: {cfg['flat_df_path']}")
    print("\n  Pipeline complete ✓")

    return {
        "light_curves":   light_curves,
        "labelled_df":    labelled_df,
        "unlabelled_df":  unlabelled_df,
        "flat_df":        flat_df,
        "all_records":    records,
    }


if __name__ == "__main__":
    setup()
    results = run_pipeline(CONFIG)
    print(results["labelled_df"].head())