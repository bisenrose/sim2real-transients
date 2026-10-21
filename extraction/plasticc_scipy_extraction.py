# Original PLAsTiCC feature-extraction code used for the paper (recovered from the Kaggle notebook
# plasticc_feature_extraction.ipynb, cell 3). The fitting code below is unchanged; the only edit is that
# the input/output paths can be set on the command line instead of being fixed to Kaggle's folders:
#   python plasticc_scipy_extraction.py --data /path/to/PLAsTiCC-2018 --out plasticc_raw_superphot_features.csv
# Inputs: training_set.csv and training_set_metadata.csv from https://www.kaggle.com/c/PLAsTiCC-2018/data
"""
PLAsTiCC raw-light-curve -> feature extraction via scipy.optimize.curve_fit,
fitting the same Villar et al. (2019) parametric SN model that
superphot-plus uses internally -- WITHOUT superphot-plus's dependency
chain (snapi/dynesty/numpyro/jax/pyts/numba).

Same output schema as the superphot_plus-based version: A, beta, gamma,
t_0, tau_rise, tau_fall, extra_sigma per band (_ZTF_g / _ZTF_r), plus
amplitude/std_flux/time_span, real (unscaled) redshift, distmod, target.

Model (piecewise, per band):
    rise/plateau (t < t0 + gamma):
        f(t) = A * (1 - beta*(t-t0)) / (1 + exp(-(t-t0)/tau_rise))
    decline (t >= t0 + gamma):
        f(t) = A * (1 - beta*gamma) * exp(-(t-t0-gamma)/tau_fall)
               / (1 + exp(-(t-t0)/tau_rise))

Dependencies: numpy, scipy, pandas, tqdm. That's it.
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit
from multiprocessing import Pool

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "lc_csv":            "/kaggle/input/competitions/PLAsTiCC-2018/training_set.csv",
    "meta_csv":          "/kaggle/input/competitions/PLAsTiCC-2018/training_set_metadata.csv",
    "output_raw":        "/kaggle/working/plasticc_raw_superphot_features.csv",
    "checkpoint_path":   "/kaggle/working/plasticc_checkpoint_features.csv",
    "save_every":         50,
    "min_observations":   10,
    "min_band_obs":        4,   # need >=4 points per band for a 6-parameter fit
    "n_workers":           os.cpu_count(),
}

PASSBAND_TO_FILTER = {0: "u", 1: "g", 2: "r", 3: "i", 4: "z", 5: "y"}
FILTER_TO_TAG = {"g": "ZTF_g", "r": "ZTF_r"}  # keep same column-name convention


# =============================================================================
# STEP 1 -- Load PLAsTiCC CSVs (identical to the superphot_plus version)
# =============================================================================
def load_plasticc(lc_path, meta_path):
    print("Step 1: Loading PLAsTiCC training_set + metadata...")
    meta = pd.read_csv(meta_path)
    lc = pd.read_csv(lc_path)

    meta["redshift_used"] = meta["hostgal_specz"]
    missing_specz = meta["redshift_used"].isna()
    meta.loc[missing_specz, "redshift_used"] = meta.loc[missing_specz, "hostgal_photoz"]

    print(f"  Metadata: {len(meta):,} objects | "
          f"spec-z available: {(~missing_specz).sum():,} | "
          f"photo-z fallback: {missing_specz.sum():,}")
    print(f"  Light curve rows: {len(lc):,}")
    return meta, lc


# =============================================================================
# STEP 2 -- Build per-object light curves (g, r only)
# =============================================================================
def build_light_curves(meta, lc, cfg):
    print("Step 2: Building per-object light curves (g+r only)...")
    lc = lc[lc["passband"].isin([1, 2])].copy()  # g=1, r=2
    lc["filter"] = lc["passband"].map(PASSBAND_TO_FILTER)

    light_curves = {}
    skipped_short, skipped_bands = 0, 0
    meta_idx = meta.set_index("object_id")

    for obj_id, grp in tqdm(lc.groupby("object_id"), desc="  Grouping", unit="obj"):
        if obj_id not in meta_idx.index:
            continue
        row = meta_idx.loc[obj_id]

        mjd = grp["mjd"].values
        band = grp["filter"].values
        flux = grp["flux"].values.astype(np.float64)
        flux_err = grp["flux_err"].values.astype(np.float64)
        flux_err = np.clip(flux_err, 1e-6, None)

        if len(mjd) < cfg["min_observations"]:
            skipped_short += 1
            continue
        if np.unique(band).size < 2:
            skipped_bands += 1
            continue

        idx = np.argsort(mjd)
        mjd, band, flux, flux_err = mjd[idx], band[idx], flux[idx], flux_err[idx]

        target = row.get("target", -1)
        light_curves[obj_id] = {
            "object_id": obj_id,
            "target": int(target) if not pd.isna(target) else -1,
            "redshift": float(row["redshift_used"]) if not pd.isna(row["redshift_used"]) else np.nan,
            "hostgal_photoz": float(row["hostgal_photoz"]) if not pd.isna(row["hostgal_photoz"]) else np.nan,
            "distmod": float(row["distmod"]) if not pd.isna(row.get("distmod", np.nan)) else np.nan,
            "mwebv": float(row["mwebv"]) if "mwebv" in row and not pd.isna(row["mwebv"]) else np.nan,
            "time": mjd - mjd[0], "mjd": mjd, "flux": flux, "flux_err": flux_err, "band": band,
        }
    print(f"  Built: {len(light_curves):,} | Skipped short: {skipped_short:,} | Skipped <2 bands: {skipped_bands:,}")
    return light_curves


# =============================================================================
# STEP 3 -- Villar-model curve_fit per band (module-level for multiprocessing)
# =============================================================================
def _villar_model(t, A, beta, gamma, t0, tau_rise, tau_fall):
    t = np.asarray(t, dtype=np.float64)
    rise_term = 1.0 / (1.0 + np.exp(-(t - t0) / tau_rise))
    is_rise = t < (t0 + gamma)
    f_rise = A * (1.0 - beta * (t - t0)) * rise_term
    f_decline = A * (1.0 - beta * gamma) * np.exp(-(t - t0 - gamma) / tau_fall) * rise_term
    return np.where(is_rise, f_rise, f_decline)


def _fit_one_band(t, flux, flux_err):
    """
    Returns dict of fitted params, or None if the fit fails / doesn't
    converge. Tries a couple of initial-guess strategies before giving up,
    since curve_fit is sensitive to starting point for this nonlinear model.
    """
    peak_idx = np.argmax(flux)
    A0 = max(flux[peak_idx], 1.0)
    t0_0 = t[peak_idx]
    span = max(t.max() - t.min(), 1.0)
    t_min, t_max = t.min(), t.max()
    # Literature-informed bounds (Superphot+/de Soto et al. 2024, Table of priors),
    # NOT ad-hoc span-scaled guesses. This model is known to be degenerate/unstable
    # under plain least-squares without these physically-motivated constraints --
    # using them is what keeps curve_fit in the same basin the Bayesian fit lives in.
    guesses = [
        (A0, 0.001, 10.0,  t0_0,       2.0,  30.0),
        (A0, 0.001, 60.0,  t0_0,       5.0,  80.0),
        (A0, 0.005, 5.0,   t0_0 - 5.0, 1.0,  15.0),
    ]
    bounds_lo = [0.0,        0.0,   1.0,   t_min - 50.0, 0.01, 1.0]
    bounds_hi = [A0 * 100.0, 0.01,  300.0, t_max + 50.0, 50.0, 300.0]

    for p0 in guesses:
        try:
            popt, _ = curve_fit(
                _villar_model, t, flux, p0=p0,
                sigma=flux_err, absolute_sigma=True,
                bounds=(bounds_lo, bounds_hi),
                maxfev=5000,
            )
            resid = flux - _villar_model(t, *popt)
            extra_sigma = float(np.std(resid))
            A, beta, gamma, t0, tau_rise, tau_fall = popt
            return {
                "A": float(A), "beta": float(beta), "gamma": float(gamma),
                "t_0": float(t0), "tau_rise": float(tau_rise), "tau_fall": float(tau_fall),
                "extra_sigma": extra_sigma,
            }
        except (RuntimeError, ValueError):
            continue
    return None


def _fit_single_curve_plasticc(lc):
    try:
        band_results = {}
        for filt, tag in FILTER_TO_TAG.items():
            mask = lc["band"] == filt
            if mask.sum() < CONFIG["min_band_obs"]:
                return None
            t_b, f_b, e_b = lc["time"][mask], lc["flux"][mask], lc["flux_err"][mask]
            fit = _fit_one_band(t_b, f_b, e_b)
            if fit is None:
                return None
            band_results[tag] = fit

        record = {
            "object_id": lc["object_id"], "target": lc["target"],
            "redshift": lc["redshift"], "hostgal_photoz": lc["hostgal_photoz"],
            "distmod": lc["distmod"], "mwebv": lc["mwebv"],
        }
        for tag, params in band_results.items():
            for pname, pval in params.items():
                record[f"{pname}_{tag}"] = pval

        all_flux = lc["flux"]
        record["amplitude"] = float(np.max(all_flux) - np.min(all_flux))
        record["std_flux"] = float(np.std(all_flux))
        record["time_span"] = float(np.max(lc["time"]) - np.min(lc["time"]))
        return record

    except Exception as e:
        import traceback
        print(f"[FIT ERROR] object_id={lc['object_id']}: {type(e).__name__}: {e!r}")
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(CONFIG["output_raw"])), "plasticc_fit_errors.log"), "a") as f:
                f.write(f"\n{'='*60}\nobject_id={lc['object_id']}\n{traceback.format_exc()}\n")
        except Exception:
            pass
        return None


def _append_checkpoint(buffer, path):
    pd.DataFrame(buffer).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def run_fit_features(light_curves, cfg):
    print("Step 3: Villar-model curve_fit per object...")
    checkpoint_path = cfg["checkpoint_path"]
    done_ids = set()
    if os.path.exists(checkpoint_path):
        done_ids = set(pd.read_csv(checkpoint_path)["object_id"].tolist())
        print(f"  Resuming -- {len(done_ids):,} already done")
    lc_list = [lc for lc in light_curves.values() if lc["object_id"] not in done_ids]
    print(f"  Remaining: {len(lc_list):,}")
    if not lc_list:
        return pd.read_csv(checkpoint_path).to_dict("records")

    buffer, n_failed = [], 0
    with Pool(cfg.get("n_workers", 4)) as pool:
        for i, result in enumerate(tqdm(
            pool.imap(_fit_single_curve_plasticc, lc_list),
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
    print(f"  Done -- {len(all_records):,} successful, {n_failed:,} failed")
    return all_records


# =============================================================================
# MAIN
# =============================================================================
def run_pipeline(cfg=CONFIG):
    print("=" * 65)
    print("  PLAsTiCC RAW Feature Extraction (scipy curve_fit, no superphot-plus)")
    print("=" * 65)

    meta, lc = load_plasticc(cfg["lc_csv"], cfg["meta_csv"])
    light_curves = build_light_curves(meta, lc, cfg)
    if not light_curves:
        raise ValueError("No light curves survived filtering.")

    records = run_fit_features(light_curves, cfg)
    df = pd.DataFrame(records)
    df.to_csv(cfg["output_raw"], index=False)

    print(f"\n  Saved: {cfg['output_raw']}")
    print(f"  Shape: {df.shape}")
    print(f"  amplitude stats (should be REAL flux units):")
    print(df["amplitude"].describe())
    print(f"\n  redshift stats (should be REAL z, 0 to ~2-3):")
    print(df["redshift"].describe())
    print("\n  Pipeline complete (no dependency hell required) ✓")
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="folder with training_set.csv and training_set_metadata.csv")
    ap.add_argument("--out", default=None, help="output CSV (default: the Kaggle working path)")
    a = ap.parse_args()
    if a.data:
        CONFIG["lc_csv"] = os.path.join(a.data, "training_set.csv")
        CONFIG["meta_csv"] = os.path.join(a.data, "training_set_metadata.csv")
    if a.out:
        CONFIG["output_raw"] = a.out
        CONFIG["checkpoint_path"] = os.path.splitext(a.out)[0] + "_checkpoint.csv"
    df = run_pipeline(CONFIG)
    print(df.head())