"""
DES light-curve feature extraction via scipy.optimize.curve_fit -- same
Villar-model fitter as plasticc_scipy_extraction.py, applied to DES's
SNANA FITS format instead of PLAsTiCC's CSV format. Reuses the FITS
loading logic from your existing des_superphot_local.py (load_fits,
build_light_curves) so the object set and quality cuts are identical to
your already-verified DES pipeline -- only the FITTING METHOD changes,
so this gives you a clean before/after comparison against your existing
superphot_plus-based des_superphot_features.csv.

Run this AFTER checking the validation comparison in
compare_fitting_methods.py -- only fully switch DES over if the
validation shows the two methods disagree meaningfully.
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit
from multiprocessing import Pool

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # same folder as this script

CONFIG = {
    "head_fits":         os.path.join(BASE_DIR, "DES-SN5YR_DES_HEAD.FITS"),
    "phot_fits":         os.path.join(BASE_DIR, "DES-SN5YR_DES_PHOT.FITS"),
    "output_path":       os.path.join(BASE_DIR, "des_superphot_features_SCIPY.csv"),
    "min_observations":   10,
    "min_bands":           2,
    "min_band_obs":        4,
    "flux_err_threshold":  0.0,
    "flux_clip_min":      -1e5,
    "flux_clip_max":       1e7,
    "n_workers":            os.cpu_count(),
    "photflag_detected":    4096,
}

# DES SNTYPE meanings: DES-SN5YR data release docs, 0_DATA "SNTYPE nomenclature".
# FIXED: SLSN-I (41, 141) -> 95 (PLAsTiCC SLSN-I; 65 is M-dwarf); SNIIb (23) -> SNII (42);
# SNIa-pec (4) removed (PLAsTiCC SNIa are normal Ia). Codes ending in '?' (101, 129, 139,
# 141, 180) are uncertain classifications; phase1/ can exclude them (des_label_policy).
SNTYPE_TO_PLASTICC = {
    1: 90, 101: 90, 20: 42, 21: 42, 22: 42, 23: 42, 29: 42, 129: 42,
    32: 62, 33: 62, 39: 62, 139: 62, 41: 95, 141: 95, 80: 88, 180: 88, 81: 15,
}
AMBIGUOUS_SNTYPES = {5, 122, 66, 82}
DES_BAND_TO_ZTF = {"g": "ZTF_g", "r": "ZTF_r"}  # matches the already-fixed mapping


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
            "hostgal_logmass": head_data["HOSTGAL_LOGMASS"].astype(float),
            "redshift_final":  head_data["REDSHIFT_FINAL"].astype(float),
            "hostgal_photoz":  head_data["HOSTGAL_PHOTOZ"].astype(float),
        })
    with fits.open(phot_path) as hdul:
        phot_data = hdul[1].data
        phot_arr = {
            "mjd":        np.array(phot_data["MJD"], dtype=np.float64),
            "band":       np.array(phot_data["BAND"], dtype=str),
            "fluxcal":    np.array(phot_data["FLUXCAL"], dtype=np.float32),
            "fluxcalerr": np.array(phot_data["FLUXCALERR"], dtype=np.float32),
            "photflag":   np.array(phot_data["PHOTFLAG"], dtype=np.int32),
        }
    labelled_mask = (head_df["sntype"] != 0) & ~head_df["sntype"].isin(AMBIGUOUS_SNTYPES)
    print(f"  HEAD: {len(head_df):,} SNe | Spec-confirmed: {labelled_mask.sum():,}")
    # FIX: only keep spectroscopically-confirmed objects -- without this,
    # this script was fitting all ~19,706 photometric-only candidates
    # instead of your actual ~568-object labelled sample (28x more work
    # than needed, and the source of the 7-hour estimate).
    head_df = head_df[labelled_mask].reset_index(drop=True)
    return head_df, phot_arr


def build_light_curves(head_df, phot_arr, cfg):
    print("Step 2: Building per-object light curves...")
    light_curves = {}
    for _, row in tqdm(head_df.iterrows(), total=len(head_df), desc="  Slicing", unit="SN"):
        snid = int(row["snid"])
        ptr_min, ptr_max = int(row["ptrobs_min"]) - 1, int(row["ptrobs_max"])
        mjd = phot_arr["mjd"][ptr_min:ptr_max]
        band = np.array([b.strip() for b in phot_arr["band"][ptr_min:ptr_max]])
        flux = phot_arr["fluxcal"][ptr_min:ptr_max].astype(np.float64)
        flux_err = phot_arr["fluxcalerr"][ptr_min:ptr_max].astype(np.float64)

        valid = mjd != -777.0
        mjd, band, flux, flux_err = mjd[valid], band[valid], flux[valid], flux_err[valid]
        good_err = flux_err > cfg["flux_err_threshold"]
        mjd, band, flux, flux_err = mjd[good_err], band[good_err], flux[good_err], flux_err[good_err]
        flux = np.clip(flux, cfg["flux_clip_min"], cfg["flux_clip_max"])
        flux_err = np.clip(flux_err, 1e-6, None)
        known = np.isin(band, list(DES_BAND_TO_ZTF.keys()))
        mjd, band, flux, flux_err = mjd[known], band[known], flux[known], flux_err[known]

        if len(mjd) < cfg["min_observations"] or np.unique(band).size < cfg["min_bands"]:
            continue

        # FIX: redshift sentinel (-999) handling, same as feature_engineering_FIXED.py
        z = float(row["redshift_final"]) if not np.isnan(row["redshift_final"]) else float(row["hostgal_photoz"])
        if z < -1:
            z = np.nan

        idx = np.argsort(mjd)
        mjd, band, flux, flux_err = mjd[idx], band[idx], flux[idx], flux_err[idx]
        light_curves[snid] = {
            "snid": snid, "sntype": int(row["sntype"]),
            "plasticc_class": SNTYPE_TO_PLASTICC.get(int(row["sntype"]), -1),
            "redshift": z, "hostgal_logmass": float(row["hostgal_logmass"]),
            "time": mjd - mjd[0], "flux": flux, "flux_err": flux_err, "band": band,
        }
    print(f"  Built: {len(light_curves):,}")
    return light_curves


def _villar_model(t, A, beta, gamma, t0, tau_rise, tau_fall):
    t = np.asarray(t, dtype=np.float64)
    # Clip exponent arguments to avoid overflow (exp(700) is near float64's
    # max) -- this happens routinely during curve_fit's search over extreme
    # parameter values and was silently making every fit slower by forcing
    # the optimizer to recover from NaN/inf residuals.
    exp_arg_rise = np.clip(-(t - t0) / tau_rise, -700, 700)
    rise_term = 1.0 / (1.0 + np.exp(exp_arg_rise))
    is_rise = t < (t0 + gamma)
    f_rise = A * (1.0 - beta * (t - t0)) * rise_term
    exp_arg_decline = np.clip(-(t - t0 - gamma) / tau_fall, -700, 700)
    f_decline = A * (1.0 - beta * gamma) * np.exp(exp_arg_decline) * rise_term
    return np.where(is_rise, f_rise, f_decline)


def _fit_one_band(t, flux, flux_err):
    """
    Fits on flux NORMALIZED to O(1) scale, then rescales A and extra_sigma
    back to real flux units afterward. Raw DES flux values run into the
    thousands-to-tens-of-thousands, which makes curve_fit's optimizer
    poorly conditioned and much slower to converge (or burn the full
    maxfev budget without converging, then retry -- compounding badly
    across 2 bands x 3 guesses x every object). beta/gamma/t0/tau_rise/
    tau_fall are scale-invariant shape params, so only A and extra_sigma
    need rescaling back.
    """
    peak_idx = np.argmax(flux)
    scale = max(abs(flux[peak_idx]), 1e-6)
    flux_n = flux / scale
    flux_err_n = flux_err / scale

    A0 = max(flux_n[peak_idx], 1.0)
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
            popt, _ = curve_fit(_villar_model, t, flux_n, p0=p0, sigma=flux_err_n,
                                 absolute_sigma=True, bounds=(bounds_lo, bounds_hi), maxfev=2000)
            resid = flux_n - _villar_model(t, *popt)
            A, beta, gamma, t0, tau_rise, tau_fall = popt
            return {"A": float(A) * scale, "beta": float(beta), "gamma": float(gamma), "t_0": float(t0),
                    "tau_rise": float(tau_rise), "tau_fall": float(tau_fall),
                    "extra_sigma": float(np.std(resid)) * scale}
        except (RuntimeError, ValueError):
            continue
    return None


def _fit_single_curve_des(lc):
    try:
        band_results = {}
        for filt, tag in DES_BAND_TO_ZTF.items():
            mask = lc["band"] == filt
            if mask.sum() < CONFIG["min_band_obs"]:
                return None
            fit = _fit_one_band(lc["time"][mask], lc["flux"][mask], lc["flux_err"][mask])
            if fit is None:
                return None
            band_results[tag] = fit

        record = {
            "snid": lc["snid"], "sntype": lc["sntype"], "plasticc_class": lc["plasticc_class"],
            "redshift": lc["redshift"], "hostgal_logmass": lc["hostgal_logmass"],
        }
        for tag, params in band_results.items():
            for pname, pval in params.items():
                record[f"{pname}_{tag}"] = pval
        record["amplitude"] = float(np.max(lc["flux"]) - np.min(lc["flux"]))
        record["std_flux"] = float(np.std(lc["flux"]))
        record["time_span"] = float(np.max(lc["time"]) - np.min(lc["time"]))
        return record
    except Exception as e:
        print(f"[FIT ERROR] snid={lc['snid']}: {type(e).__name__}: {e!r}")
        return None


def run_pipeline(cfg=CONFIG):
    head_df, phot_arr = load_fits(cfg["head_fits"], cfg["phot_fits"])
    light_curves = build_light_curves(head_df, phot_arr, cfg)
    print("Step 3: Villar-model curve_fit per object...")
    with Pool(cfg["n_workers"]) as pool:
        results = list(tqdm(pool.imap(_fit_single_curve_des, list(light_curves.values())),
                             total=len(light_curves), desc="  Fitting", unit="lc"))
    records = [r for r in results if r is not None]
    df = pd.DataFrame(records)
    df.to_csv(cfg["output_path"], index=False)
    print(f"\n  Saved: {cfg['output_path']}  Shape: {df.shape}")
    return df


if __name__ == "__main__":
    df = run_pipeline(CONFIG)
    print(df.head())