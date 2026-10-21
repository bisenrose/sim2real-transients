"""
Phase 4.1, step 2: fit the Villar model to ZTF light curves, exactly as for DES.

The fitting code is copied verbatim from des_scipy_extraction.py (same model, bounds, starting
guesses, maxfev, cuts) so ZTF and DES features come from the same procedure.
Window (pre-registered): 100 days before to 300 days after the brightest point.

Usage:  python ztf_extract.py      -> <data root>/ZTF dataset/ztf_superphot_features.csv
"""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase1"))
import common as C  # noqa: E402

ZDIR = os.path.join(C.ROOT, "ZTF dataset")
CFG = {"min_observations": 10, "min_band_obs": 4, "flux_clip_min": -1e5, "flux_clip_max": 1e7,
       "window_before": 100.0, "window_after": 300.0}
BANDS = {1: "ZTF_g", 2: "ZTF_r"}


# ---- copied from des_scipy_extraction.py (do not change one without the other) --------------
def _villar_model(t, A, beta, gamma, t0, tau_rise, tau_fall):
    t = np.asarray(t, dtype=np.float64)
    exp_arg_rise = np.clip(-(t - t0) / tau_rise, -700, 700)
    rise_term = 1.0 / (1.0 + np.exp(exp_arg_rise))
    is_rise = t < (t0 + gamma)
    f_rise = A * (1.0 - beta * (t - t0)) * rise_term
    exp_arg_decline = np.clip(-(t - t0 - gamma) / tau_fall, -700, 700)
    f_decline = A * (1.0 - beta * gamma) * np.exp(exp_arg_decline) * rise_term
    return np.where(is_rise, f_rise, f_decline)


def _fit_one_band(t, flux, flux_err):
    peak_idx = np.argmax(flux)
    scale = max(abs(flux[peak_idx]), 1e-6)
    flux_n = flux / scale
    flux_err_n = flux_err / scale
    A0 = max(flux_n[peak_idx], 1.0)
    t0_0 = t[peak_idx]
    t_min, t_max = t.min(), t.max()
    guesses = [
        (A0, 0.001, 10.0, t0_0, 2.0, 30.0),
        (A0, 0.001, 60.0, t0_0, 5.0, 80.0),
        (A0, 0.005, 5.0, t0_0 - 5.0, 1.0, 15.0),
    ]
    bounds_lo = [0.0, 0.0, 1.0, t_min - 50.0, 0.01, 1.0]
    bounds_hi = [A0 * 100.0, 0.01, 300.0, t_max + 50.0, 50.0, 300.0]
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
# -------------------------------------------------------------------------------------------


def fit_object(row):
    path = os.path.join(ZDIR, "lightcurves", f"{row['ztfid']}.csv")
    if not os.path.exists(path):
        return None
    lc = pd.read_csv(path)
    if len(lc) == 0:
        return None
    peak = lc["mjd"].iloc[int(np.argmax(lc["flux"].values))]
    lc = lc[(lc["mjd"] >= peak - CFG["window_before"]) & (lc["mjd"] <= peak + CFG["window_after"])]
    if len(lc) < CFG["min_observations"] or lc["fid"].nunique() < 2:
        return None
    t = lc["mjd"].values - lc["mjd"].values.min()
    flux = np.clip(lc["flux"].values.astype(float), CFG["flux_clip_min"], CFG["flux_clip_max"])
    err = np.clip(lc["flux_err"].values.astype(float), 1e-6, None)
    rec = {"ztfid": row["ztfid"], "bts_type": row["bts_type"], "plasticc_class": int(row["plasticc_class"]),
           "redshift": float(row["redshift"]), "lc_source": lc["source"].iloc[0] if "source" in lc else ""}
    for fid, tag in BANDS.items():
        m = lc["fid"].values == fid
        if m.sum() < CFG["min_band_obs"]:
            return None
        fit = _fit_one_band(t[m], flux[m], err[m])
        if fit is None:
            return None
        for k, v in fit.items():
            rec[f"{k}_{tag}"] = v
    rec["amplitude"] = float(flux.max() - flux.min())
    rec["std_flux"] = float(np.std(flux))
    rec["time_span"] = float(t.max() - t.min())
    return rec


def main():
    cat = pd.read_csv(os.path.join(ZDIR, "bts_selected.csv"))
    rows = cat.to_dict("records")
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        recs = [r for r in pool.imap(fit_object, rows, chunksize=20) if r is not None]
    df = pd.DataFrame(recs)
    tmp = C.ZTF_FEATURES + ".tmp"; df.to_csv(tmp, index=False); os.replace(tmp, C.ZTF_FEATURES)
    print(f"Fitted {len(df)} of {len(cat)} ZTF objects -> {C.ZTF_FEATURES}")
    if len(df):
        print(df["plasticc_class"].map(C.SHARED).value_counts().to_string())
        print("light-curve source:", df["lc_source"].value_counts().to_dict())


if __name__ == "__main__":
    main()
