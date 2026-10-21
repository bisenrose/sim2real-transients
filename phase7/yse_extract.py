"""
Phase 7: fit YSE DR1 light curves (Pan-STARRS1 g and r only) with the same code as DES and ZTF.

Input : <data root>/YSE dataset/*.snana.dat   (Zenodo 7317476, yse_dr1_zenodo.tar.gz, extracted)
Output: <data root>/YSE dataset/yse_superphot_features.csv
Rules are fixed in PREREGISTRATION_7.md.
Usage:  python yse_extract.py
"""
import glob, os, re, sys
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); XAI = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(XAI, "phase1")); sys.path.insert(0, os.path.join(XAI, "phase4"))
import common as C                     # noqa: E402
from ztf_extract import _fit_one_band  # noqa: E402  (verbatim copy of the DES fitting code)

YDIR = os.path.join(C.ROOT, "YSE dataset")
CFG = {"min_observations": 10, "min_band_obs": 4, "flux_clip_min": -1e5, "flux_clip_max": 1e7,
       "window_before": 100.0, "window_after": 300.0, "bands": {"g": "ZTF_g", "r": "ZTF_r"}}


def norm_name(s):
    s = str(s).strip().lower().replace(" ", "")
    for p in ("sn", "at"):
        if s.startswith(p) and len(s) > 4 and s[len(p):len(p) + 2].isdigit():
            s = s[len(p):]
    return s


def bts_names():
    path = os.path.join(XAI, "phase4", "bts_catalogue.csv")
    if not os.path.exists(path):
        print("WARNING: phase4/bts_catalogue.csv not found; in_bts will be False for every object")
        return set()
    cat = pd.read_csv(path)
    col = next((c for c in cat.columns if c.strip().lower() in ("iauid", "iau id", "iau_id", "tns", "tns_name")), None)
    return set(cat[col].dropna().map(norm_name)) if col else set()


def parse(path):
    head, obs, varlist = {}, [], None
    with open(path) as f:
        for line in f:
            if line.startswith("OBS:"):
                obs.append(line.split()[1:])
            elif line.startswith("VARLIST:"):
                varlist = line.split()[1:]
            elif ":" in line and not line.startswith("#"):
                k, v = line.split(":", 1); head[k.strip()] = v.strip()
    lc = pd.DataFrame(obs, columns=varlist) if obs and varlist else pd.DataFrame()
    return head, lc


def redshift(head):
    s = head.get("REDSHIFT_FINAL", "")
    m = re.match(r"\s*([-\d.eE+]+)", s)
    z = float(m.group(1)) if m else np.nan
    kind = s[s.find("(") + 1:s.find(")")] if "(" in s else ""
    spec = ("SPEC" in kind.upper() or "HOST" in kind.upper()) and "PHOT" not in kind.upper()
    return z, kind, spec


def fit_object(path, bts):
    head, lc = parse(path)
    rec = {"ysename": head.get("SNID", os.path.basename(path).split(".")[0]), "bts_type": head.get("SPEC_CLASS", "NA"),
           "spec_class_broad": head.get("SPEC_CLASS_BROAD", "NA")}
    rec["plasticc_class"] = int(C.ztf_labels(pd.DataFrame({"bts_type": [rec["bts_type"]]})).iloc[0])
    z, kind, spec = redshift(head)
    rec.update(redshift=z, redshift_kind=kind)
    try:
        rec["yse_peak_abs_mag"] = float(head.get("PEAK_ABS_MAG", "nan"))
    except ValueError:
        rec["yse_peak_abs_mag"] = np.nan
    rec["in_bts"] = norm_name(rec["ysename"]) in bts
    if rec["plasticc_class"] not in C.SHARED:
        return None, "class not shared"
    if not (spec and z > 0):
        return None, "no spectroscopic redshift"
    if lc.empty:
        return None, "no photometry"
    lc = lc[lc["FLT"].isin(CFG["bands"])].copy()
    for c in ("MJD", "FLUXCAL", "FLUXCALERR"):
        lc[c] = pd.to_numeric(lc[c], errors="coerce")
    lc = lc.dropna(subset=["MJD", "FLUXCAL", "FLUXCALERR"]).sort_values("MJD")
    if len(lc) == 0:
        return None, "no PS1 g/r points"
    peak = lc["MJD"].iloc[int(np.argmax(lc["FLUXCAL"].values))]
    lc = lc[(lc["MJD"] >= peak - CFG["window_before"]) & (lc["MJD"] <= peak + CFG["window_after"])]
    if len(lc) < CFG["min_observations"]:
        return None, "too few points"
    t = lc["MJD"].values - lc["MJD"].values.min()
    flux = np.clip(lc["FLUXCAL"].values.astype(float), CFG["flux_clip_min"], CFG["flux_clip_max"])
    err = np.clip(lc["FLUXCALERR"].values.astype(float), 1e-6, None)
    for b, tag in CFG["bands"].items():
        m = lc["FLT"].values == b
        if m.sum() < CFG["min_band_obs"]:
            return None, "too few points in a band"
        fit = _fit_one_band(t[m], flux[m], err[m])
        if fit is None:
            return None, "fit failed"
        for k, v in fit.items():
            rec[f"{k}_{tag}"] = v
    rec["amplitude"] = float(flux.max() - flux.min())
    rec["std_flux"] = float(np.std(flux))
    rec["time_span"] = float(t.max() - t.min())
    return rec, "ok"


def _work(args):
    return fit_object(*args)


def main():
    files = sorted(glob.glob(os.path.join(YDIR, "**", "*.snana.dat"), recursive=True))
    if not files:
        sys.exit(f"No *.snana.dat files under {YDIR}. Extract yse_dr1_zenodo.tar.gz there first.")
    bts = bts_names()
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        out = pool.map(_work, [(f, bts) for f in files], chunksize=20)
    recs = [r for r, s in out if r is not None]
    status = pd.Series([s for _, s in out]).value_counts()
    df = pd.DataFrame(recs)
    tmp = C.YSE_FEATURES + ".tmp"; df.to_csv(tmp, index=False); os.replace(tmp, C.YSE_FEATURES)
    print(f"{len(files)} light-curve files; fitted {len(df)} -> {C.YSE_FEATURES}")
    print(status.to_string())
    if len(df):
        print("classes:", df["plasticc_class"].map(C.SHARED).value_counts().to_dict())
        print("in BTS:", int(df["in_bts"].sum()), "of", len(df), "| redshift kinds:", df["redshift_kind"].value_counts().to_dict())


if __name__ == "__main__":
    main()
