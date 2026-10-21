"""
Phase 4.1, step 1: download ZTF light curves for the BTS sample from ALeRCE.

Before running, export the BTS catalogue:
  1. Open https://sites.astro.caltech.edu/ztf/bts/explorer.php
  2. Choose the sample with all classified transients, set "Display as" to CSV, submit.
  3. Save the result as  phase4/bts_catalogue.csv
The script finds the ZTF ID, type and redshift columns whatever their exact names.

Usage:  pip install alerce
        python ztf_download.py [--catalogue bts_catalogue.csv] [--workers 4] [--source forced|detections]
Resumable: each light curve is cached in <data root>/ZTF dataset/lightcurves/<ZTFID>.csv
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase1"))
import common as C  # noqa: E402

LC_DIR = os.path.join(C.ROOT, "ZTF dataset", "lightcurves")
FID_BAND = {1: "g", 2: "r"}


def pick(df, names):
    low = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for n in names:
        k = n.lower().replace(" ", "").replace("_", "")
        if k in low:
            return low[k]
    raise KeyError(f"none of {names} in catalogue columns {list(df.columns)}")


def load_catalogue(path):
    cat = pd.read_csv(path)
    out = pd.DataFrame({"ztfid": cat[pick(cat, ["ZTFID", "ZTF ID", "ztf_id", "oid", "name"])].astype(str).str.strip(),
                        "bts_type": cat[pick(cat, ["type", "class", "classification"])].astype(str),
                        "redshift": pd.to_numeric(cat[pick(cat, ["redshift", "z"])], errors="coerce")})
    out["plasticc_class"] = C.ztf_labels(out).values
    keep = out["plasticc_class"].isin(list(C.SHARED)) & (out["redshift"] > 0)
    print(f"Catalogue: {len(out)} rows; {int(keep.sum())} in the five shared classes with a positive redshift")
    print(out[keep]["plasticc_class"].map(C.SHARED).value_counts().to_string())
    return out[keep].drop_duplicates("ztfid").reset_index(drop=True)


def _sign(v):
    s = str(v).strip().lower()
    return -1.0 if s in ("f", "-1", "-1.0", "false", "0") else 1.0


def to_flux(df, kind):
    """Standardise ALeRCE output to mjd, fid, flux, flux_err on the zero point 27.5 scale."""
    cols = {c.lower(): c for c in df.columns}
    fid = pd.to_numeric(df[cols["fid"]], errors="coerce")
    mjd = pd.to_numeric(df[cols["mjd"]], errors="coerce")
    if "forcediffimflux" in cols and "magzpsci" in cols:          # raw ZTF forced-photometry fluxes
        f = pd.to_numeric(df[cols["forcediffimflux"]], errors="coerce")
        fe = pd.to_numeric(df[cols.get("forcediffimfluxunc", cols["forcediffimflux"])], errors="coerce")
        zp = pd.to_numeric(df[cols["magzpsci"]], errors="coerce")
        k = 10 ** (-0.4 * (zp - 27.5)); flux, err = f * k, fe * k
    else:                                                          # magnitudes + sign
        mcol = "mag" if "mag" in cols else "magpsf"
        ecol = "e_mag" if "e_mag" in cols else "sigmapsf"
        m = pd.to_numeric(df[cols[mcol]], errors="coerce"); e = pd.to_numeric(df[cols[ecol]], errors="coerce")
        sign = df[cols["isdiffpos"]].map(_sign) if "isdiffpos" in cols else 1.0
        flux = sign * 10 ** (-0.4 * (m - 27.5)); err = np.abs(flux) * 0.4 * np.log(10) * e
    out = pd.DataFrame({"mjd": mjd, "fid": fid, "flux": flux, "flux_err": err, "source": kind})
    out = out[out["fid"].isin(FID_BAND) & np.isfinite(out["flux"]) & np.isfinite(out["flux_err"]) & (out["flux_err"] > 0)]
    return out.sort_values("mjd").reset_index(drop=True)


def fetch_one(client, oid, source, retries=3):
    path = os.path.join(LC_DIR, f"{oid}.csv")
    if os.path.exists(path):
        return oid, "cached"
    for attempt in range(retries):
        try:
            lc = None
            if source == "forced":
                try:
                    fp = client.query_forced_photometry(oid, format="pandas", survey="ztf")
                except TypeError:                       # older alerce client without `survey`
                    fp = client.query_forced_photometry(oid, format="pandas")
                if fp is not None and len(fp):
                    lc = to_flux(pd.DataFrame(fp), "forced")
            if lc is None or len(lc) < 10:
                try:
                    det = client.query_detections(oid, format="pandas", survey="ztf")
                except TypeError:
                    det = client.query_detections(oid, format="pandas")
                lc = to_flux(pd.DataFrame(det), "detections")
            tmp = path + ".tmp"; lc.to_csv(tmp, index=False); os.replace(tmp, path)
            return oid, lc["source"].iloc[0] if len(lc) else "empty"
        except Exception as e:  # network hiccups, rate limits
            if attempt == retries - 1:
                return oid, f"failed: {type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=os.path.join(HERE, "bts_catalogue.csv"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--source", choices=["forced", "detections"], default="forced")
    a = ap.parse_args()
    if not os.path.exists(a.catalogue):
        sys.exit(f"{a.catalogue} not found. Export it from the BTS Sample Explorer (see the top of this file).")
    from alerce.core import Alerce
    client = Alerce()
    cat = load_catalogue(a.catalogue)
    os.makedirs(LC_DIR, exist_ok=True)
    cat.to_csv(os.path.join(C.ROOT, "ZTF dataset", "bts_selected.csv"), index=False)
    status = {}
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(fetch_one, client, oid, a.source) for oid in cat["ztfid"]]
        for i, f in enumerate(as_completed(futs), 1):
            oid, st = f.result(); status[oid] = st
            if i % 100 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} light curves ({sum(s.startswith('failed') for s in status.values())} failed)", flush=True)
    json.dump(status, open(os.path.join(C.ROOT, "ZTF dataset", "download_status.json"), "w"), indent=1)
    counts = pd.Series([s if not s.startswith("failed") else "failed" for s in status.values()]).value_counts()
    print(counts.to_string())


if __name__ == "__main__":
    main()
