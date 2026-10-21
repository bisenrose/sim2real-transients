"""
EXPLORATORY (written after seeing the first ZTF results; not pre-registered).

Why might the hybrid rule fail on ZTF? The rule assumes abs_peak_mag means the same thing in every
survey. Two things could break that on ZTF:
  1. Light-curve source: ~2/3 of ZTF light curves are alert detections only (no forced photometry),
     so "amplitude" (max - min flux) has no near-zero baseline and underestimates the peak flux.
  2. Any other measurement offset between our abs_peak_mag and the true peak absolute magnitude.

This script checks both, using only saved results (no re-training):
  A. our apparent / absolute peak magnitude vs BTS's own peakmag / peakabs, by light-curve source
  B. transfer macro-F1 split by light-curve source, for every model and preprocessing
  C. absolute-magnitude percentiles per class: PLAsTiCC train, DES, ZTF (forced) and ZTF (detections)

Usage:  python diagnose_ztf.py   -> tables/ztf_diag_*.md, results/ztf_diagnostics.json
"""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase1"))
import common as C  # noqa: E402
from ztf_download import pick  # noqa: E402


def md_table(name, df, caption):
    lines = [f"**{caption}**", "", "| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |",
             "| " + " | ".join(["---"] * (len(df.columns) + 1)) + " |"]
    for i, r in df.iterrows():
        lines.append("| " + " | ".join([str(i)] + [f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in r]) + " |")
    open(os.path.join(HERE, "tables", name + ".md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines) + "\n")


def main():
    out = {}
    ztf, lab = C.load_target("ztf")
    ztf = ztf.assign(plasticc_class=lab.values)
    ztf = ztf[ztf["plasticc_class"].isin(C.SHARED)].reset_index(drop=True)   # same order as run_experiment
    feat = [c for c in C.load_plasticc()[1]]
    Z = C.derive_features(ztf[feat])
    ztf["app_mag_ours"] = C.apparent_mag(ztf)
    ztf["abs_mag_ours"] = Z["abs_peak_mag"].values
    src = ztf["lc_source"].fillna("unknown") if "lc_source" in ztf else pd.Series("unknown", index=ztf.index)

    # ---- A. our magnitudes vs BTS's own --------------------------------------------------------
    cat_path = os.path.join(HERE, "bts_catalogue.csv")
    if os.path.exists(cat_path):
        cat = pd.read_csv(cat_path)
        cat = pd.DataFrame({"ztfid": cat[pick(cat, ["ZTFID", "ZTF ID", "ztf_id", "oid", "name"])].astype(str).str.strip(),
                            "bts_peakmag": pd.to_numeric(cat[pick(cat, ["peakmag", "peak mag", "mag"])], errors="coerce"),
                            "bts_peakabs": pd.to_numeric(cat[pick(cat, ["peakabs", "M_abs", "absmag", "abs mag"])], errors="coerce")})
        j = ztf[["ztfid", "app_mag_ours", "abs_mag_ours"]].assign(source=src.values).merge(cat, on="ztfid", how="left")
        j["d_app"] = j["app_mag_ours"] - j["bts_peakmag"]; j["d_abs"] = j["abs_mag_ours"] - j["bts_peakabs"]
        rows = {}
        for s, g in [("all", j)] + list(j.groupby("source")):
            mad = lambda x: 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x)))
            rows[s] = {"n": int(len(g)), "median Δ apparent (ours − BTS)": np.nanmedian(g["d_app"]), "scatter Δ apparent": mad(g["d_app"]),
                       "median Δ absolute (ours − BTS)": np.nanmedian(g["d_abs"]), "scatter Δ absolute": mad(g["d_abs"])}
        t = pd.DataFrame(rows).T; t.index.name = "Light-curve source"
        md_table("ztf_diag_magnitudes", t, "Our peak magnitudes vs BTS's own (positive = ours fainter); scatter = 1.4826 × MAD")
        out["magnitude_check"] = t.reset_index().to_dict("records")
    else:
        print("bts_catalogue.csv not found: skipping the magnitude check")

    # ---- B. transfer F1 by light-curve source (saved predictions) ------------------------------
    rows = {}
    for m in ["xgboost", "lightgbm", "catboost", "mlp"]:
        for p in ["consistent", "per_survey", "per_survey_drop_absmag", "hybrid"]:
            f = os.path.join(HERE, "results", f"ztf_{m}_{p}.json")
            if not os.path.exists(f):
                continue
            r = json.load(open(f)); y = np.array(r["y_des"]); labels = sorted(np.unique(y).tolist())
            if len(y) != len(ztf):
                print(f"  {f}: {len(y)} objects vs {len(ztf)} now in the feature file; skipped"); continue
            cell = {}
            for s in sorted(src.unique()):
                msk = (src == s).values
                v = [f1_score(y[msk], np.array(x["des_pred"])[msk], average="macro", labels=labels, zero_division=0) for x in r["runs"]]
                cell[f"{s} (n={int(msk.sum())})"] = f"{np.mean(v):.3f} ± {np.std(v):.3f}"
            rows[f"{m} / {p}"] = cell
    if rows:
        t = pd.DataFrame(rows).T; t.index.name = "Model / preprocessing"
        md_table("ztf_diag_by_source", t, "PLAsTiCC → ZTF transfer macro-F1 split by light-curve source")
        out["f1_by_source"] = t.reset_index().to_dict("records")

    # ---- C. absolute magnitude per class in each sample ----------------------------------------
    plas, f = C.load_plasticc(); plas = plas[plas["target"].isin(C.SHARED)]
    des = C.load_des(); dl = C.des_labels(des, "corrected"); des = des[dl.isin(C.SHARED).values].assign(cls=dl[dl.isin(C.SHARED)].values)
    samples = {"PLAsTiCC": (C.derive_features(plas[f])["abs_peak_mag"].values, plas["target"].values),
               "DES": (C.derive_features(des[f])["abs_peak_mag"].values, des["cls"].values)}
    for s in sorted(src.unique()):
        msk = (src == s).values
        samples[f"ZTF {s}"] = (ztf["abs_mag_ours"].values[msk], ztf["plasticc_class"].values[msk])
    rows = {}
    for code, name in C.SHARED.items():
        for sname, (a, c) in samples.items():
            x = a[c == code]; x = x[np.isfinite(x)]
            rows.setdefault(name, {})[sname] = f"{np.median(x):.2f} [{np.percentile(x, 10):.2f}, {np.percentile(x, 90):.2f}] n={len(x)}" if len(x) >= 5 else "--"
    t = pd.DataFrame(rows).T; t.index.name = "Class"
    md_table("ztf_diag_absmag", t, "Peak absolute magnitude (our definition): median [10th, 90th percentile]")
    out["abs_mag_by_class"] = t.reset_index().to_dict("records")
    json.dump(out, open(os.path.join(HERE, "results", "ztf_diagnostics.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
