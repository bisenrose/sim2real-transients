"""
EXPLORATORY (written after seeing the ZTF results; not pre-registered).

Hypothesis: the hybrid rule keeps redshift on PLAsTiCC's scale, assuming the target survey's
redshifts fall inside the range the classifier was trained on. ZTF BTS is a very low-redshift
sample, so many ZTF objects may sit where PLAsTiCC has few training examples. On ZTF the label-free
SHAP damage ranking puts redshift first.

This script uses only saved results (no re-training):
  A. redshift percentiles per class: PLAsTiCC train, DES, ZTF
  B. fraction of each target sample below PLAsTiCC's 5th redshift percentile, per class
  C. transfer macro-F1 by ZTF redshift tercile, for every model and preprocessing

Usage:  python diagnose_ztf_redshift.py   -> tables/ztf_diag_redshift*.md, results/ztf_redshift_diagnostics.json
"""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase1"))
import common as C  # noqa: E402


def write(name, df, caption):
    lines = [f"**{caption}**", "", "| " + " | ".join([df.index.name or ""] + list(map(str, df.columns))) + " |",
             "| " + " | ".join(["---"] * (len(df.columns) + 1)) + " |"]
    lines += ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |" for i, r in df.iterrows()]
    open(os.path.join(HERE, "tables", name + ".md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines) + "\n")


def main():
    out = {}
    plas, _ = C.load_plasticc(); plas = plas[plas["target"].isin(C.SHARED)]
    des = C.load_des(); dl = C.des_labels(des, "corrected"); m = dl.isin(C.SHARED).values
    des = des[m].assign(cls=dl[m].values)
    ztf, zl = C.load_target("ztf"); ztf = ztf.assign(cls=zl.values)
    ztf = ztf[ztf["cls"].isin(C.SHARED)].reset_index(drop=True)          # same order as run_experiment
    samples = {"PLAsTiCC": (plas["redshift"].values, plas["target"].values), "DES": (des["redshift"].values, des["cls"].values),
               "ZTF": (ztf["redshift"].values, ztf["cls"].values)}
    rows, below = {}, {}
    for code, name in C.SHARED.items():
        zp = plas["redshift"].values[plas["target"].values == code]; p5 = np.nanpercentile(zp, 5) if len(zp) else np.nan
        for s, (z, c) in samples.items():
            x = z[c == code]; x = x[np.isfinite(x)]
            rows.setdefault(name, {})[s] = f"{np.median(x):.3f} [{np.percentile(x, 10):.3f}, {np.percentile(x, 90):.3f}] n={len(x)}" if len(x) >= 5 else "--"
            if s != "PLAsTiCC" and len(x) >= 5:
                below.setdefault(name, {})[s] = f"{np.mean(x < p5):.2f}"
        below.setdefault(name, {})["PLAsTiCC 5th pct"] = f"{p5:.3f}"
    t = pd.DataFrame(rows).T; t.index.name = "Class"
    write("ztf_diag_redshift", t, "Redshift per class: median [10th, 90th percentile]")
    t2 = pd.DataFrame(below).T; t2.index.name = "Class"
    write("ztf_diag_redshift_support", t2, "Fraction of each target sample below PLAsTiCC's 5th redshift percentile for that class")
    out["redshift_by_class"] = t.reset_index().to_dict("records"); out["below_plasticc_p5"] = t2.reset_index().to_dict("records")

    edges = np.nanpercentile(ztf["redshift"], [0, 33.3, 66.7, 100])
    zb = np.digitize(ztf["redshift"].values, edges[1:-1])
    labels = [f"z {edges[i]:.3f}–{edges[i + 1]:.3f}" for i in range(3)]
    rows = {}
    for mdl in ["xgboost", "lightgbm", "catboost", "mlp"]:
        for p in ["consistent", "per_survey", "hybrid", "explore_hybrid_absmag_only"]:
            f = os.path.join(HERE, "results", f"ztf_{mdl}_{p}.json")
            if not os.path.exists(f):
                continue
            r = json.load(open(f)); y = np.array(r["y_des"])
            if len(y) != len(ztf):
                continue
            lab = sorted(np.unique(y).tolist()); cell = {}
            for b, bl in enumerate(labels):
                msk = zb == b
                v = [f1_score(y[msk], np.array(x["des_pred"])[msk], average="macro", labels=lab, zero_division=0) for x in r["runs"]]
                cell[bl] = f"{np.mean(v):.3f} ± {np.std(v):.3f}"
            rows[f"{mdl} / {p}"] = cell
    if rows:
        t3 = pd.DataFrame(rows).T; t3.index.name = "Model / preprocessing"
        write("ztf_diag_by_redshift", t3, "PLAsTiCC → ZTF transfer macro-F1 by ZTF redshift tercile")
        out["f1_by_redshift_tercile"] = t3.reset_index().to_dict("records")
    json.dump(out, open(os.path.join(HERE, "results", "ztf_redshift_diagnostics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
