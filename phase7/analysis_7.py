"""
Phase 7 tables and verdicts (Y1-Y6) from results/*.json, plus descriptive checks.
Usage:  python analysis_7.py    (after the Phase 7 configs have run)
"""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__)); XAI = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(XAI, "phase1"))
import common as C  # noqa: E402

os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
MODELS = [("xgboost", "XGBoost"), ("lightgbm", "LightGBM"), ("catboost", "CatBoost"), ("mlp", "MLP")]
TREES = [m for m, _ in MODELS[:3]]
MODES = [("consistent", "Consistent"), ("per_survey", "Per-survey"), ("per_survey_drop_absmag", "Per-survey − abs mag"),
         ("hybrid_redshift_only", "Redshift only shared"), ("hybrid_absmag_only", "M_peak only shared"),
         ("hybrid", "Hybrid rule"), ("hybrid_plus_brightness", "Hybrid + brightness shared")]


def load(k):
    p = os.path.join(HERE, "results", k + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def arr(k, m="transfer_macro_f1"):
    """Pre-registered metric: macro-F1 over classes with >= 10 objects in the evaluated set, computed
    from the saved predictions. (The engine's own transfer_macro_f1 averages over every class present,
    which on YSE includes SLSN-I with 2 objects, against the pre-registration.) For the YSE-only CV the
    same class rule is applied to the saved pooled out-of-fold per-class F1."""
    r = load(k)
    if not r:
        return np.array([])
    y = np.array(r["y_des"]); vals, cnt = np.unique(y, return_counts=True)
    keep = [r["class_order"][i] for i in vals[cnt >= 10]]
    if m == "transfer_macro_f1":
        lab = sorted(vals[cnt >= 10].tolist())
        return np.array([f1_score(y, np.array(x["des_pred"]), average="macro", labels=lab, zero_division=0) for x in r["runs"]])
    if m == "des_cv_macro_f1_mean_of_folds":
        return np.array([np.mean([x["des_cv_per_class_f1"][c] for c in keep]) for x in r["runs"] if "des_cv_per_class_f1" in x])
    return np.array([x[m] for x in r["runs"] if m in x])


def fmt(a):
    return "--" if len(a) == 0 else f"{a.mean():.3f} ± {a.std():.3f}"


def pooled(a, b):
    return np.sqrt((np.std(a) ** 2 + np.std(b) ** 2) / 2)


def gt(a, b):
    return None if len(a) == 0 or len(b) == 0 else bool(np.mean(a) - np.mean(b) > pooled(a, b))


def ge(a, b):
    return None if len(a) == 0 or len(b) == 0 else bool(np.mean(a) - np.mean(b) >= -pooled(a, b))


def write(name, header, rows, caption):
    md = [f"**{caption}**", "", "| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    open(os.path.join(HERE, "tables", name + ".md"), "w").write("\n".join(md) + "\n")
    esc = lambda s: str(s).replace("±", r"$\pm$").replace("−", "$-$").replace("_", r"\_")
    tex = [r"\begin{table}", r"\centering", r"\caption{" + esc(caption) + "}", r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}", r"\hline",
           " & ".join(map(esc, header)) + r" \\", r"\hline"] + [" & ".join(map(esc, r)) + r" \\" for r in rows] + [r"\hline", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(HERE, "tables", name + ".tex"), "w").write("\n".join(tex) + "\n"); print("\n".join(md) + "\n")


V = {}
# ---------------- main table ----------------
for tgt, cap in [("yse", "all spectroscopic YSE objects"), ("yse_no_bts", "excluding objects also in ZTF BTS (robustness)")]:
    rows = []
    for m, mn in MODELS:
        if load(f"{tgt}_{m}_consistent") is None:
            continue
        rows.append([mn] + [fmt(arr(f"{tgt}_{m}_{p}")) for p, _ in MODES] + [fmt(arr(f"{tgt}_{m}_consistent", "des_cv_macro_f1_mean_of_folds"))])
    r0 = load(f"{tgt}_xgboost_consistent")
    if r0:
        y = np.array(r0["y_des"]); counts = {n: int((y == i).sum()) for i, n in enumerate(r0["class_order"]) if (y == i).sum()}
        write(f"{tgt}_transfer", ["Model"] + [p for _, p in MODES] + ["YSE-only CV (pooled out-of-fold)"], rows,
              f"PLAsTiCC → YSE transfer macro-F1 over classes with >= 10 objects, {cap}; class counts {counts} (5 split seeds)")
for m, _ in MODELS:
    V[f"Y1_{m}_consistent_below_yse_cv"] = gt(arr(f"yse_{m}_consistent", "des_cv_macro_f1_mean_of_folds"), arr(f"yse_{m}_consistent"))
y2 = [gt(arr(f"yse_{m}_hybrid_plus_brightness"), arr(f"yse_{m}_hybrid")) for m in TREES]
y3 = [ge(arr(f"yse_{m}_hybrid_plus_brightness"), arr(f"yse_{m}_per_survey")) for m in TREES]
V["Y2_plus_brightness_gt_hybrid"] = {"per_tree_model": y2, "supported": sum(bool(x) for x in y2) >= 2}
V["Y3_plus_brightness_ge_per_survey"] = {"per_tree_model": y3, "supported": all(bool(x) for x in y3)}

# ---------------- near / far halves (label-free split at the median YSE redshift) ----------------
yse, lab = C.load_target("yse"); yse = yse.assign(cls=lab.values); yse = yse[yse["cls"].isin(C.SHARED)].reset_index(drop=True)
zmed = float(np.median(yse["redshift"])); near = (yse["redshift"].values <= zmed)


def half_f1(k, mask):
    r = load(k)
    if r is None or len(r["y_des"]) != len(mask):
        return np.array([])
    y = np.array(r["y_des"]); yy = y[mask]
    vals, cnt = np.unique(yy, return_counts=True); labels = sorted(vals[cnt >= 10].tolist())
    return np.array([f1_score(yy, np.array(x["des_pred"])[mask], average="macro", labels=labels, zero_division=0) for x in r["runs"]])


rows, y4, y5, y6 = [], [], [], []
for m, mn in MODELS:
    cells = {}
    for half, msk in [("near", near), ("far", ~near)]:
        for p in ["per_survey", "hybrid", "hybrid_plus_brightness", "consistent"]:
            cells[(half, p)] = half_f1(f"yse_{m}_{p}", msk)
    rows.append([mn] + [fmt(cells[(h, p)]) for h in ["near", "far"] for p in ["consistent", "per_survey", "hybrid", "hybrid_plus_brightness"]])
    if m in TREES:
        g_near = np.mean(cells[("near", "hybrid_plus_brightness")] - cells[("near", "hybrid")]) if len(cells[("near", "hybrid")]) else np.nan
        g_far = np.mean(cells[("far", "hybrid_plus_brightness")] - cells[("far", "hybrid")]) if len(cells[("far", "hybrid")]) else np.nan
        y4.append(None if np.isnan(g_near) or np.isnan(g_far) else bool(g_near > g_far))
        y5.append(gt(cells[("near", "per_survey")], cells[("near", "hybrid")]))
        y6.append(ge(cells[("far", "hybrid")], cells[("far", "per_survey")]))
hdr = ["Model"] + [f"{h}: {p}" for h in ["near", "far"] for p in ["consistent", "per-survey", "hybrid", "hybrid + brightness"]]
write("yse_near_far", hdr, rows, f"YSE split at its median redshift z = {zmed:.3f} (near: {int(near.sum())}, far: {int((~near).sum())}); macro-F1 over classes with >= 10 objects in each half")
V["Y4_gain_larger_near"] = {"per_tree_model": y4, "supported": sum(bool(x) for x in y4) >= 2}
V["Y5_near_hybrid_lt_per_survey"] = {"per_tree_model": y5, "supported": sum(bool(x) for x in y5) >= 2}
V["Y6_far_hybrid_ge_per_survey"] = {"per_tree_model": y6, "supported": sum(bool(x) for x in y6) >= 2}

# ---------------- descriptive: where YSE sits between DES and ZTF ----------------
plas, feat = C.load_plasticc(); plas = plas[plas["target"].isin(C.SHARED)]
des = C.load_des(); dl = C.des_labels(des, "corrected"); des = des[dl.isin(C.SHARED).values]
ztf, zl = C.load_target("ztf"); ztf = ztf[zl.isin(C.SHARED).values]
rows = []
P = C.derive_features(plas[feat])
for name, df in [("DES", des), ("YSE", yse), ("ZTF", ztf)]:
    D = C.derive_features(df[feat])
    rows.append([name, len(df), f"{np.nanmedian(df['redshift']):.3f}", f"{np.median(df['amplitude']) / np.median(plas['amplitude']):.1f}",
                 f"{C.support_overlap(P['redshift'], D['redshift']):.2f}", f"{C.support_overlap(P['abs_peak_mag'], D['abs_peak_mag']):.2f}"])
write("yse_position", ["Target", "N", "median z", "amplitude ratio to PLAsTiCC", "redshift overlap", "M_peak overlap"], rows,
      "Where YSE sits between DES and ZTF (label-free)")
D = C.derive_features(yse[feat]); d = D["abs_peak_mag"].values - yse["yse_peak_abs_mag"].values; d = d[np.isfinite(d) & (yse["yse_peak_abs_mag"].values > -90)]
V["magnitude_check_ours_minus_yse"] = {"median": float(np.median(d)) if len(d) else None, "scatter_1.4826MAD": float(1.4826 * np.median(np.abs(d - np.median(d)))) if len(d) else None, "n": int(len(d))}
V["n_in_bts"] = int(yse["in_bts"].sum()); V["n_yse"] = int(len(yse)); V["median_z"] = zmed
json.dump(V, open(os.path.join(HERE, "results", "verdicts_7.json"), "w"), indent=1)
print(json.dumps(V, indent=1))
