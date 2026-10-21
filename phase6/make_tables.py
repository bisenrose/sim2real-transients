"""Phase 6 tables and verdicts (D1 F1-F2, D2 T1-T5, D5 SS1-SS2). Run analysis_saved.py for D3/D4.
Usage: python make_tables.py"""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); XAI = os.path.dirname(HERE)
R = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(HERE, "results", "D[125]_*.json"))}
os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
MODELS = [("xgboost", "XGBoost"), ("lightgbm", "LightGBM"), ("catboost", "CatBoost"), ("mlp", "MLP")]


def arr(k, m="transfer_macro_f1", src=None):
    r = R.get(k) if src is None else (json.load(open(src)) if os.path.exists(src) else None)
    return np.array([x[m] for x in r["runs"] if m in x]) if r else np.array([])


def fmt(a):
    return "--" if len(a) == 0 else f"{a.mean():.3f} ± {a.std():.3f}"


def gt(a, b):
    return None if len(a) == 0 or len(b) == 0 else bool(a.mean() - b.mean() > np.sqrt((a.std() ** 2 + b.std() ** 2) / 2))


def within(a, b):
    return None if len(a) == 0 or len(b) == 0 else bool(abs(a.mean() - b.mean()) <= np.sqrt((a.std() ** 2 + b.std() ** 2) / 2))


def write(name, header, rows, caption):
    md = [f"**{caption}**", "", "| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    open(os.path.join(HERE, "tables", name + ".md"), "w").write("\n".join(md) + "\n")
    esc = lambda s: str(s).replace("±", r"$\pm$").replace("_", r"\_").replace("≥", r"$\geq$").replace("<", "$<$")
    tex = [r"\begin{table}", r"\centering", r"\caption{" + esc(caption) + "}", r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}", r"\hline",
           " & ".join(map(esc, header)) + r" \\", r"\hline"] + [" & ".join(map(esc, r)) + r" \\" for r in rows] + [r"\hline", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(HERE, "tables", name + ".tex"), "w").write("\n".join(tex) + "\n"); print("\n".join(md) + "\n")


V = {}
# ---------- D1 ----------
for tgt, prev in [("ztf", "phase4"), ("des", "phase2")]:
    rows = []; f1, f2 = [], []
    for m, mn in MODELS:
        if tgt == "ztf":
            ps = arr(None, src=os.path.join(XAI, prev, "results", f"ztf_{m}_per_survey.json"))
            hy = arr(None, src=os.path.join(XAI, prev, "results", f"ztf_{m}_hybrid.json"))
            ao = arr(None, src=os.path.join(XAI, prev, "results", f"ztf_{m}_explore_hybrid_absmag_only.json"))
        else:
            ps = arr(None, src=os.path.join(XAI, prev, "results", f"{m}_corrected_per_survey.json"))
            hy = arr(None, src=os.path.join(XAI, prev, "results", f"{m}_corrected_hybrid.json"))
            ao = arr(None, src=os.path.join(XAI, prev, "results", "explore_xgboost_corrected_hybrid_absmag_only.json")) if m == "xgboost" else np.array([])
        zo, hb = arr(f"D1_{tgt}_{m}_hybrid_redshift_only"), arr(f"D1_{tgt}_{m}_hybrid_plus_brightness")
        rows.append([mn, fmt(ps), fmt(zo), fmt(ao), fmt(hy), fmt(hb)])
        if tgt == "ztf":
            f1.append(gt(ao, zo)); f2.append(gt(hb, hy))
    write(f"D1_{tgt}", ["Model", "Nothing shared (per-survey)", "Redshift only shared", "M_peak only shared", "Both shared (hybrid rule)", "Both + brightness shared"], rows,
          f"Which features keep the shared scale: transfer macro-F1 on {tgt.upper()} (5 split seeds)")
    if tgt == "ztf":
        V["F1_ztf_redshift_only_lt_absmag_only"] = {"per_model": f1, "supported": sum(x for x in f1 if x) >= 3}
        V["F2_ztf_plus_brightness_gt_hybrid"] = {"per_model": f2, "supported": sum(x for x in f2 if x) >= 3}

# ---------- D2 ----------
MODES = [("consistent", "Consistent"), ("per_survey", "Per-survey"), ("hybrid", "Hybrid rule"), ("hybrid_absmag_only", "Hybrid, abs mag only")]
rows = []
for kn, kl in [("control", "Ia-rich random (control)"), ("lowz", "Ia-rich low-z")]:
    for sh, sl in [("none", "none"), ("I1", "I1 cadence + noise"), ("I2", "I2 flux calibration")]:
        for m, mn in MODELS[:2]:
            rows.append([kl, sl, mn] + [fmt(arr(f"D2_{kn}_{sh}_{m}_{p}")) for p, _ in MODES])
write("D2_two_factor", ["Population", "Instrument shift", "Model"] + [p for _, p in MODES], rows,
      "Two-factor controlled test inside PLAsTiCC (ZTF-like class priors; 5 split seeds)")
g = lambda kn, sh, m, p: arr(f"D2_{kn}_{sh}_{m}_{p}")
for m, _ in MODELS[:2]:
    V[f"T1_{m}_control_I1_per_survey_gt_consistent"] = gt(g("control", "I1", m, "per_survey"), g("control", "I1", m, "consistent"))
    V[f"T1_{m}_control_I1_hybrid_gt_consistent"] = gt(g("control", "I1", m, "hybrid"), g("control", "I1", m, "consistent"))
    V[f"T2_{m}_control_I2_hybrid_lt_per_survey"] = gt(g("control", "I2", m, "per_survey"), g("control", "I2", m, "hybrid"))
    V[f"T3_{m}_lowz_I2_hybrid_lt_per_survey"] = gt(g("lowz", "I2", m, "per_survey"), g("lowz", "I2", m, "hybrid"))
    V[f"T4_{m}_lowz_I1_hybrid_gt_per_survey"] = gt(g("lowz", "I1", m, "hybrid"), g("lowz", "I1", m, "per_survey"))
    V[f"T5_{m}_lowz_none_hybrid_gt_per_survey"] = gt(g("lowz", "none", m, "hybrid"), g("lowz", "none", m, "per_survey"))

# ---------- D5 ----------
rows = []
full = {("des", "hybrid"): os.path.join(XAI, "phase2", "results", "xgboost_corrected_hybrid.json"),
        ("des", "per_survey"): os.path.join(XAI, "phase2", "results", "xgboost_corrected_per_survey.json"),
        ("ztf", "hybrid"): os.path.join(XAI, "phase4", "results", "ztf_xgboost_hybrid.json"),
        ("ztf", "per_survey"): os.path.join(XAI, "phase4", "results", "ztf_xgboost_per_survey.json")}
for tgt in ["des", "ztf"]:
    for mode in ["per_survey", "hybrid"]:
        rows.append([tgt.upper(), mode] + [fmt(arr(f"D5_{tgt}_xgboost_n{n:03d}_{mode}")) for n in [25, 50, 100, 200]] + [fmt(arr(None, src=full[(tgt, mode)]))])
write("D5_sample_size", ["Target", "Preprocessing", "n = 25", "n = 50", "n = 100", "n = 200", "All"], rows,
      "Target statistics fitted on n unlabelled target objects (XGBoost, 5 split seeds)")
fd = arr(None, src=full[("des", "hybrid")])
V["SS1_des_hybrid_n100_within_full"] = within(arr("D5_des_xgboost_n100_hybrid"), fd)
V["SS2_des_hybrid_n25_below_full"] = gt(fd, arr("D5_des_xgboost_n025_hybrid"))
json.dump(V, open(os.path.join(HERE, "results", "verdicts_6.json"), "w"), indent=1)
print(json.dumps(V, indent=1))
