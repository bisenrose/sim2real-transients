"""
Phase 3 tables, paired bootstrap tests, calibration, and verdicts (T1, T2, B1, B2, C1).
Usage:  python stats.py     (after the configs in configs/ have been run)
"""
import glob, json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__)); XAI = os.path.dirname(HERE)
R = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(HERE, "results", "*.json")) if "verdict" not in p}
P2 = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(XAI, "phase2", "results", "*_corrected_*.json"))}
os.makedirs(os.path.join(HERE, "tables"), exist_ok=True); os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)


def arr(r, k):
    return np.array([x[k] for x in r["runs"] if k in x]) if r else np.array([])


def fmt(a, nd=3):
    return "--" if len(a) == 0 else f"{a.mean():.{nd}f} ± {a.std():.{nd}f}"


def better(a, b):
    """True/False if a beats b by more than one pooled SD; None if either result is missing."""
    if len(a) == 0 or len(b) == 0:
        return None
    return bool(a.mean() - b.mean() > np.sqrt((a.std() ** 2 + b.std() ** 2) / 2))


def write(name, header, rows, caption):
    md = [f"**{caption}**", "", "| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"] + ["| " + " | ".join(r) + " |" for r in rows]
    open(os.path.join(HERE, "tables", name + ".md"), "w").write("\n".join(md) + "\n")
    esc = lambda s: s.replace("±", r"$\pm$").replace("_", r"\_").replace("−", "$-$").replace("+", "$+$")
    tex = [r"\begin{table}", r"\centering", r"\caption{" + esc(caption) + "}", r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}", r"\hline",
           " & ".join(map(esc, header)) + r" \\", r"\hline"] + [" & ".join(map(esc, r)) + r" \\" for r in rows] + [r"\hline", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(HERE, "tables", name + ".tex"), "w").write("\n".join(tex) + "\n")


def paired_bootstrap(ra, rb, n=1000):
    """Difference in transfer macro-F1 (a − b), resampling the same DES objects for both, per split seed."""
    y = np.array(ra["y_des"]); diffs = []
    rb_by_seed = {x["split_seed"]: x for x in rb["runs"]}
    for xa in ra["runs"]:
        xb = rb_by_seed.get(xa["split_seed"])
        if xb is None or "des_pred" not in xa or "des_pred" not in xb:
            continue
        pa, pb = np.array(xa["des_pred"]), np.array(xb["des_pred"]); rng = np.random.RandomState(xa["split_seed"])
        for _ in range(n):
            i = rng.randint(0, len(y), len(y))
            diffs.append(f1_score(y[i], pa[i], average="macro", zero_division=0) - f1_score(y[i], pb[i], average="macro", zero_division=0))
    d = np.array(diffs)
    return (float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())) if len(d) else None


verd = {}

# ---- 3.1 tuning --------------------------------------------------------------------------
rows = []
for m, mn in [("xgboost", "XGBoost"), ("lightgbm", "LightGBM"), ("catboost", "CatBoost"), ("mlp", "MLP")]:
    dc, dh = P2.get(f"{m}_corrected_consistent"), P2.get(f"{m}_corrected_hybrid")
    tc, th = R.get(f"t_{m}_consistent"), R.get(f"t_{m}_hybrid")
    rows.append([mn, fmt(arr(dc, "plasticc_macro_f1")), fmt(arr(tc, "plasticc_macro_f1")),
                 fmt(arr(dc, "transfer_macro_f1")), fmt(arr(tc, "transfer_macro_f1")), fmt(arr(dh, "transfer_macro_f1")), fmt(arr(th, "transfer_macro_f1"))])
    if tc and th:
        verd[f"T1_{m}_tuned_hybrid_gt_consistent"] = better(arr(th, "transfer_macro_f1"), arr(tc, "transfer_macro_f1"))
        if m != "mlp" and dc:
            verd[f"T2_{m}_in_domain_change_lt_0.03"] = bool(abs(arr(tc, "plasticc_macro_f1").mean() - arr(dc, "plasticc_macro_f1").mean()) < 0.03)
write("tuning", ["Model", "PLAsTiCC F1 (matched)", "PLAsTiCC F1 (tuned)", "Transfer, consistent (matched)", "Transfer, consistent (tuned)",
                 "Transfer, hybrid (matched)", "Transfer, hybrid (tuned)"], rows,
      "Effect of nested hyperparameter tuning on PLAsTiCC only (5 split seeds)")

# ---- 3.2 baselines -------------------------------------------------------------------------
METHODS = [("consistent_none", "Consistent (baseline)"), ("consistent_importance_weighting", "+ importance reweighting"),
           ("consistent_iw_no_survey_features", "+ reweighting, no survey features (exploratory)"), ("consistent_coral", "+ CORAL"),
           ("consistent_selection", "+ DES-matched selection (2.2)"), ("hybrid_none", "Hybrid rule"),
           ("hybrid_importance_weighting", "Hybrid + reweighting"), ("hybrid_iw_no_survey_features", "Hybrid + reweighting, no survey features (exploratory)"),
           ("hybrid_coral", "Hybrid + CORAL")]
rows = []
for k, kn in METHODS:
    rx, rl = R.get(f"b_xgboost_{k}"), R.get(f"b_lightgbm_{k}")
    auc = arr(rx, "domain_classifier_auc")
    rows.append([kn, fmt(arr(rx, "transfer_macro_f1")), fmt(arr(rl, "transfer_macro_f1")), fmt(arr(rx, "transfer_log_loss")),
                 fmt(arr(rx, "transfer_ece")), f"{auc.mean():.3f}" if len(auc) else "--"])
write("baselines", ["Method (no DES labels used)", "XGBoost transfer F1", "LightGBM transfer F1", "XGBoost log-loss", "XGBoost ECE", "Domain-classifier AUC"],
      rows, "Hybrid rule vs standard domain-adaptation baselines (5 split seeds)")
for m in ["xgboost", "lightgbm"]:
    h = arr(R.get(f"b_{m}_hybrid_none"), "transfer_macro_f1")
    verd[f"B1_{m}_hybrid_gt_reweighting"] = better(h, arr(R.get(f"b_{m}_consistent_importance_weighting"), "transfer_macro_f1"))
    verd[f"B2_{m}_hybrid_gt_coral"] = better(h, arr(R.get(f"b_{m}_consistent_coral"), "transfer_macro_f1"))

# ---- 3.3 paired bootstrap, calibration -----------------------------------------------------
rows = []
pairs = [("b_xgboost_hybrid_none", "b_xgboost_consistent_none", "Hybrid − consistent"),
         ("b_xgboost_hybrid_none", "b_xgboost_consistent_importance_weighting", "Hybrid − reweighting"),
         ("b_xgboost_hybrid_none", "b_xgboost_consistent_iw_no_survey_features", "Hybrid − reweighting (no survey features)"),
         ("b_xgboost_hybrid_none", "b_xgboost_consistent_coral", "Hybrid − CORAL"),
         ("b_xgboost_hybrid_none", "b_xgboost_consistent_selection", "Hybrid − DES-matched selection"),
         ("t_xgboost_hybrid", "t_xgboost_consistent", "Tuned: hybrid − consistent")]
boot = {}
for a, b, lab in pairs:
    if a in R and b in R:
        res = paired_bootstrap(R[a], R[b])
        if res:
            boot[lab] = res
            rows.append([lab, f"{res[0]:+.3f}", f"[{res[1]:+.3f}, {res[2]:+.3f}]", f"{res[3]:.3f}"])
write("paired_bootstrap", ["Comparison (XGBoost)", "Mean ΔF1", "95% interval", "P(Δ > 0)"], rows,
      "Paired bootstrap on DES objects (1000 resamples × 5 split seeds)")
verd["paired_bootstrap"] = boot
c, h = R.get("b_xgboost_consistent_none"), R.get("b_xgboost_hybrid_none")
if c and h:
    verd["C1_hybrid_better_log_loss"] = better(-arr(h, "transfer_log_loss"), -arr(c, "transfer_log_loss"))
    rows = [[lab, fmt(arr(r, "transfer_log_loss")), fmt(arr(r, "transfer_brier")), fmt(arr(r, "transfer_ece")),
             fmt(arr(r, "transfer_macro_f1")), fmt(arr(r, "transfer_macro_f1_without_slsn"))] for lab, r in [("Consistent", c), ("Hybrid rule", h)]]
    write("calibration", ["XGBoost on DES", "Log-loss", "Brier", "ECE", "Macro-F1 (5 classes)", "Macro-F1 without SLSN-I"], rows,
          "Probabilistic quality of transferred predictions (lower is better for the first three)")
    names = c["class_order"]
    rows = [[n, fmt(np.array([x["transfer_per_class_f1"][n] for x in c["runs"]])), fmt(np.array([x["transfer_per_class_f1"][n] for x in h["runs"]]))] for n in names]
    write("per_class", ["Class", "Consistent", "Hybrid rule"], rows, "Per-class transfer F1, XGBoost (SLSN-I has 18 DES objects)")
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
    for lab, r, col in [("Consistent", c, "#888888"), ("Hybrid rule", h, "#1f77b4")]:
        bins = {}
        for x in r["runs"]:
            for lo, hi, cf, acc, n in x["transfer_reliability"]:
                b = bins.setdefault(lo, [0, 0, 0]); b[0] += cf * n; b[1] += acc * n; b[2] += n
        pts = sorted((v[0] / v[2], v[1] / v[2]) for v in bins.values() if v[2] > 0)
        ax.plot(*zip(*pts), marker="o", color=col, label=f"{lab} (ECE {arr(r, 'transfer_ece').mean():.3f})")
    ax.set_xlabel("Predicted confidence"); ax.set_ylabel("Observed accuracy"); ax.set_title("Reliability on DES (XGBoost, 5 seeds)", fontsize=10)
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "reliability.png"), dpi=150)

json.dump(verd, open(os.path.join(HERE, "results", "verdicts_3.json"), "w"), indent=1)
for t in ["tuning", "baselines", "paired_bootstrap", "calibration", "per_class"]:
    p = os.path.join(HERE, "tables", t + ".md")
    if os.path.exists(p):
        print(open(p).read())
print(json.dumps({k: v for k, v in verd.items() if k != "paired_bootstrap"}, indent=1))
