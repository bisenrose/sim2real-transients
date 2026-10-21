"""
Phase 4 tables, figures and verdicts (Z1-Z5 for ZTF, A1-A2 for label acquisition).
Usage:  python make_tables.py
"""
import glob, json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(HERE, "results", "ztf_*.json"))}
os.makedirs(os.path.join(HERE, "tables"), exist_ok=True); os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
MODELS = [("xgboost", "XGBoost"), ("lightgbm", "LightGBM"), ("catboost", "CatBoost"), ("mlp", "MLP")]
PREPS = [("consistent", "Consistent"), ("per_survey", "Per-survey"), ("per_survey_drop_absmag", "Per-survey − abs mag"), ("hybrid", "Hybrid (rule)")]


def arr(k, m="transfer_macro_f1"):
    r = R.get(k)
    return np.array([x[m] for x in r["runs"] if m in x]) if r else np.array([])


def fmt(a, nd=3):
    return "--" if len(a) == 0 else f"{a.mean():.{nd}f} ± {a.std():.{nd}f}"


def gt(a, b, allow_equal=False):
    if len(a) == 0 or len(b) == 0:
        return None
    pooled, d = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2), a.mean() - b.mean()
    return bool(d >= -pooled) if allow_equal else bool(d > pooled)


def write(name, header, rows, caption):
    md = [f"**{caption}**", "", "| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"] + ["| " + " | ".join(r) + " |" for r in rows]
    open(os.path.join(HERE, "tables", name + ".md"), "w").write("\n".join(md) + "\n")
    esc = lambda s: str(s).replace("±", r"$\pm$").replace("−", "$-$").replace("_", r"\_").replace("→", r"$\rightarrow$")
    tex = [r"\begin{table}", r"\centering", r"\caption{" + esc(caption) + "}", r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}", r"\hline",
           " & ".join(map(esc, header)) + r" \\", r"\hline"] + [" & ".join(map(esc, r)) + r" \\" for r in rows] + [r"\hline", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(HERE, "tables", name + ".tex"), "w").write("\n".join(tex) + "\n")


verd = {}
# ---------------- 4.1 ZTF ----------------
if R:
    any_r = next(iter(R.values()))
    y = np.array(any_r["y_des"]); names = any_r["class_order"]
    counts = {n: int((y == i).sum()) for i, n in enumerate(names)}
    rows = []
    for m, mn in MODELS:
        rows.append([mn] + [fmt(arr(f"ztf_{m}_{p}")) for p, _ in PREPS] + [fmt(arr(f"ztf_{m}_consistent", "des_cv_macro_f1_mean_of_folds"))])
        c, ps, pd_, h = (arr(f"ztf_{m}_{p}") for p, _ in PREPS)
        zcv = arr(f"ztf_{m}_consistent", "des_cv_macro_f1_mean_of_folds")
        verd[f"Z1_{m}_consistent_below_ztf_cv"] = gt(zcv, c)
        verd[f"Z2_{m}_hybrid_gt_consistent"] = gt(h, c)
        verd[f"Z3_{m}_hybrid_gt_per_survey"] = gt(h, ps)
        verd[f"Z4_{m}_hybrid_ge_drop_absmag"] = gt(h, pd_, allow_equal=True)
    write("ztf_transfer", ["Model"] + [p for _, p in PREPS] + ["ZTF-only CV"], rows,
          f"PLAsTiCC → ZTF transfer macro-F1 (classes present in ZTF: {', '.join(f'{k} {v}' for k, v in counts.items() if v)}; 5 split seeds)")
    sel = [("ztf_xgboost_consistent", "None"), ("ztf_xgboost_consistent_sel-logistic", "ZTF-matched selection"),
           ("ztf_xgboost_consistent_sel-random_logistic", "Random, same size")]
    rows = [[lab, fmt(arr(k)), fmt(arr(k, "abs_mag_ks_train_vs_des"), 2), fmt(arr(k, "abs_mag_median_gap"), 2)] for k, lab in sel if k in R]
    write("ztf_selection", ["Training selection (XGBoost, consistent)", "Transfer F1", "abs mag KS vs ZTF", "abs mag median gap"], rows,
          "Simulated ZTF brightness selection on PLAsTiCC training data")
    verd["Z5_selection_gt_none"] = gt(arr(sel[1][0]), arr(sel[0][0]))
    verd["Z5_selection_gt_random"] = gt(arr(sel[1][0]), arr(sel[2][0]))
    for k in ["ztf_xgboost_consistent", "ztf_xgboost_hybrid"]:
        if k in R:
            verd[f"calibration_{k}"] = {m: round(float(arr(k, m).mean()), 4) for m in ["transfer_log_loss", "transfer_brier", "transfer_ece"]}
    rows = []
    for n in names:
        if counts[n]:
            rows.append([f"{n} ({counts[n]})"] + [fmt(np.array([x["transfer_per_class_f1"][n] for x in R[k]["runs"]])) if k in R else "--"
                                                  for k in ["ztf_xgboost_consistent", "ztf_xgboost_hybrid"]])
    write("ztf_per_class", ["Class (ZTF n)", "Consistent", "Hybrid rule"], rows, "Per-class transfer F1 on ZTF, XGBoost")
    dm = os.path.join(HERE, "results", "diagnostic_damage_ztf.json")
    if os.path.exists(dm):
        verd["shap_damage_ztf"] = json.load(open(dm))["verdict"]

# ---------------- 4.2 acquisition ----------------
acq_p = os.path.join(HERE, "results", "acquisition.json")
if os.path.exists(acq_p):
    acq = json.load(open(acq_p)); runs = acq["runs"]
    strategies = [("random", "Random"), ("predicted_stratified", "Predicted-class stratified"), ("uncertainty", "Uncertainty"),
                  ("true_stratified", "True-class stratified (upper bound)")]
    budgets = sorted({r["budget"] for r in runs if r["budget"] > 0})
    get = lambda s, b, key="macro_f1", cls=None: np.array([(r["per_class_f1"][cls] if cls else r[key]) for r in runs if r["strategy"] == s and r["budget"] == b])
    none = get("none", 0.0)
    rows = [["No acquisition", fmt(none)] + ["" for _ in budgets[1:]]] if len(none) else []
    for s, sn in strategies:
        rows.append([sn] + [fmt(get(s, b)) for b in budgets])
    write("acquisition", ["Strategy"] + [f"{int(b * 100)}% of pool" for b in budgets], rows,
          f"Macro-F1 on the held-out DES test set ({acq['test_size']} objects) after acquiring labels from the pool ({acq['pool_size']}); 10 repeats")
    for b in (0.10, 0.20):
        verd[f"A1_predicted_stratified_gt_random_{int(b * 100)}pct"] = gt(get("predicted_stratified", b), get("random", b))
    verd["A2_true_stratified_ge_all_usable"] = all(
        get("true_stratified", b).mean() >= max(get(s, b).mean() for s in ("random", "predicted_stratified", "uncertainty")) for b in budgets)
    verd["acquisition_means"] = {s: {str(b): round(float(get(s, b).mean()), 4) for b in budgets} for s, _ in strategies}
    panels = [("Macro-F1", None), ("SNII", "SNII"), ("SNIbc", "SNIbc"), ("SLSN-I", "SLSN-I")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    colours = {"random": "#888888", "predicted_stratified": "#1f77b4", "uncertainty": "#2ca02c", "true_stratified": "#d62728"}
    for ax, (title, cls) in zip(axes, panels):
        for s, sn in strategies:
            mu = [get(s, b, cls=cls).mean() for b in budgets]; sd = [get(s, b, cls=cls).std() for b in budgets]
            x = [b * 100 for b in budgets]
            ax.plot(x, mu, marker="o", color=colours[s], label=sn, ls="--" if s == "true_stratified" else "-")
            ax.fill_between(x, np.array(mu) - sd, np.array(mu) + sd, color=colours[s], alpha=0.12)
        if len(none):
            base = np.mean([r["per_class_f1"][cls] if cls else r["macro_f1"] for r in runs if r["strategy"] == "none"])
            ax.axhline(base, color="black", lw=0.8, ls=":", label="No acquisition")
        ax.set_title(title, fontsize=10); ax.set_xlabel("Labels acquired (% of pool)"); ax.set_ylabel("F1 on held-out DES")
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "acquisition_curves.png"), dpi=150)

json.dump(verd, open(os.path.join(HERE, "results", "verdicts_4.json"), "w"), indent=1)
for t in ["ztf_transfer", "ztf_selection", "ztf_per_class", "acquisition"]:
    p = os.path.join(HERE, "tables", t + ".md")
    if os.path.exists(p):
        print(open(p).read())
print(json.dumps(verd, indent=1))
