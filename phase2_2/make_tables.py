"""Phase 2.2 tables and verdicts (Q1-Q4) from results/*.json.  Usage: python make_tables.py"""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(HERE, "results", "*.json")) if "verdict" not in p}
os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
v = lambda k, m="transfer_macro_f1": np.array([x[m] for x in R[k]["runs"]])
f = lambda a, nd=3: f"{a.mean():.{nd}f} ± {a.std():.{nd}f}"


def write(name, header, rows, caption):
    md = [f"**{caption}**", "", "| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"] + ["| " + " | ".join(r) + " |" for r in rows]
    open(os.path.join(HERE, "tables", name + ".md"), "w").write("\n".join(md) + "\n")
    esc = lambda s: s.replace("±", r"$\pm$").replace("_", r"\_").replace("→", r"$\rightarrow$")
    tex = [r"\begin{table}", r"\centering", r"\caption{" + esc(caption) + "}", r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}", r"\hline",
           " & ".join(map(esc, header)) + r" \\", r"\hline"] + [" & ".join(map(esc, r)) + r" \\" for r in rows] + [r"\hline", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(HERE, "tables", name + ".tex"), "w").write("\n".join(tex) + "\n")


SEL = [("none", "None"), ("logistic", "Logistic (DES-matched)"), ("random_logistic", "Random, same size"),
       ("hard_cut", "Hard cut (DES 99th pct)"), ("random_hard_cut", "Random, same size")]
for prefix, cap in [("xgboost_consistent", "XGBoost, consistent preprocessing"), ("xgboost_hybrid", "XGBoost, hybrid rule"),
                    ("lightgbm_consistent", "LightGBM, consistent preprocessing")]:
    rows = []
    for s, sn in SEL:
        k = f"{prefix}_sel-{s}"
        if k not in R:
            continue
        r0 = R[k]["runs"]
        n = np.mean([x.get("n_after_selection", x["n_train"]) for x in r0])
        rows.append([sn, f"{n:.0f}", f(v(k, "plasticc_macro_f1")), f(v(k)), f(v(k, "abs_mag_ks_train_vs_des"), 2), f(v(k, "abs_mag_median_gap"), 2)])
    write(f"selection_{prefix}", ["Training selection", "N train", "PLAsTiCC F1", "Transfer F1", "abs mag KS", "abs mag median gap (DES − train)"],
          rows, f"Simulated DES selection on PLAsTiCC training data: {cap} (5 split seeds)")

# per-class gains (primary setting)
names = R["xgboost_consistent_sel-none"]["class_order"]
pc = lambda k, n: np.array([x["transfer_per_class_f1"][n] for x in R[k]["runs"]])
rows, gains = [], {}
for n in names:
    a, b, c = pc("xgboost_consistent_sel-none", n), pc("xgboost_consistent_sel-logistic", n), pc("xgboost_consistent_sel-random_logistic", n)
    gains[n] = b.mean() - a.mean()
    rows.append([n, f(a), f(b), f(c), f"{gains[n]:+.3f}"])
write("per_class_selection", ["Class", "No selection", "Logistic selection", "Random, same size", "Gain (selection − none)"], rows,
      "Per-class transfer F1, XGBoost, consistent preprocessing")


def better(a, b):
    return bool(a.mean() - b.mean() > np.sqrt((a.std() ** 2 + b.std() ** 2) / 2))


none, sel, rnd = (f"xgboost_consistent_sel-{s}" for s in ["none", "logistic", "random_logistic"])
eligible = {k: g for k, g in gains.items() if k in ("SNIa", "SNII", "AGN", "SLSN-I")}
verd = {"Q1_selection_gt_none": better(v(sel), v(none)),
        "Q2_selection_gt_random_control": better(v(sel), v(rnd)),
        "Q3_abs_mag_shift_shrinks": bool(v(sel, "abs_mag_ks_train_vs_des").mean() < v(none, "abs_mag_ks_train_vs_des").mean()
                                         and abs(v(sel, "abs_mag_median_gap").mean()) < abs(v(none, "abs_mag_median_gap").mean())),
        "Q4_SNIa_largest_gain": max(eligible, key=eligible.get) == "SNIa", "per_class_gains": {k: round(g, 4) for k, g in gains.items()}}
for model in ["lightgbm"]:
    if f"{model}_consistent_sel-logistic" in R:
        verd[f"robustness_{model}_Q1"] = better(v(f"{model}_consistent_sel-logistic"), v(f"{model}_consistent_sel-none"))
        verd[f"robustness_{model}_Q2"] = better(v(f"{model}_consistent_sel-logistic"), v(f"{model}_consistent_sel-random_logistic"))
json.dump(verd, open(os.path.join(HERE, "results", "verdicts_2.2.json"), "w"), indent=1)
for t in sorted(os.listdir(os.path.join(HERE, "tables"))):
    if t.endswith(".md"):
        print(open(os.path.join(HERE, "tables", t)).read())
print(json.dumps(verd, indent=1))
