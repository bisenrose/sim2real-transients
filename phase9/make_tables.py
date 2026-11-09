"""Phase 9 tables and verdicts. Compares the ablation (survey-artefact features removed) with the
paper's main results. Usage: python make_tables.py  -> tables/ablation.md, tables/ablation.tex,
results/verdicts_9.json"""
import json, os
import numpy as np
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = [("xgboost", "XGBoost"), ("lightgbm", "LightGBM"), ("catboost", "CatBoost"), ("mlp", "MLP")]
MODES = [("consistent", "Consistent"), ("per_survey", "Per-survey"), ("hybrid", "Hybrid rule"), ("hybrid_plus_brightness", "Hybrid + brightness")]
TARGETS = [("des", "DES"), ("ztf", "ZTF"), ("yse", "YSE")]
# Published values (paper Tables 3 and 5) with all features, for the side-by-side table.
PAPER = {
 "des": {"xgboost": [.488, .528, .602, .597], "lightgbm": [.506, .518, .603, .598], "catboost": [.498, .517, .589, .617], "mlp": [.466, .564, .617, .661]},
 "ztf": {"xgboost": [.503, .510, .442, .552], "lightgbm": [.517, .526, .454, .538], "catboost": [.500, .506, .399, .516], "mlp": [.192, .511, .495, .244]},
 "yse": {"xgboost": [.586, .518, .463, .546], "lightgbm": [.571, .511, .471, .535], "catboost": [.587, .519, .449, .520], "mlp": [.273, .522, .517, .352]}}


def scores(name):
    p = os.path.join(HERE, "results", name + ".json")
    if not os.path.exists(p):
        return np.array([])
    r = json.load(open(p)); y = np.array(r["y_des"]); vals, cnt = np.unique(y, return_counts=True)
    lab = sorted(vals[cnt >= 10].tolist())   # macro-F1 over classes with >= 10 target objects
    return np.array([f1_score(y, np.array(x["des_pred"]), average="macro", labels=lab, zero_division=0) for x in r["runs"]])


def gt(a, b):
    if len(a) == 0 or len(b) == 0:
        return None
    return bool(a.mean() - b.mean() > np.sqrt((a.std() ** 2 + b.std() ** 2) / 2))


S = {(t, m, q): scores(f"noart_{t}_{m}_{q}") for t, _ in TARGETS for m, _ in MODELS for q, _ in MODES}
fmt = lambda a: "--" if len(a) == 0 else f"{a.mean():.3f} ± {a.std():.3f}"
rows = []
for t, tl in TARGETS:
    for i, (m, ml) in enumerate(MODELS):
        rows.append([tl, ml] + [f"{PAPER[t][m][k]:.3f} → {fmt(S[(t, m, q)])}" for k, (q, _) in enumerate(MODES)])
hdr = ["Target", "Model"] + [l for _, l in MODES]
md = ["**Transfer macro-F1 with all features (paper) → without the survey-artefact features (time span, t0_g, t0_r); mean ± SD over 5 split seeds**", "",
      "| " + " | ".join(hdr) + " |", "| " + " | ".join("---" for _ in hdr) + " |"] + ["| " + " | ".join(r) + " |" for r in rows]
open(os.path.join(HERE, "tables", "ablation.md"), "w").write("\n".join(md) + "\n"); print("\n".join(md))
tex = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule", " & ".join(hdr) + r" \\", r"\midrule"]
for r in rows:
    tex.append(" & ".join(x.replace("±", r"$\pm$").replace("→", r"$\to$") for x in r) + r" \\")
tex += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(HERE, "tables", "ablation.tex"), "w").write("\n".join(tex) + "\n")

trees = ["xgboost", "lightgbm", "catboost"]
V = {}
V["A1_des_hybrid_gt_consistent"] = [gt(S[("des", m, "hybrid")], S[("des", m, "consistent")]) for m, _ in MODELS]
V["A2_ztf_hybrid_lt_per_survey"] = [gt(S[("ztf", m, "per_survey")], S[("ztf", m, "hybrid")]) for m in trees]
V["A3_yse_hybrid_lt_per_survey"] = [gt(S[("yse", m, "per_survey")], S[("yse", m, "hybrid")]) for m in trees]
V["A4_yse_consistent_best"] = [None if any(len(S[("yse", m, q)]) == 0 for q, _ in MODES) else
                               bool(all(S[("yse", m, "consistent")].mean() >= S[("yse", m, q)].mean() for q, _ in MODES)) for m in trees]
V["A5_ztf_brightness_gt_hybrid"] = [gt(S[("ztf", m, "hybrid_plus_brightness")], S[("ztf", m, "hybrid")]) for m in trees]
for k in list(V):
    V[k + "_supported"] = None if None in V[k] else all(V[k])
pa = os.path.join(HERE, "results", "domain_auc.json")
if os.path.exists(pa):
    a = json.load(open(pa)); V["domain_auc"] = a
    V["A6_auc_without_artefacts_ge_0.95_supported"] = all(v["without_artefacts"] >= 0.95 for v in a.values())
json.dump(V, open(os.path.join(HERE, "results", "verdicts_9.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in V.items() if k.endswith("supported")}, indent=1))
