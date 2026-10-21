"""
Phase 2.1 tables and prediction verdicts, built only from results/*.json.
Usage:  python make_tables.py   -> tables/*.md, tables/*.tex, results/verdicts.json
"""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(HERE, "results", "*.json"))
     if "diagnostic" not in p and "verdicts" not in p}
MODELS = [("xgboost", "XGBoost"), ("lightgbm", "LightGBM"), ("catboost", "CatBoost"), ("mlp", "MLP")]
PREPS = [("consistent", "Consistent"), ("per_survey", "Per-survey"), ("per_survey_drop_absmag", "Per-survey − abs mag"), ("hybrid", "Hybrid (rule)")]
os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)


def vals(key, metric="transfer_macro_f1"):
    r = R.get(key)
    return np.array([x[metric] for x in r["runs"] if metric in x]) if r else None


def fmt(v):
    return "--" if v is None or len(v) == 0 else f"{v.mean():.3f} ± {v.std():.3f}"


def write(name, header, rows, caption):
    md = [f"**{caption}**", "", "| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    open(os.path.join(HERE, "tables", name + ".md"), "w").write("\n".join(md) + "\n")
    esc = lambda s: s.replace("±", r"$\pm$").replace("−", "$-$").replace("≥", r"$\geq$").replace(">", "$>$").replace("_", r"\_")
    tex = [r"\begin{table}", r"\centering", r"\caption{" + esc(caption) + "}", r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}",
           r"\hline", " & ".join(map(esc, header)) + r" \\", r"\hline"] + [" & ".join(map(esc, r)) + r" \\" for r in rows] + [r"\hline", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(HERE, "tables", name + ".tex"), "w").write("\n".join(tex) + "\n")


def better(a, b, allow_equal=False):
    """a > b by more than one pooled SD (or, if allow_equal, a not worse than b by more than one pooled SD)."""
    pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    d = a.mean() - b.mean()
    return bool(d >= -pooled) if allow_equal else bool(d > pooled)


verdicts = {}
for lab, lab_name in [("corrected", "all spectroscopic types"), ("confirmed_only", "confident types only")]:
    rows = []
    for m, mn in MODELS:
        cells = [fmt(vals(f"{m}_{lab}_{p}")) for p, _ in PREPS]
        cells.append(fmt(vals(f"{m}_{lab}_consistent", "des_cv_macro_f1_mean_of_folds")))
        backend = R.get(f"{m}_{lab}_hybrid", {}).get("model_backend", m)
        rows.append([mn + (" (sklearn)" if backend == "mlp-sklearn-fallback" else "")] + cells)
        h, c, ps, pd_ = (vals(f"{m}_{lab}_{p}") for p in ["hybrid", "consistent", "per_survey", "per_survey_drop_absmag"])
        if all(x is not None for x in (h, c, ps, pd_)):
            status = R[f"{m}_{lab}_hybrid"]["config"].get("status", "")
            verdicts[f"{m}_{lab}"] = {"status": status, "P1_hybrid_gt_consistent": better(h, c),
                                     "P2_hybrid_gt_per_survey": better(h, ps),
                                     "P3_hybrid_ge_drop_absmag": better(h, pd_, allow_equal=True)}
    write(f"transfer_{lab}", ["Model"] + [p for _, p in PREPS] + ["DES-only CV"], rows,
          f"PLAsTiCC → DES transfer macro-F1, DES {lab_name} (mean ± SD, 5 split seeds)")

rows = [[k.replace("_corrected", " / all types").replace("_confirmed_only", " / confident only"),
         "dev" if v["status"].startswith("development") else "confirm",
         *["yes" if v[p] else "NO" for p in ["P1_hybrid_gt_consistent", "P2_hybrid_gt_per_survey", "P3_hybrid_ge_drop_absmag"]]]
        for k, v in sorted(verdicts.items())]
write("verdicts", ["Model / labels", "Status", "P1 > consistent", "P2 > per-survey", "P3 ≥ drop abs mag"], rows,
      "Pre-registered predictions P1–P3 (supported = difference exceeds one pooled SD)")

# in-domain cost of each preprocessing (PLAsTiCC test set)
rows = [[mn] + [fmt(vals(f"{m}_corrected_{p}", "plasticc_macro_f1")) for p, _ in PREPS] for m, mn in MODELS]
write("plasticc_in_domain", ["Model"] + [p for _, p in PREPS], rows, "PLAsTiCC in-domain macro-F1 (unaffected by DES preprocessing except abs-mag removal)")

dmg = os.path.join(HERE, "results", "diagnostic_damage.json")
if os.path.exists(dmg):
    d = json.load(open(dmg))
    rows = [[r["feature"], f"{r['damage_mean']:.3f}", f"{r['importance']:.3f}", f"{r['displacement_sd']:.2f}", r["rank_per_seed"]] for r in d["table"][:10]]
    write("damage_top10", ["Feature", "SHAP damage", "Importance", "Re-centring shift (SD)", "Rank per seed"], rows,
          "Label-free SHAP damage from per-survey re-centring (XGBoost, top 10)")
    verdicts["P4_P5_diagnostic"] = d["verdict"]
json.dump(verdicts, open(os.path.join(HERE, "results", "verdicts.json"), "w"), indent=1)
print("\n\n".join(open(os.path.join(HERE, "tables", f)).read() for f in sorted(os.listdir(os.path.join(HERE, "tables"))) if f.endswith(".md")))
