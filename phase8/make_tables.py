"""Phase 8 E2 tables and verdicts (R1-R5), plus the E1 verdicts if E1 has run. Usage: python make_tables.py"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODES = [("consistent", "Consistent"), ("per_survey", "Per-survey"), ("hybrid", "Hybrid rule"), ("hybrid_plus_brightness", "Hybrid + brightness")]
POPS = [("control", "Control (all z)"), ("near3", "Moved nearer x3"), ("near6", "Moved nearer x6")]
SHIFTS = [("none", "none"), ("cal05", "0.5 mag calibration"), ("cal10", "1.0 mag calibration"), ("season", "strong season")]
MODELS = [("xgboost", "XGBoost"), ("lightgbm", "LightGBM")]


def arr(k):
    p = os.path.join(HERE, "results", k + ".json")
    if not os.path.exists(p):
        return np.array([])
    return np.array([x["transfer_macro_f1"] for x in json.load(open(p))["runs"]])


fmt = lambda a: "--" if len(a) == 0 else f"{a.mean():.3f} ± {a.std():.3f}"


def gt(a, b):
    return None if len(a) == 0 or len(b) == 0 else bool(a.mean() - b.mean() > np.sqrt((a.std() ** 2 + b.std() ** 2) / 2))


rows = []
for p, pl in POPS:
    for s, sl in SHIFTS:
        for m, ml in MODELS:
            rows.append([pl, sl, ml] + [fmt(arr(f"E2_{p}_{s}_{m}_{q}")) for q, _ in MODES])
hdr = ["Population", "Instrument shift", "Model"] + [l for _, l in MODES]
md = ["**Stronger controlled test (Ia-rich pseudo-targets, 458 objects; macro-F1, 5 split seeds)**", "",
      "| " + " | ".join(hdr) + " |", "| " + " | ".join("---" for _ in hdr) + " |"] + ["| " + " | ".join(r) + " |" for r in rows]
open(os.path.join(HERE, "tables", "E2_controlled.md"), "w").write("\n".join(md) + "\n"); print("\n".join(md) + "\n")
esc = lambda x: str(x).replace("±", r"$\pm$")
tex = [r"\begin{tabular}{lll" + "c" * len(MODES) + "}", r"\hline", " & ".join(hdr) + r" \\", r"\hline"] + [" & ".join(map(esc, r)) + r" \\" for r in rows] + [r"\hline", r"\end{tabular}"]
open(os.path.join(HERE, "tables", "E2_controlled.tex"), "w").write("\n".join(tex) + "\n")

g = lambda p, s, m, q: arr(f"E2_{p}_{s}_{m}_{q}")


def ztf_pattern(p, s, m):
    a, b = gt(g(p, s, m, "per_survey"), g(p, s, m, "hybrid")), gt(g(p, s, m, "hybrid_plus_brightness"), g(p, s, m, "hybrid"))
    return None if a is None or b is None else bool(a and b)


V = {}
for m, _ in MODELS:
    V[f"R1_{m}_near6_none_ztf_pattern"] = ztf_pattern("near6", "none", m)
    V[f"R2_{m}_near3_none_ztf_pattern"] = ztf_pattern("near3", "none", m)
    V[f"R3_{m}_control_cal10_hybrid_lt_per_survey"] = gt(g("control", "cal10", m, "per_survey"), g("control", "cal10", m, "hybrid"))
    a, b = gt(g("control", "season", m, "per_survey"), g("control", "season", m, "consistent")), gt(g("control", "season", m, "hybrid"), g("control", "season", m, "consistent"))
    V[f"R4_{m}_control_season_rescaling_gt_consistent"] = None if a is None or b is None else bool(a and b)
    V[f"R5_{m}_near6_cal10_ztf_pattern"] = ztf_pattern("near6", "cal10", m)
for r in ["R1", "R2", "R3", "R4", "R5"]:
    vals = [v for k, v in V.items() if k.startswith(r + "_")]
    V[f"{r}_supported"] = None if None in vals else all(vals)
e1 = os.path.join(HERE, "results", "E1.json")
if os.path.exists(e1):
    V.update({k: v for k, v in json.load(open(e1)).items() if k.startswith("E1")})
json.dump(V, open(os.path.join(HERE, "results", "verdicts_8.json"), "w"), indent=1)
print(json.dumps(V, indent=1))
