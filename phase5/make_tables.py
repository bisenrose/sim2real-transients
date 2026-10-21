"""Phase 5 tables and verdicts (C, W, S, K, N predictions; G descriptive). Usage: python make_tables.py"""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(HERE, "results", "*.json"))
     if "verdict" not in p and "quick" not in p}
os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
MODELS = [("xgboost", "XGBoost"), ("lightgbm", "LightGBM"), ("catboost", "CatBoost"), ("mlp", "MLP")]


def arr(k, m="transfer_macro_f1"):
    r = R.get(k)
    return np.array([x[m] for x in r["runs"] if m in x]) if r else np.array([])


def fmt(a):
    return "--" if len(a) == 0 else f"{a.mean():.3f} ± {a.std():.3f}"


def pooled(a, b):
    return np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)


def gt(a, b):
    return None if len(a) == 0 or len(b) == 0 else bool(a.mean() - b.mean() > pooled(a, b))


def eq(a, b):
    return None if len(a) == 0 or len(b) == 0 else bool(abs(a.mean() - b.mean()) <= pooled(a, b))


def write(name, header, rows, caption):
    md = [f"**{caption}**", "", "| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    open(os.path.join(HERE, "tables", name + ".md"), "w").write("\n".join(md) + "\n")
    esc = lambda s: str(s).replace("±", r"$\pm$").replace("_", r"\_").replace("τ", r"$\tau$").replace("≥", r"$\geq$")
    tex = [r"\begin{table}", r"\centering", r"\caption{" + esc(caption) + "}", r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}", r"\hline",
           " & ".join(map(esc, header)) + r" \\", r"\hline"] + [" & ".join(map(esc, r)) + r" \\" for r in rows] + [r"\hline", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(HERE, "tables", name + ".tex"), "w").write("\n".join(tex) + "\n")
    print("\n".join(md) + "\n")


def count(results):
    v = [x for x in results if x is not None]
    return f"{sum(v)}/{len(v)}"


V = {}
# ---- C: controlled support test ---------------------------------------------------------------------
MODES = [("consistent", "Consistent"), ("per_survey", "Per-survey"), ("hybrid", "Hybrid rule"), ("hybrid_absmag_only", "Hybrid, abs mag only")]
c1, c2, c3 = [], [], []
for kind, lab in [("lowz", "low-z (no redshift overlap)"), ("random", "random (control)")]:
    rows = []
    for m, mn in MODELS:
        rows.append([mn] + [fmt(arr(f"C_{kind}_{m}_{p}")) for p, _ in MODES] + [fmt(arr(f"C_{kind}_{m}_hybrid", "plasticc_macro_f1"))])
        h, ao, ps = arr(f"C_{kind}_{m}_hybrid"), arr(f"C_{kind}_{m}_hybrid_absmag_only"), arr(f"C_{kind}_{m}_per_survey")
        if kind == "lowz":
            c1.append(gt(ao, h)); c3.append(gt(ps, h))
        else:
            c2.append(eq(ao, h))
    write(f"C_{kind}", ["Model"] + [p for _, p in MODES] + ["In-domain F1"], rows,
          f"Controlled support test, PLAsTiCC pseudo-target {lab}: transfer macro-F1 (5 split seeds)")
V["C1_lowz_absmag_only_gt_hybrid"] = {"per_model": c1, "count": count(c1), "supported": sum(x for x in c1 if x is not None) >= 3}
V["C2_random_absmag_only_eq_hybrid"] = {"per_model": c2, "count": count(c2), "supported": sum(x for x in c2 if x is not None) >= 3}
V["C3_lowz_hybrid_lt_per_survey"] = {"per_model": c3, "count": count(c3), "supported": sum(x for x in c3 if x is not None) >= 3}

# ---- W and S: weighting and StratLearn --------------------------------------------------------------
rows = []
for tgt in ["des", "ztf"]:
    for m, mn in MODELS[:2]:
        b = f"B_{tgt}_{m}_consistent"
        for k, lab in [(b, "Consistent (baseline)"), (f"W_{tgt}_{m}_consistent_weighted", "+ magnitude weights"),
                       (f"S_{tgt}_{m}_stratlearn_all", "+ StratLearn, all features"),
                       (f"S_{tgt}_{m}_stratlearn_no_survey", "+ StratLearn, survey features hidden")]:
            extra = ""
            if k.startswith("W_") and k in R:
                extra = f"ESS {np.mean([x['effective_sample_size'] for x in R[k]['runs']]):.0f}"
            if k.startswith("S_") and k in R:
                extra = (f"AUC {np.mean([x['propensity_auc'] for x in R[k]['runs']]):.3f}; "
                         f"fallback {np.mean([x['target_fraction_fallback'] for x in R[k]['runs']]):.2f}")
            rows.append([tgt.upper(), mn, lab, fmt(arr(k)), fmt(arr(k, "plasticc_macro_f1")), extra])
write("W_S_baselines", ["Target", "Model", "Method", "Transfer F1", "In-domain F1", "Diagnostics"], rows,
      "Magnitude-ratio weights and StratLearn (consistent preprocessing, 5 split seeds)")
for m, _ in MODELS[:2]:
    bd, bz = arr(f"B_des_{m}_consistent"), arr(f"B_ztf_{m}_consistent")
    V[f"W1_des_{m}_weighted_gt_consistent"] = gt(arr(f"W_des_{m}_consistent_weighted"), bd)
    ind = arr(f"W_des_{m}_consistent_weighted", "plasticc_macro_f1")
    V[f"W2_des_{m}_in_domain_ge_0.76"] = None if len(ind) == 0 else bool(ind.mean() >= 0.76)
    wz = arr(f"W_ztf_{m}_consistent_weighted")
    V[f"W3_ztf_{m}_weighted_not_better"] = None if len(wz) == 0 or len(bz) == 0 else not gt(wz, bz)
    V[f"S1_des_{m}_stratlearn_all_not_better"] = None if len(bd) == 0 else not gt(arr(f"S_des_{m}_stratlearn_all"), bd)
    V[f"S2_des_{m}_stratlearn_no_survey_gt_consistent"] = gt(arr(f"S_des_{m}_stratlearn_no_survey"), bd)

# ---- K: M_peak variants ---------------------------------------------------------------------------
rows = []
for tgt in ["des", "ztf"]:
    for m, mn in MODELS[:2]:
        for var in ["bandwidth", "rband"]:
            c, ps, h = (arr(f"K_{tgt}_{m}_{var}_{p}") for p in ["consistent", "per_survey", "hybrid"])
            rows.append([tgt.upper(), mn, var, fmt(c), fmt(ps), fmt(h)])
            if tgt == "des":
                V[f"K1_des_{m}_{var}_hybrid_gt_consistent"] = gt(h, c)
            else:
                V[f"K2_ztf_{m}_{var}_hybrid_lt_per_survey"] = gt(ps, h)
write("K_absmag_variants", ["Target", "Model", "M_peak variant", "Consistent", "Per-survey", "Hybrid rule"], rows,
      "Alternative peak absolute magnitude definitions (5 split seeds)")

# ---- N: photo-z noise -------------------------------------------------------------------------------
rows, gaps = [], {}
for tgt in ["des", "ztf"]:
    for sig in ["02", "05"]:
        c, ps, h = (arr(f"N_{tgt}_xgboost_sigma{sig}_{p}") for p in ["consistent", "per_survey", "hybrid"])
        rows.append([tgt.upper(), f"0.{sig}", fmt(c), fmt(ps), fmt(h)])
        if tgt == "des":
            V[f"N_des_sigma{sig}_hybrid_gt_consistent"] = gt(h, c)
            gaps[sig] = None if len(h) == 0 or len(ps) == 0 else float(h.mean() - ps.mean())
write("N_photoz", ["Target", "σ/(1+z)", "Consistent", "Per-survey", "Hybrid rule"], rows,
      "Target redshifts replaced by noisy photometric-like values (XGBoost, 5 split seeds)")
V["N3_des_gap_smaller_at_0.05"] = None if None in gaps.values() or len(gaps) < 2 else bool(gaps["05"] < gaps["02"])
V["N3_gaps_hybrid_minus_per_survey"] = gaps

# ---- G: overlap-gated rule (exploratory) --------------------------------------------------------------
rows = []
for tgt, lab in [("des", "DES"), ("ztf", "ZTF"), ("lowz", "PLAsTiCC low-z"), ("random", "PLAsTiCC random")]:
    for m, mn in MODELS[:2]:
        for tau in ["30", "50", "70", "90"]:
            k = f"G_{tgt}_{m}_tau{tau}"
            if k not in R:
                continue
            kept = R[k]["runs"][0].get("gate_kept_shared", [])
            ov = R[k]["runs"][0].get("gate_overlap", {})
            rows.append([lab, mn, f"0.{tau}", ", ".join(kept) or "none", ", ".join(f"{a} {b:.2f}" for a, b in ov.items()), fmt(arr(k))])
write("G_gated", ["Target", "Model", "τ", "Kept on shared scale", "Overlap", "Transfer F1"], rows,
      "EXPLORATORY overlap-gated rule: sensitivity to the threshold τ (5 split seeds)")
json.dump(V, open(os.path.join(HERE, "results", "verdicts_5.json"), "w"), indent=1)
print(json.dumps(V, indent=1))
