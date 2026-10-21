"""
Phase 2.3: per-feature shift measures (label-free) vs the effect of removing each feature
(diagnostic, uses DES labels). Run AFTER the remove_* configs:
    python ../phase1/run_experiment.py --dir .      (from this folder)
    python analysis_2_3.py
Writes results/shift_measures.csv, results/feature_removal.csv, results/verdicts_2.3.json,
tables/*.md|tex, figures/removal_vs_shift.png
"""
import glob, json, os, sys, time, warnings
import numpy as np, pandas as pd, shap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance, spearmanr
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT_X = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT_X, "phase1"))
import common as C, models as M  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
SPEC = dict(type="xgboost", n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)


def shift_measures():
    plas, feat = C.load_plasticc(); des = C.load_des()
    lab = C.des_labels(des, "corrected"); m = lab.isin(C.SHARED).values
    des, ydes = des[m].reset_index(drop=True), lab[m].values
    X_des = C.derive_features(des[feat]); le = LabelEncoder().fit(sorted(C.SHARED))
    rows = []
    t0 = time.time()
    for s in SEEDS:
        print(f"  shift measures: seed {s} ({time.time() - t0:.0f}s elapsed)", flush=True)
        tr, _ = train_test_split(plas, test_size=1570 / len(plas), stratify=plas["target"], random_state=s)
        tr = tr[tr["target"].isin(C.SHARED)]
        X_tr = C.derive_features(tr[feat]); ytr = tr["target"].values
        A, (D,) = C.preprocess(X_tr, [X_des], "consistent", same_survey=[False])
        model = M.make_model(SPEC, s).fit(A.values, le.transform(ytr))
        sv = np.asarray(shap.TreeExplainer(model).shap_values(A.values))
        if sv.ndim == 3 and sv.shape[0] != A.shape[0]:
            sv = np.moveaxis(sv, 0, -1)
        imp = np.abs(sv).mean(axis=tuple(i for i in range(sv.ndim) if i != 1)); imp = imp / imp.sum()
        for j, c in enumerate(A.columns):
            a, d = A[c].values, D[c].values
            x = np.r_[a, d].reshape(-1, 1); y = np.r_[np.zeros(len(a)), np.ones(len(d))]
            prob = cross_val_predict(HistGradientBoostingClassifier(max_iter=100, random_state=s), x, y,
                                     cv=StratifiedKFold(5, shuffle=True, random_state=s), method="predict_proba")[:, 1]
            auc = roc_auc_score(y, prob)
            per_class = {f"ks_{C.SHARED[k]}": ks_2samp(a[ytr == k], d[ydes == k]).statistic for k in C.SHARED}
            rows.append(dict(seed=s, feature=c, ks=ks_2samp(a, d).statistic, wasserstein=wasserstein_distance(a, d),
                             shift=max(0.0, (auc - 0.5) * 2), importance=imp[j], **per_class))
    df = pd.DataFrame(rows).groupby("feature").mean(numeric_only=True).drop(columns="seed")
    df["risk"] = df["importance"] * df["shift"]
    dmg = os.path.join(ROOT_X, "phase2", "results", "diagnostic_damage.csv")
    if os.path.exists(dmg):
        df = df.join(pd.read_csv(dmg, index_col=0)[["damage_mean"]].rename(columns={"damage_mean": "damage"}))
    return df


def removal_effects(feats):
    out = {}
    for prep in ["consistent", "hybrid"]:
        basef = os.path.join(ROOT_X, "phase2", "results", f"xgboost_corrected_{prep}.json")
        base = np.array([r["transfer_macro_f1"] for r in json.load(open(basef))["runs"]])
        for ft in feats:
            p = os.path.join(HERE, "results", f"remove_{prep}_{ft}.json")
            if os.path.exists(p):
                v = np.array([r["transfer_macro_f1"] for r in json.load(open(p))["runs"]])
                out.setdefault(ft, {})[f"gain_{prep}"] = float(np.mean(v - base))  # paired by split seed
                out[ft][f"gain_{prep}_sd"] = float(np.std(v - base))
    return pd.DataFrame(out).T


def table(name, df, cols, caption, fmt="{:.3f}"):
    head = ["Feature"] + cols
    rows = [[i] + [fmt.format(r[c]) if pd.notna(r[c]) else "--" for c in cols] for i, r in df.iterrows()]
    md = [f"**{caption}**", "", "| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"] + ["| " + " | ".join(r) + " |" for r in rows]
    open(os.path.join(HERE, "tables", name + ".md"), "w").write("\n".join(md) + "\n")
    esc = lambda s: str(s).replace("_", r"\_")
    tex = [r"\begin{table}", r"\centering", r"\caption{" + esc(caption) + "}", r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\hline",
           " & ".join(map(esc, head)) + r" \\", r"\hline"] + [" & ".join(map(esc, r)) + r" \\" for r in rows] + [r"\hline", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(HERE, "tables", name + ".tex"), "w").write("\n".join(tex) + "\n")


def main():
    for d in ["results", "tables", "figures"]:
        os.makedirs(os.path.join(HERE, d), exist_ok=True)
    print("Computing label-free shift measures (5 seeds x 27 features)...", flush=True)
    sm = shift_measures(); sm.round(4).to_csv(os.path.join(HERE, "results", "shift_measures.csv"))
    rem = removal_effects(list(sm.index)); rem.round(4).to_csv(os.path.join(HERE, "results", "feature_removal.csv"))
    df = sm.join(rem)
    top = df.sort_values("ks", ascending=False).head(10)
    table("top10_shift", top, ["ks", "wasserstein", "shift", "importance", "risk"], "Top 10 shifted features (PLAsTiCC-standardised; mean over 5 seeds)")
    pcs = [c for c in df.columns if c.startswith("ks_")]
    table("per_class_ks", top, pcs, "Per-class KS statistic for the top 10 shifted features (descriptive; uses DES labels)", "{:.2f}")
    table("removal", df.sort_values("gain_consistent", ascending=False),
          ["gain_consistent", "gain_hybrid", "shift", "risk"] + (["damage"] if "damage" in df else []),
          "Change in transfer macro-F1 from removing each feature (diagnostic; uses DES labels)", "{:+.3f}")
    corr = {}
    for prep in ["consistent", "hybrid"]:
        g = f"gain_{prep}"
        if g in df:
            for meas in ["shift", "ks", "risk"] + (["damage"] if "damage" in df else []):
                rho, p = spearmanr(df[meas], df[g], nan_policy="omit")
                corr[f"{prep}: {meas}"] = {"spearman": round(float(rho), 3), "p": round(float(p), 4)}
    verdict = {"R1_risk_beats_shift_consistent": corr.get("consistent: risk", {}).get("spearman", np.nan) > corr.get("consistent: shift", {}).get("spearman", np.nan),
               "correlations": corr}
    json.dump(verdict, open(os.path.join(HERE, "results", "verdicts_2.3.json"), "w"), indent=1)
    # figure
    meas = ["shift", "risk"] + (["damage"] if "damage" in df else [])
    fig, axes = plt.subplots(2, len(meas), figsize=(4.6 * len(meas), 7.5))
    for i, prep in enumerate(["consistent", "hybrid"]):
        for j, mname in enumerate(meas):
            ax = axes[i, j]; x, y = df[mname], df[f"gain_{prep}"]
            ax.scatter(x, y, s=18, color="#1f77b4"); ax.axhline(0, color="#999", lw=.8)
            for ft in ["abs_peak_mag", "redshift", "extra_sigma_ZTF_g", "time_span"]:
                if ft in df.index:
                    ax.annotate(ft, (x[ft], y[ft]), fontsize=7, xytext=(3, 3), textcoords="offset points")
            r = corr.get(f"{prep}: {mname}", {}).get("spearman", np.nan)
            ax.set_title(f"{prep}: gain vs {mname}  (Spearman ρ = {r:.2f})", fontsize=9)
            ax.set_xlabel(mname); ax.set_ylabel("Δ transfer F1 when removed")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "removal_vs_shift.png"), dpi=150)
    for t in ["top10_shift", "removal"]:
        print(open(os.path.join(HERE, "tables", t + ".md")).read())
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()