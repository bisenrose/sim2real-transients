"""
Recompute the SHAP rank correlations (rho) on the CORRECTED pipeline (correct SLSN-I class,
corrected DES labels, no per-domain standardisation). Descriptive; no prediction attached.

For each tree model, preprocessing mode and PLAsTiCC split seed:
  imp_P   mean |TreeSHAP| of the PLAsTiCC-trained model on the PLAsTiCC training set
  imp_PD  mean |TreeSHAP| of the SAME model on DES                       (no DES labels)
  imp_D   mean |TreeSHAP| of a model trained on all DES (DES labels) on DES
  rho_label_free = Spearman(imp_P, imp_PD)     rho_labelled = Spearman(imp_P, imp_D)
over the 27 features, plus the rank of abs_peak_mag in each ranking.

Note: rho_label_free compares the same trees explained on two inputs, so it is high almost by
construction and is weak evidence of anything; rho_labelled is the informative one.

Usage:  python rho_recompute.py     -> tables/rho.md, results/rho.json
"""
import json, os, sys, warnings
import numpy as np, pandas as pd, shap
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); XAI = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(XAI, "phase1"))
import common as C, models as M  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
MODELS = ["xgboost", "lightgbm", "catboost"]          # TreeSHAP needs tree models
MODES = ["consistent", "hybrid"]
SPEC = lambda t: dict(type=t, n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)


def mean_abs_shap(model, X):
    sv = shap.TreeExplainer(model).shap_values(X)
    sv = np.asarray(sv)
    if sv.ndim == 3 and sv.shape[0] != X.shape[0]:      # (classes, n, f) -> (n, f, classes)
        sv = np.moveaxis(sv, 0, -1)
    return np.abs(sv).mean(axis=tuple(i for i in range(sv.ndim) if i != 1))


def main():
    plas, feat = C.load_plasticc()
    des = C.load_des(); lab = C.des_labels(des, "corrected"); keep = lab.isin(C.SHARED).values
    des = des[keep].reset_index(drop=True); le = LabelEncoder().fit(sorted(C.SHARED))
    y_des = le.transform(lab[keep].values); X_des = C.derive_features(des[feat])
    rows = []
    for m in MODELS:
        for mode in MODES:
            for s in SEEDS:
                tr, _ = train_test_split(plas, test_size=1570 / len(plas), stratify=plas["target"], random_state=s)
                tr = tr[tr["target"].isin(C.SHARED)]
                X_tr = C.derive_features(tr[feat]); y_tr = le.transform(tr["target"])
                A, (D,) = C.preprocess(X_tr, [X_des], mode, same_survey=[False])
                mp = M.make_model(SPEC(m), s).fit(A.values, y_tr)
                imp_P, imp_PD = mean_abs_shap(mp, A.values), mean_abs_shap(mp, D.values)
                Dd, _ = C.preprocess(X_des[A.columns], [], "consistent")      # DES model on DES's own scale
                md = M.make_model(SPEC(m), s).fit(Dd.values, y_des)
                imp_D = mean_abs_shap(md, Dd.values)
                r1, p1 = spearmanr(imp_P, imp_PD); r2, p2 = spearmanr(imp_P, imp_D)
                rank = lambda v: int((-v).argsort().tolist().index(list(A.columns).index("abs_peak_mag")) + 1)
                rows.append(dict(model=m, mode=mode, seed=s, rho_label_free=r1, p_label_free=p1, rho_labelled=r2, p_labelled=p2,
                                 absmag_rank_P=rank(imp_P), absmag_rank_PD=rank(imp_PD), absmag_rank_D=rank(imp_D)))
                print(f"  {m} {mode} seed {s}: rho_label_free {r1:.3f} (p={p1:.1e})  rho_labelled {r2:.3f} (p={p2:.1e})", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True); os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
    df.to_csv(os.path.join(HERE, "results", "rho_per_seed.csv"), index=False)
    g = df.groupby(["model", "mode"])
    t = pd.DataFrame({
        "rho label-free (P model: PLAsTiCC vs DES)": g.rho_label_free.agg(lambda v: f"{v.mean():.3f} ± {v.std(ddof=0):.3f}"),
        "max p": g.p_label_free.max().map(lambda p: f"{p:.1e}"),
        "rho labelled (P model vs DES model)": g.rho_labelled.agg(lambda v: f"{v.mean():.3f} ± {v.std(ddof=0):.3f}"),
        "max p ": g.p_labelled.max().map(lambda p: f"{p:.1e}"),
        "abs_peak_mag rank (P / P-on-DES / DES)": g.apply(lambda x: f"{x.absmag_rank_P.median():.0f} / {x.absmag_rank_PD.median():.0f} / {x.absmag_rank_D.median():.0f}"),
    }).reset_index()
    md = ["**SHAP rank correlations over 27 features, corrected pipeline (Spearman; mean ± SD over 5 split seeds; ranks are medians)**", "",
          "| " + " | ".join(t.columns) + " |", "| " + " | ".join("---" for _ in t.columns) + " |"]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.values]
    open(os.path.join(HERE, "tables", "rho.md"), "w").write("\n".join(md) + "\n"); print("\n".join(md))
    json.dump(df.to_dict("records"), open(os.path.join(HERE, "results", "rho.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
