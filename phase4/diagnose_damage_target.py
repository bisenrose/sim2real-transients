"""
Phase 2.1 XAI diagnostic (label-free): which features does per-survey re-centring damage?

For an XGBoost model trained on PLAsTiCC, compute TreeSHAP values on DES twice, with
consistent and with per-survey preprocessing. For each feature j:

    damage_j = mean over DES objects and classes of |SHAP_j(per_survey) - SHAP_j(consistent)|

No DES labels are used anywhere in this script. Also reports, per feature:
  importance_j   mean |SHAP| of the PLAsTiCC model on PLAsTiCC training data
  displacement_j how far re-centring moves the DES median, in PLAsTiCC standard deviations

Phase 4 version: works on any target survey (default ZTF). Descriptive only (no prediction).
Writes results/diagnostic_damage_<target>.csv and .json.
Usage:  python diagnose_damage_target.py [ztf|des]
"""
import json, os, sys, warnings
import numpy as np, pandas as pd, shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase1"))
import common as C   # noqa: E402
import models as M   # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
SPEC = dict(type="xgboost", n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)


def abs_shap(explainer, X):
    sv = np.asarray(explainer.shap_values(X))
    if sv.ndim == 3 and sv.shape[0] != X.shape[0]:   # (classes, n, features) -> (n, features, classes)
        sv = np.moveaxis(sv, 0, -1)
    return sv  # (n, features[, classes])


def main():
    plas, feat = C.load_plasticc()
    target = sys.argv[1] if len(sys.argv) > 1 else "ztf"
    des, lab = C.load_target(target)
    if lab is None:
        lab = C.des_labels(des, "corrected")
    des = des[lab.isin(C.SHARED).values].reset_index(drop=True)  # class membership only, labels unused
    X_des = C.derive_features(des[feat])
    le = LabelEncoder().fit(sorted(C.SHARED))
    rows = []
    for s in SEEDS:
        tr, _ = train_test_split(plas, test_size=1570 / len(plas), stratify=plas["target"], random_state=s)
        tr = tr[tr["target"].isin(C.SHARED)]
        X_tr = C.derive_features(tr[feat]); y_tr = le.transform(tr["target"])
        A, (D_cons,) = C.preprocess(X_tr, [X_des], "consistent", same_survey=[False])
        _, (D_ps,) = C.preprocess(X_tr, [X_des], "per_survey", same_survey=[False])
        model = M.make_model(SPEC, s).fit(A.values, y_tr)
        ex = shap.TreeExplainer(model)
        s_tr, s_c, s_p = abs_shap(ex, A.values), abs_shap(ex, D_cons.values), abs_shap(ex, D_ps.values)
        red = lambda a: a.mean(axis=tuple(i for i in range(a.ndim) if i != 1))
        imp = red(np.abs(s_tr)); damage = red(np.abs(s_p - s_c))
        disp = np.abs(D_cons.median().values)  # per-survey sets the DES median to ~0; consistent leaves it here
        for j, f in enumerate(A.columns):
            rows.append(dict(seed=s, feature=f, importance=imp[j] / imp.sum(), damage=damage[j], displacement_sd=disp[j]))
    df = pd.DataFrame(rows)
    agg = df.groupby("feature").agg(damage_mean=("damage", "mean"), damage_sd=("damage", "std"),
                                    importance=("importance", "mean"), displacement_sd=("displacement_sd", "mean"))
    agg["damage_rank"] = agg["damage_mean"].rank(ascending=False).astype(int)
    ranks = df.assign(r=df.groupby("seed")["damage"].rank(ascending=False)).pivot(index="feature", columns="seed", values="r")
    agg["rank_per_seed"] = ranks.astype(int).astype(str).agg(",".join, axis=1)
    agg = agg.sort_values("damage_mean", ascending=False)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    agg.round(4).to_csv(os.path.join(HERE, "results", f"diagnostic_damage_{target}.csv"))
    top = list(agg.index)
    verdict = {"target": target, "abs_peak_mag_rank": top.index("abs_peak_mag") + 1,
               "redshift_rank": top.index("redshift") + 1, "top5": top[:5]}
    json.dump({"verdict": verdict, "table": agg.round(4).reset_index().to_dict("records")},
              open(os.path.join(HERE, "results", f"diagnostic_damage_{target}.json"), "w"), indent=1)
    print(agg.head(10).round(4).to_string()); print(verdict)


if __name__ == "__main__":
    main()
