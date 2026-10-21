"""
Phase 4.2: honest label-acquisition experiment on DES (design fixed in PREREGISTRATION_4.md).

A fixed 30% DES test set; acquisition only from the other 70% (the pool). Strategies:
random, predicted-class stratified, uncertainty (entropy), true-class stratified (upper bound).
Acquired DES objects are added to the PLAsTiCC training set (weight 5) and the model is retrained.

Usage:  python acquisition.py      -> results/acquisition.json (resumable), then make_tables.py
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase1"))
import common as C, models as M  # noqa: E402

SPEC = dict(type="xgboost", n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
BUDGETS = [0.05, 0.10, 0.20, 0.40]
REPEATS = 10
STRATEGIES = ["random", "predicted_stratified", "uncertainty", "true_stratified"]
WEIGHT = 5.0
OUT = os.path.join(HERE, "results", "acquisition.json")


def stratified_pick(groups, budget, rng):
    """Equal share per group; groups with too few members give their share to the others."""
    pools = {g: list(rng.permutation(idx)) for g, idx in groups.items() if len(idx)}
    chosen = []
    while len(chosen) < budget and any(pools.values()):
        for g in list(pools):
            if pools[g] and len(chosen) < budget:
                chosen.append(pools[g].pop())
    return np.array(chosen, int)


def main():
    plas, feat = C.load_plasticc()
    des = C.load_des(); lab = C.des_labels(des, "corrected")
    des = des[lab.isin(C.SHARED).values].reset_index(drop=True); ydes_codes = lab[lab.isin(C.SHARED)].values
    le = LabelEncoder().fit(sorted(C.SHARED)); names = [C.SHARED[c] for c in le.classes_]
    y_des = le.transform(ydes_codes); X_des = C.derive_features(des[feat])
    idx = np.arange(len(y_des))
    pool_idx, test_idx = train_test_split(idx, test_size=0.30, stratify=y_des, random_state=0)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"test_size": int(len(test_idx)), "pool_size": int(len(pool_idx)),
                                                              "class_order": names, "runs": []}
    done = {(r["repeat"], r["strategy"], r["budget"]) for r in res["runs"]}
    for rep in range(REPEATS):
        tr, _ = train_test_split(plas, test_size=1570 / len(plas), stratify=plas["target"], random_state=rep % 5)
        tr = tr[tr["target"].isin(C.SHARED)]
        X_tr = C.derive_features(tr[feat]); y_tr = le.transform(tr["target"])
        A, (D,) = C.preprocess(X_tr, [X_des], "hybrid", same_survey=[False])   # D: all DES, hybrid-scaled
        base = M.make_model(SPEC, rep).fit(A.values, y_tr)
        prob_pool = base.predict_proba(D.values[pool_idx])
        pred_pool = prob_pool.argmax(1)
        entropy = -(prob_pool * np.log(np.clip(prob_pool, 1e-12, 1))).sum(1)
        test_f1 = lambda p: f1_score(y_des[test_idx], p, average="macro", zero_division=0)
        per_class = lambda p: dict(zip(names, map(float, f1_score(y_des[test_idx], p, average=None, labels=range(len(names)), zero_division=0))))
        if (rep, "none", 0.0) not in done:
            p = base.predict(D.values[test_idx])
            res["runs"].append({"repeat": rep, "strategy": "none", "budget": 0.0, "n_acquired": 0,
                                "macro_f1": float(test_f1(p)), "per_class_f1": per_class(p)})
        for budget in BUDGETS:
            n = int(round(budget * len(pool_idx)))
            for strat in STRATEGIES:
                if (rep, strat, budget) in done:
                    continue
                rng = np.random.RandomState(1000 * rep + int(budget * 100))
                if strat == "random":
                    pick = rng.choice(len(pool_idx), n, replace=False)
                elif strat == "uncertainty":
                    pick = np.argsort(-entropy)[:n]
                elif strat == "predicted_stratified":
                    pick = stratified_pick({c: np.where(pred_pool == c)[0] for c in range(len(names))}, n, rng)
                else:  # true_stratified: upper bound only
                    pick = stratified_pick({c: np.where(y_des[pool_idx] == c)[0] for c in range(len(names))}, n, rng)
                acq = pool_idx[pick]
                Xa = np.r_[A.values, D.values[acq]]; ya = np.r_[y_tr, y_des[acq]]
                w = np.r_[np.ones(len(y_tr)), np.full(len(acq), WEIGHT)]
                m = M.make_model(SPEC, rep).fit(Xa, ya, sample_weight=w)
                p = m.predict(D.values[test_idx])
                res["runs"].append({"repeat": rep, "strategy": strat, "budget": budget, "n_acquired": int(len(acq)),
                                    "acquired_class_counts": {names[c]: int(k) for c, k in zip(*np.unique(y_des[acq], return_counts=True))},
                                    "macro_f1": float(test_f1(p)), "per_class_f1": per_class(p)})
        tmp = OUT + ".tmp"; json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, OUT)
        print(f"  repeat {rep} done ({len(res['runs'])} runs saved)", flush=True)


if __name__ == "__main__":
    main()
