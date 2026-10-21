"""
Phase 3.1: nested hyperparameter tuning on PLAsTiCC only.

For each model family and each split seed: random search over N settings (same N for every
model), each scored by 3-fold stratified CV macro-F1 on that seed's PLAsTiCC TRAINING set
(preprocessing fitted inside each fold). DES is never touched. Writes tuning/<model>.json,
which the tuned_* configs read via "tuned_from".

Usage:  python tune.py xgboost [--trials 50] [--seeds 0,1,2,3,4]
Resumable: seeds already in tuning/<model>.json are skipped.
"""
import argparse, json, os, sys, time, warnings
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterSampler
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase1"))
import common as C, models as M  # noqa: E402

SPACES = {
    "xgboost": dict(n_estimators=[100, 200, 300, 500], max_depth=[3, 4, 5, 6, 8], learning_rate=[0.02, 0.05, 0.1],
                    subsample=[0.6, 0.8, 1.0], colsample_bytree=[0.6, 0.8, 1.0], min_child_weight=[1, 3, 5, 10],
                    reg_lambda=[0.1, 1, 5, 10]),
    "lightgbm": dict(n_estimators=[100, 200, 300, 500], max_depth=[-1, 4, 6, 8], num_leaves=[15, 31, 63],
                     learning_rate=[0.02, 0.05, 0.1], subsample=[0.6, 0.8, 1.0], colsample_bytree=[0.6, 0.8, 1.0],
                     min_child_samples=[5, 10, 20, 40], reg_lambda=[0.0, 1, 5]),
    "catboost": dict(n_estimators=[100, 200, 300, 500], max_depth=[4, 5, 6, 8], learning_rate=[0.02, 0.05, 0.1],
                     subsample=[0.6, 0.8, 1.0], colsample_bytree=[0.6, 0.8, 1.0], l2_leaf_reg=[1, 3, 5, 10]),
    "mlp": dict(lr=[0.001, 0.003, 0.01], weight_decay=[1e-5, 1e-4, 1e-3], epochs=[40, 80, 120], batch=[64, 128, 256]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seeds", default="0,1,2,3,4"); ap.add_argument("--max-seconds", type=float, default=1e9)
    a = ap.parse_args()
    t0 = time.time()
    path = os.path.join(HERE, "tuning", f"{a.model}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = {}
    if os.path.exists(path):
        try:
            done = json.load(open(path))
        except json.JSONDecodeError:   # e.g. a write interrupted by a full disk
            os.replace(path, path + ".corrupt")
            print(f"  {path} was unreadable; moved it to {path}.corrupt and starting this model again")
    plas, feat = C.load_plasticc()
    le = LabelEncoder().fit(sorted(C.SHARED))
    for s in [int(x) for x in a.seeds.split(",")]:
        if str(s) in done:
            continue
        if time.time() - t0 > a.max_seconds:
            break
        tr, _ = train_test_split(plas, test_size=1570 / len(plas), stratify=plas["target"], random_state=s)
        tr = tr[tr["target"].isin(C.SHARED)]
        X, y = C.derive_features(tr[feat]), le.transform(tr["target"])
        folds = list(StratifiedKFold(3, shuffle=True, random_state=s).split(X, y))
        trials = []
        for params in ParameterSampler(SPACES[a.model], n_iter=a.trials, random_state=s):
            params = {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}
            sc = []
            for i, j in folds:
                Ai, (Bj,) = C.preprocess(X.iloc[i], [X.iloc[j]], "consistent", same_survey=[True])
                m = M.make_model({"type": a.model, "params": params}, s).fit(Ai.values, y[i])
                sc.append(f1_score(y[j], np.asarray(m.predict(Bj.values)).ravel().astype(int), average="macro"))
            trials.append({"params": params, "cv_macro_f1": float(np.mean(sc))})
        best = max(trials, key=lambda t: t["cv_macro_f1"])
        done[str(s)] = {"best_params": best["params"], "best_cv_macro_f1": best["cv_macro_f1"],
                        "n_trials": a.trials, "backend": M.backend({"type": a.model}), "trials": trials}
        tmp = path + ".tmp"               # write to a temp file, then rename: never leaves a half-written file
        with open(tmp, "w") as fh:
            json.dump(done, fh, indent=1)
        os.replace(tmp, path)
        print(f"  {a.model} seed {s}: best inner-CV F1 {best['cv_macro_f1']:.4f} with {best['params']}")


if __name__ == "__main__":
    main()
