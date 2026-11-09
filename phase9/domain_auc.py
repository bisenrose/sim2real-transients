"""Phase 9 (A6): label-free domain-classifier AUC, PLAsTiCC training set vs each target, with all
features and with the survey-artefact features removed. Usage: python domain_auc.py"""
import json, os, sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); XAI = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(XAI, "phase1"))
import common as C  # noqa: E402

ART = ["time_span", "t_0_ZTF_g", "t_0_ZTF_r"]
plas, feat = C.load_plasticc()
tr, _ = train_test_split(plas, test_size=1570 / len(plas), stratify=plas["target"], random_state=0)
P = C.derive_features(tr[tr["target"].isin(C.SHARED)][feat])
targets = {}
des = C.load_des(); dl = C.des_labels(des, "corrected"); targets["DES"] = des[dl.isin(C.SHARED).values]
z, zl = C.load_target("ztf"); targets["ZTF"] = z[zl.isin(C.SHARED).values]
y, yl = C.load_target("yse"); targets["YSE"] = y[yl.isin(C.SHARED).values]


def auc(cols, D):
    X = pd.concat([P[cols], D[cols]]).fillna(P[cols].median()).values
    dom = np.r_[np.zeros(len(P)), np.ones(len(D))]
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, random_state=0), X, dom,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return float(roc_auc_score(dom, p))


out = {}
for name, df in targets.items():
    D = C.derive_features(df[feat])
    allc = list(P.columns); noart = [c for c in allc if c not in ART]
    out[name] = {"all_features": auc(allc, D), "without_artefacts": auc(noart, D)}
    print(f"{name}: AUC all features {out[name]['all_features']:.3f} | without artefacts {out[name]['without_artefacts']:.3f}")
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "results", "domain_auc.json"), "w"), indent=1)
