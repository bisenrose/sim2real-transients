"""
Phase 8 E1: per-feature shift of DES, ZTF and YSE relative to the PLAsTiCC training set (label-free).
Usage: python features_compare.py  -> tables/E1_*.md, figures/E1_ks_heatmap.png, results/E1.json
"""
import json, os, sys, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); XAI = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(XAI, "phase1"))
import common as C  # noqa: E402

for d in ("tables", "figures", "results"):
    os.makedirs(os.path.join(HERE, d), exist_ok=True)
GROUPS = {"source-frame": ["redshift", "abs_peak_mag"],
          "brightness": ["A_ZTF_g", "A_ZTF_r", "amplitude", "std_flux", "log_amplitude"],
          "noise": ["extra_sigma_ZTF_g", "extra_sigma_ZTF_r"],
          "cadence": ["time_span", "t_0_ZTF_g", "t_0_ZTF_r"]}

plas, feat = C.load_plasticc()
tr, _ = train_test_split(plas, test_size=1570 / len(plas), stratify=plas["target"], random_state=0)
tr = tr[tr["target"].isin(C.SHARED)]
P = C.derive_features(tr[feat])
targets = {}
des = C.load_des(); dl = C.des_labels(des, "corrected"); targets["DES"] = des[dl.isin(C.SHARED).values]
z, zl = C.load_target("ztf"); targets["ZTF"] = z[zl.isin(C.SHARED).values]
y, yl = C.load_target("yse"); targets["YSE"] = y[yl.isin(C.SHARED).values]
GROUPS["other"] = [c for c in P.columns if not any(c in g for g in GROUPS.values())]


def rel_noise(df):
    return pd.DataFrame({b: df[f"extra_sigma_ZTF_{b}"] / np.maximum(df["amplitude"], 1e-6) for b in ("g", "r")})


ks, auc, rel = {}, {}, {}
for name, df in targets.items():
    D = C.derive_features(df[feat])
    ks[name] = {c: float(ks_2samp(P[c].dropna(), D[c].dropna()).statistic) for c in P.columns}
    rel[name] = float(np.mean([ks_2samp(rel_noise(tr)[b].dropna(), rel_noise(df)[b].dropna()).statistic for b in ("g", "r")]))
    auc[name] = {}
    for g, cols in list(GROUPS.items()) + [("noise + cadence", GROUPS["noise"] + GROUPS["cadence"])]:
        X = pd.concat([P[cols], D[cols]]).fillna(P[cols].median()).values
        dom = np.r_[np.zeros(len(P)), np.ones(len(D))]
        p = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, random_state=0), X, dom,
                              cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        auc[name][g] = float(roc_auc_score(dom, p))

rows = []
for g, cols in GROUPS.items():
    rows.append([g] + [f"{np.mean([ks[n][c] for c in cols]):.2f}" for n in targets] + [f"{auc[n][g]:.2f}" for n in targets])
rows.append(["noise + cadence"] + [f"{np.mean([ks[n][c] for c in GROUPS['noise'] + GROUPS['cadence']]):.2f}" for n in targets] +
            [f"{auc[n]['noise + cadence']:.2f}" for n in targets])
rows.append(["relative noise (σ_extra / amplitude)"] + [f"{rel[n]:.2f}" for n in targets] + ["", "", ""])
hdr = ["Feature group"] + [f"mean KS: {n}" for n in targets] + [f"domain AUC: {n}" for n in targets]
md = ["**Shift of each feature group relative to the PLAsTiCC training set (label-free)**", "",
      "| " + " | ".join(hdr) + " |", "| " + " | ".join("---" for _ in hdr) + " |"] + ["| " + " | ".join(r) + " |" for r in rows]
open(os.path.join(HERE, "tables", "E1_groups.md"), "w").write("\n".join(md) + "\n"); print("\n".join(md) + "\n")

pf = pd.DataFrame(ks); pf.index.name = "feature"
order = sum(GROUPS.values(), [])
pf = pf.loc[order]
pf.round(3).to_csv(os.path.join(HERE, "tables", "E1_per_feature_ks.csv"))
fig, ax = plt.subplots(figsize=(4.2, 8))
im = ax.imshow(pf.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(pf.columns))); ax.set_xticklabels(pf.columns); ax.set_yticks(range(len(pf))); ax.set_yticklabels(pf.index, fontsize=7)
for i in range(len(pf)):
    for j in range(len(pf.columns)):
        ax.text(j, i, f"{pf.values[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if pf.values[i, j] < 0.6 else "black")
fig.colorbar(im, ax=ax, fraction=0.05, label="KS vs PLAsTiCC"); fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "E1_ks_heatmap.png"), dpi=160)

nc = GROUPS["noise"] + GROUPS["cadence"]
V = {"E1a_noise_cadence_ks_yse_lt_des": bool(np.mean([ks["YSE"][c] for c in nc]) < np.mean([ks["DES"][c] for c in nc])),
     "E1b_noise_cadence_auc_yse_lt_des": bool(auc["YSE"]["noise + cadence"] < auc["DES"]["noise + cadence"]),
     "E1c_relative_noise_ks_yse_lt_des": bool(rel["YSE"] < rel["DES"]),
     "ks": ks, "auc": auc, "relative_noise_ks": rel}
json.dump(V, open(os.path.join(HERE, "results", "E1.json"), "w"), indent=1)
print({k: v for k, v in V.items() if k.startswith("E1")})
