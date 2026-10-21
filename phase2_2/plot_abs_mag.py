"""Figure for 2.2: peak absolute magnitude per class, PLAsTiCC training set (all and DES-matched
selection) vs DES. Usage: python plot_abs_mag.py  -> figures/abs_mag_by_class.png"""
import os, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase1"))
import common as C  # noqa: E402

plas, feat = C.load_plasticc(); des = C.load_des()
lab = C.des_labels(des, "corrected"); des = des[lab.isin(C.SHARED).values].assign(cls=lab[lab.isin(C.SHARED)].values)
tr, _ = train_test_split(plas, test_size=1570 / len(plas), stratify=plas["target"], random_state=0)
tr = tr[tr["target"].isin(C.SHARED)]
keep = C.selection_mask(tr, des, {"type": "logistic"}, seed=1000)
P, D = C.derive_features(tr[feat]), C.derive_features(des[feat])
order = [90, 42, 62, 88, 95]
fig, axes = plt.subplots(1, 5, figsize=(17, 3.4), sharey=False)
for ax, c in zip(axes, order):
    a = P["abs_peak_mag"][(tr["target"] == c).values].dropna()
    s = P["abs_peak_mag"][((tr["target"] == c).values) & keep].dropna()
    d = D["abs_peak_mag"][(des["cls"] == c).values].dropna()
    bins = np.linspace(min(a.quantile(.01), d.min()) - .5, max(a.quantile(.99), d.max()) + .5, 30)
    ax.hist(a, bins, density=True, histtype="step", lw=1.6, color="#888888", label="PLAsTiCC train (all)")
    ax.hist(s, bins, density=True, histtype="step", lw=1.6, color="#1f77b4", label="PLAsTiCC train (DES-matched selection)")
    ax.hist(d, bins, density=True, histtype="stepfilled", alpha=.35, color="#d62728", label="DES")
    ax.set_title(f"{C.SHARED[c]}  (DES n={len(d)})", fontsize=10); ax.set_xlabel("Peak absolute magnitude"); ax.invert_xaxis()
axes[0].set_ylabel("Density"); axes[0].legend(fontsize=7, loc="upper left")
fig.tight_layout(); os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figures", "abs_mag_by_class.png"), dpi=150)
print("saved figures/abs_mag_by_class.png")
