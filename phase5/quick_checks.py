"""
Phase 5 quick checks from SAVED results and data (no model training):
  A. ZTF confusion matrices (XGBoost: consistent, hybrid rule, hybrid with abs mag only)
  B. Label-free support overlap of redshift and abs_peak_mag with the PLAsTiCC training set
  C. Per-class 95% bootstrap intervals on DES (XGBoost, consistent vs hybrid rule)
Usage:  python quick_checks.py      -> tables/quick_*.md, figures/ztf_confusion.png, results/quick_checks.json
"""
import json, os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__)); XAI = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(XAI, "phase1"))
import common as C  # noqa: E402

os.makedirs(os.path.join(HERE, "tables"), exist_ok=True); os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
OUT = {}


def write(name, df, caption):
    lines = [f"**{caption}**", "", "| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |",
             "| " + " | ".join(["---"] * (len(df.columns) + 1)) + " |"]
    lines += ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |" for i, r in df.iterrows()]
    open(os.path.join(HERE, "tables", name + ".md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines) + "\n")


# ---- A. ZTF confusion matrices -------------------------------------------------------------
runs = [("Consistent", "ztf_xgboost_consistent"), ("Hybrid rule", "ztf_xgboost_hybrid"),
        ("Hybrid, abs mag only (exploratory)", "ztf_xgboost_explore_hybrid_absmag_only")]
R = {lab: json.load(open(os.path.join(XAI, "phase4", "results", k + ".json"))) for lab, k in runs
     if os.path.exists(os.path.join(XAI, "phase4", "results", k + ".json"))}
if R:
    names = next(iter(R.values()))["class_order"]
    y = np.array(next(iter(R.values()))["y_des"]); present = sorted(np.unique(y))
    fig, axes = plt.subplots(1, len(R), figsize=(4.6 * len(R), 4.2))
    axes = np.atleast_1d(axes); rows = []
    for ax, (lab, r) in zip(axes, R.items()):
        cm = np.mean([x["transfer_confusion"] for x in r["runs"]], axis=0)[np.ix_(present, range(len(names)))]
        for i, c in enumerate(present):
            rows.append([lab, f"true {names[c]}"] + [f"{v:.1f}" for v in cm[i]])
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]:.0f})", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else "black")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30); ax.set_yticks(range(len(present)))
        ax.set_yticklabels([names[c] for c in present]); ax.set_xlabel("Predicted"); ax.set_title(lab, fontsize=9)
    axes[0].set_ylabel("True"); fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "ztf_confusion.png"), dpi=150)
    t = pd.DataFrame(rows, columns=["Preprocessing", "True class"] + [f"pred {n}" for n in names]).set_index("Preprocessing")
    write("quick_ztf_confusion", t, "ZTF confusion matrices, XGBoost (mean counts over 5 split seeds)")
else:
    print("Phase 4 ZTF results not found: skipping A")

# ---- B. support overlap -----------------------------------------------------------------------
plas, feat = C.load_plasticc(); pool = plas[plas["target"].isin(C.SHARED)]
tr, _ = train_test_split(plas, test_size=1570 / len(plas), stratify=plas["target"], random_state=0)
tr = tr[tr["target"].isin(C.SHARED)]
targets = {}
des = C.load_des(); dl = C.des_labels(des, "corrected"); targets["DES"] = (tr, des[dl.isin(C.SHARED).values])
try:
    z, zl = C.load_target("ztf"); targets["ZTF"] = (tr, z[zl.isin(C.SHARED).values])
except FileNotFoundError:
    pass
low = pool[pool["redshift"] < 0.12]
targets["PLAsTiCC pseudo low-z"] = (tr[(tr["redshift"] >= 0.12) & ~tr["object_id"].isin(low["object_id"])], low)
rnd = pool.sample(n=len(low), random_state=0)
targets["PLAsTiCC pseudo random"] = (tr[~tr["object_id"].isin(rnd["object_id"])], rnd)
rows = {}
for name, (src, tgt) in targets.items():
    A, B = C.derive_features(src[feat]), C.derive_features(tgt[feat])
    rows[name] = {"n target": len(B), "redshift overlap": round(C.support_overlap(A["redshift"], B["redshift"]), 3),
                  "abs_peak_mag overlap": round(C.support_overlap(A["abs_peak_mag"], B["abs_peak_mag"]), 3),
                  "target z median": round(float(np.nanmedian(B["redshift"])), 3)}
t = pd.DataFrame(rows).T; t.index.name = "Target"
write("quick_support_overlap", t, "Label-free support overlap: fraction of target objects inside the PLAsTiCC training 5th-95th percentile range")
OUT["support_overlap"] = rows

# ---- C. per-class bootstrap intervals on DES ---------------------------------------------------
rows = {}
for lab, k in [("Consistent", "b_xgboost_consistent_none"), ("Hybrid rule", "b_xgboost_hybrid_none")]:
    p = os.path.join(XAI, "phase3", "results", k + ".json")
    if not os.path.exists(p):
        continue
    r = json.load(open(p)); y = np.array(r["y_des"]); names = r["class_order"]; vals = {n: [] for n in names}
    for x in r["runs"]:
        pr = np.array(x["des_pred"]); rng = np.random.RandomState(x["split_seed"])
        for _ in range(1000):
            i = rng.randint(0, len(y), len(y))
            f = f1_score(y[i], pr[i], average=None, labels=range(len(names)), zero_division=0)
            for n, v in zip(names, f):
                vals[n].append(v)
    for n in names:
        v = np.array(vals[n]); rows.setdefault(f"{n} (n={int((y == names.index(n)).sum())})", {})[lab] = \
            f"{np.mean(v):.3f} [{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]"
if rows:
    t = pd.DataFrame(rows).T; t.index.name = "Class (DES)"
    write("quick_des_per_class_ci", t, "Per-class transfer F1 on DES with 95% bootstrap intervals (XGBoost; 1000 resamples x 5 seeds)")
    OUT["des_per_class_ci"] = rows
json.dump(OUT, open(os.path.join(HERE, "results", "quick_checks.json"), "w"), indent=1, default=str)
