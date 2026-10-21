"""
Extra Phase 3 outputs from saved results (no re-running):
  tables/confusion_transfer.md|tex   mean DES confusion matrix, consistent vs hybrid (XGBoost)
  tables/precision_recall.md|tex     per-class precision and recall on DES
  figures/confusion_transfer.png     row-normalised confusion matrices
  figures/reliability_both.png       reliability on PLAsTiCC (in-domain) and DES (transfer)
Usage:  python make_extra.py
"""
import json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = [("Consistent", "b_xgboost_consistent_none", "#888888"), ("Hybrid rule", "b_xgboost_hybrid_none", "#1f77b4")]
R = {k: json.load(open(os.path.join(HERE, "results", k + ".json"))) for _, k, _ in RUNS}
names = R[RUNS[0][1]]["class_order"]


def write(name, header, rows, caption):
    md = [f"**{caption}**", "", "| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"] + ["| " + " | ".join(r) + " |" for r in rows]
    open(os.path.join(HERE, "tables", name + ".md"), "w").write("\n".join(md) + "\n")
    esc = lambda s: s.replace("_", r"\_").replace("→", r"$\rightarrow$")
    tex = [r"\begin{table}", r"\centering", r"\caption{" + esc(caption) + "}", r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}", r"\hline",
           " & ".join(map(esc, header)) + r" \\", r"\hline"] + [" & ".join(map(esc, r)) + r" \\" for r in rows] + [r"\hline", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(HERE, "tables", name + ".tex"), "w").write("\n".join(tex) + "\n")


cms = {lab: np.mean([x["transfer_confusion"] for x in R[k]["runs"]], axis=0) for lab, k, _ in RUNS}
rows = []
for lab, cm in cms.items():
    for i, n in enumerate(names):
        rows.append([f"{lab}: true {n}"] + [f"{v:.1f}" for v in cm[i]])
write("confusion_transfer", ["", *[f"pred {n}" for n in names]], rows,
      "DES confusion matrices, XGBoost (mean counts over 5 split seeds; rows = true class)")

rows = []
for i, n in enumerate(names):
    r = [n]
    for lab, cm in cms.items():
        r += [f"{cm[i, i] / cm[:, i].sum():.2f}" if cm[:, i].sum() else "--", f"{cm[i, i] / cm[i].sum():.2f}"]
    rows.append(r)
write("precision_recall", ["Class", "Consistent precision", "Consistent recall", "Hybrid precision", "Hybrid recall"], rows,
      "Per-class precision and recall on DES, XGBoost (from mean confusion matrices)")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
for ax, (lab, cm) in zip(axes, cms.items()):
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]:.0f})", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.55 else "black")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30); ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{lab} (DES, XGBoost)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "confusion_transfer.png"), dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), sharey=True)
for ax, (pref, title) in zip(axes, [("plasticc_", "PLAsTiCC test (in-domain)"), ("transfer_", "DES (transfer)")]):
    ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
    for lab, k, col in RUNS:
        bins = {}
        for x in R[k]["runs"]:
            for lo, hi, cf, acc, n in x[pref + "reliability"]:
                b = bins.setdefault(lo, [0, 0, 0]); b[0] += cf * n; b[1] += acc * n; b[2] += n
        pts = sorted((v[0] / v[2], v[1] / v[2]) for v in bins.values() if v[2] > 0)
        ece = np.mean([x[pref + "ece"] for x in R[k]["runs"]])
        ax.plot(*zip(*pts), marker="o", color=col, label=f"{lab} (ECE {ece:.3f})")
    ax.set_title(title, fontsize=10); ax.set_xlabel("Predicted confidence"); ax.legend(fontsize=8)
axes[0].set_ylabel("Observed accuracy")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "reliability_both.png"), dpi=150)
for t in ["confusion_transfer", "precision_recall"]:
    print(open(os.path.join(HERE, "tables", t + ".md")).read())
