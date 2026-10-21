"""
Phase 6 analyses from SAVED predictions (no retraining):
  D3  EM label-shift correction (Saerens, Latinne & Decaestecker 2002) of target probabilities;
      source priors from each seed's PLAsTiCC training set; no target labels used.
  D4  per-class recall by redshift bin on ZTF.
Usage:  python analysis_saved.py   (run after the Phase 6 configs, so the D1 variants are included)
Writes tables/D3_label_shift.md, tables/D4_recall_by_z.md, results/D3_label_shift.json
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); XAI = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(XAI, "phase1"))
import common as C  # noqa: E402

os.makedirs(os.path.join(HERE, "tables"), exist_ok=True); os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
CODES = sorted(C.SHARED)                                   # label-encoder order used by the engine
PLAS, _ = C.load_plasticc()


def source_prior(seed):
    tr, _ = train_test_split(PLAS, test_size=1570 / len(PLAS), stratify=PLAS["target"], random_state=seed)
    tr = tr[tr["target"].isin(C.SHARED)]
    c = np.array([(tr["target"] == k).sum() for k in CODES], float)
    return c / c.sum()


def em_adjust(p, prior_s, iters=1000, tol=1e-8):
    prior_t = prior_s.copy()
    for _ in range(iters):
        q = p * (prior_t / np.maximum(prior_s, 1e-12)); q /= q.sum(1, keepdims=True)
        new = q.mean(0)
        if np.abs(new - prior_t).max() < tol:
            prior_t = new; break
        prior_t = new
    q = p * (prior_t / np.maximum(prior_s, 1e-12)); q /= q.sum(1, keepdims=True)
    return q, prior_t


def write(name, df, caption):
    lines = [f"**{caption}**", "", "| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |",
             "| " + " | ".join(["---"] * (len(df.columns) + 1)) + " |"]
    lines += ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |" for i, r in df.iterrows()]
    open(os.path.join(HERE, "tables", name + ".md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines) + "\n")


# ---------------- D3 ----------------
runs = []
for m in ["xgboost", "lightgbm", "catboost", "mlp"]:
    for p, lab in [("consistent", "Consistent"), ("per_survey", "Per-survey"), ("hybrid", "Hybrid rule"),
                   ("explore_hybrid_absmag_only", "Hybrid, abs mag only")]:
        runs.append(("ZTF", m, lab, os.path.join(XAI, "phase4", "results", f"ztf_{m}_{p}.json")))
    for p, lab in [("consistent", "Consistent"), ("per_survey", "Per-survey"), ("hybrid", "Hybrid rule")]:
        runs.append(("DES", m, lab, os.path.join(XAI, "phase2", "results", f"{m}_corrected_{p}.json")))
rows, out = [], {}
for tgt, m, lab, path in runs:
    if not os.path.exists(path):
        continue
    r = json.load(open(path)); y = np.array(r["y_des"]); labels = sorted(np.unique(y).tolist())
    before, after, oracle, ia_prior = [], [], [], []
    for x in r["runs"]:
        if "des_proba" not in x:
            continue
        p = np.array(x["des_proba"]); ps = source_prior(x["split_seed"])
        q, pt = em_adjust(p, ps)
        before.append(f1_score(y, p.argmax(1), average="macro", labels=labels, zero_division=0))
        # EXPLORATORY diagnostic (uses target labels for the prior only): correct with the TRUE target
        # class proportions, to ask whether known priors would close the gap at all.
        pt_true = np.array([np.mean(y == i) for i in range(len(CODES))])
        qo = p * (pt_true / np.maximum(ps, 1e-12)); qo /= np.maximum(qo.sum(1, keepdims=True), 1e-12)
        oracle.append(f1_score(y, qo.argmax(1), average="macro", labels=labels, zero_division=0))
        after.append(f1_score(y, q.argmax(1), average="macro", labels=labels, zero_division=0))
        ia_prior.append(pt[CODES.index(90)])
    if not before:
        continue
    b, a, o = np.array(before), np.array(after), np.array(oracle)
    true_ia = float(np.mean(y == CODES.index(90)))
    rows.append([tgt, m, lab, f"{b.mean():.3f} ± {b.std():.3f}", f"{a.mean():.3f} ± {a.std():.3f}",
                 f"{np.mean(ia_prior):.2f} (true {true_ia:.2f})", f"{o.mean():.3f} ± {o.std():.3f}"])
    out[f"{tgt}|{m}|{lab}"] = {"before": b.tolist(), "after": a.tolist(), "oracle_prior": o.tolist(), "est_ia_prior": float(np.mean(ia_prior))}
t = pd.DataFrame(rows, columns=["Target", "Model", "Preprocessing", "Macro-F1", "After EM prior correction", "EM-estimated SNIa prior", "After true-prior correction (exploratory)"]).set_index("Target")
write("D3_label_shift", t, "Label-shift (prior) correction by EM on saved probabilities (5 split seeds)")


def pooled(a, b):
    return np.sqrt((np.std(a) ** 2 + np.std(b) ** 2) / 2)


verd = {}
for m in ["xgboost", "lightgbm", "catboost", "mlp"]:
    h, ps = out.get(f"ZTF|{m}|Hybrid rule"), out.get(f"ZTF|{m}|Per-survey")
    if h and ps:
        verd[m] = bool(np.mean(ps["after"]) - np.mean(h["after"]) > pooled(ps["after"], h["after"]))
verd["L1_supported"] = sum(v for k, v in verd.items()) >= 3
# exploratory: does the hybrid deficit vs per-survey survive correction with the TRUE priors?
verd["exploratory_true_prior_hybrid_below_per_survey"] = {
    m: bool(np.mean(out[f"ZTF|{m}|Per-survey"]["oracle_prior"]) - np.mean(out[f"ZTF|{m}|Hybrid rule"]["oracle_prior"])
            > pooled(out[f"ZTF|{m}|Per-survey"]["oracle_prior"], out[f"ZTF|{m}|Hybrid rule"]["oracle_prior"]))
    for m in ["xgboost", "lightgbm", "catboost", "mlp"] if f"ZTF|{m}|Hybrid rule" in out and f"ZTF|{m}|Per-survey" in out}
json.dump({"runs": out, "L1": verd}, open(os.path.join(HERE, "results", "D3_label_shift.json"), "w"), indent=1)
print("L1 (hybrid still below per-survey after correction):", verd)

# ---------------- D4 ----------------
ztf, zl = C.load_target("ztf"); ztf = ztf.assign(cls=zl.values); ztf = ztf[ztf["cls"].isin(C.SHARED)].reset_index(drop=True)
z = ztf["redshift"].values; edges = [0, 0.03, 0.06, 0.10, np.inf]; bins = np.digitize(z, edges[1:-1])
blab = ["z<0.03", "0.03-0.06", "0.06-0.10", "z>=0.10"]
rows = []
for p, lab in [("consistent", "Consistent"), ("per_survey", "Per-survey"), ("hybrid", "Hybrid rule"), ("explore_hybrid_absmag_only", "Hybrid, abs mag only")]:
    path = os.path.join(XAI, "phase4", "results", f"ztf_xgboost_{p}.json")
    if not os.path.exists(path):
        continue
    r = json.load(open(path)); y = np.array(r["y_des"]); names = r["class_order"]
    if len(y) != len(ztf):
        continue
    for c in sorted(np.unique(y)):
        cells = []
        for b in range(4):
            msk = (bins == b) & (y == c)
            if msk.sum() < 10:
                cells.append(f"-- (n={int(msk.sum())})"); continue
            rec = [np.mean(np.array(x["des_pred"])[msk] == c) for x in r["runs"]]
            cells.append(f"{np.mean(rec):.2f} (n={int(msk.sum())})")
        rows.append([lab, names[c]] + cells)
t = pd.DataFrame(rows, columns=["Preprocessing", "Class"] + blab).set_index("Preprocessing")
write("D4_recall_by_z", t, "ZTF per-class recall by redshift bin (XGBoost, mean over 5 split seeds; bins with < 10 objects omitted)")
