"""
Run one experiment from a JSON config and write results/<name>.json.

Usage:  python run_experiment.py configs/benchmark_consistent.json
        python run_experiment.py --all            # every config in configs/
"""
import glob, hashlib, json, os, sys, time, warnings
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import common as C
import models as M
import adaptation as AD
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
MACRO_LABELS = None   # set per run: None for DES (all five classes present)


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def make_model(cfg, seed):
    return M.make_model(cfg["model"], seed)


def calibration_metrics(y, prob, names, prefix, n_bins=10):
    """Multi-class log-loss, Brier score and top-label expected calibration error (ECE),
    plus the reliability-diagram data (confidence vs accuracy per bin)."""
    k = prob.shape[1]
    onehot = np.eye(k)[y]
    conf, pred = prob.max(1), prob.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1); ece = 0.0; rel = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            acc, cf = float((pred[sel] == y[sel]).mean()), float(conf[sel].mean())
            ece += sel.mean() * abs(acc - cf); rel.append([float(lo), float(hi), cf, acc, int(sel.sum())])
    return {prefix + "log_loss": float(log_loss(y, np.clip(prob, 1e-15, 1), labels=range(k))),
            prefix + "brier": float(np.mean(np.sum((prob - onehot) ** 2, axis=1))),
            prefix + "ece": float(ece), prefix + "reliability": rel}


def bootstrap_ci(y, p, n=1000, seed=0):
    rng = np.random.RandomState(seed); vals = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        vals.append(f1_score(y[i], p[i], average="macro", zero_division=0, labels=MACRO_LABELS))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def run(cfg):
    classes = C.LEGACY_SHARED if cfg.get("legacy_class_codes") else C.SHARED
    plas, feat_cols = C.load_plasticc()
    target = cfg.get("target", "des")   # target survey: "des" (default) or "ztf". Result keys keep the
    #                                     historical "des"/"transfer" names whatever the target is.
    pseudo_ids = set()
    if target == "plasticc_pseudo":
        # Phase 5 controlled test: the "target survey" is a fixed subset of PLAsTiCC itself, removed
        # from training and in-domain test. kind 'lowz': all objects with z < z_cut, and training /
        # in-domain test keep only z >= z_cut (no redshift overlap). kind 'random': a random subset of
        # the same size (full overlap; the control).
        ps = cfg["pseudo_target"]
        pool = plas[plas["target"].isin(classes)]
        if ps["kind"] == "lowz":
            des = pool[pool["redshift"] < ps["z_cut"]]
        elif ps["kind"] == "random":
            n = int((pool["redshift"] < ps["z_cut"]).sum())
            des = pool.sample(n=n, random_state=ps.get("seed", 0))
        elif ps["kind"] in ("lowz_iarich", "iarich_random"):
            # Phase 6: ZTF-like class priors. All SNe Ia with z < z_cut plus SNII/SNIbc (no AGN or
            # SLSN-I, as in ZTF) so that SNe Ia make up ia_frac; 'iarich_random' draws the same class
            # counts from all redshifts (the control).
            rng = np.random.RandomState(ps.get("seed", 0))
            low = pool[pool["redshift"] < ps["z_cut"]]
            n_ia = int((low["target"] == 90).sum())
            n_other = int(round(n_ia * (1 - ps["ia_frac"]) / ps["ia_frac"]))
            others_low = low[low["target"].isin([42, 62])]
            pick_o = others_low.sample(n=n_other, random_state=rng)
            counts = {90: n_ia, **pick_o["target"].value_counts().to_dict()}
            if ps["kind"] == "lowz_iarich":
                des = pd.concat([low[low["target"] == 90], pick_o])
            else:
                des = pd.concat([pool[pool["target"] == c].sample(n=int(k), random_state=rng) for c, k in counts.items()])
        else:
            raise ValueError(ps)
        des = des.assign(plasticc_class=des["target"].values).reset_index(drop=True)
        pseudo_ids = set(des["object_id"])
        policy, target_path = f"pseudo_{ps['kind']}", C.PLASTICC_RAW
    elif target == "des":
        des = C.load_des()
        policy = "legacy" if cfg.get("legacy_class_codes") else cfg.get("des_label_policy", "corrected")
        des = des.assign(plasticc_class=C.des_labels(des, policy).values)
        target_path = C.DES_FEATURES
    else:
        des, lab = C.load_target(target)
        policy = "bts_mapping"
        des = des.assign(plasticc_class=lab.values)
        target_path = C.TARGET_PATHS.get(target, C.ZTF_FEATURES)
    des = des[des["plasticc_class"].isin(classes)].reset_index(drop=True)
    le = LabelEncoder().fit(sorted(classes))
    names = [classes[c] for c in le.classes_]

    variant = cfg.get("abs_mag_variant")
    if cfg.get("target_redshift_noise"):
        # Phase 5: replace target spectroscopic redshifts by noisy photometric-like ones,
        # z + N(0, sigma (1 + z)), clipped at 0.001, one fixed realisation per config.
        tn = cfg["target_redshift_noise"]; rng_z = np.random.RandomState(tn.get("seed", 0))
        z0 = des["redshift"].values.astype(float)
        des = des.assign(redshift=np.clip(z0 + rng_z.normal(0, tn["sigma"] * (1 + z0)), 1e-3, None))
    if cfg.get("target_instrument_shift") or cfg.get("target_population_shift"):
        # Phase 6/8: synthetic population and/or instrument shifts applied to the target's raw features only
        des = C.apply_target_shifts(des, cfg.get("target_population_shift"), cfg.get("target_instrument_shift"))
    X_des_raw = C.derive_features(des[feat_cols], abs_mag_variant=variant)
    y_des = le.transform(des["plasticc_class"])
    global MACRO_LABELS
    # DES has all five classes, so its macro-F1 is unchanged (labels=None). For a target that lacks
    # a class (ZTF has few or no AGN), macro-F1 averages over the classes present in the target.
    if target == "des":
        MACRO_LABELS = None
    elif target == "plasticc_pseudo":   # classes with at least 10 pseudo-target objects (pre-registered)
        vals, cnt = np.unique(y_des, return_counts=True); MACRO_LABELS = sorted(vals[cnt >= 10].tolist())
    else:
        MACRO_LABELS = sorted(np.unique(y_des).tolist())
    out = {"config": cfg, "des_label_policy": policy, "class_order": names, "target": target, "data_md5": {"plasticc": md5(C.PLASTICC_RAW), "des": md5(target_path)},
           "data_paths": {"plasticc": C.PLASTICC_RAW, "des": target_path},
           "versions": {"xgboost": xgb.__version__, "pandas": pd.__version__, "numpy": np.__version__},
           "model_backend": M.backend(cfg["model"]),
           "y_des": y_des.tolist(), "runs": [], "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    for split_seed in cfg["split_seeds"]:
        # Split on the FULL 14-class PLAsTiCC table first (as the original pipeline did),
        # then restrict to the shared classes, so the split is comparable across configs.
        tr_df, te_df = train_test_split(plas, test_size=1570 / len(plas), stratify=plas["target"], random_state=split_seed)
        tr_df = tr_df[tr_df["target"].isin(classes)]; te_df = te_df[te_df["target"].isin(classes)]
        if pseudo_ids:
            tr_df = tr_df[~tr_df["object_id"].isin(pseudo_ids)]; te_df = te_df[~te_df["object_id"].isin(pseudo_ids)]
            if cfg["pseudo_target"]["kind"] in ("lowz", "lowz_iarich"):
                zc = cfg["pseudo_target"]["z_cut"]
                tr_df = tr_df[tr_df["redshift"] >= zc]; te_df = te_df[te_df["redshift"] >= zc]
        assert len(set(tr_df.object_id) & set(te_df.object_id)) == 0, "train/test leakage"
        sel_info = {}
        if cfg.get("train_selection"):
            # Phase 2.2: simulate DES selection on the PLAsTiCC TRAINING set only (test set untouched)
            keep = C.selection_mask(tr_df, des, cfg["train_selection"], seed=1000 + split_seed)
            sel_info = {"n_before_selection": int(len(tr_df)), "n_after_selection": int(keep.sum()),
                        "train_class_counts_after": {classes[c]: int(n) for c, n in tr_df[keep]["target"].value_counts().items()}}
            tr_df = tr_df[keep]
        X_tr, X_te = C.derive_features(tr_df[feat_cols], variant), C.derive_features(te_df[feat_cols], variant)
        y_tr, y_te = le.transform(tr_df["target"]), le.transform(te_df["target"])
        cols = [c for c in X_tr.columns if c not in cfg.get("drop_features", [])]
        mode, calibrated, gate_info = cfg["preprocessing"], cfg.get("calibrated_features"), {}
        if mode == "gated_hybrid":
            # Phase 5, EXPLORATORY: keep a source-frame feature on the shared scale only if the target's
            # support overlaps the training set's (label-free).
            g = cfg["gate"]
            calibrated, ov = C.gated_calibrated(X_tr[cols], X_des_raw[cols], g.get("candidates", C.CALIBRATED_FEATURES),
                                                g["threshold"], tuple(g.get("quantiles", (0.05, 0.95))))
            gate_info = {"gate_overlap": ov, "gate_kept_shared": calibrated}
            mode = "hybrid"
        stats_frames = None
        if cfg.get("target_stats_n"):
            # Phase 6: fit the target's own statistics on a small unlabelled subsample only
            n_s = int(cfg["target_stats_n"]); rs = np.random.RandomState(500 + split_seed)
            stats_frames = [None, X_des_raw[cols].iloc[rs.choice(len(X_des_raw), n_s, replace=False)]]
        A, (B, D) = C.preprocess(X_tr[cols], [X_te[cols], X_des_raw[cols]], mode,
                                 same_survey=[True, False], calibrated=calibrated, stats_frames=stats_frames)

        model_seed = cfg.get("model_seed_offset", 0) + split_seed
        spec = dict(cfg["model"])
        if spec.get("tuned_from"):
            # Phase 3.1: per-split-seed hyperparameters chosen by nested CV on PLAsTiCC only
            tuned = json.load(open(os.path.join(os.path.dirname(cfg.get("_results_dir", HERE)), spec["tuned_from"])))
            spec["params"] = {**spec.get("params", {}), **tuned[str(split_seed)]["best_params"]}
        sw, adapt_info = None, dict(gate_info)
        adapt = cfg.get("adaptation")
        if cfg.get("train_weighting"):
            # Phase 5: magnitude density ratio as sample weights (keeps the whole training set)
            sw, winfo = C.selection_weights(tr_df, des, cfg["train_weighting"]); adapt_info.update(winfo)
            if adapt:
                raise ValueError("train_weighting cannot be combined with adaptation")
        if adapt and adapt["type"] == "stratlearn":
            pass   # handled after the global model is fitted (it needs the per-stratum models)
        elif adapt and adapt["type"] == "importance_weighting":
            sw, iw_info = AD.importance_weights(A, D, seed=model_seed, exclude=adapt.get("exclude_features", []))
            adapt_info.update(iw_info)
        elif adapt and adapt["type"] == "coral":
            A, (B,) = AD.coral(A, [B], D)
        elif adapt:
            raise ValueError(adapt)
        # A training set can lack a class (e.g. a strong brightness selection removes every SLSN-I).
        # Fit on the classes present, then put the probabilities back into all k columns (absent = 0).
        present_tr = np.unique(y_tr); k_all = len(names)
        remap = np.full(k_all, -1); remap[present_tr] = np.arange(len(present_tr))
        m = M.make_model(spec, model_seed)
        m = m.fit(A.values, remap[y_tr], sample_weight=sw) if sw is not None else m.fit(A.values, remap[y_tr])

        def full_proba(X):
            p = np.asarray(m.predict_proba(X)); out = np.zeros((len(X), k_all)); out[:, present_tr] = p
            return out
        prob_te, prob_des = full_proba(B.values), full_proba(D.values)
        if adapt and adapt["type"] == "stratlearn":
            # in-domain predictions stay with the global model; target predictions come from the strata
            prob_des, sl_info = AD.stratlearn(A, y_tr, D, spec, model_seed, M.make_model, k_all,
                                              n_strata=adapt.get("n_strata", 5), exclude=adapt.get("exclude_features", []),
                                              min_source=adapt.get("min_source", 30))
            adapt_info.update(sl_info)
        p_te, p_des = prob_te.argmax(1), prob_des.argmax(1)
        run = {"split_seed": split_seed, "model_seed": model_seed, "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
               "n_des": int(len(y_des)), "features": cols,
               "plasticc_macro_f1": float(f1_score(y_te, p_te, average="macro", zero_division=0)),
               "plasticc_bacc": float(balanced_accuracy_score(y_te, p_te)),
               "transfer_macro_f1": float(f1_score(y_des, p_des, average="macro", zero_division=0, labels=MACRO_LABELS)),
               "transfer_bacc": float(balanced_accuracy_score(y_des, p_des)),
               "transfer_macro_f1_ci95": bootstrap_ci(y_des, p_des, seed=split_seed),
               "plasticc_per_class_f1": dict(zip(names, map(float, f1_score(y_te, p_te, average=None, labels=range(len(names)), zero_division=0)))),
               "transfer_per_class_f1": dict(zip(names, map(float, f1_score(y_des, p_des, average=None, labels=range(len(names)), zero_division=0)))),
               "transfer_confusion": confusion_matrix(y_des, p_des, labels=range(len(names))).tolist(),
               **sel_info, **adapt_info,
               "train_class_counts": {names[c]: int((y_tr == c).sum()) for c in range(len(names))},
               "params_used": spec.get("params", {}),
               **calibration_metrics(y_des, prob_des, names, "transfer_"),
               **calibration_metrics(y_te, prob_te, names, "plasticc_"),
               "transfer_macro_f1_without_slsn": float(f1_score(y_des, p_des, average="macro", zero_division=0,
                                                       labels=[i for i, n in enumerate(names) if n != "SLSN-I"
                                                               and (MACRO_LABELS is None or i in MACRO_LABELS)])),
               "des_pred": p_des.tolist(), "des_proba": np.round(prob_des, 4).tolist()}
        if cfg.get("record_shift"):
            # raw (unscaled) abs_peak_mag shift between the PLAsTiCC training set and DES, overall and per class
            from scipy.stats import ks_2samp
            a, b = X_tr["abs_peak_mag"].dropna(), X_des_raw["abs_peak_mag"].dropna()
            run["abs_mag_ks_train_vs_des"] = float(ks_2samp(a, b).statistic)
            run["abs_mag_median_gap"] = float(b.median() - a.median())
            run["abs_mag_median_gap_per_class"] = {
                n: float(np.nanmedian(X_des_raw["abs_peak_mag"].values[y_des == i]) - np.nanmedian(X_tr["abs_peak_mag"].values[y_tr == i]))
                for i, n in enumerate(names)}

        if cfg.get("des_cv", False):
            # DES-only ceiling: 5-fold stratified CV, preprocessing fitted inside each fold.
            skf = StratifiedKFold(5, shuffle=True, random_state=split_seed)
            pooled = np.zeros(len(y_des), int); fold_f1 = []
            for a, b in skf.split(X_des_raw, y_des):
                Aa, (Bb,) = C.preprocess(X_des_raw.iloc[a][cols], [X_des_raw.iloc[b][cols]], "consistent")
                # re-index classes present in this training fold (a target may lack a class)
                present = np.unique(y_des[a]); remap = {c: i for i, c in enumerate(present)}
                md = make_model({"model": cfg["des_model"]}, model_seed).fit(Aa.values, np.array([remap[c] for c in y_des[a]]))
                pooled[b] = present[np.asarray(md.predict(Bb.values)).ravel().astype(int)]
                fold_f1.append(f1_score(y_des[b], pooled[b], average="macro", zero_division=0, labels=MACRO_LABELS))
            run["des_cv_macro_f1_mean_of_folds"] = float(np.mean(fold_f1))
            run["des_cv_macro_f1_pooled"] = float(f1_score(y_des, pooled, average="macro", zero_division=0, labels=MACRO_LABELS))
            run["des_cv_per_class_f1"] = dict(zip(names, map(float, f1_score(y_des, pooled, average=None, labels=range(len(names)), zero_division=0))))
        out["runs"].append(run)
        print(f"  [{cfg['name']}] split_seed={split_seed}: PLAsTiCC F1={run['plasticc_macro_f1']:.4f}  transfer F1={run['transfer_macro_f1']:.4f}"
              + (f"  DES-CV F1={run['des_cv_macro_f1_mean_of_folds']:.4f}" if "des_cv_macro_f1_mean_of_folds" in run else ""))

    for k in ["plasticc_macro_f1", "transfer_macro_f1", "des_cv_macro_f1_mean_of_folds", "transfer_log_loss", "transfer_ece"]:
        v = [r[k] for r in out["runs"] if k in r]
        if v:
            out[k + "_mean"], out[k + "_sd"] = float(np.mean(v)), float(np.std(v))
    out_dir = cfg.get("_results_dir", os.path.join(HERE, "results"))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, cfg["name"] + ".json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    import sklearn, astropy, numpy
    print("Project root:", C.ROOT)
    print(f"Versions: xgboost {xgb.__version__} | scikit-learn {sklearn.__version__} | pandas {pd.__version__} | "
          f"numpy {numpy.__version__} | astropy {astropy.__version__}")
    # Usage: run_experiment.py [--all | --dir <folder with configs/> | config.json ...]
    args = sys.argv[1:] or ["--all"]   # no arguments = run every Phase 1 config
    if "--dir" in args:
        base = os.path.abspath(args[args.index("--dir") + 1])
        paths = sorted(glob.glob(os.path.join(base, "configs", "*.json")))
    elif "--all" in args:
        paths = sorted(glob.glob(os.path.join(HERE, "configs", "*.json")))
    else:
        paths = args
    for p in paths:
        with open(p) as f:
            cfg = json.load(f)
        # results go in a results/ folder next to the configs/ folder the config came from
        cfg["_results_dir"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(p))), "results")
        run(cfg)
