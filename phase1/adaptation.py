"""
Standard domain-adaptation baselines (Phase 3.2). Both are label-free: they use PLAsTiCC
labels and unlabelled DES features only.
"""
import numpy as np
from scipy import linalg
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold


def importance_weights(A, D, seed, clip_percentile=99, exclude=()):
    """Covariate-shift weights w = p/(1-p) for each PLAsTiCC training row, where p is the
    out-of-fold probability (domain classifier, 5-fold) that the row looks like DES.
    `exclude`: features the domain classifier may not use (e.g. survey-design features that
    separate the surveys perfectly and make every weight identical)."""
    cols = [c for c in A.columns if c not in exclude]
    X = np.r_[A[cols].values, D[cols].values]; y = np.r_[np.zeros(len(A)), np.ones(len(D))]
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, random_state=seed), X, y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:len(A), 1]
    p = np.clip(p, 1e-4, 1 - 1e-4)
    w = p / (1 - p)
    w = np.minimum(w, np.percentile(w, clip_percentile))
    w = w / w.mean()
    from sklearn.metrics import roc_auc_score
    p_all = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, random_state=seed), X, y,
                              cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return w, {"domain_classifier_auc": float(roc_auc_score(y, p_all)), "weight_max": float(w.max()),
               "effective_sample_size": float(w.sum() ** 2 / np.sum(w ** 2))}


def coral(A, others, D, eps=1.0):
    """CORAL (Sun & Saenko 2016): whiten source features, re-colour with target covariance.
    Returns the transformed training frame and the same transform applied to `others`
    (e.g. the PLAsTiCC test set, so in-domain evaluation uses the model's own input space)."""
    Cs = np.cov(A.values, rowvar=False) + eps * np.eye(A.shape[1])
    Ct = np.cov(D.values, rowvar=False) + eps * np.eye(A.shape[1])
    T = np.real(linalg.fractional_matrix_power(Cs, -0.5) @ linalg.fractional_matrix_power(Ct, 0.5))
    tf = lambda F: F.__class__(F.values @ T, columns=F.columns, index=F.index)
    return tf(A), [tf(o) for o in others]



def stratlearn(A, y, D, spec, seed, make_model, k_all, n_strata=5, exclude=(), min_source=30):
    """StratLearn (Autenrieth, van Dyk, Trotta & Stenning 2024): estimate propensity scores
    (probability of belonging to the target) with logistic regression, cut source + target into
    `n_strata` strata at the quantiles of the combined scores, fit a learner on the source objects
    of each stratum and predict the target objects in that stratum. A stratum with fewer than
    `min_source` source objects (or a single class) falls back to a model fitted on all source
    data; the fallback rate is reported. Returns full k_all-column probabilities for D."""
    from sklearn.linear_model import LogisticRegression
    cols = [c for c in A.columns if c not in exclude]
    X = np.r_[A[cols].values, D[cols].values]; dom = np.r_[np.zeros(len(A)), np.ones(len(D))]
    ps = LogisticRegression(max_iter=2000, C=1.0).fit(X, dom).predict_proba(X)[:, 1]
    edges = np.quantile(ps, np.linspace(0, 1, n_strata + 1))[1:-1]
    strata = np.digitize(ps, edges); s_src, s_tgt = strata[:len(A)], strata[len(A):]

    def fit_predict(Xs, ys, Xt):
        present = np.unique(ys); remap = np.full(k_all, -1); remap[present] = np.arange(len(present))
        mdl = make_model(spec, seed).fit(Xs, remap[ys])
        out = np.zeros((len(Xt), k_all)); out[:, present] = np.asarray(mdl.predict_proba(Xt)); return out

    prob = fit_predict(A.values, y, D.values)            # global model = fallback
    info = {"n_strata": n_strata, "strata": []}
    n_fallback = 0
    for s in range(n_strata):
        src, tgt = s_src == s, s_tgt == s
        rec = {"stratum": s, "n_source": int(src.sum()), "n_target": int(tgt.sum())}
        if tgt.sum() == 0:
            info["strata"].append(rec); continue
        if src.sum() < min_source or len(np.unique(y[src])) < 2:
            rec["fallback"] = True; n_fallback += int(tgt.sum())
        else:
            prob[tgt] = fit_predict(A.values[src], y[src], D.values[tgt]); rec["fallback"] = False
        info["strata"].append(rec)
    info["target_fraction_fallback"] = n_fallback / max(1, len(D))
    from sklearn.metrics import roc_auc_score
    info["propensity_auc"] = float(roc_auc_score(dom, ps))
    return prob, info
