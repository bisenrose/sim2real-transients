"""
Shared data loading, feature derivation and preprocessing for the Phase 1 audit.

Everything that turns raw feature files into model inputs lives here, so that
PLAsTiCC and DES are guaranteed to go through the same code.
"""
import os
import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths (relative to the project root, i.e. the folder that contains
# "superphot features/" and "DES dataset/"). Override with env var XAI_ROOT.
# ---------------------------------------------------------------------------
def _find_root():
    """Find the project root: the folder that contains plasticc_raw_superphot_features.csv
    (inside 'superphot features/') and des_superphot_features.csv (inside 'DES dataset/').
    Searches XAI_ROOT, then the folders around this file, then the whole tree below them."""
    here = os.path.dirname(os.path.abspath(__file__))
    starts = [os.environ.get("XAI_ROOT"), os.path.dirname(here), here,]
    starts = [s for s in starts if s and os.path.isdir(s)]
    for s in starts:  # fast path: the expected layout
        if os.path.isfile(os.path.join(s, "superphot features", "plasticc_raw_superphot_features.csv")):
            return s
    for s in starts:  # slow path: search below, in case folders are named or nested differently
        for dirpath, dirnames, filenames in os.walk(s):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("__pycache__", "phase1")]
            if "plasticc_raw_superphot_features.csv" in filenames:
                global PLASTICC_RAW_FOUND
                PLASTICC_RAW_FOUND = os.path.join(dirpath, "plasticc_raw_superphot_features.csv")
                return os.path.dirname(dirpath)
    raise FileNotFoundError(
        "Could not find plasticc_raw_superphot_features.csv under " + ", ".join(starts) +
        ". Check the file exists, then run with XAI_ROOT=/path/to/folder/containing/'superphot features'.")


PLASTICC_RAW_FOUND = None
ROOT = _find_root()
PLASTICC_RAW = PLASTICC_RAW_FOUND or os.path.join(ROOT, "superphot features", "plasticc_raw_superphot_features.csv")
DES_FEATURES = os.path.join(ROOT, "DES dataset", "des_superphot_features.csv")

# ---------------------------------------------------------------------------
# Class codes.
# PLAsTiCC: 95 = SLSN-I, 65 = M-dwarf stellar flare (Kessler et al. 2019).
# The original src/data_loader.py mapped 65 -> 'SLSN-I', which is WRONG.
# ---------------------------------------------------------------------------
SHARED = {90: "SNIa", 42: "SNII", 62: "SNIbc", 88: "AGN", 95: "SLSN-I"}
LEGACY_SHARED = {90: "SNIa", 42: "SNII", 62: "SNIbc", 88: "AGN", 65: "SLSN-I"}  # buggy, for reproduction only

# ---------------------------------------------------------------------------
# DES SNTYPE -> PLAsTiCC class. DES labels are always rebuilt from the 'sntype'
# column, never from the stored 'plasticc_class' column.
# SNTYPE meanings from the DES-SN5YR data release documentation (0_DATA, "SNTYPE
# nomenclature"): 1 SNIa, 101 SNIa?, 3 SNIax, 4 SNIa-pec, 5 SNI, 20 SNIIP, 21 SNIIL,
# 22 SNIIn, 23 SNIIb, 29 SNII, 122 SNIIn?, 129 SNII?, 32 SNIb, 33 SNIc, 39 SNIbc,
# 139 SNIbc?, 41 SLSN-I, 42 SLSN-II, 66 SLSN-II?, 141 SLSN-I?, 80 AGN, 81 TDE,
# 82 M-Star, 180 AGN?.  A trailing '?' = uncertain spectroscopic classification.
# ---------------------------------------------------------------------------
# The mapping the original extraction scripts used (SNIIb -> SNIbc, SNIa-pec -> SNIa, SLSN-I -> 65)
DES_SNTYPE_LEGACY = {1: 90, 4: 90, 29: 42, 129: 42, 32: 62, 33: 62, 39: 62, 139: 62,
                     23: 62, 41: 65, 141: 65, 80: 88, 180: 88, 81: 15, 101: 90}
# Corrected: SNIIb is a type II subtype -> SNII; SNIa-pec dropped (PLAsTiCC SNIa are normal Ia);
# SLSN-I -> 95. SNII subtypes 20/21/22 are listed for completeness (none in this sample).
DES_SNTYPE_CORRECTED = {1: 90, 101: 90, 20: 42, 21: 42, 22: 42, 23: 42, 29: 42, 129: 42,
                        32: 62, 33: 62, 39: 62, 139: 62, 41: 95, 141: 95, 80: 88, 180: 88, 81: 15}
DES_UNCERTAIN_SNTYPES = {101, 129, 139, 141, 180}


def des_labels(des, policy):
    """policy: 'legacy' | 'corrected' | 'confirmed_only' (corrected, minus uncertain '?' types)."""
    if policy == "legacy":
        return des["sntype"].map(DES_SNTYPE_LEGACY).fillna(-1).astype(int)
    lab = des["sntype"].map(DES_SNTYPE_CORRECTED).fillna(-1).astype(int)
    if policy == "confirmed_only":
        lab[des["sntype"].isin(DES_UNCERTAIN_SNTYPES)] = -1
    elif policy != "corrected":
        raise ValueError(policy)
    return lab

ID_COLS = ["object_id", "target", "hostgal_photoz", "distmod", "mwebv"]
SURVEY_ARTEFACTS = ["t_0_ZTF_g", "t_0_ZTF_r", "time_span"]


def load_plasticc():
    df = pd.read_csv(PLASTICC_RAW)
    feats = [c for c in df.columns if c not in ID_COLS]
    return df, feats


def load_des():
    return pd.read_csv(DES_FEATURES)


def derive_features(df, abs_mag_variant=None):
    """Derived features, identical to src/feature_engineering.py but WITHOUT scaling,
    clipping or imputation. Applied identically to every domain.

    abs_mag_variant (Phase 5 robustness check of peak absolute magnitude):
      None        27.5 - 2.5 log10(amplitude) - mu(z)            (default, as in Phases 1-4)
      'bandwidth' default + 2.5 log10(1+z): the bandwidth part of the K-correction for a
                  flat-spectrum source (the colour part needs spectral templates)
      'rband'     apparent magnitude from the fitted r-band peak flux A_r instead of
                  max - min of all g and r flux points
    """
    out = df.copy()
    z = np.clip(out["redshift"].values, 1e-4, 5.0)
    if abs_mag_variant == "rband":
        app_mag = 27.5 - 2.5 * np.log10(np.maximum(1e-4, out["A_ZTF_r"]))
    else:
        app_mag = 27.5 - 2.5 * np.log10(np.maximum(1e-4, out["amplitude"]))
    dist_mod = 5.0 * np.log10(cosmo.luminosity_distance(z).to(u.pc).value / 10.0)
    out["abs_peak_mag"] = app_mag - dist_mod
    if abs_mag_variant == "bandwidth":
        out["abs_peak_mag"] = out["abs_peak_mag"] + 2.5 * np.log10(1.0 + z)
    elif abs_mag_variant not in (None, "rband"):
        raise ValueError(abs_mag_variant)
    out["tau_ratio_g"] = out["tau_rise_ZTF_g"] / (out["tau_fall_ZTF_g"] + 1e-8)
    out["tau_ratio_r"] = out["tau_rise_ZTF_r"] / (out["tau_fall_ZTF_r"] + 1e-8)
    out["is_fast_transient"] = ((out["tau_rise_ZTF_g"] < 5.0) & (out["tau_fall_ZTF_g"] < 10.0)).astype(float)
    out["log_amplitude"] = np.log1p(np.maximum(0, out["amplitude"]))
    out["g_r_peak_color"] = out["A_ZTF_g"] - out["A_ZTF_r"]
    out["g_r_amp_ratio"] = out["A_ZTF_g"] / (out["A_ZTF_r"] + 1e-8)
    out["beta_diff_gr"] = out["beta_ZTF_g"] - out["beta_ZTF_r"]
    out["tau_fall_diff_gr"] = out["tau_fall_ZTF_g"] - out["tau_fall_ZTF_r"]
    return out


def _fit_stats(df):
    med, sd = df.median(), df.std()
    return med, sd


def _apply(df, med, sd, scaler=None):
    df = df.fillna(med)
    for c in df.columns:
        if sd[c] > 1e-8:
            df[c] = df[c].clip(med[c] - 20 * sd[c], med[c] + 20 * sd[c])
    if scaler is None:
        scaler = StandardScaler().fit(df)
    return pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index), scaler


# Source-frame physical quantities (see phase2/PREREGISTRATION_2.1.md). Fixed by physics,
# not tuned on DES labels.
CALIBRATED_FEATURES = ["redshift", "abs_peak_mag"]


def preprocess(train, others, mode, same_survey=None, calibrated=None, stats_frames=None):
    """
    Fill/clip/scale features. `others` is a list of frames; `same_survey[i]` says whether
    others[i] comes from the same survey as `train` (e.g. the PLAsTiCC test set).

    mode = 'consistent' : statistics from the training set, applied to every set.
    mode = 'per_domain' : LEGACY. Every set uses its own statistics, including the PLAsTiCC
                          test set. Kept only to reproduce the original pipeline.
    mode = 'per_survey' : statistics per survey. Same-survey sets use the training
                          statistics; other surveys use their own (transductive, label-free).
    mode = 'hybrid'     : per_survey, except `calibrated` features (default
                          CALIBRATED_FEATURES) use the training statistics everywhere.
    stats_frames: optional list parallel to `others`; when stats_frames[i] is given, the "own"
    statistics of others[i] are fitted on that frame (e.g. a small unlabelled subsample, Phase 6)
    and then applied to the whole of others[i].
    """
    if same_survey is None:
        same_survey = [False] * len(others)
    calibrated = CALIBRATED_FEATURES if calibrated is None else calibrated
    med, sd = _fit_stats(train)
    tr, sc = _apply(train.copy(), med, sd)
    outs = []
    for o, same in zip(others, same_survey):
        o = o[train.columns].copy()
        shared = _apply(o.copy(), med, sd, sc)[0]
        if mode == "consistent" or (mode in ("per_survey", "hybrid") and same):
            outs.append(shared)
        elif mode in ("per_domain", "per_survey", "hybrid"):
            sf = stats_frames[len(outs)] if stats_frames is not None else None
            if sf is None:
                m2, s2 = _fit_stats(o)
                own = _apply(o.copy(), m2, s2)[0]
            else:
                sf = sf[train.columns]
                m2, s2 = _fit_stats(sf)
                _, sc2 = _apply(sf.copy(), m2, s2)
                own = _apply(o.copy(), m2, s2, sc2)[0]
            if mode == "hybrid":
                keep = [c for c in calibrated if c in own.columns]
                own[keep] = shared[keep]
            outs.append(own)
        else:
            raise ValueError(mode)
    return tr, outs


# ---------------------------------------------------------------------------
# Phase 2.2: simulate DES's flux-limited spectroscopic selection on PLAsTiCC.
# Selection variable: apparent peak magnitude m = 27.5 - 2.5 log10(amplitude), the same
# quantity that enters abs_peak_mag. Uses unlabelled DES features only.
# ---------------------------------------------------------------------------
def apparent_mag(df):
    return 27.5 - 2.5 * np.log10(np.maximum(df["amplitude"].values, 1e-3))


def selection_mask(train_df, des_df, spec, seed):
    """
    spec['type']:
      'logistic'        keep each PLAsTiCC object with probability a(m) proportional to the
                        DES/PLAsTiCC density ratio in m, estimated by logistic regression on
                        (m, m^2) with domain as the target (label-free), capped at 1.
      'hard_cut'        keep m < quantile(DES m, spec.get('quantile', 0.99)).
      'random_logistic' / 'random_hard_cut'
                        random subset with the SAME size as the corresponding selection
                        (sample-size control).
    """
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(seed)
    m_tr, m_des = apparent_mag(train_df), apparent_mag(des_df)
    kind = spec["type"]
    base = kind.replace("random_", "")
    if base == "logistic":
        mu, sd = m_tr.mean(), m_tr.std()
        feats = lambda m: np.c_[(m - mu) / sd, ((m - mu) / sd) ** 2]
        X = np.r_[feats(m_tr), feats(m_des)]; y = np.r_[np.zeros(len(m_tr)), np.ones(len(m_des))]
        clf = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)
        p = clf.predict_proba(feats(m_tr))[:, 1]
        ratio = p / (1 - p)
        accept = np.clip(ratio / np.percentile(ratio, spec.get("cap_percentile", 99)), 0, 1)
        keep = rng.uniform(size=len(m_tr)) < accept
    elif base == "hard_cut":
        keep = m_tr < np.quantile(m_des, spec.get("quantile", 0.99))
    else:
        raise ValueError(kind)
    if kind.startswith("random_"):
        n = int(keep.sum())
        keep = np.zeros(len(m_tr), bool); keep[rng.choice(len(m_tr), n, replace=False)] = True
    return keep


# ---------------------------------------------------------------------------
# Phase 4.1: ZTF Bright Transient Survey as a second real target survey.
# Features are produced by phase4/ztf_extract.py with the same light-curve fit as DES.
# BTS types are TNS classifications ("SN Ia", "SN IIb", ...). Mapping fixed in
# phase4/PREREGISTRATION_4.md before any ZTF data was downloaded.
# ---------------------------------------------------------------------------
ZTF_FEATURES = os.path.join(ROOT, "ZTF dataset", "ztf_superphot_features.csv")
BTS_TYPE_TO_PLASTICC = {
    "Ia": 90, "Ia-norm": 90, "Ia-91T": 90, "Ia-99aa": 90,
    "II": 42, "IIP": 42, "II-P": 42, "IIL": 42, "II-L": 42, "IIn": 42, "IIb": 42,
    "Ib": 62, "Ic": 62, "Ib/c": 62, "Ibc": 62, "Ic-BL": 62,
    "SLSN-I": 95,
    "AGN": 88,
    "TDE": 15,
}
# Explicitly excluded (not shared with PLAsTiCC's five classes, or peculiar subtypes):
# Ia-pec, Ia-91bg, Iax, Ia-CSM, Ia-SC, Ia-03fg, II-pec, Ib-pec, Ic-pec, Ibn, Icn, SLSN-II,
# anything with '?', and non-SN types (CV, nova, star, LBV, ILRT, LRN, ...).


def normalise_bts_type(t):
    t = str(t).strip()
    for prefix in ("SN ", "SN"):
        if t.startswith(prefix) and not t.startswith("SLSN"):
            t = t[len(prefix):].strip()
    return t


def ztf_labels(df):
    """PLAsTiCC class code for each ZTF object (-1 = not used), from the BTS/TNS type."""
    t = df["bts_type"].map(normalise_bts_type)
    lab = t.map(BTS_TYPE_TO_PLASTICC).fillna(-1).astype(int)
    lab[t.str.contains(r"\?", regex=True)] = -1
    return lab


def load_target(name):
    """Return (dataframe, label Series) for a target survey: 'des' or 'ztf'."""
    if name == "des":
        return load_des(), None
    if name == "ztf":
        if not os.path.isfile(ZTF_FEATURES):
            raise FileNotFoundError(f"{ZTF_FEATURES} not found. Run phase4/ztf_download.py and phase4/ztf_extract.py first.")
        df = pd.read_csv(ZTF_FEATURES)
        return df, ztf_labels(df)
    if name in ("yse", "yse_no_bts"):
        # Phase 7: YSE DR1, Pan-STARRS g and r only (phase7/yse_extract.py). Spectroscopic types are
        # stored in the 'bts_type' column so the same TNS-type mapping is used as for ZTF.
        if not os.path.isfile(YSE_FEATURES):
            raise FileNotFoundError(f"{YSE_FEATURES} not found. Run phase7/yse_extract.py first.")
        df = pd.read_csv(YSE_FEATURES)
        if name == "yse_no_bts":
            df = df[~df["in_bts"].astype(bool)].reset_index(drop=True)
        return df, ztf_labels(df)
    raise ValueError(name)


YSE_FEATURES = os.path.join(ROOT, "YSE dataset", "yse_superphot_features.csv")
TARGET_PATHS = {"ztf": ZTF_FEATURES, "yse": YSE_FEATURES, "yse_no_bts": YSE_FEATURES}



# ---------------------------------------------------------------------------
# Phase 5
# ---------------------------------------------------------------------------
def selection_weights(train_df, des_df, spec):
    """Phase 5 (reviewer request): the Phase 2.2 apparent-magnitude density ratio used as sample
    WEIGHTS instead of sub-sampling, so the whole training set is kept. Same logistic model on
    (m, m^2), ratio capped at the given percentile, normalised to mean 1. Label-free."""
    from sklearn.linear_model import LogisticRegression
    m_tr, m_des = apparent_mag(train_df), apparent_mag(des_df)
    mu, sd = m_tr.mean(), m_tr.std()
    feats = lambda m: np.c_[(m - mu) / sd, ((m - mu) / sd) ** 2]
    X = np.r_[feats(m_tr), feats(m_des)]; y = np.r_[np.zeros(len(m_tr)), np.ones(len(m_des))]
    p = LogisticRegression(C=1e6, max_iter=1000).fit(X, y).predict_proba(feats(m_tr))[:, 1]
    w = p / (1 - p)
    w = np.minimum(w, np.percentile(w, spec.get("cap_percentile", 99)))
    w = w / w.mean()
    return w, {"weight_max": float(w.max()), "effective_sample_size": float(w.sum() ** 2 / np.sum(w ** 2))}


def support_overlap(train_col, target_col, q=(0.05, 0.95)):
    """Fraction of target values inside the training set's [q_lo, q_hi] quantile range. Label-free."""
    a = np.asarray(train_col, float); b = np.asarray(target_col, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    lo, hi = np.quantile(a, q[0]), np.quantile(a, q[1])
    return float(np.mean((b >= lo) & (b <= hi))) if len(b) else float("nan")


def gated_calibrated(train, target, candidates, threshold, q=(0.05, 0.95)):
    """EXPLORATORY (formulated after the ZTF result): keep a source-frame feature on the shared
    scale only if its support overlap with the training set is at least `threshold`."""
    ov = {c: support_overlap(train[c], target[c], q) for c in candidates if c in train.columns}
    return [c for c, v in ov.items() if v >= threshold], ov



# ---------------------------------------------------------------------------
# Phase 6: synthetic instrument shifts for controlled tests (applied to raw target features)
# ---------------------------------------------------------------------------
FLUX_FEATURES = ["A_ZTF_g", "A_ZTF_r", "amplitude", "std_flux", "extra_sigma_ZTF_g", "extra_sigma_ZTF_r"]


def instrument_shift(df, spec):
    """spec['type']:
      'cadence_noise'    multiply columns by the factors in spec['factors'] (pre-registered)
      'flux_calibration' multiply every flux feature by 10^(-0.4 delta_mag): a calibration offset,
                         so the derived abs_peak_mag shifts by +delta_mag
    """
    out = df.copy()
    if spec["type"] == "cadence_noise":
        for c, f in spec["factors"].items():
            out[c] = out[c] * f
    elif spec["type"] == "flux_calibration":
        k = 10 ** (-0.4 * spec["delta_mag"])
        for c in FLUX_FEATURES:
            out[c] = out[c] * k
    elif spec["type"] != "none":
        raise ValueError(spec)
    return out


# ---------------------------------------------------------------------------
# Phase 8: synthetic population shift "move nearer" (applied to raw target features)
# ---------------------------------------------------------------------------
TIMESCALE_FEATURES = ["tau_rise_ZTF_g", "tau_rise_ZTF_r", "tau_fall_ZTF_g", "tau_fall_ZTF_r", "gamma_ZTF_g", "gamma_ZTF_r"]


def population_shift(df, spec):
    """'move_nearer': place each object at z' = z / factor, as if observed nearer by the same survey.
    Flux features (fitted scales, amplitude, spread and residual scatter) scale by
    (d_L(z) / d_L(z'))^2, observer-frame timescales by (1 + z') / (1 + z); the derived abs_peak_mag
    is then unchanged up to the K-correction we do not model. Cadence features are left alone."""
    if spec["type"] != "move_nearer":
        raise ValueError(spec)
    out = df.copy()
    z = np.clip(out["redshift"].values.astype(float), 1e-4, 5.0); z2 = z / float(spec["factor"])
    k = (cosmo.luminosity_distance(z).value / cosmo.luminosity_distance(z2).value) ** 2
    for c in FLUX_FEATURES:
        out[c] = out[c].values * k
    for c in TIMESCALE_FEATURES:
        out[c] = out[c].values * (1 + z2) / (1 + z)
    out["redshift"] = z2
    return out


def apply_target_shifts(df, population=None, instrument=None):
    """Population shift first, then one or more instrument shifts (a dict or a list of dicts)."""
    if population:
        df = population_shift(df, population)
    if instrument:
        for spec in (instrument if isinstance(instrument, list) else [instrument]):
            df = instrument_shift(df, spec)
    return df
