"""
Model families, with matched settings (same number of trees, depth and learning rate for
the three boosted-tree models; the MLP matches src/experiments/exp12's architecture).

Note for the paper: the neural model in exp12 was labelled "SuperNNova", but it is a plain
feed-forward MLP on the 27 features, not SuperNNova (an RNN on light curves). Call it "MLP".
"""
import numpy as np


class _MLPTorch:
    """Same architecture/training as exp12: 128-64 with BatchNorm, ReLU, Dropout; AdamW, 80 epochs."""
    def __init__(self, seed, epochs=80, lr=0.003, batch=128, weight_decay=1e-4):
        self.seed, self.epochs, self.lr, self.batch, self.wd = seed, epochs, lr, batch, weight_decay

    def fit(self, X, y):
        import torch, torch.nn as nn
        torch.manual_seed(self.seed)
        k = int(np.max(y)) + 1
        self.net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, k))
        Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        loss = nn.CrossEntropyLoss(); g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            self.net.train()
            for idx in torch.randperm(len(Xt), generator=g).split(self.batch):
                if len(idx) < 2:
                    continue
                opt.zero_grad(); loss(self.net(Xt[idx]), yt[idx]).backward(); opt.step()
        return self

    def predict_proba(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            return torch.softmax(self.net(torch.tensor(X, dtype=torch.float32)), 1).numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def make_model(spec, seed):
    """spec: dict with 'type' (xgboost | lightgbm | catboost | mlp) and hyperparameters."""
    t = spec.get("type", "xgboost")
    extra = dict(spec.get("params", {}))   # tuned or extra hyperparameters, passed straight through
    spec = dict(spec)
    for k in ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"):
        if k in extra:                      # tuned values override the matched defaults
            spec[k] = extra.pop(k)
    n, d, lr = spec.get("n_estimators", 250), spec.get("max_depth", 5), spec.get("learning_rate", 0.05)
    sub, col = spec.get("subsample", 1.0), spec.get("colsample_bytree", 1.0)
    if t == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=n, max_depth=d, learning_rate=lr, subsample=sub,
                                 colsample_bytree=col, random_state=seed, n_jobs=-1, verbosity=0, **extra)
    if t == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=n, max_depth=d, learning_rate=lr, subsample=sub,
                                  subsample_freq=1 if sub < 1 else 0, colsample_bytree=col,
                                  random_state=seed, n_jobs=-1, verbose=-1, **extra)
    if t == "catboost":
        from catboost import CatBoostClassifier
        kw = dict(bootstrap_type="Bernoulli", subsample=sub) if sub < 1 else {}
        return CatBoostClassifier(iterations=n, depth=d, learning_rate=lr, rsm=col, random_seed=seed,
                                  verbose=0, thread_count=-1, allow_writing_files=False, **kw, **extra)
    if t == "mlp":
        try:
            import torch  # noqa: F401
            return _MLPTorch(seed, **extra)
        except ImportError:  # fallback when PyTorch is unavailable (recorded in results)
            from sklearn.neural_network import MLPClassifier
            kw = dict(alpha=extra.get("weight_decay", 1e-4), learning_rate_init=extra.get("lr", 0.003),
                      max_iter=extra.get("epochs", 80), batch_size=extra.get("batch", 128))
            return MLPClassifier(hidden_layer_sizes=(128, 64), random_state=seed, **kw)
    raise ValueError(t)


def supports_weights(spec):
    return spec.get("type", "xgboost") in ("xgboost", "lightgbm", "catboost")


def backend(spec):
    if spec.get("type") != "mlp":
        return spec.get("type", "xgboost")
    try:
        import torch  # noqa: F401
        return "mlp-torch"
    except ImportError:
        return "mlp-sklearn-fallback"
