"""Shared machinery: splits, metrics, operating points, prior adaptation, bootstrap.

Conventions
  * label 1 = fake/false/rumour/clickbait
  * every model outputs a probability calibrated on the SOURCE validation split (Platt)
  * the default operating point is t = 0.5 on that calibrated probability
"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             balanced_accuracy_score, matthews_corrcoef)
import config as C

CAP_TRAIN, CAP_VAL, CAP_TEST = 8000, 1500, 3000


def load_corpora():
    return pd.read_parquet(f"{C.ARTIFACTS}/corpora.parquet")


def _cap(df, n, seed):
    if len(df) <= n:
        return df
    return train_test_split(df, train_size=n, stratify=df.label, random_state=seed)[0]


def split(df, seed, cap=True):
    """Stratified 70/10/20 split of one corpus for one seed."""
    tr, te = train_test_split(df, test_size=0.2, stratify=df.label, random_state=seed)
    tr, va = train_test_split(tr, test_size=0.125, stratify=tr.label, random_state=seed)
    if cap:
        tr, va, te = _cap(tr, CAP_TRAIN, seed), _cap(va, CAP_VAL, seed), _cap(te, CAP_TEST, seed)
    return tr, va, te


# ------------------------------------------------------------------ calibration / priors
class Platt:
    def fit(self, s, y):
        self.m = LogisticRegression(C=1e4, max_iter=1000).fit(np.asarray(s).reshape(-1, 1), y)
        return self

    def __call__(self, s):
        return self.m.predict_proba(np.asarray(s).reshape(-1, 1))[:, 1]


def em_prior(p, pi_s, iters=200, tol=1e-6):
    """Saerens et al. (2002) EM: re-estimate the target prior from unlabelled posteriors.
    Returns (pi_t, adjusted posteriors)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    pi_t = pi_s
    for _ in range(iters):
        a = (pi_t / pi_s) * p
        b = ((1 - pi_t) / (1 - pi_s)) * (1 - p)
        q = a / (a + b)
        new = q.mean()
        if abs(new - pi_t) < tol:
            pi_t = new
            break
        pi_t = new
    a = (pi_t / pi_s) * p
    b = ((1 - pi_t) / (1 - pi_s)) * (1 - p)
    return pi_t, a / (a + b)


# ------------------------------------------------------------------ metrics
def ece(y, p, bins=15):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def at_threshold(y, p, t):
    yhat = (p >= t).astype(int)
    f1s = f1_score(y, yhat, average=None, labels=[0, 1], zero_division=0)
    return dict(macro_f1=float(f1s.mean()), f1_real=float(f1s[0]), f1_fake=float(f1s[1]),
                bal_acc=float(balanced_accuracy_score(y, yhat)),
                mcc=float(matthews_corrcoef(y, yhat)) if len(set(yhat)) > 1 else 0.0,
                ppr=float(yhat.mean()))


def oracle_f1(y, p):
    """Best macro-F1 over all thresholds on the target (an upper bound given the ranking)."""
    qs = np.unique(np.quantile(p, np.linspace(0.005, 0.995, 199)))
    return max(f1_score(y, (p >= t).astype(int), average="macro", zero_division=0) for t in qs)


def prior_matched_threshold(p, pi):
    """Threshold that makes the predicted-positive rate equal to the prior pi."""
    return float(np.quantile(p, 1 - pi))


def evaluate(y, p, pi_s):
    """All target-side metrics for one (model, source, target, seed)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    pi_t = float(y.mean())
    out = dict(n=len(y), prior=pi_t, src_prior=float(pi_s),
               auc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)),
               ece=float(ece(y, p)))
    d = at_threshold(y, p, 0.5)
    out.update({f"def_{k}": v for k, v in d.items()})
    pm = at_threshold(y, p, prior_matched_threshold(p, pi_t))
    out.update({f"pm_{k}": v for k, v in pm.items()})
    out["oracle_macro_f1"] = float(oracle_f1(y, p))
    pi_em, q = em_prior(p, pi_s)
    em = at_threshold(y, q, 0.5)
    out.update({f"em_{k}": v for k, v in em.items()})
    out["em_prior_est"] = float(pi_em)
    out["ece_em"] = float(ece(y, q))
    return out


def bootstrap(y, p, fn, n=200, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        vals.append(fn(y[i], p[i]))
    return np.array(vals)


def macro_f1_at(t):
    return lambda y, p: f1_score(y, (p >= t).astype(int), average="macro", zero_division=0)


def auc_fn(y, p):
    return roc_auc_score(y, p)
