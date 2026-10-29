"""Constructive part: can label-free repairs close the transfer gap?

Setting: leave-one-corpus-out (LOCO) over the seven veracity corpora. For every held-out
target T and seed, the training pool is the union of the other six corpora (each capped at
1,500 training / 300 validation items, stratified). No target labels are used by any method.

Methods (all TF-IDF based; dense variants use a 300-d SVD of the same TF-IDF space)
  pooled            logistic regression on the pooled corpora
  pooled_bal        + sample weights equalising every (corpus, class) cell
  pooled_mask       + masking of source tags, datelines, URLs, handles, weekdays, months, numbers
  corpus_feat_drop  + delete the 2,000 features most predictive of corpus identity
  svd               dense baseline: logistic regression on SVD-300 features
  inlp              + iterative nullspace projection removing corpus-identity directions
  coral             + CORAL alignment of source to (unlabelled) target feature covariance
On top of every method, Saerens EM prior adaptation is reported by common.evaluate (em_*).

Writes results/fix_loco.csv
Run: python src/fix.py
"""
import numpy as np, pandas as pd
from scipy import linalg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
import config as C
from common import load_corpora, split, evaluate, Platt, _cap
from mechanism import mask_texts

VERACITY = ["ISOT", "WELFake", "PolitiFact", "GossipCop", "LIAR", "Twitter15", "Twitter16"]
PER_TRAIN, PER_VAL, MAXCH = 1500, 300, 6000
GRID = [0.25, 1.0, 4.0]


def fit_lr(X, y, Xv, yv, w=None):
    best, bauc = None, -1
    for c in GRID:
        m = LogisticRegression(C=c, max_iter=3000, solver="liblinear").fit(X, y, sample_weight=w)
        a = roc_auc_score(yv, m.decision_function(Xv))
        if a > bauc:
            best, bauc = m, a
    cal = Platt().fit(best.decision_function(Xv), yv)
    return lambda Z: cal(best.decision_function(Z))


def inlp(Z, corp_ids, Zv, cv, iters=15):
    """Iterative nullspace projection (Ravfogel et al., 2020) for a multi-class attribute."""
    d = Z.shape[1]
    P = np.eye(d)
    hist = []
    for _ in range(iters):
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(Z @ P.T, corp_ids)
        acc = clf.score(Zv @ P.T, cv)
        hist.append(acc)
        W = clf.coef_
        _, s, vt = np.linalg.svd(W, full_matrices=False)
        B = vt[s > 1e-8]
        P = (np.eye(d) - B.T @ B) @ P
    return P, hist


def coral(Zs, Zt):
    cs = np.cov(Zs, rowvar=False) + np.eye(Zs.shape[1])
    ct = np.cov(Zt, rowvar=False) + np.eye(Zs.shape[1])
    A = np.real(linalg.fractional_matrix_power(cs, -0.5) @ linalg.fractional_matrix_power(ct, 0.5))
    return lambda Z: Z @ A


def main():
    D = load_corpora()
    D = D[D.corpus.isin(VERACITY)]
    rows, done = [], set()
    out = f"{C.RESULTS}/fix_loco.csv"
    import os
    if os.path.exists(out):                       # resume: keep seeds that are complete
        old = pd.read_csv(out)
        done = {int(s) for s, g in old.groupby("seed") if len(g) >= 7 * len(VERACITY)}
        rows = old[old.seed.isin(done)].to_dict("records")
    for seed in C.SEEDS:
        if seed in done:
            print("seed", seed, "done, skipping", flush=True)
            continue
        splits = {c: split(D[D.corpus == c], seed) for c in VERACITY}
        for T in VERACITY:
            srcs = [c for c in VERACITY if c != T]
            tr = pd.concat([_cap(splits[c][0], PER_TRAIN, seed) for c in srcs])
            va = pd.concat([_cap(splits[c][1], PER_VAL, seed) for c in srcs])
            te = splits[T][2]
            pat = "|".join(fr"(?:^|;){c}(?:;|$)" for c in srcs)
            te = te[~te.dup_in.fillna("").str.contains(pat)]
            ttr, tva, tte = ([str(t)[:MAXCH] for t in x.text] for x in (tr, va, te))
            ytr, yva, yte = tr.label.values, va.label.values, te.label.values
            pi_s = float(ytr.mean())

            def record(name, p, extra=None):
                r = evaluate(yte, p, pi_s)
                r.update(method=name, target=T, seed=seed, n_train=len(tr), **(extra or {}))
                rows.append(r)

            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)
            X, Xv, Xt = vec.fit_transform(ttr), vec.transform(tva), vec.transform(tte)
            record("pooled", fit_lr(X, ytr, Xv, yva)(Xt))
            # (corpus, class) balanced weights
            cell = tr.corpus.astype(str) + "_" + tr.label.astype(str)
            w = cell.map(1.0 / cell.value_counts()).values
            w = w / w.mean()
            record("pooled_bal", fit_lr(X, ytr, Xv, yva, w)(Xt))
            # corpus-identity feature deletion
            cid = OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=2000, solver="liblinear")).fit(
                X, tr.corpus.values)
            score = np.abs(np.vstack([e.coef_.ravel() for e in cid.estimators_])).max(0)
            keep = np.argsort(score)[:-2000]
            record("corpus_feat_drop", fit_lr(X[:, keep], ytr, Xv[:, keep], yva, w)(Xt[:, keep]))
            # masking
            mtr, mva, mte = (mask_texts(x, "source", MAXCH) for x in (ttr, tva, tte))
            mtr, mva, mte = (mask_texts(x, "time", MAXCH) for x in (mtr, mva, mte))
            mvec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)
            M, Mv, Mt = mvec.fit_transform(mtr), mvec.transform(mva), mvec.transform(mte)
            record("pooled_mask", fit_lr(M, ytr, Mv, yva, w)(Mt))
            # dense: SVD, INLP, CORAL
            svd = TruncatedSVD(300, random_state=seed).fit(X)
            Z, Zv, Zt = svd.transform(X), svd.transform(Xv), svd.transform(Xt)
            mu, sd = Z.mean(0), Z.std(0) + 1e-6
            Z, Zv, Zt = (Z - mu) / sd, (Zv - mu) / sd, (Zt - mu) / sd
            record("svd", fit_lr(Z, ytr, Zv, yva, w)(Zt))
            P, hist = inlp(Z, tr.corpus.values, Zv, va.corpus.values)
            record("inlp", fit_lr(Z @ P.T, ytr, Zv @ P.T, yva, w)(Zt @ P.T),
                   dict(corpus_id_acc_before=hist[0], corpus_id_acc_after=hist[-1]))
            A = coral(Z, Zt)
            record("coral", fit_lr(A(Z), ytr, A(Zv), yva, w)(Zt))
            print(f"seed={seed} T={T:<10} " + " ".join(
                f"{r['method']}={r['auc']:.3f}" for r in rows if r['seed'] == seed and r['target'] == T),
                flush=True)
        pd.DataFrame(rows).to_csv(f"{C.RESULTS}/fix_loco.csv", index=False)


if __name__ == "__main__":
    import os, sys
    done = f"{C.RESULTS}/fix_loco.csv"
    if os.path.exists(done) and len(pd.read_csv(done)) >= 7 * 7 * len(C.SEEDS) and "--force" not in sys.argv:
        sys.exit("fix_loco.csv is complete; use --force to rerun")
    lock = f"{C.ARTIFACTS}/fix.lock"
    if os.path.exists(lock):
        try:
            os.kill(int(open(lock).read()), 0)
            sys.exit("another fix.py is running; exiting")
        except (OSError, ValueError):
            pass
    open(lock, "w").write(str(os.getpid()))
    try:
        main()
    finally:
        os.remove(lock)
