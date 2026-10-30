"""Robustness checks on the decomposition and on rate transfer, computed from saved scores.

  corpus_bootstrap  95% intervals for the ranking share and for the recovery rates, resampling
                    the seven corpora (not items), so the interval reflects which corpora we happened
                    to collect
  prior_shift       re-sample each target corpus to a chosen base rate (source prior + delta) and
                    compare default, EM, rate transfer, true-rate and oracle thresholds
  ba_decomp         the same decomposition with balanced accuracy instead of macro-F1
  slack             how far the default threshold is from the in-domain optimum
  per_model         recovery rates of every operating rule for every detector
Writes results/summary_robust.json, results/robust_prior_shift.csv, results/robust_per_model.csv
Run: python src/analysis_robust.py
"""
import glob, re, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, balanced_accuracy_score
import config as C
from common import em_prior

VER = ["ISOT", "WELFake", "PolitiFact", "GossipCop", "LIAR", "Twitter15", "Twitter16"]
OUT = {}
QS = np.linspace(0.005, 0.995, 199)


def mf1(y, yhat):
    return f1_score(y, yhat, average="macro", zero_division=0)


def ba(y, yhat):
    return balanced_accuracy_score(y, yhat)


def qt(p, rate):
    return float(np.quantile(p, 1 - np.clip(rate, 0, 1)))


def _curve(y, p, ts):
    """Macro-F1 and balanced accuracy at each threshold in ts (vectorised)."""
    P, N = y.sum(), len(y) - y.sum()
    o = np.argsort(p)
    ps, ys = p[o], y[o]
    below = np.searchsorted(ps, ts, side="left")          # items with p < t are predicted 0
    pos_below = np.r_[0, np.cumsum(ys)][below]
    fn_ = pos_below; tn = below - pos_below
    tp = P - fn_; fp = N - tn
    f1p = np.where(2 * tp + fp + fn_ > 0, 2 * tp / np.maximum(2 * tp + fp + fn_, 1), 0)
    f1n = np.where(2 * tn + fn_ + fp > 0, 2 * tn / np.maximum(2 * tn + fn_ + fp, 1), 0)
    return (f1p + f1n) / 2, (tp / max(P, 1) + tn / max(N, 1)) / 2


def best(y, p, fn):
    ts = np.unique(np.quantile(p, QS))
    f, b = _curve(y, p, ts)
    return float((f if fn is mf1 else b).max())


def load_preds(models):
    T = pd.concat([pd.read_csv(f) for f in glob.glob(f"{C.RESULTS}/transfer_*.csv")])
    T = T[(T.regime == "full")]
    sp = T.groupby(["model", "source", "seed"]).src_prior.first()
    P = {}
    for f in glob.glob(f"{C.ARTIFACTS}/preds/*_full_*_*_*.npz"):
        m = re.match(r".*/(.+)_full_(.+?)_(.+?)_(\d+)\.npz$", f)
        mo, s, t, seed = m.group(1), m.group(2), m.group(3), int(m.group(4))
        if mo not in models or s not in VER or t not in VER:
            continue
        z = np.load(f)
        P[(mo, s, t, seed)] = (z["y"].astype(int), z["p"], float(sp.get((mo, s, seed), np.nan)))
    return P


def corpus_bootstrap(B=2000):
    rng = np.random.default_rng(0)
    D = pd.read_csv(f"{C.RESULTS}/summary_decomposition_cells.csv")
    D = D[(D.model == "tfidf_lr") & (D.regime == "full")]
    d = D.groupby(["source", "target"])[["ranking", "threshold", "prior"]].mean()
    O = pd.read_csv(f"{C.RESULTS}/summary_operating_points_cells.csv")
    O = O[(O.model == "tfidf_lr") & ~O.in_domain]
    o = O.groupby(["source", "target"])[["default", "em", "src_rate", "true_rate", "oracle"]].mean()

    def stats(w):
        dd = d.mul(w, axis=0).sum()
        oo = o.mul(w, axis=0).sum()
        loss = oo.oracle - oo.default
        return dict(rank_share=dd.ranking / dd.sum(), rec_rank=(oo.src_rate - oo.default) / loss,
                    rec_em=(oo.em - oo.default) / loss, rec_prior=(oo.true_rate - oo.default) / loss)
    point = stats(pd.Series(1.0, index=d.index))
    draws = []
    for _ in range(B):
        k = pd.Series(rng.choice(VER, len(VER))).value_counts()
        w = pd.Series([k.get(s, 0) * k.get(t, 0) for s, t in d.index], index=d.index, dtype=float)
        if w.sum() == 0:
            continue
        draws.append(stats(w))
    dr = pd.DataFrame(draws)
    OUT["corpus_bootstrap"] = {k: dict(point=float(point[k]), lo=float(dr[k].quantile(0.025)),
                                       hi=float(dr[k].quantile(0.975))) for k in point}
    # leave-one-corpus-out sensitivity of the ranking share
    lo = {}
    for c in VER:
        w = pd.Series([0.0 if c in (s, t) else 1.0 for s, t in d.index], index=d.index)
        lo[c] = float(stats(w)["rank_share"])
    OUT["rank_share_drop_one_corpus"] = lo


def prior_shift(P, deltas=(-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3), n=1000, reps=3):
    rng = np.random.default_rng(1)
    rows = []
    for (mo, s, t, seed), (y, p, pi_s) in P.items():
        if mo != "tfidf_lr" or s == t or y.min() == y.max():
            continue
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        for dlt in deltas:
            pi = float(np.clip(pi_s + dlt, 0.05, 0.95))
            for r in range(reps):
                k = int(round(n * pi))
                idx = np.r_[rng.choice(pos, k), rng.choice(neg, n - k)]
                yy, pp = y[idx], p[idx]
                _, q = em_prior(pp, pi_s)
                ys, ps, _ = P[(mo, s, s, seed)]
                rows.append(dict(source=s, target=t, seed=seed, delta=dlt, pi_t=pi, rep=r,
                                 default=mf1(yy, (pp >= .5).astype(int)), em=mf1(yy, (q >= .5).astype(int)),
                                 rank=mf1(yy, (pp >= qt(pp, pi_s)).astype(int)),
                                 acc=mf1(yy, (pp >= qt(pp, acc_prior(pp, ys, ps))).astype(int)),
                                 half=mf1(yy, (pp >= qt(pp, 0.5)).astype(int)),
                                 true_rate=mf1(yy, (pp >= qt(pp, pi)).astype(int)), oracle=best(yy, pp, mf1)))
    R = pd.DataFrame(rows)
    R.to_csv(f"{C.RESULTS}/robust_prior_shift.csv", index=False)
    g = R.groupby("delta")[["default", "em", "rank", "acc", "half", "true_rate", "oracle"]].mean()
    g["rec_rank"] = (g["rank"] - g.default) / (g.oracle - g.default)
    g["rec_em"] = (g.em - g.default) / (g.oracle - g.default)
    g["rec_acc"] = (g.acc - g.default) / (g.oracle - g.default)
    g["rec_half"] = (g.half - g.default) / (g.oracle - g.default)
    g["rank_minus_default"] = g["rank"] - g.default
    OUT["prior_shift"] = {f"{k:+.1f}": {c: float(v) for c, v in row.items()} for k, row in g.iterrows()}
    print(g.round(3).to_string())


def ba_and_slack(P):
    rows = []
    ind = {}
    for (mo, s, t, seed), (y, p, pi_s) in P.items():
        if mo == "tfidf_lr" and s == t:
            ind[(t, seed)] = dict(f1=mf1(y, (p >= .5).astype(int)), f1_opt=best(y, p, mf1),
                                  ba=ba(y, (p >= .5).astype(int)), ba_opt=best(y, p, ba))
    for (mo, s, t, seed), (y, p, pi_s) in P.items():
        if mo != "tfidf_lr" or s == t or (t, seed) not in ind:
            continue
        b = ind[(t, seed)]
        rows.append(dict(source=s, target=t, seed=seed, ba_bb=b["ba"], ba_def=ba(y, (p >= .5).astype(int)),
                         ba_pm=ba(y, (p >= qt(p, y.mean())).astype(int)), ba_or=best(y, p, ba),
                         ba_rank=ba(y, (p >= qt(p, pi_s)).astype(int)),
                         f1_slack=b["f1_opt"] - b["f1"], ba_slack=b["ba_opt"] - b["ba"]))
    X = pd.DataFrame(rows)
    rank, thr, pri = (X.ba_bb - X.ba_or), (X.ba_or - X.ba_pm), (X.ba_pm - X.ba_def)
    tot = rank + thr + pri
    OUT["ba_decomp"] = dict(gap=float(tot.mean()), ranking=float(rank.mean()), threshold=float(thr.mean()),
                            prior=float(pri.mean()), rank_share=float(rank.sum() / tot.sum()),
                            rec_rank=float((X.ba_rank - X.ba_def).mean() / (X.ba_or - X.ba_def).mean()))
    S = pd.DataFrame(ind).T
    OUT["indomain_slack"] = dict(f1_mean=float((S.f1_opt - S.f1).mean()), f1_max=float((S.f1_opt - S.f1).max()),
                                 ba_mean=float((S.ba_opt - S.ba).mean()))
    print(json.dumps({k: OUT[k] for k in ["ba_decomp", "indomain_slack"]}, indent=1))


def per_model():
    O = pd.read_csv(f"{C.RESULTS}/summary_operating_points_cells.csv")
    O = O[~O.in_domain]
    rows = []
    for m, g in O.groupby("model"):
        loss = (g.oracle - g.default).mean()
        r = dict(model=m, cells=len(g), default=g.default.mean(), oracle=g.oracle.mean(), loss=loss)
        for c in ["em", "src_rate", "half_rate", "gmm", "true_rate"]:
            r[f"rec_{c}"] = (g[c] - g.default).mean() / loss
        r["rank_worse_share"] = float((g.src_rate < g.default - 1e-9).mean())
        r["rank_gain_vs_gap_rho"] = spearmanr(g.prior_gap, g.src_rate - g.default).correlation
        rows.append(r)
    M = pd.DataFrame(rows)
    M.to_csv(f"{C.RESULTS}/robust_per_model.csv", index=False)
    OUT["per_model"] = M.set_index("model").to_dict("index")
    print(M.round(3).to_string())


def acc_prior(p_t, y_s, p_s, t=0.5):
    """Adjusted classify-and-count (Forman 2008): invert the source confusion rates at threshold t."""
    tpr = ((p_s >= t) & (y_s == 1)).sum() / max((y_s == 1).sum(), 1)
    fpr = ((p_s >= t) & (y_s == 0)).sum() / max((y_s == 0).sum(), 1)
    if tpr - fpr < 0.05:
        return float((p_t >= t).mean())
    return float(np.clip(((p_t >= t).mean() - fpr) / (tpr - fpr), 0.01, 0.99))


def rate_variants(models):
    """Other label-free rates for the quantile rule, and an honest (cross-fitted) oracle.
    src_ppr  flag the share the source model flags on its own held-out test split
    acc      flag the target prior estimated by adjusted classify-and-count
    oracle_cf  threshold tuned on one random half of the target, scored on the other half"""
    P = load_preds(models)
    rng = np.random.default_rng(2)
    rows = []
    for (mo, s, t, seed), (y, p, pi_s) in P.items():
        if s == t or y.min() == y.max() or (mo, s, s, seed) not in P:
            continue
        ys, ps, _ = P[(mo, s, s, seed)]
        ppr_s = float((ps >= .5).mean())
        pi_acc = acc_prior(p, ys, ps)
        _, q = em_prior(p, pi_s)
        cf = []
        for _ in range(5):
            h = rng.permutation(len(y)) < len(y) // 2
            for a, b in [(h, ~h), (~h, h)]:
                if y[a].min() == y[a].max() or y[b].min() == y[b].max():
                    continue
                qs = np.unique(np.quantile(p[a], QS))
                tb = qs[int(np.argmax(_curve(y[a], p[a], qs)[0]))]
                cf.append(float(_curve(y[b], p[b], np.array([tb]))[0][0]))
        rows.append(dict(model=mo, source=s, target=t, seed=seed, n=len(y), pi_s=pi_s, pi_t=y.mean(),
                         ppr_s=ppr_s, pi_acc=pi_acc, pi_em=float(q.mean()),
                         default=mf1(y, (p >= .5).astype(int)), em=mf1(y, (q >= .5).astype(int)),
                         rank=mf1(y, (p >= qt(p, pi_s)).astype(int)),
                         src_ppr=mf1(y, (p >= qt(p, ppr_s)).astype(int)),
                         acc=mf1(y, (p >= qt(p, pi_acc)).astype(int)),
                         true_rate=mf1(y, (p >= qt(p, y.mean())).astype(int)),
                         oracle=best(y, p, mf1), oracle_cf=float(np.mean(cf)) if cf else np.nan))
    V = pd.DataFrame(rows)
    V.to_csv(f"{C.RESULTS}/robust_rate_variants.csv", index=False)
    res = {}
    for m, g in V.groupby("model"):
        loss = (g.oracle - g.default).mean()
        r = {k: float((g[k] - g.default).mean() / loss) for k in ["em", "rank", "src_ppr", "acc", "true_rate"]}
        r["oracle_cf_minus_oracle"] = float((g.oracle_cf - g.oracle).mean())
        r["err_src_prior"] = float((g.pi_s - g.pi_t).abs().mean())
        r["err_acc"] = float((g.pi_acc - g.pi_t).abs().mean())
        r["err_em"] = float((g.pi_em - g.pi_t).abs().mean())
        r["err_src_ppr"] = float((g.ppr_s - g.pi_t).abs().mean())
        # decomposition with the cross-fitted oracle (tfidf_lr only, filled below)
        res[m] = r
    OUT["rate_variants"] = res
    g = V[V.model == "tfidf_lr"]
    OUT["oracle_cf_tfidf_lr"] = dict(oracle=float(g.oracle.mean()), oracle_cf=float(g.oracle_cf.mean()),
                                     small_targets=float((g[g.n < 250].oracle - g[g.n < 250].oracle_cf).mean()))
    print(pd.DataFrame(res).T.round(3).to_string())
    return V


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "variants":
        OUT.update(json.load(open(f"{C.RESULTS}/summary_robust.json")))
        rate_variants({"tfidf_lr", "tfidf_svm", "nbsvm", "glove_lr", "style_gb", "roberta_lr", "roberta_ft"})
        json.dump(OUT, open(f"{C.RESULTS}/summary_robust.json", "w"), indent=1, default=float)
        sys.exit()
    corpus_bootstrap()
    per_model()
    P = load_preds({"tfidf_lr"})
    ba_and_slack(P)
    prior_shift(P)
    json.dump(OUT, open(f"{C.RESULTS}/summary_robust.json", "w"), indent=1, default=float)
    print(json.dumps(OUT["corpus_bootstrap"], indent=1))
    print(json.dumps(OUT["rank_share_drop_one_corpus"], indent=1))
