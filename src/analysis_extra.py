"""Second-stage aggregation (uses saved predictions and the mechanism / repair / Bengali outputs).

  operating points  label-free thresholds computed from saved target scores:
                      default t=0.5 | EM prior adaptation | PPR = source prior (rate transfer)
                      PPR = 0.5 | 2-component GMM on target logits
                    and with one number known: PPR = true target prior; plus the oracle bound
  pad               proxy A-distance vs. transfer AUC (Spearman, corpus-pair bootstrap CI)
  mechanism         compact tables for learning curve, top-K, masking, topic, vocab
  repairs           LOCO repair table with paired Wilcoxon tests vs. pooled
  bengali           official-split comparison with Mahtab et al. (2023), LOSO, translate-test
Run: python src/analysis_extra.py
"""
import glob, os, re, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture
from statsmodels.stats.multitest import multipletests
import config as C
from common import em_prior

VER = ["ISOT", "WELFake", "PolitiFact", "GossipCop", "LIAR", "Twitter15", "Twitter16"]
OUT = {}


def mf1(y, p, t):
    return f1_score(y, (p >= t).astype(int), average="macro", zero_division=0)


def q_thresh(p, rate):
    return float(np.quantile(p, 1 - rate))


def gmm_thresh(p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    g = GaussianMixture(2, random_state=0).fit(z.reshape(-1, 1))
    hi = int(np.argmax(g.means_.ravel()))
    grid = np.linspace(z.min(), z.max(), 400)
    post = g.predict_proba(grid.reshape(-1, 1))[:, hi]
    cross = grid[np.argmin(np.abs(post - 0.5))]
    return 1 / (1 + np.exp(-cross))


def operating_points():
    T = pd.concat([pd.read_csv(f) for f in glob.glob(f"{C.RESULTS}/transfer_*.csv")])
    T = T[(T.regime == "full") & T.source.isin(VER) & T.target.isin(VER)]
    src_prior = T.groupby(["model", "source", "seed"]).src_prior.first()
    rows = []
    for f in glob.glob(f"{C.ARTIFACTS}/preds/*_full_*_*_*.npz"):
        m = re.match(r".*/(.+)_full_(.+?)_(.+?)_(\d+)\.npz$", f)
        model, s, t, seed = m.group(1), m.group(2), m.group(3), int(m.group(4))
        if s not in VER or t not in VER or model == "tfidf_lr_sub":
            continue
        z = np.load(f)
        y, p = z["y"].astype(int), z["p"]
        if y.min() == y.max():
            continue
        pi_s = src_prior.get((model, s, seed), np.nan)
        if np.isnan(pi_s):
            continue
        _, q = em_prior(p, pi_s)
        qs = np.unique(np.quantile(p, np.linspace(0.005, 0.995, 199)))
        rows.append(dict(model=model, source=s, target=t, seed=seed, in_domain=s == t,
                         default=mf1(y, p, 0.5), em=mf1(y, q, 0.5),
                         src_rate=mf1(y, p, q_thresh(p, pi_s)), half_rate=mf1(y, p, q_thresh(p, 0.5)),
                         gmm=mf1(y, p, gmm_thresh(p)), true_rate=mf1(y, p, q_thresh(p, y.mean())),
                         oracle=max(mf1(y, p, t_) for t_ in qs), prior_gap=abs(y.mean() - pi_s)))
    O = pd.DataFrame(rows)
    O.to_csv(f"{C.RESULTS}/summary_operating_points_cells.csv", index=False)
    off = O[~O.in_domain]
    cols = ["default", "em", "src_rate", "half_rate", "gmm", "true_rate", "oracle"]
    agg = off.groupby(["model", "seed"])[cols].mean().groupby("model").agg(["mean", "std"])
    agg.to_csv(f"{C.RESULTS}/summary_operating_points.csv")
    # share of the operating-point loss recovered, per method (tfidf_lr)
    g = off[off.model == "tfidf_lr"]
    loss = (g.oracle - g.default).mean()
    OUT["op_recovery_tfidf_lr"] = {c: float((g[c] - g.default).mean() / loss) for c in cols[1:]}
    # paired test vs default, per method, Holm-corrected (cells x seeds)
    tests = {}
    for c in ["em", "src_rate", "half_rate", "gmm"]:
        tests[c] = wilcoxon(g[c], g.default).pvalue
    adj = multipletests(list(tests.values()), method="holm")[1]
    OUT["op_wilcoxon_holm_tfidf_lr"] = dict(zip(tests, map(float, adj)))
    # when does EM help? correlation with prior gap
    OUT["em_gain_vs_prior_gap_spearman"] = float(spearmanr(g.prior_gap, g.em - g.default).correlation)
    # why EM fails: its prior estimate vs. simply assuming the source prior
    t = T[(T.model == "tfidf_lr") & (T.source != T.target)]
    OUT["em_prior_abs_err"] = float((t.em_prior_est - t.prior).abs().mean())
    OUT["src_prior_abs_err"] = float((t.src_prior - t.prior).abs().mean())
    OUT["default_ppr_abs_err"] = float((t.def_ppr - t.prior).abs().mean())
    OUT["target_ece_mean"] = float(t.ece.mean())
    OUT["target_ece_em_mean"] = float(t.ece_em.mean())
    ind = T[(T.model == "tfidf_lr") & (T.source == T.target)]
    OUT["indomain_ece_mean"] = float(ind.ece.mean())
    print(agg.round(3).to_string())
    print(json.dumps(OUT["op_recovery_tfidf_lr"], indent=1))
    return O


def pad():
    f = f"{C.RESULTS}/mech_pad.csv"
    if not os.path.exists(f):
        return
    P = pd.read_csv(f).groupby(["a", "b"]).pad.mean().reset_index()
    sym = {**{(r.a, r.b): r.pad for r in P.itertuples()}, **{(r.b, r.a): r.pad for r in P.itertuples()}}
    M = pd.read_csv(f"{C.RESULTS}/summary_matrix.csv")
    M = M[(M.model == "tfidf_lr") & (M.source != M.target) & M.source.isin(VER) & M.target.isin(VER)]
    M["pad"] = [sym.get((s, t), np.nan) for s, t in zip(M.source, M.target)]
    M = M.dropna(subset=["pad"]).rename(columns={"auc_mean": "auc"})
    M[["source", "target", "pad", "auc"]].to_csv(f"{C.RESULTS}/summary_pad_cells.csv", index=False)
    rho = spearmanr(M.pad, M.auc).correlation
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):                       # resample corpora (not cells): cells share corpora
        cs = rng.choice(VER, len(VER), replace=True)
        sub = M[M.source.isin(cs) & M.target.isin(cs)]
        if len(sub) > 5:
            boots.append(spearmanr(sub.pad, sub.auc).correlation)
    OUT["pad_vs_auc"] = dict(spearman=float(rho), ci=[float(np.nanpercentile(boots, 2.5)),
                                                      float(np.nanpercentile(boots, 97.5))], n_cells=len(M))
    # in-domain AUC vs mean transfer AUC as source ("promise vs delivery")
    print("PAD vs transfer AUC:", OUT["pad_vs_auc"])


def mechanism():
    R = C.RESULTS
    if os.path.exists(f"{R}/mech_lc.csv"):
        L = pd.read_csv(f"{R}/mech_lc.csv")
        t = L.groupby(["corpus", "n"]).auc.agg(["mean", "std"]).round(3).reset_index()
        t.to_csv(f"{R}/summary_lc.csv", index=False)
        OUT["lc_auc_at_64"] = L[L.n == 64].groupby("corpus").auc.mean().round(3).to_dict()
        OUT["lc_auc_at_256"] = L[L.n == 256].groupby("corpus").auc.mean().round(3).to_dict()
    if os.path.exists(f"{R}/mech_topk.csv"):
        K = pd.read_csv(f"{R}/mech_topk.csv")
        K["kind"] = np.where(K.source == K.target, "in", "out")
        t = K.groupby(["source", "K", "kind"]).auc.mean().unstack().round(3)
        t.to_csv(f"{R}/summary_topk.csv")
        OUT["topk"] = {f"{s}|{k}": v for (s, k), v in t["in"].to_dict().items()}
        OUT["topk_out"] = {f"{s}|{k}": v for (s, k), v in t["out"].to_dict().items()}
    for name in ["mask", "topic", "vocab"]:
        if os.path.exists(f"{R}/mech_{name}.csv"):
            X = pd.read_csv(f"{R}/mech_{name}.csv")
            X["kind"] = np.where(X.source == X.target, "in", "out")
            t = X.groupby(["source", "variant", "kind"]).auc.mean().unstack().round(3)
            t.to_csv(f"{R}/summary_{name}.csv")
            OUT[name] = {f"{s}|{v}": dict(r) for (s, v), r in t.iterrows()}
            if name == "mask":
                pair = X[(X.source == "ISOT") & (X.target == "WELFake") | (X.source == "WELFake") & (X.target == "ISOT")]
                OUT["mask_pair"] = pair.groupby(["source", "variant"]).auc.mean().round(3).to_dict()
                OUT["mask_pair"] = {f"{a}|{b}": v for (a, b), v in OUT["mask_pair"].items()}
    if os.path.exists(f"{R}/mech_group.csv"):
        G = pd.read_csv(f"{R}/mech_group.csv")
        OUT["group_only_auc"] = G.groupby("corpus").auc.mean().round(3).to_dict()


def repairs():
    f = f"{C.RESULTS}/fix_loco.csv"
    if not os.path.exists(f):
        return
    F = pd.read_csv(f)
    tab = F.groupby("method")[["auc", "ap", "def_macro_f1", "em_macro_f1", "pm_macro_f1", "oracle_macro_f1"]].mean()
    tab.to_csv(f"{C.RESULTS}/summary_fix.csv")
    base = F[F.method == "pooled"].set_index(["target", "seed"]).auc
    tests = {}
    for m in F.method.unique():
        if m == "pooled":
            continue
        x = F[F.method == m].set_index(["target", "seed"]).auc.reindex(base.index)
        tests[m] = dict(delta=float((x - base).mean()), p=float(wilcoxon(x, base).pvalue))
    adj = multipletests([v["p"] for v in tests.values()], method="holm")[1]
    for (k, v), a in zip(tests.items(), adj):
        v["p_holm"] = float(a)
    OUT["repairs_vs_pooled_auc"] = tests
    per_t = F.groupby(["target", "method"]).auc.mean().unstack().round(3)
    per_t.to_csv(f"{C.RESULTS}/summary_fix_per_target.csv")
    # gap closure vs best single source and vs target in-domain ceiling
    M = pd.read_csv(f"{C.RESULTS}/summary_matrix.csv")
    M = M[M.model == "tfidf_lr"]
    ceil = M[M.source == M.target].set_index("target").auc_mean
    best = M[M.source != M.target].groupby("target").auc_mean.max()
    mean_single = M[M.source != M.target].groupby("target").auc_mean.mean()
    per_t["ceiling"], per_t["best_single_source"], per_t["mean_single_source"] = ceil, best, mean_single
    per_t.to_csv(f"{C.RESULTS}/summary_fix_per_target.csv")
    print(tab.round(3).to_string())
    print(json.dumps(tests, indent=1))


def bengali():
    R = C.RESULTS
    if os.path.exists(f"{R}/bangla_official.csv"):
        B = pd.read_csv(f"{R}/bangla_official.csv")
        OUT["bangla_official"] = B.round(4).to_dict("records")
    if os.path.exists(f"{R}/bangla_seeds.csv"):
        S = pd.read_csv(f"{R}/bangla_seeds.csv")
        agg = S.groupby("model")[["f1_clickbait", "macro_f1", "auc"]].agg(["mean", "std"]).round(4)
        OUT["bangla_seeds"] = {m: {f"{a}_{b}": float(v) for (a, b), v in r.items()} for m, r in agg.iterrows()}
    if os.path.exists(f"{R}/bangla_loso.csv"):
        L = pd.read_csv(f"{R}/bangla_loso.csv")
        w = L.n / L.n.sum()
        OUT["bangla_loso"] = dict(weighted_accuracy=float((L.accuracy * w).sum()),
                                  mean_auc=float(L.auc.mean()), n_domains=len(L),
                                  domains_single_class=int((L.clickbait_rate.isin([0, 1])).sum()))
    if os.path.exists(f"{R}/bangla_xling.csv"):
        X = pd.read_csv(f"{R}/bangla_xling.csv")
        OUT["bangla_xling"] = X.groupby(["source", "target"])[["auc", "def_macro_f1", "pm_macro_f1"]].mean().round(3).reset_index().to_dict("records")


if __name__ == "__main__":
    operating_points()
    pad()
    mechanism()
    repairs()
    bengali()
    json.dump(OUT, open(f"{C.RESULTS}/summary_extra.json", "w"), indent=1, default=str)
