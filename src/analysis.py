"""Aggregate raw results into the tables reported in the paper (results/summary_*.csv/json).

Transfer gap decomposition for a cell A -> B (A != B), per seed, using macro-F1:
    G  = F1_AA(def) - F1_AB(def)                       total gap a benchmark user experiences
       = [F1_AA(def) - F1_BB(def)]                     (i)   difficulty difference of B vs A
       + [F1_BB(def) - F1_AB(oracle)]                  (ii)  ranking deficit vs B's own ceiling
       + [F1_AB(oracle) - F1_AB(prior-matched)]        (iii) residual threshold loss
       + [F1_AB(prior-matched) - F1_AB(def)]           (iv)  prior / calibration loss
(i)+(ii) need a better representation; (iii)+(iv) only need a better operating point.
Run: python src/analysis.py
"""
import glob, json, os, re
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score, f1_score
import config as C

VER = ["ISOT", "WELFake", "PolitiFact", "GossipCop", "LIAR", "Twitter15", "Twitter16"]
PAIRS = [("ISOT", "WELFake"), ("WELFake", "ISOT"), ("PolitiFact", "GossipCop"), ("GossipCop", "PolitiFact"),
         ("Twitter15", "Twitter16"), ("Twitter16", "Twitter15"), ("LIAR", "PolitiFact"), ("PolitiFact", "LIAR")]
MODELS = ["tfidf_lr", "tfidf_svm", "nbsvm", "glove_lr", "style_gb", "roberta_lr"]
OUT = {}


def load_transfer():
    fr = [pd.read_csv(f) for f in glob.glob(f"{C.RESULTS}/transfer_*.csv")]
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()


def ms(x):
    return f"{np.mean(x):.3f} ± {np.std(x, ddof=1):.3f}" if len(x) > 1 else f"{np.mean(x):.3f}"


def pooled_ci(model, regime, src, tgt, fn, n_boot=200):
    vals = []
    for f in sorted(glob.glob(f"{C.ARTIFACTS}/preds/{model}_{regime}_{src}_{tgt}_*.npz")):
        z = np.load(f)
        y, p = z["y"], z["p"]
        rng = np.random.default_rng(int(re.findall(r"_(\d+)\.npz$", f)[0]))
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() != y[i].max():
                vals.append(fn(y[i], p[i]))
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


def f1_def(y, p):
    return f1_score(y, (p >= 0.5).astype(int), average="macro", zero_division=0)


def matrix_tables(T):
    rows = []
    for model, g in T[T.regime == "full"].groupby("model"):
        for (s, t), c in g.groupby(["source", "target"]):
            r = dict(model=model, source=s, target=t, n_seeds=len(c))
            for k in ["auc", "ap", "def_macro_f1", "def_bal_acc", "def_mcc", "def_f1_real", "def_f1_fake",
                      "def_ppr", "prior", "pm_macro_f1", "em_macro_f1", "oracle_macro_f1", "ece", "ece_em",
                      "em_prior_est"]:
                r[k + "_mean"], r[k + "_sd"] = c[k].mean(), c[k].std(ddof=1)
            if model in ("tfidf_lr", "roberta_lr"):
                r["auc_ci_lo"], r["auc_ci_hi"] = pooled_ci(model, "full", s, t, roc_auc_score)
                r["f1_ci_lo"], r["f1_ci_hi"] = pooled_ci(model, "full", s, t, f1_def)
            rows.append(r)
    M = pd.DataFrame(rows)
    M.to_csv(f"{C.RESULTS}/summary_matrix.csv", index=False)
    return M


def decomposition(T, model="tfidf_lr", regime="full"):
    g = T[(T.model == model) & (T.regime == regime)]
    ind = g[g.source == g.target].set_index(["seed", "source"])
    rows = []
    for _, r in g[g.source != g.target].iterrows():
        aa = ind.loc[(r.seed, r.source), "def_macro_f1"]
        bb = ind.loc[(r.seed, r.target), "def_macro_f1"]
        rows.append(dict(model=model, regime=regime, seed=r.seed, source=r.source, target=r.target,
                         gap=aa - r.def_macro_f1, gap_target=bb - r.def_macro_f1,
                         difficulty=aa - bb, ranking=bb - r.oracle_macro_f1,
                         threshold=r.oracle_macro_f1 - r.pm_macro_f1, prior=r.pm_macro_f1 - r.def_macro_f1,
                         em_recovered=r.em_macro_f1 - r.def_macro_f1,
                         op_loss=r.oracle_macro_f1 - r.def_macro_f1,
                         auc_gap=ind.loc[(r.seed, r.source), "auc"] - r.auc,
                         auc_target_gap=ind.loc[(r.seed, r.target), "auc"] - r.auc,
                         same_task=(r.source, r.target) in PAIRS))
    return pd.DataFrame(rows)


def mcnemar_tests(T):
    """Paired McNemar on default-threshold predictions, model vs tfidf_lr, same items."""
    rows = []
    for m in [x for x in MODELS if x != "tfidf_lr"]:
        for (s, t) in PAIRS + [(v, v) for v in VER]:
            for seed in C.SEEDS:
                fa = f"{C.ARTIFACTS}/preds/tfidf_lr_full_{s}_{t}_{seed}.npz"
                fb = f"{C.ARTIFACTS}/preds/{m}_full_{s}_{t}_{seed}.npz"
                if not (os.path.exists(fa) and os.path.exists(fb)):
                    continue
                a, b = np.load(fa), np.load(fb)
                if len(a["y"]) != len(b["y"]) or not (a["y"] == b["y"]).all():
                    continue
                ca = (a["p"] >= .5) == a["y"]
                cb = (b["p"] >= .5) == b["y"]
                tab = [[int((ca & cb).sum()), int((ca & ~cb).sum())], [int((~ca & cb).sum()), int((~ca & ~cb).sum())]]
                rows.append(dict(model=m, source=s, target=t, seed=seed,
                                 p=mcnemar(tab, exact=False, correction=True).pvalue,
                                 acc_lr=ca.mean(), acc_model=cb.mean()))
    R = pd.DataFrame(rows)
    if len(R):
        R["p_holm"] = multipletests(R.p, method="holm")[1]
    R.to_csv(f"{C.RESULTS}/summary_mcnemar.csv", index=False)
    return R


def main():
    T = load_transfer()
    T = T[T.source.isin(VER) & T.target.isin(VER)]
    M = matrix_tables(T)
    # headline per model: in-domain vs off-diagonal vs same-task pairs
    head = []
    for (model, regime), g in T.groupby(["model", "regime"]):
        ind = g[g.source == g.target]
        off = g[g.source != g.target]
        pair = off[[tuple(x) in PAIRS for x in off[["source", "target"]].values]]
        by_seed = lambda d, k: d.groupby("seed")[k].mean()
        head.append(dict(model=model, regime=regime,
                         in_auc=ms(by_seed(ind, "auc")), off_auc=ms(by_seed(off, "auc")),
                         pair_auc=ms(by_seed(pair, "auc")),
                         in_f1=ms(by_seed(ind, "def_macro_f1")), off_f1=ms(by_seed(off, "def_macro_f1")),
                         off_f1_em=ms(by_seed(off, "em_macro_f1")), off_f1_oracle=ms(by_seed(off, "oracle_macro_f1")),
                         off_cells_auc_below_05=float((off.groupby(["source", "target"]).auc.mean() < 0.5).mean()),
                         n_rows=len(g)))
    H = pd.DataFrame(head)
    H.to_csv(f"{C.RESULTS}/summary_headline.csv", index=False)
    OUT["headline"] = H.to_dict("records")
    # decomposition
    decs = [decomposition(T, m, rg) for m in T.model.unique() for rg in ["full", "matched"]
            if len(T[(T.model == m) & (T.regime == rg)])]
    Dc = pd.concat(decs)
    Dc.to_csv(f"{C.RESULTS}/summary_decomposition_cells.csv", index=False)
    agg = Dc.groupby(["model", "regime", "same_task"])[["gap", "gap_target", "difficulty", "ranking", "threshold",
                                                        "prior", "em_recovered", "op_loss", "auc_gap",
                                                        "auc_target_gap"]].mean().reset_index()
    agg["em_share_of_op_loss"] = agg.em_recovered / agg.op_loss
    agg["ranking_share_of_target_gap"] = agg.ranking / agg.gap_target
    agg["op_share_of_target_gap"] = (agg.threshold + agg.prior) / agg.gap_target
    agg.to_csv(f"{C.RESULTS}/summary_decomposition.csv", index=False)
    OUT["decomposition"] = agg.to_dict("records")
    # same-task pair table (tfidf_lr full) with CIs
    pt = M[(M.model == "tfidf_lr")]
    rows = []
    for s, t in PAIRS:
        a = pt[(pt.source == s) & (pt.target == t)].iloc[0]
        i = pt[(pt.source == s) & (pt.target == s)].iloc[0]
        rows.append(dict(source=s, target=t, in_auc=i.auc_mean, auc=a.auc_mean, auc_sd=a.auc_sd,
                         auc_ci=(a.auc_ci_lo, a.auc_ci_hi), f1_def=a.def_macro_f1_mean, f1_em=a.em_macro_f1_mean,
                         f1_pm=a.pm_macro_f1_mean, f1_oracle=a.oracle_macro_f1_mean, ppr=a.def_ppr_mean,
                         prior=a.prior_mean, ece=a.ece_mean, ece_em=a.ece_em_mean))
    P = pd.DataFrame(rows)
    P.to_csv(f"{C.RESULTS}/summary_pairs.csv", index=False)
    # permutation control
    perm = T[T.regime == "permuted"]
    OUT["permutation_auc"] = dict(mean=float(perm.auc.mean()), sd=float(perm.auc.std()),
                                  min=float(perm.auc.min()), max=float(perm.auc.max()),
                                  in_domain_mean=float(perm[perm.source == perm.target].auc.mean()))
    # model comparison
    R = mcnemar_tests(T)
    if len(R):
        OUT["mcnemar"] = R.groupby("model").apply(lambda g: dict(
            n=len(g), frac_sig_holm=float((g.p_holm < 0.05).mean()),
            mean_acc_lr=float(g.acc_lr.mean()), mean_acc_model=float(g.acc_model.mean()))).to_dict()
    json.dump(OUT, open(f"{C.RESULTS}/summary.json", "w"), indent=1, default=str)
    print(H.to_string(index=False))
    print(agg.round(3).to_string(index=False))
    print(P.round(3).to_string(index=False))
    print(json.dumps(OUT["permutation_auc"], indent=1))


if __name__ == "__main__":
    main()
