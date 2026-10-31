"""Cross-check: recompute headline numbers from raw result files (independently of
paper_numbers.py) and assert every directional claim made in the manuscript.
Exit code 1 if any check fails.  Run: python src/verify_claims.py
"""
import json, os, re, sys, glob
import numpy as np, pandas as pd
import config as C

R = C.RESULTS
VER = ["ISOT", "WELFake", "PolitiFact", "GossipCop", "LIAR", "Twitter15", "Twitter16"]
K = json.load(open(f"{R}/key_numbers.json"))
fails, n = [], 0


def num(k):
    v = K[k]
    v = v.replace("−", "-").replace("%", "").replace(",", "").replace("<", "").replace("[", "").split()[0]
    return float(v)


def check(cond, msg):
    global n
    n += 1
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def close(a, b, tol, msg):
    check(abs(a - b) <= tol, f"{msg}: recomputed {a:.4f} vs reported {b:.4f}")


T = pd.read_csv(f"{R}/transfer_tfidf_lr.csv")
T = T[T.regime == "full"]
ind = T[T.source == T.target].groupby("seed").auc.mean().mean()
off = T[T.source != T.target].groupby("seed").auc.mean().mean()
close(ind, num("InAUC"), 0.0006, "in-domain AUC")
close(off, num("OffAUC"), 0.0006, "off-diagonal AUC")
cells = T[T.source != T.target].groupby(["source", "target"]).auc.mean()
close((cells < 0.5).sum(), num("NcellsBelowHalf"), 0, "cells with AUC<0.5")
perm = pd.read_csv(f"{R}/transfer_tfidf_lr.csv")
perm = perm[perm.regime == "permuted"]
check(abs(perm.auc.mean() - 0.5) < 0.03, f"permutation control near 0.5 ({perm.auc.mean():.3f})")

# decomposition shares, recomputed from raw rows
ind_f1 = T[T.source == T.target].set_index(["seed", "source"]).def_macro_f1
o = T[T.source != T.target].copy()
o["bb"] = [ind_f1[(s, t)] for s, t in zip(o.seed, o.target)]
rank = (o.bb - o.oracle_macro_f1).sum()
op = (o.oracle_macro_f1 - o.def_macro_f1).sum()
close(100 * rank / (rank + op), num("RankShareAll"), 0.51, "ranking share of gap (%)")

# operating points
O = pd.read_csv(f"{R}/summary_operating_points_cells.csv")
g = O[(O.model == "tfidf_lr") & ~O.in_domain]
check(g.em.mean() < g.default.mean(), "EM prior adaptation lowers off-diagonal macro-F1")
rec = (g.src_rate - g.default).mean() / (g.oracle - g.default).mean()
close(100 * rec, num("RecRank"), 0.51, "rate transfer recovery (%)")
check(rec > 0.5, "rate transfer recovers most of the operating-point loss")
check(num("ECEout") > 3 * num("ECEin"), "calibration degrades on target (ECE)")
check(num("SrcPriorErr") < num("EMpriorErr"), "source prior is a better guess than EM prior")

# models
sub = pd.read_csv(f"{R}/subset_roberta_vs_tfidf.csv")
sub = sub[sub.source.isin(VER) & sub.target.isin(VER) & (sub.source != sub.target)]
rob = sub[sub.model == "roberta_lr"].groupby("seed").auc.mean().mean()
lr = sub[sub.model == "tfidf_lr_sub"].groupby("seed").auc.mean().mean()
check(rob <= lr + 0.01, f"RoBERTa does not transfer better than TF-IDF ({rob:.3f} vs {lr:.3f})")
check(0 < num("OffAUCmin") - num("StyleOffAUC") < 0.06, "style transfer a few points below lexical models")

# repairs
F = pd.read_csv(f"{R}/fix_loco.csv")
m = F.groupby("method").auc.mean()
close(m["pooled"], num("LocoPooled"), 0.0006, "LOCO pooled AUC")
check(m["pooled"] > num("LocoMeanSingle") + 0.05, "pooling beats the average single source")
check(abs(m["pooled_mask"] - m["pooled"]) < 0.01 and abs(m["pooled_bal"] - m["pooled"]) < 0.01,
      "balancing and masking leave average AUC unchanged")
for k in ["inlp", "coral", "corpus_feat_drop"]:
    check(m[k] < m["pooled"], f"{k} lowers AUC vs pooled")
ex = json.load(open(f"{R}/summary_extra.json"))
for k in ["inlp", "coral", "corpus_feat_drop"]:
    check(ex["repairs_vs_pooled_auc"][k]["p_holm"] < 0.05, f"{k} decrease significant (Holm)")
pt = pd.read_csv(f"{R}/summary_fix_per_target.csv", index_col=0)
gain = pt.pooled_mask - pt.pooled
check(gain.idxmax() == "ISOT" and gain.drop("ISOT").max() < 0.015, "masking helps clearly only on ISOT")

# mechanism
Tp = pd.read_csv(f"{R}/mech_topic.csv")
for s, tol in [("ISOT", 0.01), ("WELFake", 0.01), ("LIAR", 0.01)]:
    a = Tp[(Tp.source == s) & (Tp.target == s)].groupby("variant").auc.mean()
    check(abs(a["topic_balanced"] - a["random"]) < tol, f"topic balancing leaves {s} in-domain AUC unchanged")
a = Tp[(Tp.source == "GossipCop") & (Tp.target == "GossipCop")].groupby("variant").auc.mean()
check(a["random"] - a["topic_balanced"] > 0.03, "GossipCop loses in-domain AUC under topic balancing")
Kt = pd.read_csv(f"{R}/mech_topk.csv")
for s, o2 in [("ISOT", "WELFake"), ("WELFake", "ISOT")]:
    a = Kt[(Kt.source == s) & (Kt.target == s)].groupby("K").auc.mean()
    b = Kt[(Kt.source == s) & (Kt.target == o2)].groupby("K").auc.mean()
    check(a[0] - a[3000] < 0.04, f"top-K barely moves {s} in-domain AUC ({a[0]:.3f}->{a[3000]:.3f})")
    check(b[0] - b[3000] > 0.05, f"top-K lowers {s}->{o2} transfer ({b[0]:.3f}->{b[3000]:.3f})")
check(abs(num("PADrho")) < 0.1, "no association between PAD and transfer")
if os.path.exists(f"{R}/mech_mask.csv"):
    X = pd.read_csv(f"{R}/mech_mask.csv")
    for s in ["ISOT", "WELFake"]:
        a = X[(X.source == s) & (X.target == s)].groupby("variant").auc.mean()
        check(a["none"] - a["all"] < 0.05, f"masking everything leaves {s} in-domain AUC high ({a['all']:.3f})")
if os.path.exists(f"{R}/mech_vocab.csv"):
    V = pd.read_csv(f"{R}/mech_vocab.csv")
    for s in ["ISOT", "WELFake"]:
        a = V[(V.source == s) & (V.target == s)].groupby("variant").auc.mean()
        check(a.max() - a.min() < 0.05, f"vocabulary ablation moves {s} in-domain AUC only slightly")

# Bengali
B = pd.read_csv(f"{R}/bangla_official.csv")
b = B[(B.version == "as_released") & (B.field == "title") & (B.model == "char_lr")].iloc[0]
check(0 < 66.28 - 100 * b.f1_clickbait < 2, "char-LR within two points of the published 66.28")
d = B[(B.version == "deduplicated") & (B.field == "title") & (B.model == "char_lr")].iloc[0]
check(d.f1_clickbait < b.f1_clickbait, "deduplication lowers Bengali F1")
L = pd.read_csv(f"{R}/bangla_loso.csv")
S = pd.read_csv(f"{R}/bangla_seeds.csv")
check(S[S.model == "char_lr"].auc.mean() - L.auc.mean() < 0.05, "LOSO close to random-split AUC")
Xl = pd.read_csv(f"{R}/bangla_xling.csv").groupby(["source", "target"]).auc.mean()
check(Xl[("ClickbaitEN", "BanglaBait-EN")] > 0.65, "English->Bengali translate-test well above chance")

# robustness of the decomposition and of rate transfer (independent recomputation where cheap)
RB = json.load(open(f"{R}/summary_robust.json"))
cb = RB["corpus_bootstrap"]
check(cb["rank_share"]["lo"] > 0.5, f"ranking is the majority even at the lower corpus-bootstrap bound ({cb['rank_share']['lo']:.2f})")
check(cb["rec_em"]["hi"] < 0, "EM recovery interval stays below zero")
check(cb["rec_rank"]["lo"] > 0.5, "rate-transfer recovery interval stays above one half")
d1 = RB["rank_share_drop_one_corpus"].values()
check(min(d1) > 0.6, "ranking share > 60% whichever corpus is dropped")
check(RB["indomain_slack"]["f1_mean"] < 0.02, "in-domain threshold slack is small")
check(RB["ba_decomp"]["rank_share"] > 0.7, "ranking dominates under balanced accuracy too")
V = pd.read_csv(f"{R}/robust_rate_variants.csv")
v = V[V.model == "tfidf_lr"]
check(v.oracle_cf.mean() < v.oracle.mean(), "cross-fitted oracle is below the in-sample oracle")
check(v["rank"].mean() >= v.oracle_cf.mean() - 0.005, "rate transfer is at least as good as the cross-fitted oracle")
check(abs(v.pi_s - v.pi_t).mean() < abs(v.pi_acc - v.pi_t).mean(), "source prior beats adjusted classify-and-count")
PM = pd.read_csv(f"{R}/robust_per_model.csv").set_index("model")
check((PM.rec_em < 0).all(), "EM makes the operating point worse for every detector")
check((PM.rec_src_rate > 0.5).all(), "rate transfer recovers most of the loss for every detector")
check(abs(PM.loc["tfidf_lr", "rec_src_rate"] - PM.loc["tfidf_lr", "rec_half_rate"]) < 0.05,
      "rate 0.5 recovers almost as much as rate transfer (stated in the text)")
PS = pd.read_csv(f"{R}/robust_prior_shift.csv").groupby("delta")[["default", "em", "rank", "acc", "true_rate"]].mean()
check((PS["rank"] > PS.default).all(), "rate transfer beats the default threshold at every prior shift")
check((PS.em < PS.default).all() and (PS.acc < PS.default).all(), "EM and ACC stay below default at every shift")
check(PS["rank"].idxmax() in (-0.1, 0.0, 0.1), "rate transfer peaks near zero shift")
check(PS.loc[0.3, "true_rate"] - PS.loc[0.3, "rank"] > 0.03, "rate transfer falls behind the true rate at +0.3")
dt = pd.read_csv(f"{R}/data_table.csv").set_index("corpus").loc[VER].positive_rate
check(abs(dt.min() - 0.23) < 0.006 and abs(dt.max() - 0.51) < 0.006, "veracity base rates span 0.23-0.51 (text)")
check(abs(num("RankShareAll") - 70) < 0.6, "highlight '70%' matches the ranking share")
if os.path.exists(f"{R}/transfer_roberta_ft.csv"):
    FT = pd.read_csv(f"{R}/transfer_roberta_ft.csv")
    FS = pd.read_csv(f"{R}/transfer_tfidf_lr_ftsub.csv")
    fo = FT[FT.source != FT.target].auc.mean()
    lo = FS[(FS.source != FS.target) & FS.seed.isin(FT.seed.unique())].auc.mean()
    check(fo <= lo + 0.02, f"fine-tuned RoBERTa does not transfer clearly better than paired TF-IDF ({fo:.3f} vs {lo:.3f})")
    if "roberta_ft" in PM.index:
        check(PM.loc["roberta_ft", "rec_em"] < 0, "EM worse for the fine-tuned model too")
        check(PM.loc["roberta_ft", "rec_src_rate"] > 0.5, "rate transfer recovers most for the fine-tuned model")
    Dc = pd.read_csv(f"{R}/summary_decomposition_cells.csv")
    x = Dc[Dc.model == "roberta_ft"]
    if len(x):
        check(x.ranking.sum() / (x.ranking + x.threshold + x.prior).sum() > 0.5, "fine-tuned gap mostly ranking")

# zero-shot LLM and published generalisation methods
if os.path.exists(f"{R}/llm_zeroshot_qwen3_0.6b.csv"):
    L = pd.read_csv(f"{R}/llm_zeroshot_qwen3_0.6b.csv")
    FS = pd.read_csv(f"{R}/transfer_tfidf_lr_ftsub.csv")
    tg = L.groupby("target").auc.mean() - FS[FS.source != FS.target].groupby("target").auc.mean()
    check((tg > 0).all(), "zero-shot LLM ranks every corpus better than the average transferred TF-IDF")
    check(L.auc.mean() < FS[FS.source == FS.target].auc.mean() - 0.1, "zero-shot LLM far below in-domain detector")
    check(L.def_ppr.mean() < 0.01, "LLM's own answer almost never says fake")
    check(L.half_macro_f1.mean() - L.def_macro_f1.mean() > 0.2, "flagging half rescues the LLM operating point")
    fbb = FS[FS.source == FS.target].set_index(["target", "seed"]).def_macro_f1
    x = L.set_index(["target", "seed"])
    rk = (fbb.loc[x.index] - x.oracle_macro_f1).sum(); op = (x.oracle_macro_f1 - x.def_macro_f1).sum()
    check(rk / (rk + op) < 0.5, "LLM loss is mostly operating point")
if os.path.exists(f"{R}/dg_loco.csv"):
    D = pd.read_csv(f"{R}/dg_loco.csv")
    m = D.groupby("model").auc.mean()
    check(all(m[k] <= m["erm_lr"] for k in ["eann", "mdfend", "endef", "erm_mlp"]), "no published DG method beats pooled LR")
    check(((D.em_macro_f1 - D.def_macro_f1).mean()) < 0, "EM lowers macro-F1 for the DG models too")

# data audit
P = pd.read_csv(f"{R}/label_probe.csv").set_index(["corpus", "probe"])
nyt = P.loc[("WELFake", "title ends '- The New York Times' (reliable)")]
check(nyt.raw_label_1 == nyt.n and nyt.n > 1000, "WELFake reliable signature sits entirely under raw label 1")

# manuscript hygiene
paper = os.environ.get("PAPER_DIR", os.path.join(C.ROOT, "paper"))
if os.path.isdir(paper):
    check(not os.path.exists(f"{paper}/pending.tex"), "no pending placeholders file")
    tex = "".join(open(f).read() for f in glob.glob(f"{paper}/sections/*.tex") + glob.glob(f"{paper}/tables/*.tex"))
    check("??" not in tex and "pending" not in tex.lower(), "no '??' or 'pending' in manuscript sources")
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", open(f"{paper}/numbers.tex").read()))
    used = {u for u in re.findall(r"\\([A-Z][A-Za-z]+)", tex)} - {"Delta"}
    check(used <= defined, f"all number macros defined (missing: {sorted(used - defined)})")

print(f"\n{n - len(fails)}/{n} checks passed")
sys.exit(1 if fails else 0)
