"""Write every number and table used in the manuscript from results/ (no hand-typed numbers).

Outputs (to $PAPER_DIR, default paper/ inside the repository):
  numbers.tex          \\newcommand macros for every number quoted in the text
  tables/*.tex         LaTeX tables
Run: python src/paper_numbers.py
"""
import os, json, re
import numpy as np, pandas as pd
import config as C

P = os.environ.get("PAPER_DIR", os.path.join(C.ROOT, "paper"))
os.makedirs(f"{P}/tables", exist_ok=True)
R = C.RESULTS
VER = ["ISOT", "WELFake", "PolitiFact", "GossipCop", "LIAR", "Twitter15", "Twitter16"]
MACROS = {}


def fmt_int(x):
    return f"{int(x):,}".replace(",", "{,}")


def f3(x):
    return f"{x:.3f}"


def f2(x):
    return f"{x:.2f}"


def pct(x):
    v = f"{100 * x:.0f}"
    return (f"$-${v[1:]}" if v.startswith("-") else v) + "\\%"


def cell(v):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def mac(name, val):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    MACROS[name] = val


def write_table(name, body):
    open(f"{P}/tables/{name}.tex", "w").write(body)


# ------------------------------------------------------------------ data
def data_part():
    dt = pd.read_csv(f"{R}/data_table.csv").set_index("corpus")
    rep = json.load(open(f"{R}/dedup_report.json"))
    ov = pd.read_csv(f"{R}/cross_corpus_overlap.csv")
    probe = pd.read_csv(f"{R}/label_probe.csv")
    order = VER + ["PHEME", "ClickbaitEN", "BanglaBait"]
    unit = {"ISOT": "article", "WELFake": "article", "PolitiFact": "headline", "GossipCop": "headline",
            "LIAR": "claim", "Twitter15": "tweet", "Twitter16": "tweet", "PHEME": "tweet",
            "ClickbaitEN": "headline", "BanglaBait": "headline"}
    cons = {"PHEME": "rumour", "ClickbaitEN": "clickbait", "BanglaBait": "clickbait"}
    lines = []
    for c in order:
        r, d = dt.loc[c], rep[c]
        removed = d["raw"] - d["final"]
        conflicts = d["exact_conflict_rows"] + d["near_conflict_rows"]
        lines.append(f"{c} & {unit[c]} & {cons.get(c, 'veracity')} & {'bn' if c == 'BanglaBait' else 'en'} & "
                     f"{fmt_int(d['raw'])} & {fmt_int(d['exact_dups_removed'])} & {fmt_int(d['near_dups_removed'])} & "
                     f"{fmt_int(conflicts)} & {fmt_int(r.n)} & {r.positive_rate:.2f} \\\\")
    write_table("corpora", r"""\begin{table}[t]
\centering
\caption{Corpora after harmonisation and deduplication. ``Exact'' and ``near'' count removed duplicates; ``conflict'' counts items removed because duplicates disagreed on the label; ``pos.'' is the share labelled fake, false, rumour or clickbait. For LIAR and Twitter15/16, ``raw'' counts the items in the binary view (LIAR: pants-fire/false vs.\ mostly-true/true; Twitter: false vs.\ true).}\label{tab:corpora}
\small
\begin{tabular}{@{}llllrrrrrr@{}}
\toprule
Corpus & Unit & Construct & Lang. & Raw & Exact & Near & Conflict & Final & Pos. \\
\midrule
""" + "\n".join(lines[:7]) + "\n\\midrule\n" + "\n".join(lines[7:]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    tot = int(dt.loc[order].n.sum())
    mac("Ntotal", fmt_int(tot))
    mac("BBraw", fmt_int(rep["BanglaBait"]["raw"]))
    mac("BBfinal", fmt_int(rep["BanglaBait"]["final"]))
    mac("BBexact", fmt_int(rep["BanglaBait"]["exact_dups_removed"]))
    mac("GCexact", fmt_int(rep["GossipCop"]["exact_dups_removed"]))
    mac("GCconflict", fmt_int(rep["GossipCop"]["exact_conflict_rows"]))
    mac("WELnear", fmt_int(rep["WELFake"]["near_dups_removed"]))

    def ovl(a, b):
        r = ov[((ov.corpus_a == a) & (ov.corpus_b == b)) | ((ov.corpus_a == b) & (ov.corpus_b == a))]
        return int(r.items_in_a_dup_in_b.iloc[0]) if len(r) else 0
    mac("DupIW", fmt_int(ovl("ISOT", "WELFake")))
    mac("DupPT", fmt_int(ovl("PHEME", "Twitter15")))
    mac("DupPTs", fmt_int(ovl("PHEME", "Twitter16")))
    t15 = {l.split(":")[1].strip() for l in open(C.TW15_LAB) if ":" in l}
    t16 = {l.split(":")[1].strip() for l in open(C.TW16_LAB) if ":" in l}
    mac("TwShared", fmt_int(len(t15 & t16)))
    pr = probe.set_index(["corpus", "probe"])
    k_reu = "text contains '(Reuters)' (reliable)"
    k_fiv = "text contains 'Featured image via' (fake-site template)"
    k_nyt = "title ends '- The New York Times' (reliable)"
    mac("ISOTreuters", fmt_int(pr.loc[("ISOT", k_reu), "n"]))
    mac("ISOTreutersReal", fmt_int(pr.loc[("ISOT", k_reu), "raw_label_0"]))
    mac("ISOTfiv", fmt_int(pr.loc[("ISOT", k_fiv), "n"]))
    mac("WELnyt", fmt_int(pr.loc[("WELFake", k_nyt), "n"]))
    mac("WELfiv", fmt_int(pr.loc[("WELFake", k_fiv), "n"]))
    rows = []
    for p in probe.probe.unique():
        a = probe[(probe.corpus == "ISOT") & (probe.probe == p)].iloc[0]
        b = probe[(probe.corpus == "WELFake") & (probe.probe == p)].iloc[0]
        rows.append(f"{p.split(' (')[0]} & {a.expected_class} & {fmt_int(a.raw_label_0)} / {fmt_int(a.raw_label_1)} & "
                    f"{fmt_int(b.raw_label_0)} / {fmt_int(b.raw_label_1)} \\\\")
    write_table("probe", r"""\begin{table}[t]
\centering
\caption{Label-direction probes on the raw files: number of items matching each source signature under raw label 0 / raw label 1. A reliable-outlet signature must sit under ``real'', a fake-site signature under ``fake''. ISOT codes $1=$ fake; the WELFake file codes $1=$ real.}\label{tab:probe}
\small
\begin{tabular}{@{}llrr@{}}
\toprule
Signature & Expected class & ISOT (0 / 1) & WELFake (0 / 1) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


MODEL_NAME = {"tfidf_lr": r"\textsc{tfidf-lr}", "tfidf_svm": r"\textsc{tfidf-svm}", "nbsvm": r"\textsc{nb-svm}",
              "glove_lr": r"\textsc{glove-lr}", "style_gb": r"\textsc{style-gb}", "roberta_lr": r"\textsc{roberta-lr}"}
PAIRS = [("ISOT", "WELFake"), ("WELFake", "ISOT"), ("PolitiFact", "GossipCop"), ("GossipCop", "PolitiFact"),
         ("Twitter15", "Twitter16"), ("Twitter16", "Twitter15"), ("LIAR", "PolitiFact"), ("PolitiFact", "LIAR")]


def pm_str(s):
    """'0.914 ± 0.006' -> '0.914\\,{\\scriptsize$\\pm$0.006}'"""
    a, b = [x.strip() for x in s.split("±")]
    return f"{a}\\,{{\\scriptsize$\\pm${b}}}"


def transfer_part():
    H = pd.read_csv(f"{R}/summary_headline.csv")
    rows = []
    for m in ["tfidf_lr", "tfidf_svm", "nbsvm", "glove_lr", "style_gb", "roberta_lr"]:
        for rg in ["full", "matched", "permuted"]:
            h = H[(H.model == m) & (H.regime == rg)]
            if not len(h):
                continue
            h = h.iloc[0]
            rows.append(f"{MODEL_NAME[m]} & {rg} & {pm_str(h.in_auc)} & {pm_str(h.off_auc)} & {pm_str(h.pair_auc)} & "
                        f"{pm_str(h.in_f1)} & {pm_str(h.off_f1)} & {pm_str(h.off_f1_oracle)} \\\\")
        rows.append("\\addlinespace")
    write_table("headline", r"""\begin{table}[t]
\centering
\caption{In-domain versus transfer performance on the seven veracity corpora (mean $\pm$ s.d.\ over five seeds of the per-seed mean over cells). ``Pairs'' averages the eight directed same-task cells. Off-diagonal macro-F1 is given at the default threshold and at the oracle threshold. \textsc{roberta-lr} is evaluated on the 1{,}500-item subsets (Section~\ref{sec:models}).}\label{tab:headline}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llcccccc@{}}
\toprule
 & & \multicolumn{3}{c}{ROC-AUC} & \multicolumn{3}{c}{macro-F1} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Model & Regime & In-domain & Off-diag. & Pairs & In-domain & Off-diag. & Off-diag.\ oracle \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    g = lambda m, rg: H[(H.model == m) & (H.regime == rg)].iloc[0]
    lr = g("tfidf_lr", "full")
    num = lambda s: s.split("±")[0].strip()
    mac("InAUC", num(lr.in_auc)); mac("OffAUC", num(lr.off_auc)); mac("PairAUC", num(lr.pair_auc))
    mac("InFone", num(lr.in_f1)); mac("OffFone", num(lr.off_f1)); mac("OffFoneOracle", num(lr.off_f1_oracle))
    mac("CellsBelowHalf", pct(lr.off_cells_auc_below_05))
    mac("NcellsBelowHalf", str(int(round(lr.off_cells_auc_below_05 * 42))))
    mt = g("tfidf_lr", "matched")
    mac("MatchedInAUC", num(mt.in_auc)); mac("MatchedOffAUC", num(mt.off_auc))
    js = json.load(open(f"{R}/summary.json"))
    pa = js["permutation_auc"]
    mac("PermAUC", f3(pa["mean"])); mac("PermAUCsd", f3(pa["sd"])); mac("PermInAUC", f3(pa["in_domain_mean"]))
    # pairs table
    S = pd.read_csv(f"{R}/summary_pairs.csv")
    rows = []
    for _, r in S.iterrows():
        lo, hi = [float(x) for x in re.findall(r"\d\.\d+", r.auc_ci)][:2]
        rows.append(f"{r.source}$\\to${r.target} & {r.in_auc:.3f} & {r.auc:.3f} [{lo:.2f}, {hi:.2f}] & "
                    f"{r.prior:.2f} & {r.ppr:.2f} & {r.f1_def:.3f} & {r.f1_pm:.3f} & {r.f1_oracle:.3f} & {r.ece:.2f} \\\\")
        key = (r.source + r.target).replace("Twitter", "Tw").replace("PolitiFact", "PF").replace("GossipCop", "GC")
        key = re.sub(r"[^A-Za-z]", "", key.replace("15", "A").replace("16", "B"))
        mac(f"pair{key}AUC", f"{r.auc:.2f}")
        mac(f"pair{key}CI", f"[{lo:.2f}, {hi:.2f}]")
        mac(f"pair{key}Fdef", f"{r.f1_def:.3f}"); mac(f"pair{key}Fpm", f"{r.f1_pm:.3f}")
        mac(f"pair{key}Forc", f"{r.f1_oracle:.3f}"); mac(f"pair{key}PPR", f"{r.ppr:.2f}")
        mac(f"pair{key}Prior", f"{r.prior:.2f}"); mac(f"pair{key}In", f"{r.in_auc:.3f}")
    write_table("pairs", r"""\begin{table}[t]
\centering
\caption{Same-task pairs (\textsc{tfidf-lr}, five seeds). AUC with pooled seed-bootstrap 95\% CI; ``In'' is the source's in-domain AUC; prior is the target fake rate and PPR the predicted-positive rate at $t=0.5$; macro-F1 at the default, prior-matched and oracle thresholds; ECE of the source-calibrated scores on the target.}\label{tab:pairs}
\small
\begin{tabular}{@{}lcccccccc@{}}
\toprule
Pair & In & AUC [95\% CI] & Prior & PPR & F1$_{0.5}$ & F1$_{\pi}$ & F1$^\star$ & ECE \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    # decomposition
    Dg = pd.read_csv(f"{R}/summary_decomposition.csv")
    d = Dg[(Dg.model == "tfidf_lr") & (Dg.regime == "full")].set_index("same_task")
    for flag, tag in [(True, "Pair"), (False, "Other")]:
        x = d.loc[flag]
        mac(f"Gap{tag}", f2(x.gap_target)); mac(f"Rank{tag}", f2(x.ranking)); mac(f"Thr{tag}", f2(x.threshold))
        mac(f"Prior{tag}", f2(x.prior)); mac(f"RankShare{tag}", pct(x.ranking_share_of_target_gap))
        mac(f"OpShare{tag}", pct(x.op_share_of_target_gap)); mac(f"PromiseGap{tag}", f2(x.gap))
    cells = pd.read_csv(f"{R}/summary_decomposition_cells.csv")
    c = cells[(cells.model == "tfidf_lr") & (cells.regime == "full")]
    tot = c[["ranking", "threshold", "prior"]].sum(1)
    mac("RankShareAll", pct(c.ranking.sum() / tot.sum()))
    mac("OpShareAll", pct((c.threshold.sum() + c.prior.sum()) / tot.sum()))
    rows = []
    for m in ["tfidf_lr", "tfidf_svm", "nbsvm", "glove_lr", "style_gb", "roberta_lr"]:
        for flag in [True, False]:
            x = Dg[(Dg.model == m) & (Dg.regime == "full") & (Dg.same_task == flag)]
            if not len(x):
                continue
            x = x.iloc[0]
            rows.append(f"{MODEL_NAME[m]} & {'pairs' if flag else 'other cells'} & {x.gap_target:.3f} & {x.ranking:.3f} & "
                        f"{x.threshold:.3f} & {x.prior:.3f} & {pct(x.ranking_share_of_target_gap)} \\\\")
    write_table("decomp", r"""\begin{table}[t]
\centering
\caption{Transfer gap decomposition (Eq.~\ref{eq:decomp}), macro-F1 points, mean over cells and seeds. The gap is relative to the target's own in-domain score.}\label{tab:decomp}
\small
\begin{tabular}{@{}llrrrrr@{}}
\toprule
Model & Cells & Gap & Ranking & Threshold & Prior/cal. & Ranking share \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def operating_part():
    O = pd.read_csv(f"{R}/summary_operating_points.csv", header=[0, 1], index_col=0)
    ex = json.load(open(f"{R}/summary_extra.json"))
    rec = ex["op_recovery_tfidf_lr"]
    mac("RecEM", pct(rec["em"])); mac("RecRank", pct(rec["src_rate"])); mac("RecHalf", pct(rec["half_rate"]))
    mac("RecGMM", pct(rec["gmm"])); mac("RecPrior", pct(rec["true_rate"]))
    mac("EMgainPriorRho", f2(ex["em_gain_vs_prior_gap_spearman"]))
    mac("EMpriorErr", f2(ex["em_prior_abs_err"])); mac("SrcPriorErr", f2(ex["src_prior_abs_err"]))
    mac("DefPPRErr", f2(ex["default_ppr_abs_err"]))
    mac("ECEin", f3(ex["indomain_ece_mean"])); mac("ECEout", f3(ex["target_ece_mean"]))
    mac("ECEem", f3(ex["target_ece_em_mean"]))
    lr = O.loc["tfidf_lr"]
    for k, n in [("default", "Def"), ("em", "EM"), ("src_rate", "Rank"), ("half_rate", "Half"), ("gmm", "GMM"),
                 ("true_rate", "Prior"), ("oracle", "Oracle")]:
        mac(f"OP{n}", f3(lr[(k, "mean")]))
    rows = []
    for m in [x for x in ["tfidf_lr", "tfidf_svm", "nbsvm", "glove_lr", "style_gb", "roberta_lr"] if x in O.index]:
        r = O.loc[m]
        cells = " & ".join(f"{r[(k, 'mean')]:.3f}" for k in
                           ["default", "em", "src_rate", "half_rate", "gmm", "true_rate", "oracle"])
        rows.append(f"{MODEL_NAME[m]} & {cells} \\\\")
    pw = ex["op_wilcoxon_holm_tfidf_lr"]
    write_table("oppoints", r"""\begin{table}[t]
\centering
\caption{Off-diagonal macro-F1 under different operating points (mean over the 42 cells and five seeds). Label-free: default $t=0.5$, EM prior adaptation, rank transfer (predicted-positive rate set to the source prior), predicted-positive rate 0.5, and a 2-component GMM on target logits. With one number known: rate set to the true target prior. Upper bound: oracle threshold. For \textsc{tfidf-lr}, Holm-corrected Wilcoxon $p$ versus default: EM """ + f"{pw['em']:.1e}" + r""", rank transfer """ + f"{pw['src_rate']:.1e}" + r""", rate 0.5 """ + f"{pw['half_rate']:.1e}" + r""", GMM """ + f"{pw['gmm']:.1e}" + r""".}\label{tab:oppoints}
\small
\begin{tabular}{@{}lccccccc@{}}
\toprule
 & \multicolumn{5}{c}{label-free} & prior known & bound \\
\cmidrule(lr){2-6}\cmidrule(lr){7-7}\cmidrule(lr){8-8}
Model & $t=0.5$ & EM & rank transfer & rate 0.5 & GMM & rate $=\pi_B$ & oracle \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def appendix_matrices():
    M = pd.read_csv(f"{R}/summary_matrix.csv")
    out = []
    for m in [x for x in ["tfidf_lr", "tfidf_svm", "nbsvm", "glove_lr", "style_gb", "roberta_lr"] if x in set(M.model)]:
        for key, lab in [("auc", "ROC-AUC"), ("def_macro_f1", "macro-F1 at $t=0.5$")]:
            g = M[M.model == m].pivot(index="source", columns="target", values=f"{key}_mean").reindex(index=VER, columns=VER)
            s = M[M.model == m].pivot(index="source", columns="target", values=f"{key}_sd").reindex(index=VER, columns=VER)
            body = []
            for a in VER:
                cells = []
                for b in VER:
                    v, e = g.loc[a, b], s.loc[a, b]
                    txt = "--" if np.isnan(v) else f"{v:.2f}\\,{{\\tiny$\\pm${e:.2f}}}"
                    cells.append(f"\\textbf{{{txt}}}" if a == b else txt)
                body.append(f"{a} & " + " & ".join(cells) + " \\\\")
            out.append(r"""\begin{table}[h]
\centering
\caption{""" + f"{MODEL_NAME[m]}, {lab}" + r""" (rows: training corpus; columns: test corpus; mean $\pm$ s.d.\ over five seeds).}
\scriptsize
\begin{tabular}{@{}l""" + "c" * len(VER) + r"""@{}}
\toprule
 & """ + " & ".join(v.replace("Twitter", "Tw") for v in VER) + r""" \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    write_table("appendix_matrices", "\n".join(out))


def bangla_part():
    B = pd.read_csv(f"{R}/bangla_official.csv")
    S = pd.read_csv(f"{R}/bangla_seeds.csv")
    L = pd.read_csv(f"{R}/bangla_loso.csv")
    X = pd.read_csv(f"{R}/bangla_xling.csv")
    G = pd.read_csv(f"{R}/mech_group.csv")
    b = lambda v, f, m: B[(B.version == v) & (B.field == f) & (B.model == m)].iloc[0]
    rel, ded = b("as_released", "title", "char_lr"), b("deduplicated", "title", "char_lr")
    mac("BBrelFone", f"{100 * rel.f1_clickbait:.2f}"); mac("BBdedFone", f"{100 * ded.f1_clickbait:.2f}")
    mac("BBrelAcc", f"{100 * rel.accuracy:.2f}"); mac("BBrelAUC", f3(rel.auc)); mac("BBdedAUC", f3(ded.auc))
    sv = b("as_released", "title", "char_svm")
    mac("BBrelSVMFone", f"{100 * sv.f1_clickbait:.2f}")
    ts = b("as_released", "title+content", "char_lr")
    mac("BBrelTCFone", f"{100 * ts.f1_clickbait:.2f}")
    s = S[S.model == "char_lr"]
    mac("BBseedAUC", f3(s.auc.mean())); mac("BBseedAUCsd", f3(s.auc.std()))
    mac("BBseedFone", f3(s.f1_clickbait.mean()))
    mac("BBlosoAUC", f3(L.auc.mean())); mac("BBlosoMin", f3(L.auc.min())); mac("BBlosoMax", f3(L.auc.max()))
    mac("BBlosoN", str(len(L)))
    mac("BBgroupAUC", f3(G[G.corpus == "BanglaBait"].auc.mean()))
    xg = X.groupby(["source", "target"]).auc.mean()
    mac("XlBN", f3(xg[("BanglaBait-BN", "BanglaBait-BN")])); mac("XlBBEN", f3(xg[("BanglaBait-EN", "BanglaBait-EN")]))
    mac("XlENtoBB", f3(xg[("ClickbaitEN", "BanglaBait-EN")])); mac("XlBBtoEN", f3(xg[("BanglaBait-EN", "ClickbaitEN")]))
    mac("XlEN", f3(xg[("ClickbaitEN", "ClickbaitEN")]))
    mac("XlTransCost", f2(xg[("BanglaBait-BN", "BanglaBait-BN")] - xg[("BanglaBait-EN", "BanglaBait-EN")]))
    big = L[(L.n >= 100) & ~L.domain.str.startswith("other")]
    mac("BBrateMin", pct(big.clickbait_rate.min())); mac("BBrateMax", pct(big.clickbait_rate.max()))
    mac("BBnPub", str(len(big)))
    rows = []
    for v, vl in [("as_released", "as released"), ("deduplicated", "deduplicated")]:
        for f in ["title", "title+content"]:
            for m, ml in [("char_lr", "LR, char 3--5"), ("char_svm", "SVM, char 3--5"), ("word_lr", "LR, word 1--2")]:
                r = b(v, f, m)
                rows.append(f"{vl} & {f} & {ml} & {100 * r.f1_clickbait:.2f} & {100 * r.precision:.2f} & "
                            f"{100 * r.recall:.2f} & {100 * r.accuracy:.2f} & {r.auc:.3f} \\\\")
        rows.append("\\addlinespace")
    write_table("bangla", r"""\begin{table}[t]
\centering
\caption{BanglaBait, official split. Clickbait-class F1, precision and recall and accuracy (\%), as reported by \citet{mahtab2023banglabait}, plus ROC-AUC. Reference values from that paper (as released): LR char 3--5-grams F1 66.28, BanglaBERT 71.72, SS-GAN BanglaBERT 75.13.}\label{tab:bangla}
\small
\begin{tabular}{@{}lllrrrrr@{}}
\toprule
Test set & Input & Model & F1 & P & R & Acc. & AUC \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def models_part():
    H = pd.read_csv(f"{R}/summary_headline.csv")
    H = H[H.regime == "full"].set_index("model")
    num = lambda s: float(s.split("±")[0])
    offs = {m: num(H.loc[m, "off_auc"]) for m in H.index}
    ins = {m: num(H.loc[m, "in_auc"]) for m in H.index}
    nonrob = {m: v for m, v in offs.items() if m not in ("roberta_lr", "style_gb")}
    mac("OffAUCmin", f3(min(nonrob.values()))); mac("OffAUCmax", f3(max(nonrob.values())))
    mac("InAUCmin", f3(min(v for m, v in ins.items() if m not in ("roberta_lr", "style_gb"))))
    mac("InAUCmax", f3(max(v for m, v in ins.items() if m not in ("roberta_lr", "style_gb"))))
    if "style_gb" in H.index:
        mac("StyleInAUC", f3(ins["style_gb"])); mac("StyleOffAUC", f3(offs["style_gb"]))
    if "glove_lr" in H.index:
        mac("GloveInAUC", f3(ins["glove_lr"])); mac("GloveOffAUC", f3(offs["glove_lr"]))
    sub = pd.read_csv(f"{R}/subset_roberta_vs_tfidf.csv")
    sub = sub[sub.source.isin(VER) & sub.target.isin(VER)]
    for m, tag in [("roberta_lr", "Rob"), ("tfidf_lr_sub", "SubLR")]:
        g = sub[sub.model == m]
        i = g[g.source == g.target].groupby("seed").auc.mean()
        o = g[g.source != g.target].groupby("seed").auc.mean()
        mac(f"{tag}InAUC", f3(i.mean())); mac(f"{tag}OffAUC", f3(o.mean()))
        mac(f"{tag}OffAUCsd", f3(o.std()))
        mac(f"{tag}OffForc", f3(g[g.source != g.target].oracle_macro_f1.mean()))
        mac(f"{tag}OffFdef", f3(g[g.source != g.target].def_macro_f1.mean()))
    # paired comparison on the subset: per (cell, seed) AUC difference
    a = sub[sub.model == "roberta_lr"].set_index(["source", "target", "seed"]).auc
    b = sub[sub.model == "tfidf_lr_sub"].set_index(["source", "target", "seed"]).auc
    d = (a - b).dropna()
    off = d[[s != t for s, t, _ in d.index]]
    from scipy.stats import wilcoxon
    mac("RobMinusLRoff", f3(off.mean())); mac("RobMinusLRp", f"{wilcoxon(off).pvalue:.2f}")
    mac("RobBetterCells", str(int((off.groupby(level=[0, 1]).mean() > 0).sum())))
    js = json.load(open(f"{R}/summary.json"))
    if "mcnemar" in js:
        mc = js["mcnemar"]
        mac("McFracSig", ", ".join(f"{MODEL_NAME[m]} {pct(v['frac_sig_holm'])}" for m, v in mc.items() if m in MODEL_NAME))


def mechanism_part():
    M = pd.read_csv(f"{R}/summary_matrix.csv")
    lr = M[M.model == "tfidf_lr"]
    st = M[M.model == "style_gb"]
    L = pd.read_csv(f"{R}/mech_lc.csv")
    G = pd.read_csv(f"{R}/mech_group.csv").groupby("corpus").auc.mean()
    lc64 = L[L.n == 64].groupby("corpus").auc.mean()
    lc256 = L[L.n == 256].groupby("corpus").auc.mean()
    rows, ratio = [], []
    for c in VER:
        ind = lr[(lr.source == c) & (lr.target == c)].auc_mean.iloc[0]
        sty = st[(st.source == c) & (st.target == c)].auc_mean.iloc[0] if len(st) else np.nan
        out = lr[(lr.source == c) & (lr.target != c)].auc_mean.mean()
        inn = lr[(lr.target == c) & (lr.source != c)].auc_mean.mean()
        g = G.get(c, np.nan)
        rows.append(f"{c} & {ind:.3f} & {sty:.3f} & {cell(lc64.get(c, np.nan))} & {cell(lc256.get(c, np.nan))} & "
                    f"{'--' if np.isnan(g) else f'{g:.3f}'} & {out:.3f} & {inn:.3f} \\\\")
        ratio.append((sty - 0.5) / (ind - 0.5))
    write_table("shortcuts", r"""\begin{table}[t]
\centering
\caption{Shortcut indicators (ROC-AUC, mean of five seeds). In-domain AUC of \textsc{tfidf-lr} with the full training split, of the surface-statistics \textsc{style-gb}, and of \textsc{tfidf-lr} trained on only 64 or 256 items; AUC of a group-identity baseline (publisher domain for PolitiFact and GossipCop, speaker for LIAR); and the mean off-diagonal AUC with the corpus as source and as target.}\label{tab:shortcuts}
\small
\begin{tabular}{@{}lccccccc@{}}
\toprule
 & \multicolumn{5}{c}{in-domain} & \multicolumn{2}{c}{transfer} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-8}
Corpus & full & style only & $n=64$ & $n=256$ & group only & as source & as target \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    mac("StyleShare", pct(np.nanmean(ratio)))
    sty_in = st[st.source == st.target].set_index("source").auc_mean
    mac("StyleISOT", f3(sty_in["ISOT"])); mac("StyleWEL", f3(sty_in["WELFake"]))
    mac("LCsixfourISOT", f3(lc64["ISOT"])); mac("LCsixfourWEL", f3(lc64["WELFake"]))
    mac("LCsixfourGC", f3(lc64["GossipCop"])); mac("LCsixfourLIAR", f3(lc64["LIAR"]))
    for c, k in [("PolitiFact", "PF"), ("GossipCop", "GC"), ("LIAR", "LIAR"), ("PHEME", "PHEME")]:
        mac(f"Group{k}", f3(G[c]))
    P = pd.read_csv(f"{R}/mech_pad.csv").groupby(["a", "b"]).pad.mean()
    mac("PADmin", f2(P.min())); mac("PADmax", f2(P.max()))
    mac("PADiw", f2(P.get(("ISOT", "WELFake"), np.nan)))
    ex = json.load(open(f"{R}/summary_extra.json"))
    if "pad_vs_auc" in ex:
        mac("PADrho", f2(ex["pad_vs_auc"]["spearman"]))
        lo, hi = ex["pad_vs_auc"]["ci"]
        mac("PADrhoCI", f"[{lo:.2f}, {hi:.2f}]")

    # topic balance
    T = pd.read_csv(f"{R}/mech_topic.csv")
    for s, k in [("ISOT", "ISOT"), ("WELFake", "WEL"), ("GossipCop", "GC"), ("LIAR", "LIAR")]:
        g = T[T.source == s]
        mac(f"Topic{k}RandIn", f3(g[(g.target == s) & (g.variant == "random")].auc.mean()))
        mac(f"Topic{k}BalIn", f3(g[(g.target == s) & (g.variant == "topic_balanced")].auc.mean()))
        mac(f"Topic{k}RandOut", f3(g[(g.target != s) & (g.variant == "random")].auc.mean()))
        mac(f"Topic{k}BalOut", f3(g[(g.target != s) & (g.variant == "topic_balanced")].auc.mean()))
        mac(f"Topic{k}Purity", f2(g.mean_topic_purity.mean())); mac(f"Topic{k}Dropped", pct(g.frac_dropped.mean()))


def topk_part():
    if not os.path.exists(f"{R}/mech_topk.csv"):
        print('pending', 'mech_topk.csv'); return
    # top-K
    K = pd.read_csv(f"{R}/mech_topk.csv")
    for s, k in [("ISOT", "ISOT"), ("WELFake", "WEL")]:
        g = K[(K.source == s)]
        ii = g[g.target == s].groupby("K").auc.mean()
        oo = g[(g.target == ("WELFake" if s == "ISOT" else "ISOT"))].groupby("K").auc.mean()
        mac(f"TopK{k}In", f3(ii[0])); mac(f"TopK{k}InThousand", f3(ii[1000])); mac(f"TopK{k}InMax", f3(ii[3000]))
        mac(f"TopK{k}Pair", f3(oo[0])); mac(f"TopK{k}PairThousand", f3(oo[1000])); mac(f"TopK{k}PairMax", f3(oo[3000]))


def mask_part():
    if not os.path.exists(f"{R}/mech_mask.csv"):
        print('pending', 'mech_mask.csv'); return
    # masking
    X = pd.read_csv(f"{R}/mech_mask.csv")
    for s, k in [("ISOT", "ISOT"), ("WELFake", "WEL")]:
        o = "WELFake" if s == "ISOT" else "ISOT"
        for v in ["none", "source", "time", "ents", "all"]:
            vi = X[(X.source == s) & (X.target == s) & (X.variant == v)].auc.mean()
            vo = X[(X.source == s) & (X.target == o) & (X.variant == v)].auc.mean()
            vn = re.sub(r"[^A-Za-z]", "", v.title())
            mac(f"Mask{k}{vn}In", f3(vi)); mac(f"Mask{k}{vn}Pair", f3(vo))
    rows = []
    for s in ["ISOT", "WELFake"]:
        for v, vl in [("none", "none"), ("source", "source tags, datelines, URLs, handles"), ("time", "dates, weekdays, numbers"),
                      ("ents", "named entities"), ("all", "all of the above")]:
            g = X[(X.source == s) & (X.variant == v)].groupby("target").auc.mean()
            rows.append(f"{s} & {vl} & " + " & ".join(f"{g.get(t, np.nan):.3f}" for t in
                                                       ["ISOT", "WELFake", "PolitiFact", "GossipCop", "LIAR"]) + " \\\\")
        rows.append("\\addlinespace")
    write_table("mask", r"""\begin{table}[t]
\centering
\caption{Masking ablation (\textsc{tfidf-lr}, ROC-AUC, three seeds, first 3{,}000 characters). Rows: training corpus and what is masked in every corpus; columns: test corpus (the training corpus's own column is in-domain).}\label{tab:mask}
\small
\begin{tabular}{@{}llccccc@{}}
\toprule
Train & Masked & ISOT & WELFake & PolitiFact & GossipCop & LIAR \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def vocab_part():
    if not os.path.exists(f"{R}/mech_vocab.csv"):
        print('pending', 'mech_vocab.csv'); return
    # vocab
    V = pd.read_csv(f"{R}/mech_vocab.csv")
    rows = []
    for s in ["ISOT", "WELFake"]:
        o = "WELFake" if s == "ISOT" else "ISOT"
        for v in V.variant.unique():
            g = V[(V.source == s) & (V.variant == v)]
            rows.append(f"{s} & {v.replace('_', ' ').replace('char wb', 'char')} & {g[g.target == s].auc.mean():.3f} & "
                        f"{g[g.target == o].auc.mean():.3f} & {g[~g.target.isin([s])].auc.mean():.3f} \\\\")
    write_table("vocab", r"""\begin{table}[h]
\centering
\caption{Vocabulary and n-gram ablation (\textsc{tfidf-lr}, ROC-AUC, three seeds).}\label{tab:vocab}
\small
\begin{tabular}{@{}llccc@{}}
\toprule
Train & Features & In-domain & Pair & All targets \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def repair_part():
    F = pd.read_csv(f"{R}/fix_loco.csv")
    ex = json.load(open(f"{R}/summary_extra.json"))
    tests = ex["repairs_vs_pooled_auc"]
    per = pd.read_csv(f"{R}/summary_fix_per_target.csv", index_col=0)
    names = {"pooled": "pooled (6 corpora)", "pooled_bal": "+ (corpus, class) balancing",
             "pooled_mask": "+ balancing + masking", "corpus_feat_drop": "+ balancing + corpus-feature deletion",
             "svd": "SVD-300 + balancing", "inlp": "SVD-300 + INLP", "coral": "SVD-300 + CORAL"}
    rows = []
    g = F.groupby("method")[["auc", "ap", "def_macro_f1", "oracle_macro_f1"]].mean()
    sd = F.groupby(["method", "seed"]).auc.mean().groupby("method").std()
    for m in names:
        if m not in g.index:
            continue
        t = tests.get(m)
        pv = "--" if t is None else (f"{t['delta']:+.3f} ({t['p_holm']:.3f})" if t['p_holm'] >= 0.001 else f"{t['delta']:+.3f} ($<$0.001)")
        rows.append(f"{names[m]} & {g.loc[m, 'auc']:.3f}\\,{{\\scriptsize$\\pm${sd[m]:.3f}}} & {g.loc[m, 'ap']:.3f} & "
                    f"{g.loc[m, 'def_macro_f1']:.3f} & {g.loc[m, 'oracle_macro_f1']:.3f} & {pv} \\\\")
    ms, bs, ce = per.mean_single_source.mean(), per.best_single_source.mean(), per.ceiling.mean()
    write_table("repair", r"""\begin{table}[t]
\centering
\caption{Leave-one-corpus-out repairs (\textsc{tfidf-lr} features; mean over the seven held-out veracity corpora and five seeds; s.d.\ across seeds). $\Delta$AUC versus pooled training with Holm-corrected Wilcoxon $p$ over the 35 (target, seed) pairs. For reference, the mean single-source transfer AUC to the same targets is """ + f"{ms:.3f}" + r""", the best single source (chosen with target labels) """ + f"{bs:.3f}" + r""", and the targets' own in-domain AUC """ + f"{ce:.3f}" + r""".}\label{tab:repair}
\small
\begin{tabular}{@{}lccccc@{}}
\toprule
Training & AUC & AP & F1$_{0.5}$ & F1$^\star$ & $\Delta$AUC ($p$) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    mac("LocoPooled", f3(g.loc["pooled", "auc"])); mac("LocoMeanSingle", f3(ms)); mac("LocoBestSingle", f3(bs))
    mac("LocoCeiling", f3(ce))
    for m, k in [("pooled_bal", "Bal"), ("pooled_mask", "Mask"), ("corpus_feat_drop", "Drop"), ("svd", "SVD"),
                 ("inlp", "INLP"), ("coral", "CORAL")]:
        if m in g.index:
            mac(f"Loco{k}", f3(g.loc[m, "auc"]))
            t = tests[m]
            mac(f"Loco{k}Delta", f"{t['delta']:+.3f}")
            mac(f"Loco{k}P", "<0.001" if t["p_holm"] < 0.001 else f"{t['p_holm']:.3f}")
    if "inlp" in F.method.unique():
        x = F[F.method == "inlp"]
        mac("INLPaccBefore", f2(x.corpus_id_acc_before.mean())); mac("INLPaccAfter", f2(x.corpus_id_acc_after.mean()))
    # per-target table
    cols = ["pooled", "pooled_mask", "inlp", "coral"]
    rows = []
    for t in VER:
        r = per.loc[t]
        rows.append(f"{t} & {r.mean_single_source:.3f} & {r.best_single_source:.3f} & " +
                    " & ".join(f"{r[c]:.3f}" for c in cols) + f" & {r.ceiling:.3f} \\\\")
    write_table("repair_targets", r"""\begin{table}[h]
\centering
\caption{Leave-one-corpus-out AUC per held-out corpus (mean of five seeds), against single-source transfer and the in-domain ceiling.}\label{tab:repair-targets}
\small
\begin{tabular}{@{}lccccccc@{}}
\toprule
Held out & mean single & best single & pooled & + masking & INLP & CORAL & in-domain \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def supplement_part():
    # per-cell decomposition (tfidf_lr, full)
    Dc = pd.read_csv(f"{R}/summary_decomposition_cells.csv")
    d = Dc[(Dc.model == "tfidf_lr") & (Dc.regime == "full")].groupby(["source", "target"]).mean(numeric_only=True)
    rows = [f"{s} & {t} & {r.gap_target:.3f} & {r.ranking:.3f} & {r.threshold:.3f} & {r.prior:.3f} & {r.difficulty:.3f} & "
            f"{r.em_recovered:+.3f} \\\\" for (s, t), r in d.iterrows()]
    write_table("supp_decomp_cells", r"""\begin{longtable}{@{}llrrrrrr@{}}
\caption{Per-cell transfer gap decomposition, \textsc{tfidf-lr}, mean of five seeds (macro-F1 points). Gap = target in-domain F1 minus transfer F1 at $t=0.5$ = ranking + threshold + prior; ``difficulty'' is source minus target in-domain F1; ``EM'' is the change produced by EM prior adaptation.}\label{tab:supp-decomp}\\
\toprule
Source & Target & Gap & Ranking & Threshold & Prior & Difficulty & EM \\
\midrule
\endfirsthead
\toprule
Source & Target & Gap & Ranking & Threshold & Prior & Difficulty & EM \\
\midrule
\endhead
""" + "\n".join(rows) + r"""
\bottomrule
\end{longtable}
""")
    # operating points per model
    O = pd.read_csv(f"{R}/summary_operating_points_cells.csv")
    O = O[~O.in_domain]
    rows = []
    for (m, s, t), g in O.groupby(["model", "source", "target"]):
        if m != "tfidf_lr":
            continue
        r = g.mean(numeric_only=True)
        rows.append(f"{s} & {t} & {r.default:.3f} & {r.em:.3f} & {r.src_rate:.3f} & {r.half_rate:.3f} & {r.gmm:.3f} & "
                    f"{r.true_rate:.3f} & {r.oracle:.3f} \\\\")
    write_table("supp_oppoints_cells", r"""\begin{longtable}{@{}llrrrrrrr@{}}
\caption{Per-cell macro-F1 under each operating point, \textsc{tfidf-lr}, mean of five seeds.}\label{tab:supp-op}\\
\toprule
Source & Target & $t=0.5$ & EM & rank & 0.5 rate & GMM & $\pi_B$ & oracle \\
\midrule
\endfirsthead
\toprule
Source & Target & $t=0.5$ & EM & rank & 0.5 rate & GMM & $\pi_B$ & oracle \\
\midrule
\endhead
""" + "\n".join(rows) + r"""
\bottomrule
\end{longtable}
""")
    # learning curves
    L = pd.read_csv(f"{R}/mech_lc.csv")
    L = L[~L.full]
    tab = L.groupby(["corpus", "n"]).auc.agg(["mean", "std"])
    ns = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    rows = []
    for c in VER + ["PHEME", "ClickbaitEN"]:
        cells = []
        for n in ns:
            cells.append(f"{tab.loc[(c, n), 'mean']:.3f}" if (c, n) in tab.index else "--")
        rows.append(f"{c} & " + " & ".join(cells) + " \\\\")
    write_table("supp_lc", r"""\begin{table}[h]
\centering
\caption{Learning curves: in-domain ROC-AUC of \textsc{tfidf-lr} trained on $n$ items (mean of five seeds).}\label{tab:supp-lc}
\small
\begin{tabular}{@{}l""" + "c" * len(ns) + r"""@{}}
\toprule
Corpus & """ + " & ".join(str(n) for n in ns) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    # Bengali leave-one-publisher-out
    Lo = pd.read_csv(f"{R}/bangla_loso.csv").sort_values("n", ascending=False)
    rows = [f"{r.domain.replace('_', '-')} & {fmt_int(r.n)} & {r.clickbait_rate:.2f} & {r.accuracy:.3f} & {r.auc:.3f} & {r.pred_rate:.2f} \\\\"
            for r in Lo.itertuples()]
    write_table("supp_loso", r"""\begin{table}[h]
\centering
\caption{BanglaBait leave-one-publisher-out (character 3--5-gram LR on titles): each row is a held-out publisher.}\label{tab:supp-loso}
\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Publisher & $n$ & clickbait rate & accuracy & AUC & predicted rate \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    # dedup report
    rep = json.load(open(f"{R}/dedup_report.json"))
    keys = ["raw", "empty_removed", "exact_conflict_rows", "exact_dups_removed", "near_conflict_rows",
            "near_dups_removed", "final"]
    rows = [f"{c} & " + " & ".join(fmt_int(rep[c][k]) for k in keys) + " \\\\" for c in rep]
    write_table("supp_dedup", r"""\begin{table}[h]
\centering
\caption{Deduplication report per corpus.}\label{tab:supp-dedup}
\small
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Corpus & raw & empty & exact conflict & exact dup. & near conflict & near dup. & final \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


PARTS = [data_part, transfer_part, operating_part, appendix_matrices, bangla_part, models_part, mechanism_part,
         repair_part, supplement_part, topk_part, mask_part, vocab_part]


def main():
    for f in PARTS:
        try:
            f()
        except FileNotFoundError as e:
            print("skipped", f.__name__, e)
    with open(f"{P}/numbers.tex", "w") as fh:
        fh.write("% generated by src/paper_numbers.py - do not edit by hand\n")
        for k, v in sorted(MACROS.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(len(MACROS), "macros written to", f"{P}/numbers.tex")
    plain = {k: re.sub(r"\{,\}", ",", v).replace("\\%", "%").replace("$-$", "−") for k, v in MACROS.items()}
    json.dump(plain, open(f"{R}/key_numbers.json", "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
