"""Paper figures (vector PDF + 300-dpi PNG) from the results/ tables.
Palette: validated reference palette (light mode); diverging blue<->red with a grey midpoint
for AUC around 0.5; categorical slots used in fixed order.
Run: python src/figures.py
"""
import os, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve
import config as C

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
VER = ["ISOT", "WELFake", "PolitiFact", "GossipCop", "LIAR", "Twitter15", "Twitter16"]
DIVERGE = LinearSegmentedColormap.from_list("d", ["#b03030", RED, "#f0efec", BLUE, "#104281"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
                     "lines.linewidth": 2, "pdf.fonttype": 42})


def save(fig, name):
    os.makedirs(C.FIGURES, exist_ok=True)
    fig.savefig(f"{C.FIGURES}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{C.FIGURES}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def heatmap(M, model="tfidf_lr", key="auc_mean", name="fig_heatmap_auc", title="ROC-AUC"):
    g = M[M.model == model].pivot(index="source", columns="target", values=key).loc[VER, VER]
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    im = ax.imshow(g.values, cmap=DIVERGE,
                   norm=matplotlib.colors.TwoSlopeNorm(vcenter=0.5, vmin=0.2, vmax=1.0))
    for i in range(len(VER)):
        for j in range(len(VER)):
            v = g.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.2,
                    color="white" if (v > 0.85 or v < 0.3) else INK,
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(len(VER)), VER, rotation=40, ha="right")
    ax.set_yticks(range(len(VER)), VER)
    ax.set_xlabel("test corpus")
    ax.set_ylabel("training corpus")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(f"{title} (mean of 5 seeds)", color=INK2)
    cb.outline.set_visible(False)
    save(fig, name)


def decomposition_bars(Dc, name="fig_gap_decomposition"):
    pairs = [("ISOT", "WELFake"), ("WELFake", "ISOT"), ("PolitiFact", "GossipCop"), ("GossipCop", "PolitiFact"),
             ("Twitter15", "Twitter16"), ("Twitter16", "Twitter15"), ("LIAR", "PolitiFact"), ("PolitiFact", "LIAR")]
    d = Dc[(Dc.model == "tfidf_lr") & (Dc.regime == "full")]
    comps = [("difficulty", "target harder (i)", MUTED), ("ranking", "ranking deficit (ii)", BLUE),
             ("threshold", "threshold residual (iii)", AQUA), ("prior", "prior / calibration (iv)", ORANGE)]
    rows = [d[(d.source == s) & (d.target == t)][[c for c, _, _ in comps] + ["gap", "em_recovered"]].mean()
            for s, t in pairs]
    rows.append(d[~d.same_task][[c for c, _, _ in comps] + ["gap", "em_recovered"]].mean())
    labels = [f"{s}→{t}" for s, t in pairs] + ["all other cells"]
    R = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    y = np.arange(len(R))[::-1]
    pos, neg = np.zeros(len(R)), np.zeros(len(R))
    for c, lab, col in comps:
        v = R[c].values
        left = np.where(v >= 0, pos, neg + v)
        ax.barh(y, np.abs(v), left=left, height=0.62, color=col, label=lab, edgecolor="white", linewidth=1.2)
        pos = np.where(v >= 0, pos + v, pos)
        neg = np.where(v < 0, neg + v, neg)
    ax.scatter(R.gap.values, y, marker="D", s=22, color=INK, zorder=5, label="total gap (sum)")
    ax.scatter(R.em_recovered.values, y, marker="|", s=90, color=RED, zorder=6, linewidths=2,
               label="recovered by label-free EM")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("macro-F1 points lost relative to the source's in-domain score")
    ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, 1.2))
    ax.grid(axis="y", visible=False)
    save(fig, name)


def operating_points(M, name="fig_operating_points"):
    O = pd.read_csv(f"{C.RESULTS}/summary_operating_points_cells.csv")
    g = O[(O.model == "tfidf_lr") & (~O.in_domain)].groupby(["source", "target"])[
        ["default", "em", "src_rate", "oracle"]].mean().reset_index().sort_values("oracle")
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    ax.vlines(x, np.minimum(g["default"], g["em"]), g["oracle"], color=GRID, lw=2)
    for k, lab, col, mk in [("default", "default threshold 0.5", ORANGE, "o"),
                            ("em", "EM prior adaptation (label-free)", RED, "x"),
                            ("src_rate", "rate transfer (label-free)", BLUE, "s"),
                            ("oracle", "oracle threshold (upper bound)", INK, "_")]:
        ax.scatter(x, g[k], s=16 if mk not in "_x" else 40 if mk == "_" else 18, marker=mk, color=col,
                   label=lab, zorder=3, linewidths=1.6 if mk in "_x" else 0)
    box = dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.9)
    ax.axhline(1 / 3, color=MUTED, ls=":", lw=1, zorder=1)
    ax.text(-1.6, 1 / 3, "one class ≈ 0.33", ha="right", va="center", fontsize=6.5, color=INK2, bbox=box)
    ax.axhline(0.5, color=MUTED, ls="--", lw=0.8, zorder=1)
    ax.text(-1.6, 0.5, "random ≈ 0.5", ha="right", va="center", fontsize=6.5, color=INK2, bbox=box)
    ax.set_xlim(-1, len(g))
    ax.set_xticks([])
    ax.set_xlabel("42 off-diagonal train→test cells, sorted by oracle-threshold macro-F1")
    ax.set_ylabel("macro-F1 (mean of 5 seeds)")
    ax.legend(ncol=2, fontsize=7, loc="upper left")
    save(fig, name)


def learning_curve(L, name="fig_learning_curve"):
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    hi = {"ISOT": BLUE, "WELFake": ORANGE, "GossipCop": AQUA, "LIAR": YELLOW}
    grey_done = False
    for c, g in L.groupby("corpus"):
        g = g[~g.full].groupby("n").auc.agg(["mean", "std"]).reset_index()
        col = hi.get(c, "#c9c8c3")
        lab = c if c in hi else (None if grey_done else "other corpora")
        grey_done = grey_done or c not in hi
        ax.plot(g.n, g["mean"], color=col, lw=2 if c in hi else 1.2, zorder=3 if c in hi else 2,
                marker="o", ms=3, label=lab)
    h, l = ax.get_legend_handles_labels()
    order = [l.index(x) for x in ["ISOT", "WELFake", "GossipCop", "LIAR", "other corpora"] if x in l]
    ax.legend([h[i] for i in order], [l[i] for i in order], fontsize=6.8, loc="lower right")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("training examples (log scale)")
    ax.set_ylabel("in-domain ROC-AUC")
    ax.set_ylim(0.45, 1.01)
    save(fig, name)


def topk(K, name="fig_topk"):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    for ax, src in zip(axes, ["ISOT", "WELFake"]):
        g = K[K.source == src].groupby(["K", "target"]).auc.mean().unstack()
        for i, (t, col) in enumerate(zip(g.columns, [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET])):
            lw = 3.0 if t == src else 1.4
            ax.plot(np.maximum(g.index, 1), g[t], marker="o", ms=3, lw=lw, color=col, label=t)
        ax.set_xscale("log")
        ax.set_title(f"trained on {src}", color=INK, fontsize=8.5)
        ax.set_xlabel("top-K features deleted (K=0 plotted at 1)")
        ax.axhline(0.5, color=MUTED, ls="--", lw=0.8)
    axes[0].set_ylabel("ROC-AUC (mean of 5 seeds)")
    axes[1].legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.0, 0.5), title="test corpus\n(thick = in-domain)",
                   title_fontsize=6.5)
    save(fig, name)


def weights(W, name="fig_weights", corpora=("ISOT", "WELFake"), top=12):
    fig, axes = plt.subplots(1, len(corpora), figsize=(6.6, 3.2))
    for ax, c in zip(axes, corpora):
        g = W[W.corpus == c]
        f = g[(g.direction == "fake") & (g["rank"] <= top)].sort_values("weight")
        r = g[(g.direction == "real") & (g["rank"] <= top)].sort_values("weight", ascending=False)
        d = pd.concat([r, f])
        ax.barh(range(len(d)), d.weight, color=[ORANGE if w > 0 else BLUE for w in d.weight], height=0.7)
        ax.set_yticks(range(len(d)), d.feature, fontsize=6.6)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(f"{c}: strongest cues", color=INK, fontsize=8.5)
        ax.set_xlabel("weight (← real   fake →)")
        ax.grid(axis="y", visible=False)
    save(fig, name)


def pad_scatter(Pc, name="fig_pad"):
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.scatter(Pc.pad, Pc.auc, s=18, color=BLUE, alpha=0.85, edgecolors="white", linewidths=0.6)
    ax.set_xlabel("proxy A-distance between corpora (label-free)")
    ax.set_ylabel("transfer ROC-AUC (TF-IDF LR)")
    ax.axhline(0.5, color=MUTED, ls="--", lw=0.8)
    save(fig, name)


def roc_panel(name="fig_roc"):
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.8), sharey=True)
    for ax, src in zip(axes, ["ISOT", "WELFake"]):
        for t, col in [(src, INK), ("WELFake" if src == "ISOT" else "ISOT", BLUE), ("PolitiFact", ORANGE),
                       ("GossipCop", AQUA), ("LIAR", YELLOW)]:
            f = f"{C.ARTIFACTS}/preds/tfidf_lr_full_{src}_{t}_0.npz"
            if not os.path.exists(f):
                continue
            z = np.load(f)
            fp, tp, _ = roc_curve(z["y"], z["p"])
            ax.plot(fp, tp, color=col, lw=2 if t == src else 1.5, label=t + (" (in-domain)" if t == src else ""))
        ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=0.8)
        ax.set_title(f"trained on {src} (seed 0)", color=INK, fontsize=8.5)
        ax.set_xlabel("false-positive rate")
        ax.legend(fontsize=6.5, loc="lower right")
    axes[0].set_ylabel("true-positive rate")
    save(fig, name)


def fix_bars(F, name="fig_fix"):
    order = ["pooled", "pooled_bal", "pooled_mask", "corpus_feat_drop", "svd", "inlp", "coral"]
    lab = {"pooled": "pooled", "pooled_bal": "+ balanced", "pooled_mask": "+ masking",
           "corpus_feat_drop": "+ corpus-feature drop", "svd": "SVD-300", "inlp": "+ INLP", "coral": "+ CORAL"}
    g = F.groupby("method")[["auc", "def_macro_f1", "em_macro_f1"]].mean().loc[order]
    s = F.groupby(["method", "seed"])[["auc", "def_macro_f1", "em_macro_f1"]].mean().groupby("method").std().loc[order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    for i, (k, col, l) in enumerate([("auc", BLUE, "ROC-AUC"), ("def_macro_f1", ORANGE, "macro-F1, default t=0.5"),
                                     ("em_macro_f1", AQUA, "macro-F1, + EM prior adaptation")]):
        ax.bar(x + (i - 1) * 0.26, g[k], 0.24, yerr=s[k], color=col, label=l, error_kw=dict(lw=0.8, ecolor=INK2))
    ax.set_xticks(x, [lab[o] for o in order], rotation=20, ha="right")
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("held-out corpus (mean of 7×5)")
    ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="x", visible=False)
    save(fig, name)


def prior_shift(R, name="fig_prior_shift"):
    """Macro-F1 when each target is re-sampled to the source prior + delta (tfidf-lr, 42 cells x 5 seeds)."""
    g = R.groupby("delta")[["default", "em", "rank", "acc", "true_rate", "oracle"]].mean()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for k, lab, col, mk, ls in [("oracle", "oracle threshold (bound)", INK, "_", ":"),
                                ("true_rate", "true target base rate known", AQUA, "D", "-"),
                                ("rank", "rate transfer (source base rate)", BLUE, "s", "-"),
                                ("default", "default threshold 0.5", ORANGE, "o", "-"),
                                ("acc", "adjusted classify-and-count rate", YELLOW, "^", "-"),
                                ("em", "EM prior adaptation", RED, "x", "-")]:
        ax.plot(g.index, g[k], color=col, marker=mk, ms=4 if mk != "_" else 9, ls=ls, lw=1.8, label=lab)
    ax.axvline(0, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("target base rate minus source base rate")
    ax.set_ylabel("macro-F1 (off-diagonal mean)")
    ax.set_xticks(g.index, [f"{d:+.1f}" if d else "0" for d in g.index])
    ax.legend(fontsize=6.4, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    save(fig, name)


if __name__ == "__main__":
    R = C.RESULTS
    M = pd.read_csv(f"{R}/summary_matrix.csv")
    heatmap(M)
    heatmap(M, key="def_macro_f1_mean", name="fig_heatmap_f1", title="macro-F1 at t=0.5")
    if (M.model == "roberta_lr").any():
        heatmap(M, model="roberta_lr", name="fig_heatmap_auc_roberta")
    decomposition_bars(pd.read_csv(f"{R}/summary_decomposition_cells.csv"))
    operating_points(M)
    roc_panel()
    for f, fn in [("mech_lc.csv", learning_curve), ("mech_topk.csv", topk), ("mech_weights.csv", weights),
                  ("fix_loco.csv", fix_bars), ("summary_pad_cells.csv", pad_scatter),
                  ("robust_prior_shift.csv", prior_shift)]:
        if os.path.exists(f"{R}/{f}"):
            fn(pd.read_csv(f"{R}/{f}"))
    print("figures written to", C.FIGURES)
