"""Explanatory figures for the manuscript. They replace most result tables in the main text; the
exact values stay in the Supplementary Material. Every number drawn is read from results/; the only
synthetic data are the two illustrative score distributions in fig_concept (panels a and b).

Each figure is checked automatically for overlapping text and for text outside the canvas
(check_layout), and the run fails if any figure has a problem.
Run: python src/figures_paper.py
"""
import json, os, re, itertools
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D
import config as C
from figures import BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED, INK, INK2, MUTED, GRID

R = C.RESULTS
VER = ["ISOT", "WELFake", "PolitiFact", "GossipCop", "LIAR", "Twitter15", "Twitter16"]
MODELS = ["tfidf_lr", "tfidf_svm", "nbsvm", "glove_lr", "style_gb", "roberta_lr", "roberta_ft"]
MNAME = {"tfidf_lr": "TF-IDF + LR", "tfidf_svm": "TF-IDF + SVM", "nbsvm": "NB-SVM", "glove_lr": "GloVe + LR",
         "style_gb": "Style model (22 stats)", "roberta_lr": "RoBERTa, frozen †", "roberta_ft": "RoBERTa, fine-tuned ‡"}
SOFT = {"blue": "#e3eefb", "orange": "#fde8df", "aqua": "#dcf3ea", "yellow": "#fdf1d3", "violet": "#e8e5fa", "grey": "#f1f0ed"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.6, "axes.titlesize": 8.2, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.labelsize": 7.6, "xtick.labelsize": 7, "ytick.labelsize": 7.2,
                     "legend.fontsize": 6.8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "pdf.fonttype": 42, "figure.facecolor": "white"})
num = lambda s: float(str(s).split("±")[0])
PROBLEMS = {}


# ------------------------------------------------------------------ layout checker
def check_layout(fig, name, ignore=()):
    """Report pairs of overlapping text objects and text that leaves the canvas."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    hidden = set()                       # tick labels of ticks outside the view are never drawn
    for ax in fig.axes:
        for axis, lim in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            lo, hi = min(lim), max(lim)
            for tk in axis.get_major_ticks():
                if not (lo - 1e-9 <= tk.get_loc() <= hi + 1e-9):
                    hidden.update({id(tk.label1), id(tk.label2)})
                if not tk.label1.get_visible():
                    hidden.add(id(tk.label1))
                if not tk.label2.get_visible():
                    hidden.add(id(tk.label2))
    items = []
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text().strip()
        if not s or not t.get_visible() or s in ignore or id(t) in hidden:
            continue
        try:
            bb = t.get_window_extent(r)
        except Exception:
            continue
        if bb.width < 1 or bb.height < 1:
            continue
        items.append((s, bb))
    issues = []
    for s, bb in items:
        if bb.x0 < -0.4 * W or bb.y0 < -0.4 * H or bb.x1 > 1.4 * W or bb.y1 > 1.4 * H:
            issues.append(f"far outside canvas: {s!r}")
    for (s1, b1), (s2, b2) in itertools.combinations(items, 2):
        ix = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
        iy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
        if ix > 1.0 and iy > 1.0:
            issues.append(f"overlap: {s1!r} <-> {s2!r}")
    if issues:
        PROBLEMS[name] = issues


def save(fig, name, ignore=()):
    check_layout(fig, name, ignore)
    os.makedirs(C.FIGURES, exist_ok=True)
    fig.savefig(f"{C.FIGURES}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{C.FIGURES}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel(ax, s, title=None, x=-0.01):
    ax.text(x, 1.035, s, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom", color=INK)
    if title:
        ax.set_title(title, pad=6)


def K():
    return json.load(open(f"{R}/key_numbers.json"))


# ------------------------------------------------------------------ 1. study overview
def overview(name="fig_overview"):
    k = K()
    fig = plt.figure(figsize=(6.9, 2.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(2, 36); ax.axis("off")
    cards = [
        ("1", "Audit the data", ["10 public corpora", "English and Bengali", "articles, headlines,", "claims, tweets"],
         ["labels checked", "duplicates removed"], SOFT["yellow"], YELLOW),
        ("2", "Train on one,\ntest on all", ["7 trained detectors", "+ zero-shot LLM", "42 corpus pairs", "seeds, CIs, controls"],
         [f"in-domain AUC {k['InAUC']}", f"other corpora {k['OffAUC']}"], SOFT["blue"], BLUE),
        ("3", "Split the\nlost F1", ["ranking deficit:", "  no threshold fixes it", "operating point:", "  a threshold fixes it"],
         [f"ranking  {k['RankShareAll']}", f"operating point  {k['OpShareAll']}"], SOFT["violet"], VIOLET),
        ("4", "Repair\neach part", ["threshold: flag the", "  source base rate", "ranking: pool corpora,", "  invariance methods"],
         [f"rate rule fixes {k['RecRank']}", f"EM fixes {k['RecEM']}"], SOFT["aqua"], AQUA),
        ("5", "Check another\nlanguage", ["Bengali clickbait", "unseen publishers", "translate-test to", "  English and back"],
         [f"new publisher {k['BBlosoAUC']}", f"EN → BN {k['XlENtoBB']}"], SOFT["orange"], ORANGE),
    ]
    w, gap, x0, top, h = 18.2, 2.0, 0.6, 35.0, 17.5
    for i, (n, title, lines, chips, face, edge) in enumerate(cards):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, top - h), w, h, boxstyle="round,pad=0,rounding_size=1.4", fc=face, ec=edge, lw=1.2))
        ax.add_patch(plt.Circle((x + 2.6, top - 3.4), 1.8, color=edge, zorder=3))
        ax.text(x + 2.6, top - 3.45, n, ha="center", va="center", fontsize=7.6, color="white", fontweight="bold", zorder=4)
        ax.text(x + 5.2, top - 3.4, title, ha="left", va="center", fontsize=7.6, fontweight="bold", color=INK, linespacing=1.05)
        ax.text(x + 1.4, top - 8.6, "\n".join(lines), ha="left", va="top", fontsize=6.5, color=INK2, linespacing=1.4)
        for j, c in enumerate(chips):
            cy = top - h - 3.4 - j * 4.4
            ax.add_patch(FancyBboxPatch((x + 0.4, cy - 1.7), w - 0.8, 3.4, boxstyle="round,pad=0,rounding_size=1.6",
                                        fc="white", ec=edge, lw=0.9))
            ax.text(x + w / 2, cy, c, ha="center", va="center", fontsize=6.2, color=INK, fontweight="bold")
        if i < len(cards) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.15, top - h / 2), (x + w + gap - 0.15, top - h / 2),
                                         arrowstyle="-|>", mutation_scale=9, lw=1.1, color=MUTED))
    ax.text(50, 2.4, "Released: code, seeds, per-item scores, reporting protocol, and a script that regenerates every number in the paper",
            ha="center", va="bottom", fontsize=6.4, color=INK2, style="italic")
    save(fig, name)


# ------------------------------------------------------------------ 2. concept
def concept(name="fig_concept"):
    rng = np.random.default_rng(3)
    fig = plt.figure(figsize=(6.9, 3.9))
    gt = fig.add_gridspec(1, 2, wspace=0.12, left=0.04, right=0.97, bottom=0.62, top=0.93)
    gb = fig.add_gridspec(1, 1, left=0.34, right=0.92, bottom=0.1, top=0.40)
    a, b = fig.add_subplot(gt[0, 0]), fig.add_subplot(gt[0, 1])
    c = fig.add_subplot(gb[0, 0])
    bins = np.linspace(0, 1, 26)

    def dist(ax, real, fake, note):
        h1 = ax.hist(real, bins, color=BLUE, alpha=0.55)[0]
        h2 = ax.hist(fake, bins, color=ORANGE, alpha=0.55)[0]
        ax.set_ylim(0, 1.55 * max(h1.max(), h2.max()))
        ax.set_xlim(0, 1); ax.set_yticks([]); ax.grid(False); ax.spines["left"].set_visible(False)
        ax.set_xlabel("detector score on a new corpus")
        ax.text(0.02, 0.97, note, transform=ax.transAxes, ha="left", va="top", fontsize=6.4, color=INK2, zorder=5,
                bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.95))

    dist(a, rng.beta(5, 5, 700), rng.beta(5.6, 4.4, 700), "real and fake overlap:\nno cut-off can separate them")
    a.axvline(0.5, color=INK, lw=1.1, ls="--")
    panel(a, "a", "Ranking deficit")
    real, fake = rng.beta(6.5, 3.0, 700), rng.beta(16, 1.6, 700)
    dist(b, real, fake, "the classes separate, but\nthe cut-off 0.5 flags nearly all")
    b.axvline(0.5, color=INK, lw=1.1, ls="--")
    b.axvline(np.quantile(np.r_[real, fake], 0.5), color=AQUA, lw=2)
    panel(b, "b", "Operating-point loss")
    leg = [Patch(color=BLUE, alpha=0.55, label="real items"), Patch(color=ORANGE, alpha=0.55, label="fake items"),
           Line2D([], [], color=INK, ls="--", label="default cut-off 0.5"), Line2D([], [], color=AQUA, lw=2, label="rate-based cut-off")]
    a.legend(handles=leg, loc="upper center", bbox_to_anchor=(1.06, -0.36), ncol=4, fontsize=6.4, handlelength=1.4)
    D = pd.read_csv(f"{R}/summary_decomposition_cells.csv")
    D = D[(D.model == "tfidf_lr") & (D.regime == "full")]
    T = pd.read_csv(f"{R}/transfer_tfidf_lr.csv")
    T = T[T.regime == "full"]
    fbb = T[T.source == T.target].def_macro_f1.mean()
    rk, th, pr = D.ranking.mean(), D.threshold.mean(), D.prior.mean()
    steps = [("model trained on the target corpus", 0, fbb, "#c9c8c3"),
             ("ranking deficit (no cut-off can recover it)", fbb - rk, fbb, BLUE),
             ("threshold residual", fbb - rk - th, fbb - rk, AQUA),
             ("prior / calibration loss", fbb - rk - th - pr, fbb - rk - th, ORANGE),
             ("model from another corpus, cut-off 0.5", 0, fbb - rk - th - pr, "#8a8984")]
    y = np.arange(len(steps))[::-1]
    for yi, (lab, lo, hi, col) in zip(y, steps):
        c.barh(yi, hi - lo, left=lo, color=col, height=0.62)
        txt = f"{hi:.2f}" if lo == 0 else f"\u2212{hi - lo:.2f}"
        c.text(hi + 0.01, yi, txt, va="center", fontsize=6.6, color=INK)
    c.set_yticks(y, [st[0] for st in steps])
    c.set_xlim(0, 1.0); c.set_xlabel("macro-F1, TF-IDF + LR, mean over the 42 transfer cells")
    c.grid(axis="y", visible=False)
    panel(c, "c", "Measured in this study", x=-0.01)
    save(fig, name)


# ------------------------------------------------------------------ 3. audit
def audit(name="fig_audit"):
    P = pd.read_csv(f"{R}/label_probe.csv")
    rep = json.load(open(f"{R}/dedup_report.json"))
    fig = plt.figure(figsize=(6.9, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.8, left=0.25, right=0.99, bottom=0.2, top=0.86)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    probes = [("text contains '(Reuters)' (reliable)", "'(Reuters)'", "real"),
              ("title ends '- The New York Times' (reliable)", "'- The New York Times'", "real"),
              ("text contains 'Featured image via' (fake-site template)", "'Featured image via'", "fake"),
              ("text contains '21st Century Wire' (fake site)", "'21st Century Wire'", "fake")]
    rows = []
    for key, lab, exp in probes:
        for corp in ["ISOT", "WELFake"]:
            r = P[(P.corpus == corp) & (P.probe == key)].iloc[0]
            if r.n >= 100:
                rows.append((f"{corp}  {lab}", exp, r.raw_label_0 / r.n, int(r.n)))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, exp, share0, n) in zip(y, rows):
        a.barh(yi, share0, color=AQUA, height=0.6)
        a.barh(yi, 1 - share0, left=share0, color=VIOLET, height=0.6)
        a.text(1.03, yi, f"expects {exp}", va="center", fontsize=6.2, color=BLUE if exp == "real" else ORANGE)
    a.set_yticks(y, [r[0] for r in rows], fontsize=6.4)
    a.set_xlim(0, 1); a.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    a.set_xlabel("share of items carrying the signature")
    a.grid(axis="y", visible=False)
    panel(a, "a", "Which raw label do signatures carry?", x=-0.02)
    a.legend(handles=[Patch(color=AQUA, label="raw label 0"), Patch(color=VIOLET, label="raw label 1")],
             loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.2)
    order = VER + ["PHEME", "ClickbaitEN", "BanglaBait"]
    yb = np.arange(len(order))[::-1]
    for yi, c in zip(yb, order):
        d = rep[c]
        ex, nr = d["exact_dups_removed"] / d["raw"], d["near_dups_removed"] / d["raw"]
        cf = (d["exact_conflict_rows"] + d["near_conflict_rows"]) / d["raw"]
        b.barh(yi, ex, color=VIOLET, height=0.6)
        b.barh(yi, nr, left=ex, color=AQUA, height=0.6)
        b.barh(yi, cf, left=ex + nr, color=RED, height=0.6)
        b.text(ex + nr + cf + 0.004, yi, f"{100 * (ex + nr + cf):.1f}%", va="center", fontsize=6.2, color=INK2)
    b.set_yticks(yb, order, fontsize=6.6)
    b.set_xlim(0, 0.16); b.set_xticks([0, 0.05, 0.1, 0.15], ["0%", "5%", "10%", "15%"])
    b.set_xlabel("share of items removed")
    b.grid(axis="y", visible=False)
    panel(b, "b", "Duplicates removed", x=-0.02)
    b.legend(handles=[Patch(color=VIOLET, label="exact copy"), Patch(color=AQUA, label="near copy"),
                      Patch(color=RED, label="label conflict")], loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=3,
             fontsize=6.2)
    save(fig, name)


# ------------------------------------------------------------------ 4. same-task pairs
def pairs(name="fig_pairs"):
    P = pd.read_csv(f"{R}/summary_pairs.csv")
    P["lo"] = P.auc_ci.map(lambda s: float(re.findall(r"\d\.\d+", s)[0]))
    P["hi"] = P.auc_ci.map(lambda s: float(re.findall(r"\d\.\d+", s)[1]))
    fig = plt.figure(figsize=(6.9, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.06, left=0.2, right=0.99, bottom=0.3, top=0.91)
    a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1], sharey=a)
    y = np.arange(len(P))[::-1]
    lab = [f"{s} → {t}" for s, t in zip(P.source, P.target)]
    a.hlines(y, P.lo, P.hi, color=ORANGE, lw=2)
    a.scatter(P.auc, y, color=ORANGE, s=26, zorder=3)
    a.scatter(P.in_auc, y, color=BLUE, s=24, marker="s", zorder=3)
    a.axvline(0.5, color=MUTED, ls="--", lw=0.9)
    a.set_yticks(y, lab); a.set_xlim(0.38, 1.03); a.set_xlabel("ROC-AUC")
    a.grid(axis="y", visible=False)
    panel(a, "a", "Ranking")
    a.legend(handles=[Line2D([], [], marker="s", ls="", color=BLUE, label="source, in-domain"),
                      Line2D([], [], marker="o", color=ORANGE, label="on the target, 95% CI")],
             loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=6.3, handletextpad=0.3, columnspacing=0.8)
    cols = ["f1_def", "f1_pm", "f1_oracle"]
    b.hlines(y, P[cols].min(1), P[cols].max(1), color=GRID, lw=3, zorder=1)
    b.scatter(P.f1_def, y, color=ORANGE, s=26, zorder=3)
    b.scatter(P.f1_pm, y, color=AQUA, marker="D", s=22, zorder=3)
    b.scatter(P.f1_oracle, y, color=INK, marker="|", s=80, lw=1.8, zorder=4)
    b.axvline(1 / 3, color=MUTED, ls=":", lw=0.9); b.axvline(0.5, color=MUTED, ls="--", lw=0.9)
    b.set_xlim(0.28, 0.82); b.set_xlabel("macro-F1 on the target")
    b.tick_params(axis="y", labelleft=False)
    b.grid(axis="y", visible=False)
    panel(b, "b", "Operating point")
    b.legend(handles=[Line2D([], [], marker="o", ls="", color=ORANGE, label="cut-off 0.5"),
                      Line2D([], [], marker="D", ls="", color=AQUA, label="true base rate"),
                      Line2D([], [], marker="|", ls="", color=INK, markersize=9, label="best cut-off")],
             loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=6.3, handletextpad=0.3, columnspacing=0.8)
    save(fig, name)


# ------------------------------------------------------------------ 5. decomposition shares per detector
def decomp_models(name="fig_decomp_models"):
    D = pd.read_csv(f"{R}/summary_decomposition_cells.csv")
    D = D[D.regime == "full"]
    rows = [m for m in MODELS if m in set(D.model)]
    L = pd.read_csv(f"{R}/llm_zeroshot_qwen3_0.6b.csv")
    FS = pd.read_csv(f"{R}/transfer_tfidf_lr_ftsub.csv")
    fbb = FS[FS.source == FS.target].set_index(["target", "seed"]).def_macro_f1
    x = L.set_index(["target", "seed"])
    llm = dict(r=(fbb.loc[x.index] - x.oracle_macro_f1).sum(), t=(x.oracle_macro_f1 - x.pm_macro_f1).sum(),
               p=(x.pm_macro_f1 - x.def_macro_f1).sum())
    fig, ax = plt.subplots(figsize=(6.4, 2.75))
    labels = [MNAME[m] for m in rows] + ["Zero-shot LLM §"]
    y = np.arange(len(labels))[::-1]
    vals = []
    for m in rows:
        g = D[D.model == m]
        vals.append((g.ranking.sum(), g.threshold.sum(), g.prior.sum()))
    vals.append((llm["r"], llm["t"], llm["p"]))
    for yi, (r, t, p) in zip(y, vals):
        tot, left = r + t + p, 0
        for v, col in [(r, BLUE), (t, AQUA), (p, ORANGE)]:
            ax.barh(yi, v / tot, left=left, color=col, height=0.62, edgecolor="white", lw=1.2)
            if v / tot >= 0.09:
                ax.text(left + v / tot / 2, yi, f"{100 * v / tot:.0f}%", ha="center", va="center", fontsize=6.6,
                        color="white" if col == BLUE else INK, fontweight="bold")
            left += v / tot
    ax.axhline(0.5, color=MUTED, lw=0.8, ls="-")
    ax.axvline(0.5, color=INK, lw=0.8, ls=":")
    ax.set_yticks(y, labels); ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of the macro-F1 lost on a new corpus")
    ax.grid(False)
    ax.legend(handles=[Patch(color=BLUE, label="ranking deficit (no cut-off fixes it)"),
                       Patch(color=AQUA, label="threshold residual"), Patch(color=ORANGE, label="prior / calibration")],
              loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3, fontsize=6.5)
    save(fig, name)


# ------------------------------------------------------------------ 6. operating-point rules per detector
def oprules(name="fig_oprules"):
    M = pd.read_csv(f"{R}/robust_per_model.csv").set_index("model")
    rows = [m for m in MODELS if m in M.index]
    rules = [("rec_em", "EM prior adaptation", RED, "X"), ("rec_gmm", "two-component mixture", YELLOW, "^"),
             ("rec_half_rate", "flag half of the items", VIOLET, "v"), ("rec_src_rate", "rate transfer", BLUE, "o"),
             ("rec_true_rate", "true base rate known", AQUA, "D")]
    fig, ax = plt.subplots(figsize=(6.4, 2.75))
    y = np.arange(len(rows))[::-1]
    ax.axvspan(-1.0, 0, color="#f8e6e6", zorder=0, lw=0)
    ax.axvline(0, color=INK, lw=0.9); ax.axvline(1, color=INK, lw=0.9, ls="--")
    for j, (k, l, col, mk) in enumerate(rules):
        ax.scatter(M.loc[rows, k], y + (j - 2) * 0.11, color=col, marker=mk, s=30, zorder=3, label=l, edgecolor="white", lw=0.5)
    ax.set_yticks(y, [MNAME[m] for m in rows]); ax.set_xlim(-1.0, 1.05); ax.set_ylim(-0.6, len(rows) - 0.25)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1], ["−100%", "−50%", "0%\n(cut-off 0.5)", "50%", "100%\n(best cut-off)"])
    ax.set_xlabel("share of the operating-point loss recovered (no target labels unless stated)")
    ax.text(-0.98, len(rows) - 0.55, "worse than doing nothing", ha="left", va="top", fontsize=6.4, color=RED)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=5, fontsize=6.3, handletextpad=0.15, columnspacing=0.8)
    save(fig, name)


# ------------------------------------------------------------------ 7. detectors, LLM included
def models(name="fig_models"):
    H = pd.read_csv(f"{R}/summary_headline.csv")
    H = H[H.regime == "full"].set_index("model")
    rows = [m for m in MODELS if m in H.index]
    L = pd.read_csv(f"{R}/llm_zeroshot_qwen3_0.6b.csv")
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    labels = [MNAME[m] for m in rows] + ["Zero-shot LLM §"]
    y = np.arange(len(labels))[::-1]
    for yi, m in zip(y, rows):
        i, o, p = num(H.loc[m, "in_auc"]), num(H.loc[m, "off_auc"]), num(H.loc[m, "pair_auc"])
        ax.plot([o, i], [yi, yi], color=GRID, lw=3, zorder=1, solid_capstyle="round")
        ax.scatter(i, yi, s=34, color=BLUE, zorder=3)
        ax.scatter(o, yi, s=34, color=ORANGE, zorder=3)
        ax.scatter(p, yi, s=30, facecolor="white", edgecolor=ORANGE, lw=1.4, zorder=3)
        ax.text(i + 0.012, yi, f"{i:.2f}", va="center", fontsize=6.4, color=INK2)
        ax.text(o - 0.012, yi, f"{o:.2f}", va="center", ha="right", fontsize=6.4, color=INK2)
    a = L.auc.mean()
    ax.scatter(a, y[-1], s=40, marker="D", color=VIOLET, zorder=3)
    ax.text(a + 0.012, y[-1], f"{a:.2f} on every corpus (no training)", va="center", fontsize=6.4, color=INK2)
    ax.axhline(0.5, color=MUTED, lw=0.7)
    ax.axvline(0.5, color=MUTED, ls="--", lw=0.9)
    ax.set_yticks(y, labels); ax.set_xlim(0.40, 1.02); ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlabel("ROC-AUC (mean over cells)")
    ax.grid(axis="y", visible=False)
    h = [Line2D([], [], marker="o", ls="", color=BLUE, label="in-domain"),
         Line2D([], [], marker="o", ls="", color=ORANGE, label="other corpora (42 cells)"),
         Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=ORANGE, label="same-task pairs (8)"),
         Line2D([], [], marker="D", ls="", color=VIOLET, label="zero-shot")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=4, fontsize=6.4, handletextpad=0.3)
    save(fig, name)


# ------------------------------------------------------------------ 8. zero-shot LLM against trained detectors
def llm(name="fig_llm"):
    Lm = pd.read_csv(f"{R}/llm_zeroshot_qwen3_0.6b.csv").groupby("target")[
        ["auc", "def_macro_f1", "half_macro_f1", "pm_macro_f1", "oracle_macro_f1", "def_ppr", "prior"]].mean()
    FT = pd.read_csv(f"{R}/transfer_roberta_ft.csv")
    TS = pd.read_csv(f"{R}/transfer_tfidf_lr_ftsub.csv")
    ftx = FT[FT.source != FT.target].groupby("target").auc.mean()
    tsx = TS[TS.source != TS.target].groupby("target").auc.mean()
    tsi = TS[TS.source == TS.target].groupby("target").auc.mean()
    fig = plt.figure(figsize=(6.9, 3.1))
    gs = fig.add_gridspec(1, 2, wspace=0.06, left=0.12, right=0.99, bottom=0.33, top=0.9)
    a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1], sharey=a)
    y = np.arange(len(VER))[::-1]
    for s, l, col, mk in [(tsi, "TF-IDF trained here", BLUE, "s"), (tsx, "TF-IDF from other corpora", ORANGE, "o"),
                          (ftx, "fine-tuned RoBERTa, other corpora", YELLOW, "^"), (Lm.auc, "zero-shot LLM", VIOLET, "D")]:
        a.scatter(s.loc[VER], y, color=col, marker=mk, s=30, zorder=3, label=l, edgecolor="white", lw=0.4)
    a.axvline(0.5, color=MUTED, ls="--", lw=0.9)
    a.set_yticks(y, VER); a.set_xlim(0.38, 1.02); a.set_xlabel("ROC-AUC on the same test items")
    a.grid(axis="y", visible=False)
    panel(a, "a", "Ranking")
    a.legend(loc="upper center", bbox_to_anchor=(0.45, -0.24), ncol=2, fontsize=6.2, handletextpad=0.2, columnspacing=0.6)
    cols = ["def_macro_f1", "half_macro_f1", "oracle_macro_f1"]
    b.hlines(y, Lm.loc[VER, cols].min(1), Lm.loc[VER, cols].max(1), color=GRID, lw=3, zorder=1)
    b.scatter(Lm.loc[VER, "def_macro_f1"], y, color=ORANGE, s=28, zorder=3)
    b.scatter(Lm.loc[VER, "half_macro_f1"], y, color=AQUA, marker="D", s=24, zorder=3)
    b.scatter(Lm.loc[VER, "oracle_macro_f1"], y, color=INK, marker="|", s=80, lw=1.8, zorder=4)
    b.axvline(1 / 3, color=MUTED, ls=":", lw=0.9)
    b.set_xlim(0.28, 0.78); b.set_xlabel("zero-shot LLM macro-F1")
    b.tick_params(axis="y", labelleft=False); b.grid(axis="y", visible=False)
    panel(b, "b", "Operating point")
    b.legend(handles=[Line2D([], [], marker="o", ls="", color=ORANGE, label=f"its own answer (flags {100 * Lm.def_ppr.mean():.0f}%)"),
                      Line2D([], [], marker="D", ls="", color=AQUA, label="flag half the items"),
                      Line2D([], [], marker="|", ls="", color=INK, markersize=9, label="best cut-off")],
             loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=6.2, handletextpad=0.3, columnspacing=0.6)
    save(fig, name)


# ------------------------------------------------------------------ 9. shortcut indicators per corpus
def shortcuts(name="fig_shortcuts"):
    M = pd.read_csv(f"{R}/summary_matrix.csv")
    lr, st = M[M.model == "tfidf_lr"], M[M.model == "style_gb"]
    L = pd.read_csv(f"{R}/mech_lc.csv")
    G = pd.read_csv(f"{R}/mech_group.csv").groupby("corpus").auc.mean()
    lc64 = L[L.n == 64].groupby("corpus").auc.mean()
    rows = []
    for c in VER:
        rows.append(dict(c=c, full=lr[(lr.source == c) & (lr.target == c)].auc_mean.iloc[0],
                         style=st[(st.source == c) & (st.target == c)].auc_mean.iloc[0],
                         n64=lc64.get(c, np.nan), group=G.get(c, np.nan)))
    S = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    y = np.arange(len(S))[::-1]
    for yi, r in zip(y, S.itertuples()):
        vals = [v for v in [r.full, r.style, r.n64, r.group] if not np.isnan(v)]
        ax.plot([0.5, max(vals)], [yi, yi], color=GRID, lw=2.5, zorder=1)
    for k, l, col, mk in [("full", "full TF-IDF model", BLUE, "o"), ("style", "22 surface statistics only", VIOLET, "s"),
                          ("n64", "TF-IDF, 64 training items", AQUA, "D"), ("group", "publisher or speaker only", YELLOW, "^")]:
        ax.scatter(S[k], y, color=col, marker=mk, s=34, zorder=3, label=l, edgecolor="white", lw=0.5)
    ax.axvline(0.5, color=MUTED, ls="--", lw=0.9)
    ax.set_yticks(y, S.c); ax.set_xlim(0.48, 1.01); ax.set_ylim(-0.6, len(S) - 0.4)
    ax.set_xlabel("in-domain ROC-AUC")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=4, fontsize=6.3, handletextpad=0.2, columnspacing=0.8)
    save(fig, name)


# ------------------------------------------------------------------ 10. masking
def masking(name="fig_masking"):
    X = pd.read_csv(f"{R}/mech_mask.csv")
    lab = {"none": "nothing masked", "source": "source tags, datelines", "time": "dates, weekdays, numbers",
           "ents": "named entities", "all": "all of the above"}
    order = ["none", "source", "time", "ents", "all"]
    fig = plt.figure(figsize=(6.9, 2.5))
    gs = fig.add_gridspec(1, 2, wspace=0.06, left=0.2, right=0.99, bottom=0.34, top=0.9)
    axes = [fig.add_subplot(gs[0, 0])]
    axes.append(fig.add_subplot(gs[0, 1], sharey=axes[0]))
    y = np.arange(len(order))[::-1]
    for i, (ax, (s, t)) in enumerate(zip(axes, [("ISOT", "WELFake"), ("WELFake", "ISOT")])):
        vi = [X[(X.source == s) & (X.target == s) & (X.variant == v)].auc.mean() for v in order]
        vo = [X[(X.source == s) & (X.target == t) & (X.variant == v)].auc.mean() for v in order]
        ax.hlines(y, vo, vi, color=GRID, lw=3)
        ax.scatter(vi, y, color=BLUE, s=30, zorder=3)
        ax.scatter(vo, y, color=ORANGE, s=30, zorder=3)
        for yi, v in zip(y, vo):
            ax.text(v - 0.012, yi, f"{v:.2f}", ha="right", va="center", fontsize=6.4, color=INK2)
        ax.set_xlim(0.45, 1.02); ax.grid(axis="y", visible=False)
        ax.set_xlabel("ROC-AUC")
        panel(ax, "ab"[i], f"Trained on {s}")
        if i == 0:
            ax.set_yticks(y, [lab[o] for o in order])
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label=f"in-domain ({s})"),
                           Line2D([], [], marker="o", ls="", color=ORANGE, label=f"{s} → {t}")],
                  loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, fontsize=6.3, handletextpad=0.2)
    save(fig, name)


# ------------------------------------------------------------------ 11. ranking repairs (TF-IDF repairs + published methods)
def repairs(name="fig_repairs"):
    per = pd.read_csv(f"{R}/summary_fix_per_target.csv", index_col=0)
    ex = json.load(open(f"{R}/summary_extra.json"))["repairs_vs_pooled_auc"]
    D = pd.read_csv(f"{R}/dg_loco.csv")
    fig = plt.figure(figsize=(6.9, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.55, left=0.17, right=0.98, bottom=0.3, top=0.91)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    meth = [("pooled", "pooled training"), ("pooled_bal", "+ class balancing"), ("pooled_mask", "+ masking signatures"),
            ("corpus_feat_drop", "− corpus features"), ("svd", "SVD-300"), ("inlp", "SVD + INLP"), ("coral", "SVD + CORAL")]
    y = np.arange(len(meth))[::-1]
    pooled = per.pooled.mean()
    for yi, (k, l) in zip(y, meth):
        v = per[k]
        a.scatter(v, np.full(len(v), yi) + np.linspace(-0.2, 0.2, len(v)), s=7, color=MUTED, zorder=2)
        sig = k in ex and ex[k]["p_holm"] < 0.05
        col = BLUE if k == "pooled" else (RED if sig else INK2)
        a.scatter(v.mean(), yi, s=46, marker="D", color=col, zorder=3, edgecolor="white")
    a.axvline(per.mean_single_source.mean(), color=MUTED, ls=":", lw=1)
    a.axvline(per.best_single_source.mean(), color=MUTED, ls="--", lw=1)
    a.set_yticks(y, [l for _, l in meth]); a.set_xlim(0.45, 1.0)
    a.set_xlabel("held-out-corpus ROC-AUC")
    a.grid(axis="y", visible=False)
    panel(a, "a", "TF-IDF repairs (5 seeds)")
    a.legend(handles=[Line2D([], [], color=MUTED, ls=":", label="one source, average"),
                      Line2D([], [], color=MUTED, ls="--", label="best source (needs labels)"),
                      Line2D([], [], marker="D", ls="", color=RED, label="worse than pooling *"),
                      Line2D([], [], marker="o", ls="", color=MUTED, markersize=3, label="each held-out corpus")],
             loc="upper center", bbox_to_anchor=(1.0, -0.22), ncol=4, fontsize=6.1, handletextpad=0.3, columnspacing=0.8)
    dm = [("erm_lr", "pooled LR"), ("erm_mlp", "pooled MLP"), ("eann", "EANN-style"), ("mdfend", "MDFEND-style"),
          ("endef", "ENDEF-style")]
    yb = np.arange(len(dm))[::-1]
    for yi, (k, l) in zip(yb, dm):
        g = D[D.model == k].groupby("target").auc.mean()
        b.scatter(g.values, np.full(len(g), yi) + np.linspace(-0.2, 0.2, len(g)), s=7, color=MUTED, zorder=2)
        b.scatter(g.mean(), yi, s=46, marker="D", color=BLUE if k == "erm_lr" else INK2, zorder=3, edgecolor="white")
    b.set_yticks(yb, [l for _, l in dm]); b.set_xlim(0.45, 1.0)
    b.set_xlabel("held-out-corpus ROC-AUC")
    b.grid(axis="y", visible=False)
    panel(b, "b", "Published methods (RoBERTa, 3 seeds)")
    save(fig, name)


# ------------------------------------------------------------------ 12. Bengali
def bengali(name="fig_bengali"):
    B = pd.read_csv(f"{R}/bangla_official.csv")
    L = pd.read_csv(f"{R}/bangla_loso.csv")
    S = pd.read_csv(f"{R}/bangla_seeds.csv")
    X = pd.read_csv(f"{R}/bangla_xling.csv").groupby(["source", "target"]).auc.mean()
    fig = plt.figure(figsize=(6.9, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 0.9, 1.25], wspace=0.55, left=0.07, right=0.97, bottom=0.3, top=0.86)
    a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))
    g = B[B.field == "title"]
    mods = [("char_lr", "char LR"), ("char_svm", "char SVM"), ("word_lr", "word LR")]
    x = np.arange(len(mods))
    for i, (v, col) in enumerate([("as_released", BLUE), ("deduplicated", ORANGE)]):
        vals = [100 * g[(g.version == v) & (g.model == m)].f1_clickbait.iloc[0] for m, _ in mods]
        a.bar(x + (i - 0.5) * 0.38, vals, 0.36, color=col)
        for xi, vv in zip(x, vals):
            a.text(xi + (i - 0.5) * 0.38, vv - 2.0, f"{vv:.0f}", ha="center", va="top", fontsize=5.8, color="white",
                   fontweight="bold")
    a.axhline(66.28, color=INK, ls="--", lw=0.9)
    a.set_xticks(x, [l for _, l in mods], fontsize=6.6); a.set_ylim(0, 85); a.set_ylabel("clickbait-class F1 (%)")
    a.grid(axis="x", visible=False)
    panel(a, "a", "Official split")
    a.legend(handles=[Patch(color=BLUE, label="as released"), Patch(color=ORANGE, label="deduplicated"),
                      Line2D([], [], color=INK, ls="--", label="published (66.28)")],
             loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=6.1)
    Ls = L.sort_values("auc")
    b.scatter(Ls.auc, np.arange(len(Ls)), s=np.clip(Ls.n / 14, 6, 50), color=AQUA, edgecolor="white", lw=0.4, zorder=3)
    rs = S[S.model == "char_lr"].auc.mean()
    b.axvline(rs, color=INK, ls="--", lw=0.9)
    b.axvline(0.5, color=MUTED, ls=":", lw=0.9)
    b.set_yticks([]); b.set_xlim(0.45, 1.0); b.set_xlabel("ROC-AUC")
    b.grid(axis="y", visible=False)
    panel(b, "b", f"{len(Ls)} unseen publishers")
    b.legend(handles=[Line2D([], [], marker="o", ls="", color=AQUA, label="held-out publisher"),
                      Line2D([], [], color=INK, ls="--", label="random split")],
             loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=6.1)
    items = [(("BanglaBait-BN", "BanglaBait-BN"), "BN → BN"), (("BanglaBait-EN", "BanglaBait-EN"), "BN-tr → BN-tr"),
             (("ClickbaitEN", "BanglaBait-EN"), "EN → BN-tr"), (("BanglaBait-EN", "ClickbaitEN"), "BN-tr → EN"),
             (("ClickbaitEN", "ClickbaitEN"), "EN → EN")]
    items = [(k, l) for k, l in items if k in X.index]
    yy = np.arange(len(items))[::-1]
    vals = [X[k] for k, _ in items]
    cols = [BLUE if k[0] == k[1] else ORANGE for k, _ in items]
    c.barh(yy, vals, color=cols, height=0.6)
    for yi, v in zip(yy, vals):
        c.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=6.4, color=INK2)
    c.set_yticks(yy, [l for _, l in items], fontsize=6.6); c.set_xlim(0.5, 1.07); c.set_xlabel("ROC-AUC")
    c.grid(axis="y", visible=False)
    panel(c, "c", "Across languages")
    c.legend(handles=[Patch(color=BLUE, label="same language"), Patch(color=ORANGE, label="across languages")],
             loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=1, fontsize=6.1)
    c.text(1.0, -0.62, "BN: Bengali; BN-tr: machine-translated Bengali titles; EN: English",
           transform=c.transAxes, ha="right", fontsize=5.8, color=INK2)
    save(fig, name)


ALL = [overview, concept, audit, pairs, decomp_models, oprules, models, llm, shortcuts, masking, repairs, bengali]

if __name__ == "__main__":
    import sys
    want = sys.argv[1:]
    for f in ALL:
        if not want or f.__name__ in want:
            f()
            print("ok", f.__name__, flush=True)
    for k, v in PROBLEMS.items():
        print(f"LAYOUT {k}:")
        for x in v:
            print("   ", x)
    if PROBLEMS:
        sys.exit(1)
