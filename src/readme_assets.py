"""Animated SVG graphics for the README, drawn from the result files.
Writes assets/banner.svg and assets/decomposition.svg. Run: python src/readme_assets.py
Animations are CSS keyframes; every element also carries its final state as a plain attribute,
so a viewer that ignores the animation still shows the complete picture.
"""
import json, os
import pandas as pd
import config as C

OUT = os.path.join(C.ROOT, "assets")
os.makedirs(OUT, exist_ok=True)
K = json.load(open(f"{C.RESULTS}/key_numbers.json"))
pct = lambda k: float(K[k].replace("−", "-").replace("%", ""))

VER = ["ISOT", "WELFake", "PolitiFact", "GossipCop", "LIAR", "Twitter15", "Twitter16"]
SHORT = ["ISOT", "WEL", "PF", "GC", "LIAR", "T15", "T16"]
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
BLUE, ORANGE, GREEN, RED, GREY = "#3987e5", "#eb6834", "#1baf7a", "#e5484d", "#8fa3bd"
INK, SUB = "#f0efec", "#c3c2b7"
BG = """<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#0d1b2e"/><stop offset="1" stop-color="#172840"/>
  </linearGradient>
  <linearGradient id="rule" x1="0" x2="1">
    <stop offset="0" stop-color="#3987e5"/><stop offset="0.55" stop-color="#1baf7a"/><stop offset="1" stop-color="#eb6834"/>
  </linearGradient>
</defs>"""


def banner():
    T = pd.read_csv(f"{C.RESULTS}/transfer_tfidf_lr.csv")
    T = T[T.regime == "full"].groupby(["source", "target"]).auc.mean()
    W, H, cell, gap = 1200, 330, 30, 4
    x0, y0 = 92, 62
    cells, labels = [], []
    order = []
    for i, s in enumerate(VER):
        for j, t in enumerate(VER):
            a = T[(s, t)]
            col = BLUE if a >= 0.5 else ORANGE
            op = max(0.07, min(1.0, abs(a - 0.5) / 0.5))
            order.append((0 if i == j else 1, abs(i - j), i, j, col, op, a))
    order.sort()
    for k, (diag, _, i, j, col, op, a) in enumerate(order):
        d = 0.15 + 0.045 * k
        cells.append(f'<rect class="c" style="animation-delay:{d:.2f}s;--o:{op:.2f}" x="{x0 + j * (cell + gap)}" '
                     f'y="{y0 + i * (cell + gap)}" width="{cell}" height="{cell}" rx="6" fill="{col}" opacity="{op:.2f}">'
                     f'<title>{VER[i]} → {VER[j]}: AUC {a:.2f}</title></rect>')
    for j, s in enumerate(SHORT):
        labels.append(f'<text x="{x0 + j * (cell + gap) + cell / 2}" y="{y0 - 10}" text-anchor="middle">{s}</text>')
        labels.append(f'<text x="{x0 - 9}" y="{y0 + j * (cell + gap) + cell / 2 + 4}" text-anchor="end">{s}</text>')
    mx = x0 + 7 * (cell + gap) - gap
    facts = [f"{K['RankShareAll']} of the lost F1 is ranking; {K['OpShareAll']} is the threshold",
             f"EM label-shift correction makes the threshold worse ({K['RecEM']} recovered)",
             f"Flagging at the source base rate recovers {K['RecRank']}, with no target labels",
             f"In-domain AUC {K['InAUC']}, on a new corpus {K['OffAUC']}"]
    n, per = len(facts), 4.0
    cyc = n * per
    fact_svg = []
    for k, f in enumerate(facts):
        a0 = 100 * k * per / cyc
        a1 = 100 * (k * per + 0.6) / cyc
        a2 = 100 * ((k + 1) * per - 0.6) / cyc
        a3 = 100 * (k + 1) * per / cyc
        kf = (f"@keyframes f{k}{{0%{{opacity:0}}{a0:.2f}%{{opacity:0;transform:translateY(8px)}}"
              f"{a1:.2f}%{{opacity:1;transform:translateY(0)}}{a2:.2f}%{{opacity:1;transform:translateY(0)}}"
              f"{a3:.2f}%{{opacity:0;transform:translateY(-6px)}}100%{{opacity:0}}}}")
        fact_svg.append((kf, f'<text class="fact" style="animation:f{k} {cyc:.0f}s linear 2.6s infinite both" x="430" y="242" '
                             f'opacity="{1 if k == 0 else 0}">{f}</text>'))
    style = f"""<style>
  text {{ font-family: {FONT}; }}
  .lab text {{ font-size: 11px; fill: {GREY}; font-weight: 600; letter-spacing: .3px; }}
  .c {{ animation: pop .55s cubic-bezier(.2,.8,.2,1) both; transform-box: fill-box; transform-origin: center; }}
  @keyframes pop {{ from {{ opacity: 0; transform: scale(.4); }} to {{ opacity: var(--o); transform: scale(1); }} }}
  .t1 {{ animation: rise .8s ease-out .2s both; }}
  .t2 {{ animation: rise .8s ease-out .5s both; }}
  .chips {{ animation: rise .8s ease-out 1.1s both; }}
  @keyframes rise {{ from {{ opacity: 0; transform: translateY(12px); }} to {{ opacity: 1; transform: translateY(0); }} }}
  .rule {{ animation: sweep 1.4s cubic-bezier(.6,0,.2,1) .7s both; transform-box: fill-box; transform-origin: left; }}
  @keyframes sweep {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}
  .fact {{ font-size: 19px; fill: {INK}; }}
  {''.join(k for k, _ in fact_svg)}
  @media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; }} }}
</style>"""
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d">
<title id="t">Rank or threshold? Why fake-news detectors fail on new corpora</title>
<desc id="d">Animated banner: the seven-by-seven transfer matrix of ROC-AUC (diagonal in-domain, off-diagonal across corpora) beside the paper title and its main findings.</desc>
{BG}
{style}
<rect width="{W}" height="{H}" rx="20" fill="url(#bg)"/>
<g>{''.join(cells)}</g>
<g class="lab">{''.join(labels)}</g>
<text x="{(x0 + mx) / 2}" y="{y0 + 7 * (cell + gap) + 22}" text-anchor="middle" font-size="12.5" fill="{GREY}">train (rows) → test (columns), ROC-AUC</text>
<g>
  <text class="t1" x="430" y="122" font-size="56" font-weight="800" fill="{INK}" letter-spacing="-1">Rank or threshold?</text>
  <text class="t2" x="432" y="166" font-size="25" fill="{SUB}">Why fake-news detectors fail on new corpora</text>
  <rect class="rule" x="432" y="188" width="560" height="5" rx="2.5" fill="url(#rule)"/>
  {''.join(t for _, t in fact_svg)}
  <g class="chips" font-size="13.5" fill="{GREY}">
    <text x="432" y="292">10 corpora  ·  English and Bengali  ·  7 detectors + a zero-shot LLM  ·  5 seeds  ·  label-free repairs</text>
  </g>
</g>
</svg>"""
    open(f"{OUT}/banner.svg", "w", encoding="utf8").write(svg)


def decomposition():
    W, H = 1200, 470
    L, R = 300, 1140                  # bar area
    rank, op = pct("RankShareAll"), pct("OpShareAll")
    xs = L + (R - L) * rank / 100
    rec = [("EM prior adaptation", pct("RecEM"), RED, "the textbook fix"),
           ("Flag half of the items", pct("RecHalf"), "#5ea4ef", "no labels"),
           ("Rate transfer", pct("RecRank"), GREEN, "no labels"),
           ("Known target base rate", pct("RecPrior"), GREY, "needs labels")]
    z = L + (R - L) * 0.5               # zero of the recovery axis (-100% .. 100%)
    unit = (R - L) * 0.5 / 100
    cyc = 14
    t = lambda s: f"{100 * s / cyc:.2f}%"

    def grow(name, start, dur, origin="left"):
        return (f".{name}{{transform-box:fill-box;transform-origin:{origin};animation:{name} {cyc}s cubic-bezier(.3,.7,.2,1) infinite both}}"
                f"@keyframes {name}{{0%{{transform:scaleX(0)}}{t(start)}{{transform:scaleX(0)}}{t(start + dur)}{{transform:scaleX(1)}}"
                f"{t(cyc - .5)}{{transform:scaleX(1);opacity:1}}100%{{transform:scaleX(1);opacity:0}}}}")

    def fade(name, start, dur=.6):
        return (f".{name}{{animation:{name} {cyc}s ease-out infinite both}}"
                f"@keyframes {name}{{0%{{opacity:0}}{t(start)}{{opacity:0}}{t(start + dur)}{{opacity:1}}"
                f"{t(cyc - .5)}{{opacity:1}}100%{{opacity:0}}}}")

    css = [grow("gA", .2, 1.0), fade("lA", .5), grow("gR", 1.6, .9), grow("gO", 2.3, .7), fade("lR", 2.1), fade("lO", 2.8),
           fade("ax", 3.4), fade("zoom", 3.2)]
    rows = []
    for k, (name, v, col, note) in enumerate(rec):
        y = 300 + k * 38
        w = abs(v) * unit
        x = z if v >= 0 else z - w
        cls = f"r{k}"
        css.append(grow(cls, 3.8 + .45 * k, .8, "left" if v >= 0 else "right"))
        css.append(fade(f"v{k}", 4.4 + .45 * k))
        tx = x + w + 10 if v >= 0 else x - 10
        rows.append(f'<text class="v{k}" x="{L - 16}" y="{y + 16}" text-anchor="end" font-size="15" fill="{INK}">{name}'
                    f'<tspan fill="{GREY}" font-size="12.5">  {note}</tspan></text>'
                    f'<rect class="{cls}" x="{x:.1f}" y="{y}" width="{w:.1f}" height="22" rx="5" fill="{col}"/>'
                    f'<text class="v{k}" x="{tx:.1f}" y="{y + 16}" text-anchor="{"start" if v >= 0 else "end"}" font-size="15" '
                    f'font-weight="700" fill="{col}">{K[["RecEM", "RecHalf", "RecRank", "RecPrior"][k]]}</text>')
    style = (f"<style>text{{font-family:{FONT}}}{''.join(css)}"
             "@media (prefers-reduced-motion: reduce){*{animation:none!important}}</style>")
    ticks = "".join(f'<line x1="{z + p * unit:.1f}" y1="292" x2="{z + p * unit:.1f}" y2="452" stroke="#2c3e58" stroke-width="1"/>'
                    f'<text x="{z + p * unit:.1f}" y="466" text-anchor="middle" font-size="11.5" fill="{GREY}">{p:+d}%</text>'
                    for p in (-100, -50, 0, 50, 100)).replace("+0%", "0%")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H + 12}" width="{W}" height="{H + 12}" role="img" aria-labelledby="t d">
<title id="t">Where the lost macro-F1 goes, and what repairs it</title>
<desc id="d">Animated chart. The macro-F1 a detector loses on a new corpus splits into {K['RankShareAll']} ranking deficit and {K['OpShareAll']} operating-point loss. Of the operating-point loss, EM prior adaptation recovers {K['RecEM']}, flagging half of the items {K['RecHalf']}, rate transfer {K['RecRank']} and the known target base rate {K['RecPrior']}.</desc>
{BG}
{style}
<rect width="{W}" height="{H + 12}" rx="20" fill="url(#bg)"/>
<text x="60" y="58" font-size="26" font-weight="800" fill="{INK}">Where the lost macro-F1 goes</text>
<text x="60" y="86" font-size="15" fill="{GREY}">TF-IDF + LR, 42 train → test pairs, 5 seeds. Every number is read from the result files.</text>

<text class="lA" x="{L - 16}" y="146" text-anchor="end" font-size="15" fill="{INK}">Lost on a new corpus</text>
<rect class="gA" x="{L}" y="128" width="{R - L}" height="26" rx="6" fill="#4a5d7a"/>
<text class="lA" x="{R}" y="122" text-anchor="end" font-size="12.5" fill="{GREY}">100% of the gap</text>

<text class="lR" x="{L - 16}" y="204" text-anchor="end" font-size="15" fill="{INK}">split by cause</text>
<rect class="gR" x="{L}" y="186" width="{xs - L - 3:.1f}" height="26" rx="6" fill="{BLUE}"/>
<rect class="gO" x="{xs + 3:.1f}" y="186" width="{R - xs - 3:.1f}" height="26" rx="6" fill="{ORANGE}"/>
<text class="lR" x="{L + 12}" y="204" font-size="15" font-weight="700" fill="#ffffff">Ranking {K['RankShareAll']}
  <tspan font-weight="400" fill="#e6efff">  no threshold can fix it</tspan></text>
<text class="lO" x="{xs + 14:.1f}" y="204" font-size="15" font-weight="700" fill="#ffffff">Threshold {K['OpShareAll']}</text>
<text class="lO" x="{R}" y="234" text-anchor="end" font-size="13" fill="{ORANGE}">the right cut-off fixes this part ↓</text>

<g class="zoom">
  <path d="M{xs + 3:.1f} 214 L{R} 214 L{R} 262 L{L} 262 Z" fill="{ORANGE}" opacity=".07"/>
  <text x="60" y="282" font-size="17" font-weight="700" fill="{INK}">Share of the threshold loss recovered</text>
</g>
<g class="ax">{ticks}<line x1="{z}" y1="292" x2="{z}" y2="452" stroke="{GREY}" stroke-width="1.5"/></g>
{''.join(rows)}
</svg>"""
    open(f"{OUT}/decomposition.svg", "w", encoding="utf8").write(svg)


def rate_transfer():
    """Schematic: why a score threshold breaks on a new corpus and a rate threshold does not."""
    import numpy as np
    W, H = 1200, 400
    pi = 0.40                                  # source base rate (illustrative)
    xs = np.linspace(0, 1, 4001)
    pdf = lambda m, s: np.exp(-0.5 * ((xs - m) / s) ** 2)
    src = (1 - pi) * pdf(0.25, 0.11) + pi * pdf(0.78, 0.11)
    tgt = (1 - pi) * pdf(0.18, 0.10) + pi * pdf(0.42, 0.12)   # scores shrink towards "real" on the new corpus
    src, tgt = src / src.max(), tgt / tgt.max()

    def cdf_q(d, q):
        c = np.cumsum(d); c /= c[-1]
        return float(xs[np.searchsorted(c, q)])

    def flagged(d, t):
        return float(d[xs >= t].sum() / d.sum())

    t_src, t_rate = 0.5, cdf_q(tgt, 1 - flagged(src, 0.5))
    pane = [(60, "Training corpus", src, [(t_src, ORANGE, "default cut-off 0.5")]),
            (630, "New corpus", tgt, [(t_src, RED, "same cut-off 0.5"), (t_rate, GREEN, "rate transfer")])]
    pw, ph, base = 510, 190, 300
    X = lambda x0, v: x0 + v * pw
    Y = lambda v: base - v * ph
    out, css = [], []
    cyc = 12
    tt = lambda s: f"{100 * s / cyc:.2f}%"
    css.append(f".slide{{animation:slide {cyc}s cubic-bezier(.5,0,.2,1) infinite both}}"
               f"@keyframes slide{{0%{{transform:translateX({(t_src - t_rate) * pw:.1f}px)}}{tt(3)}{{transform:translateX({(t_src - t_rate) * pw:.1f}px)}}"
               f"{tt(5)}{{transform:translateX(0)}}{tt(11.4)}{{transform:translateX(0)}}100%{{transform:translateX({(t_src - t_rate) * pw:.1f}px)}}}}")
    for nm, a, b in [("before", 0, 3.2), ("after", 4.8, 11.4)]:
        if nm == "before":
            css.append(f".{nm}{{animation:{nm} {cyc}s linear infinite both}}@keyframes {nm}{{0%{{opacity:1}}{tt(b)}{{opacity:1}}"
                       f"{tt(b + .6)}{{opacity:0}}{tt(11.5)}{{opacity:0}}100%{{opacity:1}}}}")
        else:
            css.append(f".{nm}{{animation:{nm} {cyc}s linear infinite both}}@keyframes {nm}{{0%{{opacity:0}}{tt(a)}{{opacity:0}}"
                       f"{tt(a + .6)}{{opacity:1}}{tt(b)}{{opacity:1}}{tt(b + .5)}{{opacity:0}}100%{{opacity:0}}}}")
    for x0, title, d, cuts in pane:
        pts = " ".join(f"{X(x0, x):.1f},{Y(v):.1f}" for x, v in zip(xs, d))
        out.append(f'<text x="{x0}" y="78" font-size="19" font-weight="700" fill="{INK}">{title}</text>')
        out.append(f'<line x1="{x0}" y1="{base}" x2="{x0 + pw}" y2="{base}" stroke="{GREY}" stroke-width="1"/>')
        for v in (0, 0.5, 1):
            out.append(f'<text x="{X(x0, v):.1f}" y="{base + 20}" text-anchor="middle" font-size="12" fill="{GREY}">{v:g}</text>')
        out.append(f'<text x="{x0 + pw}" y="{base + 40}" text-anchor="end" font-size="12.5" fill="{GREY}">score: probability of "fake"</text>')
        for t, col, lab in cuts:
            share = flagged(d, t)
            area = " ".join(f"{X(x0, x):.1f},{Y(v):.1f}" for x, v in zip(xs, d) if x >= t)
            poly = f"{X(x0, t):.1f},{base} {area} {X(x0, 1):.1f},{base}"
            if title == "Training corpus":
                cls, line_cls = "", ""
            else:
                cls = "before" if col == RED else "after"
                line_cls = cls
            out.append(f'<polygon class="{cls}" points="{poly}" fill="{col}" opacity=".28"/>' if cls == "before" else
                       f'<g class="{cls}"><polygon points="{poly}" fill="{col}" opacity=".28"/></g>')
            grp = "slide" if col == GREEN else ""
            out.append(f'<g class="{line_cls}"><g class="{grp}">'
                       f'<line x1="{X(x0, t):.1f}" y1="{base}" x2="{X(x0, t):.1f}" y2="{base - ph - 18}" stroke="{col}" stroke-width="2.5" stroke-dasharray="6 4"/>'
                       f'<text x="{X(x0, t) + 8:.1f}" y="{base - ph - 8}" font-size="13.5" font-weight="700" fill="{col}">{lab}</text></g>'
                       f'<text x="{x0 + pw}" y="78" text-anchor="end" font-size="16" font-weight="600" fill="{col}">flags {100 * share:.0f}% of items</text></g>')
        out.append(f'<polyline points="{pts}" fill="none" stroke="{INK}" stroke-width="2.2"/>')
    style = (f"<style>text{{font-family:{FONT}}}{''.join(css)}"
             "@media (prefers-reduced-motion: reduce){*{animation:none!important}}</style>")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d">
<title id="t">Rate transfer in one picture</title>
<desc id="d">Schematic. On the training corpus the default cut-off of 0.5 flags about {100 * flagged(src, .5):.0f}% of items, close to the base rate. On a new corpus the scores drift, so the same cut-off flags only {100 * flagged(tgt, .5):.0f}%. Rate transfer moves the cut-off until the new corpus is flagged at the training base rate again.</desc>
{BG}
{style}
<rect width="{W}" height="{H}" rx="20" fill="url(#bg)"/>
<text x="60" y="42" font-size="13" fill="{GREY}" letter-spacing="1.2">SCHEMATIC  ·  WHY THE THRESHOLD BREAKS, AND THE LABEL-FREE FIX</text>
{''.join(out)}
<text x="60" y="378" font-size="14.5" fill="{SUB}">Rate transfer: <tspan fill="{INK}" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">t = quantile(new_scores, 1 − source_base_rate)</tspan>. The ranking is kept; only the cut-off moves.</text>
</svg>"""
    open(f"{OUT}/rate_transfer.svg", "w", encoding="utf8").write(svg)


if __name__ == "__main__":
    banner()
    decomposition()
    rate_transfer()
    print("assets/banner.svg, assets/decomposition.svg, assets/rate_transfer.svg written")
