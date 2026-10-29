"""Mechanism tests: what signal do the detectors actually use?

  lc        learning curve: in-domain AUC with n = 32 ... full training examples
  topk      delete the K highest-|weight| features (K = 10 ... 3000), retrain, re-test
  mask      mask dates/weekdays/months/numbers/URLs/handles/source tags/named entities
  topic     topic-balanced training (classes balanced within each topic cluster)
  group     group-only baseline (publisher domain / speaker / event identity -> label)
  pad       proxy A-distance between every pair of corpora (label-free divergence)
  weights   top features per class for every corpus (TF-IDF LR, seed 0)
  vocab     vocabulary-size and n-gram ablation on the article pair
All with 5 seeds. Writes results/mech_<test>.csv
Run: python src/mechanism.py --tests lc,topk,mask,topic,group,pad,weights,vocab
"""
import re, argparse, json
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
import config as C
from common import load_corpora, split, evaluate
from models import TfidfLinear

VERACITY = ["ISOT", "WELFake", "PolitiFact", "GossipCop", "LIAR", "Twitter15", "Twitter16"]
SEEDS = C.SEEDS


def corp(D, c):
    return D[D.corpus == c]


def eval_targets(m, src, splits, pi_s, tag, extra=None):
    rows = []
    for tgt, (_, _, te) in splits.items():
        if tgt != src:
            te = te[~te.dup_in.fillna("").str.contains(fr"(?:^|;){src}(?:;|$)")]
        r = evaluate(te.label.values, m.predict_proba(te.text.tolist()), pi_s)
        r.update(source=src, target=tgt, variant=tag, **(extra or {}))
        rows.append(r)
    return rows


# ---------------------------------------------------------------- learning curve
def t_lc(D):
    rows = []
    for seed in SEEDS:
        for c in VERACITY + ["PHEME", "ClickbaitEN"]:
            tr, va, te = split(corp(D, c), seed)
            for n in [32, 64, 128, 256, 512, 1024, 2048, 4096, len(tr)]:
                if n > len(tr):
                    continue
                sub = tr if n == len(tr) else train_test_split(tr, train_size=n, stratify=tr.label,
                                                               random_state=seed)[0]
                m = TfidfLinear("lr").fit(sub.text.tolist(), sub.label.values, va.text.tolist(), va.label.values)
                p = m.predict_proba(te.text.tolist())
                rows.append(dict(corpus=c, seed=seed, n=n, full=n == len(tr),
                                 auc=roc_auc_score(te.label.values, p)))
        print("lc seed", seed, flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- top-K deletion
def t_topk(D, sources=("ISOT", "WELFake", "PolitiFact", "GossipCop")):
    """Top-K deletion; training capped at 4,000, test at 1,500, 5 seeds."""
    rows = []
    for seed in SEEDS:
        splits = {c: small_split(corp(D, c), seed) for c in VERACITY}
        for src in sources:
            tr, va, _ = splits[src]
            base = TfidfLinear("lr").fit(tr.text.tolist(), tr.label.values, va.text.tolist(), va.label.values)
            names = base.feature_names()
            order = np.argsort(-np.abs(base.m.coef_.ravel()))
            for K in [0, 10, 30, 100, 300, 1000, 3000]:
                if K > len(names) // 2:            # small vocabularies (PolitiFact): skip
                    continue
                m = base if K == 0 else TfidfLinear("lr", vocab_drop=names[order[:K]]).fit(
                    tr.text.tolist(), tr.label.values, va.text.tolist(), va.label.values)
                rows += eval_targets(m, src, splits, tr.label.mean(), f"top{K}", dict(K=K, seed=seed))
        print("topk seed", seed, flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- masking
MONTHS = r"\b(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(t|tember)?|oct(ober)?|nov(ember)?|dec(ember)?)\b"
WEEK = r"\b(mon|tues|wednes|thurs|fri|satur|sun)day\b"
SOURCE_TAGS = [r"\(reuters\)", r"\breuters\b", r"\bassociated press\b", r"\(ap\)", r"featured image via",
               r"image via", r"getty images?", r"\bvia\b", r"21st century wire", r"21wire",
               r"\bbreitbart\b", r"new york times", r"\bnytimes\b", r"washington post", r"\bcnn\b",
               r"\bfox news\b", r"pic\.twitter\.com\S*", r"twitter\.com\S*", r"youtube\.com\S*"]
_NLP = None


def ner():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "tagger", "attribute_ruler"])
    return _NLP


def mask_texts(texts, level, maxch=3000):
    out = []
    texts = [str(t)[:maxch] for t in texts]         # same truncation for every variant
    docs = ner().pipe(texts, batch_size=64) if level in ("ents", "all") else [None] * len(texts)
    for t, doc in zip(texts, docs):
        if doc is not None:
            pieces, last = [], 0
            for e in doc.ents:
                if e.label_ in ("PERSON", "ORG", "GPE", "NORP", "LOC", "FAC", "EVENT", "DATE", "TIME"):
                    pieces += [t[last:e.start_char], f" ENT{e.label_} "]
                    last = e.end_char
            t = "".join(pieces) + t[last:]
        if level in ("time", "all"):
            t = re.sub(WEEK, " WEEKDAY ", t, flags=re.I)
            t = re.sub(MONTHS, " MONTH ", t, flags=re.I)
            t = re.sub(r"\d+([.,:]\d+)*", " NUM ", t)
        if level in ("source", "all"):
            for p in SOURCE_TAGS:
                t = re.sub(p, " SRC ", t, flags=re.I)
            t = re.sub(r"https?://\S+|www\.\S+", " URL ", t)
            t = re.sub(r"@\w+", " HANDLE ", t)
            t = re.sub(r"^\s*[A-Z][A-Z .,'/-]{2,40}\s*(\([^)]{1,25}\))?\s*[-—]\s*", " DATELINE ", t)
        out.append(t)
    return out


def small_split(d, seed, ntr=4000, nva=1000, nte=1500):
    tr, va, te = split(d, seed)
    sub = lambda x, n: x if len(x) <= n else train_test_split(x, train_size=n, stratify=x.label, random_state=seed)[0]
    return sub(tr, ntr), sub(va, nva), sub(te, nte)


def t_mask(D, sources=("ISOT", "WELFake"), targets=("ISOT", "WELFake", "PolitiFact", "GossipCop", "LIAR"),
           seeds=(0, 1, 2)):
    """Masking ablation. Training capped at 4,000, test at 1,500, 3 seeds. Masked text is
    computed once per document (cached) so every seed sees identical masking."""
    rows, cache = [], {}
    sps = {seed: {c: small_split(corp(D, c), seed) for c in targets} for seed in seeds}
    allrows = pd.concat([x for sp in sps.values() for t in sp.values() for x in t]).drop_duplicates("id")
    ids, texts = allrows.id.tolist(), allrows.text.tolist()
    for level in ["none", "source", "time", "ents", "all"]:
        if level == "none":
            cache[level] = dict(zip(ids, [str(t)[:3000] for t in texts]))
        elif level == "all":
            cache[level] = dict(zip(ids, mask_texts(mask_texts([cache["ents"][i] for i in ids], "time"), "source")))
        else:
            cache[level] = dict(zip(ids, mask_texts(texts, level)))
        print("masked", level, len(ids), flush=True)
    for seed in seeds:
        for level in cache:
            msp = {c: tuple(x.assign(text=x.id.map(cache[level])) for x in t) for c, t in sps[seed].items()}
            for src in sources:
                tr, va, _ = msp[src]
                m = TfidfLinear("lr").fit(tr.text.tolist(), tr.label.values, va.text.tolist(), va.label.values)
                rows += eval_targets(m, src, msp, tr.label.mean(), level, dict(seed=seed))
        print("mask seed", seed, flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- topic-balanced training
def topic_balance(df, k, seed):
    X = TfidfVectorizer(max_features=20000, stop_words="english", min_df=2).fit_transform(
        [str(t)[:5000] for t in df.text])
    Z = TruncatedSVD(100, random_state=seed).fit_transform(X)
    lab = KMeans(k, n_init=4, random_state=seed).fit_predict(Z)
    df = df.assign(topic=lab)
    keep = []
    for _, g in df.groupby("topic"):
        n = g.label.value_counts()
        if len(n) < 2:
            continue
        m = n.min()
        keep += [g[g.label == 0].sample(m, random_state=seed), g[g.label == 1].sample(m, random_state=seed)]
    return pd.concat(keep), df


def t_topic(D, sources=("ISOT", "WELFake", "GossipCop", "LIAR"), k=50):
    rows = []
    for seed in SEEDS:
        splits = {c: split(corp(D, c), seed) for c in VERACITY}
        for src in sources:
            tr, va, te = splits[src]
            bal, tagged = topic_balance(pd.concat([tr, va]), k, seed)
            purity = tagged.groupby("topic").label.mean().apply(lambda x: max(x, 1 - x))
            btr, bva = train_test_split(bal, test_size=0.125, stratify=bal.label, random_state=seed)
            for tag, (a, b) in {"random": (tr.sample(len(btr), random_state=seed), va),
                                "topic_balanced": (btr, bva)}.items():
                m = TfidfLinear("lr").fit(a.text.tolist(), a.label.values, b.text.tolist(), b.label.values)
                rows += eval_targets(m, src, splits, a.label.mean(), tag,
                                     dict(seed=seed, n_train=len(a), mean_topic_purity=float(purity.mean()),
                                          frac_dropped=1 - len(bal) / len(tagged)))
        print("topic seed", seed, flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- group-only baseline
def t_group(D):
    """Out-of-fold target encoding of group identity (publisher domain, speaker, event)."""
    rows = []
    for c, gname in [("PolitiFact", "publisher domain"), ("GossipCop", "publisher domain"),
                     ("LIAR", "speaker"), ("PHEME", "event"), ("BanglaBait", "publisher domain")]:
        d = corp(D, c).reset_index(drop=True)
        g = d.group.fillna("").replace("", "<none>")
        for seed in SEEDS:
            oof = np.zeros(len(d))
            for tri, tei in StratifiedKFold(5, shuffle=True, random_state=seed).split(d, d.label):
                prior = d.label.iloc[tri].mean()
                stats = d.iloc[tri].groupby(g.iloc[tri]).label.agg(["sum", "count"])
                enc = (stats["sum"] + 2 * prior) / (stats["count"] + 2)     # smoothed
                oof[tei] = g.iloc[tei].map(enc).fillna(prior).values
            sizes = g.value_counts()
            share_pure = d.groupby(g).label.mean().apply(lambda x: x in (0, 1))
            rows.append(dict(corpus=c, group=gname, seed=seed, auc=roc_auc_score(d.label, oof),
                             n_groups=int(g.nunique()),
                             share_items_in_single_class_groups=float(g.map(share_pure).mean()),
                             share_items_in_groups_ge5=float(g.map(sizes).ge(5).mean())))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- proxy A-distance
def t_pad(D, names=VERACITY + ["PHEME", "ClickbaitEN"], n=1500):
    rows = []
    for seed in SEEDS[:3]:
        samp = {c: corp(D, c).sample(min(n, len(corp(D, c))), random_state=seed) for c in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                da, db = samp[a], samp[b]
                k = min(len(da), len(db))
                da, db = da.sample(k, random_state=seed), db.sample(k, random_state=seed)
                texts = [str(t)[:5000] for t in pd.concat([da.text, db.text])]
                y = np.r_[np.zeros(k), np.ones(k)]
                X = TfidfVectorizer(max_features=20000, min_df=2, sublinear_tf=True).fit_transform(texts)
                oof = np.zeros(len(y))
                for tri, tei in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
                    oof[tei] = LogisticRegression(C=1, max_iter=2000, solver="liblinear").fit(
                        X[tri], y[tri]).predict(X[tei])
                err = (oof != y).mean()
                rows.append(dict(a=a, b=b, seed=seed, err=err, pad=2 * (1 - 2 * err)))
        print("pad seed", seed, flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- weights
def sparse_mad(X):
    """Exact mean |x_ij - mean_j| per column of a sparse matrix, without densifying."""
    X = X.tocsc()
    n = X.shape[0]
    mu = np.asarray(X.mean(0)).ravel()
    out = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        v = X.data[X.indptr[j]:X.indptr[j + 1]]
        out[j] = (np.abs(v - mu[j]).sum() + (n - len(v)) * mu[j]) / n
    return out


def t_weights(D, top=25):
    rows = []
    for c in VERACITY + ["PHEME", "ClickbaitEN"]:
        tr, va, _ = split(corp(D, c), 0)
        m = TfidfLinear("lr").fit(tr.text.tolist(), tr.label.values, va.text.tolist(), va.label.values)
        w, names = m.m.coef_.ravel(), m.feature_names()
        X = m.feat(tr.text.tolist())
        # linear SHAP magnitude: |w_j| * mean |x_ij - mean_j|  (TF-IDF, independent features)
        shap = np.abs(w) * sparse_mad(X)
        for direction, idx in [("fake", np.argsort(-w)[:top]), ("real", np.argsort(w)[:top])]:
            for r, j in enumerate(idx):
                rows.append(dict(corpus=c, direction=direction, rank=r + 1, feature=names[j],
                                 weight=float(w[j]), shap_mag=float(shap[j])))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- vocab / n-gram ablation
def t_vocab(D, sources=("ISOT", "WELFake")):
    rows = []
    cfgs = [("word", (1, 1), 1000), ("word", (1, 1), 5000), ("word", (1, 1), 20000), ("word", (1, 2), 100000),
            ("word", (1, 3), 200000), ("char_wb", (3, 5), 100000)]
    for seed in SEEDS[:3]:
        splits = {c: small_split(corp(D, c), seed) for c in ["ISOT", "WELFake", "PolitiFact", "GossipCop", "LIAR"]}
        for src in sources:
            tr, va, _ = splits[src]
            for an, ng, mf in cfgs:
                m = TfidfLinear("lr", ngram=ng, max_features=mf, analyzer=an).fit(
                    [t[:6000] for t in tr.text], tr.label.values, [t[:6000] for t in va.text], va.label.values)
                rows += eval_targets(m, src, splits, tr.label.mean(), f"{an}{ng}-{mf}",
                                     dict(seed=seed, analyzer=an, ngram=str(ng), max_features=mf))
        print("vocab seed", seed, flush=True)
    return pd.DataFrame(rows)


TESTS = dict(lc=t_lc, topk=t_topk, mask=t_mask, topic=t_topic, group=t_group, pad=t_pad,
             weights=t_weights, vocab=t_vocab)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default=",".join(TESTS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    D = load_corpora()
    import os
    for t in a.tests.split(","):
        if os.path.exists(f"{C.RESULTS}/mech_{t}.csv") and not a.force:
            print("exists, skipping", t, flush=True)
            continue
        df = TESTS[t](D)
        df.to_csv(f"{C.RESULTS}/mech_{t}.csv", index=False)
        print("wrote", t, len(df), flush=True)
