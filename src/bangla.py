"""Bengali clickbait (BanglaBait): replication, source confound, cross-lingual transfer.

  official   official 70/10/20 split, raw (as released) and after our near-duplicate removal;
             clickbait-class F1/P/R and accuracy, the metrics reported by Mahtab et al. (2023)
  seeds      5 random stratified re-splits of the deduplicated corpus (mean, s.d.)
  loso       leave-one-source(domain)-out: train on all other publisher domains
  xling      translate-test: English clickbait (Chakraborty et al., 2016) <-> BanglaBait
             machine-translated English titles
Writes results/bangla_*.csv
Run: python src/bangla.py
"""
import os
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import config as C
from common import load_corpora, split, evaluate
from models import TfidfLinear


def char_lr():
    return TfidfLinear("lr", ngram=(3, 5), analyzer="char_wb", max_features=200000)


def char_svm():
    return TfidfLinear("svm", ngram=(3, 5), analyzer="char_wb", max_features=200000)


def word_lr():
    return TfidfLinear("lr", ngram=(1, 2), analyzer="word", max_features=100000)


MODELS = {"char_lr": char_lr, "char_svm": char_svm, "word_lr": word_lr}


def scores(y, p, t=0.5):
    yhat = (p >= t).astype(int)
    return dict(f1_clickbait=f1_score(y, yhat), precision=precision_score(y, yhat, zero_division=0),
                recall=recall_score(y, yhat), accuracy=accuracy_score(y, yhat),
                macro_f1=f1_score(y, yhat, average="macro"), auc=roc_auc_score(y, p))


def official():
    rows = []
    raw = {s: pd.read_csv(os.path.join(C.BANGLA, f"{s}.csv")) for s in ["train", "dev", "test"]}
    D = load_corpora()
    dd = D[D.corpus == "BanglaBait"]
    ded = {s: dd[dd.split == s] for s in ["train", "dev", "test"]}
    for version, parts in [("as_released", {s: raw[s].assign(text=raw[s].title.fillna("")) for s in raw}),
                           ("deduplicated", {s: ded[s].assign(text=ded[s].title) for s in ded})]:
        tr, va, te = parts["train"], parts["dev"], parts["test"]
        for field in ["title", "title+content"]:
            def txt(x):
                return (x.text if field == "title" else
                        (x.title.fillna("") + " " + x.get("content", x.get("body", "")).fillna("")))
            for name, mk in MODELS.items():
                m = mk().fit(txt(tr).tolist(), tr.label.values, txt(va).tolist(), va.label.values)
                p = m.predict_proba(txt(te).tolist())
                r = scores(te.label.values, p)
                r.update(version=version, field=field, model=name, n_train=len(tr), n_test=len(te))
                rows.append(r)
                print(version, field, name, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)},
                      flush=True)
    return pd.DataFrame(rows)


def seeds():
    D = load_corpora()
    d = D[D.corpus == "BanglaBait"].assign(text=lambda x: x.title)
    rows = []
    for seed in C.SEEDS:
        tr, va, te = split(d, seed, cap=False)
        for name, mk in MODELS.items():
            m = mk().fit(tr.text.tolist(), tr.label.values, va.text.tolist(), va.label.values)
            r = scores(te.label.values, m.predict_proba(te.text.tolist()))
            r.update(model=name, seed=seed)
            rows.append(r)
    return pd.DataFrame(rows)


def loso():
    D = load_corpora()
    d = D[D.corpus == "BanglaBait"].assign(text=lambda x: x.title)
    size = d.group.map(d.group.value_counts())
    d = d.assign(group=np.where(size >= 100, d.group, "other (<100 items each)"))
    rows = []
    for dom, g in d.groupby("group"):
        rest = d[d.group != dom]
        tr, va = train_test_split(rest, test_size=0.125, stratify=rest.label, random_state=0)
        m = char_lr().fit(tr.text.tolist(), tr.label.values, va.text.tolist(), va.label.values)
        p = m.predict_proba(g.text.tolist())
        yhat = (p >= 0.5).astype(int)
        rows.append(dict(domain=dom, n=len(g), clickbait_rate=float(g.label.mean()),
                         accuracy=accuracy_score(g.label, yhat),
                         auc=roc_auc_score(g.label, p) if g.label.nunique() == 2 else np.nan,
                         pred_rate=float(yhat.mean())))
    # matched random hold-out of equal sizes, for comparison
    return pd.DataFrame(rows)


def xling():
    D = load_corpora()
    en = D[D.corpus == "ClickbaitEN"]
    bb = D[D.corpus == "BanglaBait"].assign(text=lambda x: x.title_en.fillna(""))
    bn = D[D.corpus == "BanglaBait"].assign(text=lambda x: x.title)
    rows = []
    for seed in C.SEEDS:
        s_en, s_bb, s_bn = split(en, seed), split(bb, seed), split(bn, seed)
        for src_name, (tr, va, _) in [("ClickbaitEN", s_en), ("BanglaBait-EN", s_bb)]:
            m = TfidfLinear("lr").fit(tr.text.tolist(), tr.label.values, va.text.tolist(), va.label.values)
            for tgt_name, (_, _, te) in [("ClickbaitEN", s_en), ("BanglaBait-EN", s_bb)]:
                r = evaluate(te.label.values, m.predict_proba(te.text.tolist()), tr.label.mean())
                r.update(source=src_name, target=tgt_name, seed=seed, lang="en")
                rows.append(r)
        tr, va, te = s_bn
        m = char_lr().fit(tr.text.tolist(), tr.label.values, va.text.tolist(), va.label.values)
        r = evaluate(te.label.values, m.predict_proba(te.text.tolist()), tr.label.mean())
        r.update(source="BanglaBait-BN", target="BanglaBait-BN", seed=seed, lang="bn")
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for name, fn in [("official", official), ("seeds", seeds), ("loso", loso), ("xling", xling)]:
        df = fn()
        df.to_csv(f"{C.RESULTS}/bangla_{name}.csv", index=False)
        print("wrote", name, len(df), flush=True)
