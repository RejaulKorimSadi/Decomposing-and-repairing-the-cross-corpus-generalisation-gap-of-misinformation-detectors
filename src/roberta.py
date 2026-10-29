"""RoBERTa-base representations (weights as packaged in spaCy en_core_web_trf 3.8.0).

  embed    mean-pooled last-layer RoBERTa states for a stratified subset (<= 1,500 items
           per corpus, seed 0); text truncated to 600 characters
  matrix   logistic regression on those embeddings, train-on-one / test-on-all, 5 seeds
           (70/10/20 splits of the subset), same metrics and near-duplicate exclusion
Writes artifacts/roberta_subset.parquet, artifacts/roberta_emb.npy,
       results/transfer_roberta_lr.csv, artifacts/preds/roberta_lr_full_*.npz
Run: python src/roberta.py embed && python src/roberta.py matrix
"""
import sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import config as C
from common import load_corpora, evaluate
from models import DenseLR, TfidfLinear

VER = ["ISOT", "WELFake", "PolitiFact", "GossipCop", "LIAR", "Twitter15", "Twitter16"]
EXTRA = ["ClickbaitEN"]
PER, MAXCH = 1500, 600


def subset():
    D = load_corpora()
    parts = []
    for c in VER + EXTRA + ["BanglaBait"]:
        d = D[D.corpus == c]
        if c == "BanglaBait":
            d = d.assign(corpus="BanglaBait-EN", text=d.title_en.fillna(""))
        if len(d) > PER:
            d = train_test_split(d, train_size=PER, stratify=d.label, random_state=0)[0]
        parts.append(d)
    S = pd.concat(parts, ignore_index=True)
    S["row"] = np.arange(len(S))
    S.to_parquet(f"{C.ARTIFACTS}/roberta_subset.parquet", index=False)
    return S


def embed():
    import os, spacy, torch
    torch.set_num_threads(1)
    S = subset()
    nlp = spacy.load("en_core_web_trf", exclude=["tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
    path = f"{C.ARTIFACTS}/roberta_emb.npy"
    out = np.load(path) if os.path.exists(path) else np.zeros((len(S), 768), np.float32)
    todo = np.where(~out.any(1))[0]                              # resume: rows not yet embedded
    t0 = time.time()
    texts = [str(S.text.iloc[i])[:MAXCH] for i in todo]
    for k, (i, doc) in enumerate(zip(todo, nlp.pipe(texts, batch_size=8))):
        out[i] = np.asarray(doc._.trf_data.last_hidden_layer_state.dataXd).mean(0)
        if k % 500 == 0:
            print(f"{k}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
            np.save(path, out)
    np.save(path, out)


def matrix():
    S = pd.read_parquet(f"{C.ARTIFACTS}/roberta_subset.parquet")
    E = np.load(f"{C.ARTIFACTS}/roberta_emb.npy")
    names = VER + EXTRA + ["BanglaBait-EN"]
    rows = []
    for seed in C.SEEDS:
        sp = {}
        for c in names:
            d = S[S.corpus == c]
            tr, te = train_test_split(d, test_size=0.2, stratify=d.label, random_state=seed)
            tr, va = train_test_split(tr, test_size=0.125, stratify=tr.label, random_state=seed)
            sp[c] = (tr, va, te)
        for src in names:
            tr, va, _ = sp[src]
            fitted = {"roberta_lr": DenseLR(E).fit(tr.row.values, tr.label.values, va.row.values, va.label.values),
                      "tfidf_lr_sub": TfidfLinear("lr").fit(tr.text.tolist(), tr.label.values,
                                                            va.text.tolist(), va.label.values)}
            for tgt in names:
                # cross-construct cells (clickbait <-> veracity) are skipped
                if (src in VER) != (tgt in VER):
                    continue
                te = sp[tgt][2]
                if tgt != src:
                    te = te[~te.dup_in.fillna("").str.contains(fr"(?:^|;){src}(?:;|$)")]
                for mname, m in fitted.items():
                    p = m.predict_proba(te.row.values if mname == "roberta_lr" else te.text.tolist())
                    r = evaluate(te.label.values, p, tr.label.mean())
                    r.update(model=mname, regime="full", source=src, target=tgt, seed=seed, n_train=len(tr))
                    rows.append(r)
                    np.savez_compressed(f"{C.ARTIFACTS}/preds/{mname}_full_{src}_{tgt}_{seed}.npz",
                                        y=te.label.values, p=p)
        print("roberta matrix seed", seed, flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(f"{C.RESULTS}/subset_roberta_vs_tfidf.csv", index=False)
    R[R.model == "roberta_lr"].to_csv(f"{C.RESULTS}/transfer_roberta_lr.csv", index=False)


if __name__ == "__main__":
    {"embed": embed, "matrix": matrix}[sys.argv[1]]()
