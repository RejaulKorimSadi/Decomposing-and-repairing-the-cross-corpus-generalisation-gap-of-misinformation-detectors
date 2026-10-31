"""Published generalisation methods for fake-news detection, re-implemented on one backbone and
tested on corpora they never saw (leave-one-corpus-out over the seven veracity corpora).

Backbone: frozen RoBERTa-base sentence vectors (src/roberta.py subset, first 600 characters).
For each held-out corpus T and seed 0-2, models train on the six other corpora (the 70% training
part of each corpus's subset), select on their validation parts, and are tested on T's test part
with near-duplicates of any training corpus removed. No method sees a label of T.

  erm_lr   pooled logistic regression (reference)
  erm_mlp  pooled two-layer MLP
  eann     MLP with a corpus discriminator behind a gradient-reversal layer, the event-adversarial
           idea of EANN (Wang et al., KDD 2018) / DANN (Ganin et al., 2016) with corpus as the domain
  mdfend   mixture of five MLP experts with a learned gate, the domain-gated design of MDFEND
           (Nan et al., CIKM 2021); the gate is fed the text vector because the target domain is
           unknown at test time
  endef    entity debiasing of ENDEF (Zhu et al., SIGIR 2022): an entity-only model (TF-IDF of the
           named entities found by spaCy) is trained jointly with the main MLP on the summed logits,
           with an auxiliary entity loss; at test time only the main model is used

Writes results/dg_loco.csv and artifacts/preds/<method>_loco_<T>_<seed>.npz
Run: python src/dg.py ents && python src/dg.py run
"""
import os, sys, copy, time
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
import config as C
from common import evaluate, Platt

VER = ["ISOT", "WELFake", "PolitiFact", "GossipCop", "LIAR", "Twitter15", "Twitter16"]
SEEDS = [0, 1, 2]
torch.set_num_threads(2)
ENTS = f"{C.ARTIFACTS}/subset_entities.parquet"


def ents():
    import spacy
    S = pd.read_parquet(f"{C.ARTIFACTS}/roberta_subset.parquet")
    S = S[S.corpus.isin(VER)]
    nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "attribute_ruler"])
    out = []
    for doc in nlp.pipe((str(t)[:600] for t in S.text), batch_size=64):
        out.append(" ; ".join(e.text.lower().replace(" ", "_") for e in doc.ents))
    pd.DataFrame(dict(row=S.row.values, ents=out)).to_parquet(ENTS, index=False)
    print("entities for", len(out), "texts; empty:", sum(1 for o in out if not o))


def splits(S, seed):
    sp = {}
    for c in VER:
        d = S[S.corpus == c]
        tr, te = train_test_split(d, test_size=0.2, stratify=d.label, random_state=seed)
        tr, va = train_test_split(tr, test_size=0.125, stratify=tr.label, random_state=seed)
        sp[c] = (tr, va, te)
    return sp


class GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def mlp(d_in, d_h=256, p=0.2):
    return torch.nn.Sequential(torch.nn.Linear(d_in, d_h), torch.nn.ReLU(), torch.nn.Dropout(p))


class Net(torch.nn.Module):
    def __init__(self, kind, d_in, n_dom, d_ent=0):
        super().__init__()
        self.kind = kind
        if kind == "mdfend":
            self.experts = torch.nn.ModuleList([mlp(d_in) for _ in range(5)])
            self.gate = torch.nn.Sequential(torch.nn.Linear(d_in, 64), torch.nn.ReLU(), torch.nn.Linear(64, 5))
        else:
            self.enc = mlp(d_in)
        self.head = torch.nn.Linear(256, 1)
        if kind == "eann":
            self.dom = torch.nn.Sequential(torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, n_dom))
        if kind == "endef":
            self.ent = torch.nn.Linear(d_ent, 1)

    def features(self, x):
        if self.kind == "mdfend":
            w = torch.softmax(self.gate(x), -1)
            return sum(w[:, k:k + 1] * e(x) for k, e in enumerate(self.experts))
        return self.enc(x)

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


def train_net(kind, X, y, dom, Xv, yv, E=None, seed=0, epochs=30, lam_max=0.5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_dom = int(dom.max()) + 1
    net = Net(kind, X.shape[1], n_dom, 0 if E is None else E.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt, dt = torch.tensor(X), torch.tensor(y, dtype=torch.float32), torch.tensor(dom)
    Et = None if E is None else torch.tensor(E)
    pos = float(y.mean())
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos))
    ce = torch.nn.CrossEntropyLoss()
    best, state, bad = -1, None, 0
    steps, k = epochs * int(np.ceil(len(X) / 64)), 0
    for ep in range(epochs):
        net.train()
        for b in np.array_split(np.random.permutation(len(X)), int(np.ceil(len(X) / 64))):
            k += 1
            h = net.features(Xt[b])
            z = net.head(h).squeeze(-1)
            if kind == "endef":
                ze = net.ent(Et[b]).squeeze(-1)
                loss = bce(z + ze, yt[b]) + 0.2 * bce(ze, yt[b])
            else:
                loss = bce(z, yt[b])
            if kind == "eann":
                lam = lam_max * (2 / (1 + np.exp(-10 * k / steps)) - 1)
                loss = loss + ce(net.dom(GradRev.apply(h, lam)), dt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            a = roc_auc_score(yv, net(torch.tensor(Xv)).numpy())
        if a > best + 1e-4:
            best, state, bad = a, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= 5:
                break
    net.load_state_dict(state)
    net.eval()
    return lambda Z: net(torch.tensor(Z)).detach().numpy()


def run():
    S = pd.read_parquet(f"{C.ARTIFACTS}/roberta_subset.parquet")
    S = S[S.corpus.isin(VER)]
    Emb = np.load(f"{C.ARTIFACTS}/roberta_emb.npy").astype(np.float32)
    ent = pd.read_parquet(ENTS).set_index("row").ents
    rows = []
    for seed in SEEDS:
        sp = splits(S, seed)
        for T in VER:
            t0 = time.time()
            src = [c for c in VER if c != T]
            tr = pd.concat([sp[c][0] for c in src])
            va = pd.concat([sp[c][1] for c in src])
            te = sp[T][2]
            pat = "|".join(fr"(?:^|;){c}(?:;|$)" for c in src)
            te = te[~te.dup_in.fillna("").str.contains(pat)]
            mu, sd = Emb[tr.row].mean(0), Emb[tr.row].std(0) + 1e-6
            f = lambda d: (Emb[d.row] - mu) / sd
            X, Xv, Xt = f(tr), f(va), f(te)
            y, yv, yt = tr.label.values, va.label.values, te.label.values
            dom = pd.Categorical(tr.corpus, categories=src).codes.astype(np.int64)
            vec = TfidfVectorizer(min_df=2, max_features=20000, token_pattern=r"[^ ;]+")
            Ein = vec.fit_transform(ent.loc[tr.row].values).toarray().astype(np.float32)
            scorers = {}
            fits = [LogisticRegression(C=c, max_iter=3000).fit(X, y) for c in (0.0025, 0.01, 0.04)]
            lr = max(fits, key=lambda m: roc_auc_score(yv, m.decision_function(Xv)))
            scorers["erm_lr"] = lr.decision_function
            scorers["erm_mlp"] = train_net("mlp", X, y, dom, Xv, yv, seed=seed)
            scorers["eann"] = train_net("eann", X, y, dom, Xv, yv, seed=seed)
            scorers["mdfend"] = train_net("mdfend", X, y, dom, Xv, yv, seed=seed)
            scorers["endef"] = train_net("endef", X, y, dom, Xv, yv, E=Ein, seed=seed)
            for name, fn in scorers.items():
                cal = Platt().fit(fn(Xv), yv)
                p = cal(fn(Xt))
                r = evaluate(yt, p, float(y.mean()))
                r.update(model=name, target=T, seed=seed, n_train=len(tr), val_auc=roc_auc_score(yv, fn(Xv)))
                rows.append(r)
                np.savez_compressed(f"{C.ARTIFACTS}/preds/{name}_loco_{T}_{seed}.npz", y=yt, p=p)
            print(f"seed={seed} T={T:<10} " + " ".join(f"{x['model']}={x['auc']:.3f}" for x in rows[-5:]) +
                  f" {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(f"{C.RESULTS}/dg_loco.csv", index=False)


if __name__ == "__main__":
    {"ents": ents, "run": run}[sys.argv[1]]()
