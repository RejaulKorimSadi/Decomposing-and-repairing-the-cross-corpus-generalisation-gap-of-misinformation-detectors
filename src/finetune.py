"""Fine-tuned RoBERTa-base (top four of twelve layers + classification head).

Weights are those packaged in spaCy's en_core_web_trf 3.8.0. To make fine-tuning feasible on a
CPU, the embeddings and the lower eight layers are frozen and their outputs are computed once
per text and cached; the upper four layers and a mean-pooling linear head are trained.

Design (per veracity corpus and seed 0-2, using the same 70/10/20 splits as transfer.py):
  train 800 items, validation 200, test 300 per target corpus (all stratified)
  max 128 word pieces for articles, 64 otherwise; AdamW, lr 3e-5 (layers) / 1e-3 (head),
  batch 16, 3 epochs, epoch selected by validation ROC-AUC, Platt scaling on validation.
Outputs: results/transfer_roberta_ft.csv, artifacts/preds/roberta_ft_full_<src>_<tgt>_<seed>.npz
Run: python src/finetune.py cache && python src/finetune.py train
"""
import os, sys, copy, time, math
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import config as C
from common import load_corpora, split, evaluate, Platt

VER = ["ISOT", "WELFake", "PolitiFact", "GossipCop", "LIAR", "Twitter15", "Twitter16"]
SEEDS = [0, 1, 2]
N_TR, N_VA, N_TE = 800, 200, 300
FROZEN = 8
CACHE = f"{C.ARTIFACTS}/ft_cache"
torch.set_num_threads(2)


def maxlen(c):
    return 128 if c in ("ISOT", "WELFake") else 64


def sub(df, n, seed):
    return df if len(df) <= n else train_test_split(df, train_size=n, stratify=df.label, random_state=seed)[0]


def plan():
    """Items each (corpus, seed) needs; identical splits to transfer.py, then subsampled."""
    D = load_corpora()
    P = {}
    for c in VER:
        d = D[D.corpus == c]
        for s in SEEDS:
            tr, va, te = split(d, s)
            P[(c, s)] = (sub(tr, N_TR, s), sub(va, N_VA, s), sub(te, N_TE, s))
    return P


def encoder():
    import spacy
    nlp = spacy.load("en_core_web_trf", exclude=["tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
    m = nlp.get_pipe("transformer").model
    proc = [n for n in m.walk() if n.name == "byte_bpe_encoder"][0].attrs["byte_bpe_processor"]
    ce = [n for n in m.walk() if n.name == "pytorch"][0].shims[0]._model.curated_encoder
    ce.eval()
    for p in ce.parameters():
        p.requires_grad_(False)
    return proc, ce


def tokenize(proc, texts, L):
    ids = np.ones((len(texts), L), np.int64)               # pad id 1
    mask = np.zeros((len(texts), L), bool)
    for i, t in enumerate(texts):
        x = [0] + proc.encode_as_ids(str(t)[:1500])[: L - 2] + [2]
        ids[i, :len(x)], mask[i, :len(x)] = x, True
    return ids, mask


def cache():
    from curated_transformers.models.attention import AttentionMask
    proc, ce = encoder()
    os.makedirs(CACHE, exist_ok=True)
    P = plan()
    for c in VER:
        if os.path.exists(f"{CACHE}/{c}_ids.npy"):
            continue
        items = pd.concat([x for s in SEEDS for x in P[(c, s)]]).drop_duplicates("id")
        L = maxlen(c)
        ids, mask = tokenize(proc, items.text.tolist(), L)
        H = np.lib.format.open_memmap(f"{CACHE}/{c}_h.npy", "w+", np.float16, (len(items), L, 768))
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(items), 32):
                x = torch.tensor(ids[i:i + 32])
                am = AttentionMask(torch.tensor(mask[i:i + 32]))
                h = ce.embeddings(x, None, None)
                for layer in ce.layers[:FROZEN]:
                    h = layer(h, am)
                H[i:i + 32] = h.numpy().astype(np.float16)
        H.flush()
        np.save(f"{CACHE}/{c}_mask.npy", mask)
        np.save(f"{CACHE}/{c}_ids.npy", items.id.values.astype(str))
        print(f"cached {c}: {len(items)} items in {time.time() - t0:.0f}s", flush=True)


class Head(torch.nn.Module):
    def __init__(self, top):
        super().__init__()
        self.top = top
        self.drop = torch.nn.Dropout(0.1)
        self.out = torch.nn.Linear(768, 1)

    def forward(self, h, mask):
        from curated_transformers.models.attention import AttentionMask
        am = AttentionMask(mask)
        for layer in self.top:
            h = layer(h, am)
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.out(self.drop(pooled)).squeeze(-1)


def load_cache(c):
    H = np.load(f"{CACHE}/{c}_h.npy", mmap_mode="r")
    M = np.load(f"{CACHE}/{c}_mask.npy")
    idx = {str(k): i for i, k in enumerate(np.load(f"{CACHE}/{c}_ids.npy", allow_pickle=True))}
    return H, M, idx


def batches(H, M, rows, bs):
    for i in range(0, len(rows), bs):
        r = np.sort(rows[i:i + bs])
        yield torch.tensor(np.asarray(H[r], np.float32)), torch.tensor(M[r]), r


def scores(model, H, M, rows):
    model.eval()
    out = np.zeros(len(rows))
    order = np.argsort(rows)
    srt = np.asarray(rows)[order]
    res = []
    with torch.no_grad():
        for h, m, _ in batches(H, M, srt, 64):
            res.append(model(h, m).numpy())
    out[order] = np.concatenate(res)
    return out


def train():
    proc, ce = encoder()
    P = plan()
    caches = {c: load_cache(c) for c in VER}
    out_csv = f"{C.RESULTS}/transfer_roberta_ft.csv"
    rows, done = [], set()
    if os.path.exists(out_csv):
        old = pd.read_csv(out_csv)
        done = {(r.source, int(r.seed)) for r in old.drop_duplicates(["source", "seed"]).itertuples()
                if len(old[(old.source == r.source) & (old.seed == r.seed)]) == len(VER)}
        rows = old[[(s, int(e)) in done for s, e in zip(old.source, old.seed)]].to_dict("records")
    for seed in SEEDS:
        for src in VER:
            if (src, seed) in done:
                continue
            torch.manual_seed(seed)
            np.random.seed(seed)
            H, M, idx = caches[src]
            tr, va, _ = P[(src, seed)]
            rtr = np.array([idx[i] for i in tr.id]); ytr = tr.label.values.astype(np.float32)
            rva = np.array([idx[i] for i in va.id]); yva = va.label.values
            model = Head(copy.deepcopy(ce.layers[FROZEN:]))
            for p in model.parameters():
                p.requires_grad_(True)
            opt = torch.optim.AdamW([{"params": model.top.parameters(), "lr": 3e-5},
                                     {"params": model.out.parameters(), "lr": 1e-3}], weight_decay=0.01)
            steps = 3 * math.ceil(len(rtr) / 16)
            sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: max(0.0, 1 - k / steps))
            pos = float(ytr.mean())
            lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos))
            best, best_state, t0 = -1, None, time.time()
            for ep in range(3):
                model.train()
                perm = np.random.permutation(len(rtr))
                for i in range(0, len(perm), 16):
                    b = perm[i:i + 16]
                    r = rtr[b]
                    o = np.argsort(r)
                    h = torch.tensor(np.asarray(H[r[o]], np.float32))
                    m = torch.tensor(M[r[o]])
                    loss = lossf(model(h, m), torch.tensor(ytr[b][o]))
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sch.step()
                auc = roc_auc_score(yva, scores(model, H, M, rva))
                if auc > best:
                    best, best_state = auc, copy.deepcopy(model.state_dict())
            model.load_state_dict(best_state)
            cal = Platt().fit(scores(model, H, M, rva), yva)
            for tgt in VER:
                Ht, Mt, it = caches[tgt]
                te = P[(tgt, seed)][2]
                if tgt != src:
                    te = te[~te.dup_in.fillna("").str.contains(fr"(?:^|;){src}(?:;|$)")]
                p = cal(scores(model, Ht, Mt, np.array([it[i] for i in te.id])))
                r = evaluate(te.label.values, p, float(ytr.mean()))
                r.update(model="roberta_ft", regime="full", source=src, target=tgt, seed=seed, n_train=len(tr),
                         val_auc=best)
                rows.append(r)
                np.savez_compressed(f"{C.ARTIFACTS}/preds/roberta_ft_full_{src}_{tgt}_{seed}.npz",
                                    y=te.label.values, p=p)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            ind = [x for x in rows if x["source"] == src and x["target"] == src and x["seed"] == seed][0]
            print(f"seed={seed} {src:<10} val_auc={best:.3f} in-domain auc={ind['auc']:.3f} "
                  f"{time.time() - t0:.0f}s", flush=True)


def tfidf():
    """Paired baseline: TF-IDF + LR on exactly the same train/validation/test items, first 600 characters
    (roughly what 128 word pieces cover), so the fine-tuned model is compared on equal terms."""
    from models import make
    P = plan()
    rows = []
    for seed in SEEDS:
        for src in VER:
            tr, va, _ = P[(src, seed)]
            m = make("tfidf_lr").fit([t[:600] for t in tr.text], tr.label.values,
                                     [t[:600] for t in va.text], va.label.values)
            for tgt in VER:
                te = P[(tgt, seed)][2]
                if tgt != src:
                    te = te[~te.dup_in.fillna("").str.contains(fr"(?:^|;){src}(?:;|$)")]
                p = m.predict_proba([t[:600] for t in te.text])
                r = evaluate(te.label.values, p, float(tr.label.mean()))
                r.update(model="tfidf_lr_ftsub", regime="full", source=src, target=tgt, seed=seed, n_train=len(tr))
                rows.append(r)
                np.savez_compressed(f"{C.ARTIFACTS}/preds/tfidf_lr_ftsub_full_{src}_{tgt}_{seed}.npz",
                                    y=te.label.values, p=p)
        print("tfidf seed", seed, "done", flush=True)
    pd.DataFrame(rows).to_csv(f"{C.RESULTS}/transfer_tfidf_lr_ftsub.csv", index=False)


if __name__ == "__main__":
    {"cache": cache, "train": train, "tfidf": tfidf}[sys.argv[1]]()
