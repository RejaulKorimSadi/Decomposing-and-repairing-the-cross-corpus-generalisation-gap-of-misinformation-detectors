"""Train on each corpus, test on every corpus, for several models, regimes and seeds.

Regimes
  full      source training split capped at 8,000
  matched   source training split subsampled to n = 250 (equal size for every source)
  permuted  source train/val labels shuffled (sanity control: AUC should be ~0.5)

Target items that are near-duplicates of any item in the source corpus are excluded.
Writes results/transfer_<model>.csv (one row per model/regime/source/target/seed) and
artifacts/preds/<model>_<regime>_<src>_<tgt>_<seed>.npz (y, p) for bootstrap CIs.

Run: python src/transfer.py --models tfidf_lr,tfidf_svm,nbsvm,glove_lr,style_gb
"""
import os, argparse, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import config as C
from common import load_corpora, split, evaluate
from models import make

VERACITY = ["ISOT", "WELFake", "PolitiFact", "GossipCop", "LIAR", "Twitter15", "Twitter16"]
MATCHED_N = 250


def run(model_name, regimes, seeds, corpora, out_csv):
    D = load_corpora()
    D = D[D.corpus.isin(corpora)]
    os.makedirs(f"{C.ARTIFACTS}/preds", exist_ok=True)
    rows, done = [], set()
    if os.path.exists(out_csv):                       # resume: keep fully finished seeds
        old = pd.read_csv(out_csv)
        need = len(regimes) * len(corpora) * len(corpora)
        for s, g in old.groupby("seed"):
            if len(g) >= need:
                done.add(int(s))
        rows = old[old.seed.isin(done)].to_dict("records")
    for seed in seeds:
        if seed in done:
            print(f"[{model_name}] seed {seed} already done, skipping", flush=True)
            continue
        splits = {c: split(D[D.corpus == c], seed) for c in corpora}
        for regime in regimes:
            for src in corpora:
                tr, va, _ = splits[src]
                if regime == "matched":
                    tr = train_test_split(tr, train_size=min(MATCHED_N, len(tr) - 2),
                                          stratify=tr.label, random_state=seed)[0]
                ytr, yva = tr.label.values, va.label.values
                if regime == "permuted":
                    rng = np.random.default_rng(seed)
                    ytr, yva = rng.permutation(ytr), rng.permutation(yva)
                t0 = time.time()
                m = make(model_name).fit(tr.text.tolist(), ytr, va.text.tolist(), yva)
                pi_s = float(np.mean(ytr))
                for tgt in corpora:
                    te = splits[tgt][2]
                    if tgt != src:
                        te = te[~te.dup_in.fillna("").str.contains(fr"(?:^|;){src}(?:;|$)")]
                    p = m.predict_proba(te.text.tolist())
                    r = evaluate(te.label.values, p, pi_s)
                    r.update(model=model_name, regime=regime, source=src, target=tgt, seed=seed,
                             n_train=len(tr))
                    rows.append(r)
                    np.savez_compressed(f"{C.ARTIFACTS}/preds/{model_name}_{regime}_{src}_{tgt}_{seed}.npz",
                                        y=te.label.values, p=p)
                print(f"[{model_name}] seed={seed} {regime:<8} {src:<10} fit+eval {time.time()-t0:5.1f}s "
                      f"in-domain AUC={[x for x in rows if x['source']==src and x['target']==src and x['seed']==seed and x['regime']==regime][-1]['auc']:.3f}",
                      flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tfidf_lr")
    ap.add_argument("--regimes", default="full,matched,permuted")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--corpora", default=",".join(VERACITY))
    a = ap.parse_args()
    for mn in a.models.split(","):
        run(mn, a.regimes.split(","), [int(s) for s in a.seeds.split(",")], a.corpora.split(","),
            f"{C.RESULTS}/transfer_{mn}.csv")
