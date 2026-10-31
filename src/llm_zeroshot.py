"""Zero-shot instruction-tuned LLM baseline on the seven veracity corpora.

The model is asked whether an item is real or fake and the score is the log-odds of the next
token being "fake" rather than "real" (case variants pooled). No training data are used, so the
same model is applied to every corpus; the items are the test items of the fine-tuning
experiment (finetune.plan(), seeds 0-2), which makes the comparison with roberta_ft and the
paired TF-IDF model item-for-item.

Model file: a GGUF checkpoint given by $LLM_GGUF (default: Qwen3-0.6B Q4_K_M), loaded with
transformers (de-quantised to float32, CPU). Thinking mode is disabled for Qwen3.
Outputs: artifacts/llm_scores_<tag>.csv (id, score), results/llm_zeroshot_<tag>.csv
Run: LLM_GGUF=/path/model.gguf LLM_TAG=qwen3_0.6b python src/llm_zeroshot.py
"""
import os, time
import numpy as np, pandas as pd, torch
import config as C
from finetune import plan, VER, SEEDS
from common import evaluate

GGUF = os.environ.get("LLM_GGUF", os.path.join(C.ROOT, "models", "Qwen3-0.6B-Q4_K_M.gguf"))
TAG = os.environ.get("LLM_TAG", "qwen3_0.6b")
UNIT = {"ISOT": "news article", "WELFake": "news article", "PolitiFact": "news headline",
        "GossipCop": "news headline", "LIAR": "claim", "Twitter15": "tweet", "Twitter16": "tweet"}
PROMPT = ("Is the following {unit} real or fake? Answer with one word: real or fake.\n\n"
          "{unit_cap}: {text}")
MAXCH = 600
torch.set_num_threads(2)


def load():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    d, f = os.path.dirname(GGUF), os.path.basename(GGUF)
    tok = AutoTokenizer.from_pretrained(d, gguf_file=f)
    m = AutoModelForCausalLM.from_pretrained(d, gguf_file=f, dtype=torch.float32).eval()
    ids = lambda ws: sorted({tok.encode(w, add_special_tokens=False)[0] for w in ws})
    return tok, m, ids(["fake", "Fake", " fake", " Fake", "FAKE"]), ids(["real", "Real", " real", " Real", "REAL"])


def score_items(items):
    out_f = f"{C.ARTIFACTS}/llm_scores_{TAG}.csv"
    done = pd.read_csv(out_f, dtype={"id": str}) if os.path.exists(out_f) else pd.DataFrame(columns=["id", "score"])
    todo = items[~items.id.isin(set(done.id))]
    if not len(todo):
        return done
    tok, m, fake_ids, real_ids = load()
    rows, t0 = [], time.time()
    for k, r in enumerate(todo.itertuples()):
        unit = UNIT[r.corpus]
        msg = PROMPT.format(unit=unit, unit_cap=unit[0].upper() + unit[1:], text=str(r.text)[:MAXCH].strip())
        p = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
        x = tok(p, return_tensors="pt")
        with torch.no_grad():
            lo = torch.log_softmax(m(**x).logits[0, -1].float(), -1)
        s = float(torch.logsumexp(lo[fake_ids], 0) - torch.logsumexp(lo[real_ids], 0))
        rows.append(dict(id=r.id, score=s))
        if (k + 1) % 200 == 0 or k + 1 == len(todo):
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_csv(out_f, index=False)
            rows = []
            print(f"{k + 1}/{len(todo)} scored, {time.time() - t0:.0f}s", flush=True)
    return done


def main():
    P = plan()
    items = pd.concat([P[(c, s)][2] for c in VER for s in SEEDS]).drop_duplicates("id")
    S = score_items(items[["id", "corpus", "text"]]).drop_duplicates("id").set_index("id").score.astype(float)
    rows = []
    for seed in SEEDS:
        for tgt in VER:
            te = P[(tgt, seed)][2]
            z = S.loc[te.id].astype(float).values
            p = 1 / (1 + np.exp(-z))                       # model's own answer <=> p >= 0.5
            r = evaluate(te.label.values, p, 0.5)
            q = np.quantile(p, 0.5)
            from sklearn.metrics import f1_score
            r["half_macro_f1"] = f1_score(te.label.values, (p >= q).astype(int), average="macro", zero_division=0)
            r.update(model=f"llm_{TAG}", regime="zeroshot", source="none", target=tgt, seed=seed)
            rows.append(r)
            np.savez_compressed(f"{C.ARTIFACTS}/preds/llm_{TAG}_zeroshot_none_{tgt}_{seed}.npz", y=te.label.values, p=p)
    R = pd.DataFrame(rows)
    R.to_csv(f"{C.RESULTS}/llm_zeroshot_{TAG}.csv", index=False)
    print(R.groupby("target")[["auc", "def_macro_f1", "pm_macro_f1", "oracle_macro_f1", "def_ppr", "prior"]].mean().round(3))


if __name__ == "__main__":
    main()
