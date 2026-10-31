"""Reconstruct the two cleaned article files from the original releases and report every step.

Needs the original releases:
  ISOT_RAW     folder with Fake.csv and True.csv (ISOT Fake News Dataset, University of Victoria)
  WELFAKE_RAW  WELFake_Dataset.csv (Zenodo 10.5281/zenodo.4561253)
Compares them with the working copies in $DATA_DIR and writes results/provenance.json.
Run: ISOT_RAW=... WELFAKE_RAW=... python src/provenance.py
"""
import os, json, hashlib
import pandas as pd
import config as C

s = lambda x: x.fillna("").astype(str).str.strip()
key = lambda d: s(d.title) + "||" + s(d.text)
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
out = {}

r = os.environ["ISOT_RAW"]
iso = pd.concat([pd.read_csv(os.path.join(r, "Fake.csv")).assign(label=1),
                 pd.read_csv(os.path.join(r, "True.csv")).assign(label=0)], ignore_index=True)
ic = pd.read_csv(C.ISOT)
idd = iso[~key(iso).duplicated()]
out["isot"] = dict(release_rows=len(iso), release_fake=int(iso.label.sum()), release_real=int((iso.label == 0).sum()),
                   working_rows=len(ic), working_rows_in_release=float(key(ic).isin(set(key(iso))).mean()),
                   working_label_matches_release=float(ic.merge(idd[["title", "text", "label"]], on=["title", "text"],
                                                                suffixes=("", "_r")).pipe(lambda m: (m.label == m.label_r).mean())),
                   exact_title_text_duplicates=len(iso) - len(idd), after_exact_dedup=len(idd),
                   further_removed=len(idd) - len(ic),
                   further_removed_sharing_title_with_kept=int(s(idd[~key(idd).isin(set(key(ic)))].title).isin(set(s(ic.title))).sum()),
                   further_removed_real=int((idd[~key(idd).isin(set(key(ic)))].label == 0).sum()),
                   working_empty_texts=int((s(ic.text) == "").sum()))

p = os.environ["WELFAKE_RAW"]
w = pd.read_csv(p)
wc = pd.read_csv(C.WELFAKE)
nyt, fiv = s(w.title).str.endswith("- The New York Times"), s(w.text).str.contains("Featured image via")
reu = s(w.text).str.contains(r"\(Reuters\)")
wd = w[~key(w).duplicated()]
it, iti = set(s(iso.text)) - {""}, set(s(iso.title)) - {""}
a = wd[~s(wd.text).isin(it)]
b = a[~s(a.title).isin(iti)]
m = wc.merge(wd[["title", "text", "label"]], on=["title", "text"], suffixes=("", "_r"))
out["welfake"] = dict(release_md5=md5(p), release_rows=len(w), release_label1=int(w.label.sum()),
                      release_label0=int((w.label == 0).sum()),
                      nyt_label0=int((w[nyt].label == 0).sum()), nyt_n=int(nyt.sum()),
                      featured_label1=int((w[fiv].label == 1).sum()), featured_n=int(fiv.sum()),
                      reuters_label0=int((w[reu].label == 0).sum()), reuters_n=int(reu.sum()),
                      working_rows=len(wc), working_rows_in_release=float(key(wc).isin(set(key(w))).mean()),
                      working_label_is_inverted=float((m.label == 1 - m.label_r).mean()),
                      exact_title_text_duplicates=len(w) - len(wd), after_exact_dedup=len(wd),
                      text_shared_with_isot=len(wd) - len(a), title_shared_with_isot=len(a) - len(b),
                      after_isot_removal=len(b), unexplained_further_removed=len(b) - len(wc),
                      working_rows_not_in_reconstruction=int((~key(wc).isin(set(key(b)))).sum()))
json.dump(out, open(f"{C.RESULTS}/provenance.json", "w"), indent=1)
print(json.dumps(out, indent=1))
