"""Label-direction probe: count source signatures per RAW label value in ISOT and WELFake.

A reliable outlet's signature (e.g. a "- The New York Times" title suffix, a "(Reuters)"
dateline) must sit under the 'real' label; a fake-site template ("Featured image via",
"BREAKING" titles, "21st Century Wire") under the 'fake' label. Run on the raw files, before
harmonisation. The working copies in $DATA_DIR are always probed; the original releases are
probed too when ISOT_RAW (folder with Fake.csv and True.csv) and WELFAKE_RAW (WELFake_Dataset.csv)
are set. Writes results/label_probe.csv
Run: python src/label_probe.py
"""
import os
import pandas as pd
import config as C

PROBES = {
    "title ends '- The New York Times' (reliable)": ("title", r"- The New York Times\s*$", "real"),
    "title ends '- Breitbart' (Kaggle-reliable split)": ("title", r"- Breitbart\s*$", "real"),
    "text contains '(Reuters)' (reliable)": ("text", r"\(Reuters\)", "real"),
    "text contains 'Featured image via' (fake-site template)": ("text", r"Featured image via", "fake"),
    "title contains 'BREAKING' (fake-site style)": ("title", r"BREAKING", "fake"),
    "text contains '21st Century Wire' (fake site)": ("text", r"21st Century Wire", "fake"),
    "text contains 'Activist Post' (fake site)": ("text", r"Activist Post", "fake"),
}
rows = []
files = [("ISOT", C.ISOT), ("WELFake", C.WELFAKE)]
if os.environ.get("WELFAKE_RAW"):
    files.append(("WELFake release", os.environ["WELFAKE_RAW"]))
if os.environ.get("ISOT_RAW"):
    files.append(("ISOT release", None))
for name, path in files:
    if name == "ISOT release":            # the ISOT release has no label column: file name gives the class
        r = os.environ["ISOT_RAW"]
        d = pd.concat([pd.read_csv(os.path.join(r, "Fake.csv")).assign(label=1),
                       pd.read_csv(os.path.join(r, "True.csv")).assign(label=0)], ignore_index=True)
    else:
        d = pd.read_csv(path)
    d["title"], d["text"] = d.title.fillna(""), d.text.fillna("")
    for probe, (col, pat, expect) in PROBES.items():
        m = d[col].str.contains(pat, case=False, regex=True)
        vc = d.loc[m, "label"].value_counts()
        rows.append(dict(corpus=name, probe=probe, expected_class=expect, n=int(m.sum()),
                         raw_label_0=int(vc.get(0, 0)), raw_label_1=int(vc.get(1, 0))))
R = pd.DataFrame(rows)
R.to_csv(f"{C.RESULTS}/label_probe.csv", index=False)
print(R.to_string(index=False))
