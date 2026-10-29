"""Render README.md from README.template.md and results/key_numbers.json.
Placeholders look like {{MacroName}}; every number in the README therefore matches the paper.
Run: python src/make_readme.py
"""
import json, re, os
import config as C

K = json.load(open(f"{C.RESULTS}/key_numbers.json"))
tpl = open(os.path.join(C.ROOT, "README.template.md"), encoding="utf8").read()
missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", tpl)) - set(K))
if missing:
    raise SystemExit(f"missing numbers: {missing}")
out = re.sub(r"\{\{(\w+)\}\}", lambda m: K[m.group(1)], tpl)
open(os.path.join(C.ROOT, "README.md"), "w", encoding="utf8").write(out)
print("README.md written")
