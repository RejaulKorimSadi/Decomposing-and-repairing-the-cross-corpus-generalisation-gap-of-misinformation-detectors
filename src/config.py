"""Paths and global settings. Point DATA_DIR at the folder described in DATA.md."""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))
ARTIFACTS = os.environ.get("ARTIFACTS", os.path.join(ROOT, "artifacts"))
RESULTS = os.environ.get("RESULTS", os.path.join(ROOT, "results"))
FIGURES = os.environ.get("FIGURES", os.path.join(ROOT, "figures"))

ISOT = os.path.join(DATA_DIR, "isot", "ISOT_FINAL_CLEANED.csv")
WELFAKE = os.path.join(DATA_DIR, "welfake", "WELFake_ULTIMATE_LEAKAGE_CONTROLLED.csv")
FNN = os.path.join(DATA_DIR, "fakenewsnet")
LIAR = os.path.join(DATA_DIR, "liar_plus")
TW15_LAB = os.path.join(DATA_DIR, "twitter15", "label.txt")
TW15_SRC = os.path.join(DATA_DIR, "twitter15", "source_tweets.txt")
TW16_LAB = os.path.join(DATA_DIR, "twitter16", "label.txt")
TW16_SRC = os.path.join(DATA_DIR, "twitter16", "source_tweets.txt")
RVNN = os.path.join(DATA_DIR, "rvnn")                      # data.TD_RvNN.vol_5000.txt etc.
PHEME_RNR = os.path.join(DATA_DIR, "pheme-rnr-dataset")
CLICKBAIT_EN = os.path.join(DATA_DIR, "clickbait_en")
BANGLA = os.path.join(DATA_DIR, "banglabait")
GLOVE = os.path.join(DATA_DIR, "models", "glove-wiki-gigaword-100.gz")

SEEDS = [0, 1, 2, 3, 4]
for p in (ARTIFACTS, RESULTS, FIGURES):
    os.makedirs(p, exist_ok=True)
