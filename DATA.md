# Data statement

No dataset is redistributed in this repository. Every corpus is public and is obtained from its
original release. `src/prep.py` expects the layout below under `$DATA_DIR`.

```
$DATA_DIR/
├── isot/ISOT_FINAL_CLEANED.csv                    title, text, label (1 = fake)
├── welfake/WELFake_ULTIMATE_LEAKAGE_CONTROLLED.csv  title, text, label (inverted: 1 = real, see below)
├── fakenewsnet/{politifact,gossipcop}_{fake,real}.csv
├── liar_plus/{train2,val2,test2}.tsv
├── twitter15/{label.txt,source_tweets.txt}
├── twitter16/{label.txt,source_tweets.txt}
├── pheme-rnr-dataset/<event>/{rumours,non-rumours}/<id>/source-tweet/*.json
├── clickbait_en/{clickbait_data.gz,non_clickbait_data.gz}
├── banglabait/{train,dev,test}.csv
└── models/glove-wiki-gigaword-100.gz
```

### The two article files

The ISOT and WELFake files above are cleaned working copies made earlier in this project, not
the raw releases:

* `ISOT_FINAL_CLEANED.csv`: 38,833 of the 44,898 released articles, all with their release
  labels. Relative to the release it lacks 5,795 exact title+text duplicates and 270 further
  articles, 267 of which repeat the title of a kept article. 447 articles have an empty body;
  they keep their headline and are scored on it.
* `WELFake_ULTIMATE_LEAKAGE_CONTROLLED.csv`: 24,530 of the 72,134 released articles
  (`WELFake_Dataset.csv`, MD5 `73c9675a4b3d09f86a6933d0b8d7d908`). Dropping 8,458 exact
  title+text duplicates, the 38,656 articles whose text also appears in the ISOT release and 480
  more whose title does leaves 24,540; the copy is that set minus 10 articles, and none of its
  articles lies outside it. WELFake merges a Reuters-based collection like ISOT's real class,
  which is why the overlap step exists. Only 18 near-duplicates of ISOT remain, and transfer
  tests drop them.

`src/provenance.py` reconstructs both copies from the releases and writes the counts above to
`results/provenance.json`:

```bash
export ISOT_RAW=/path/to/isot          # folder with Fake.csv and True.csv
export WELFAKE_RAW=/path/to/WELFake_Dataset.csv
python src/provenance.py
```

To run the whole pipeline from the original releases instead, keep those two variables set and
run `python src/prep.py`: it deduplicates with its own normalisation and MinHash and then drops
WELFake articles shared with ISOT, so the counts are close to, but not identical with, the
working copies.

| Corpus | Unit | Construct | Source | Licence / terms |
|---|---|---|---|---|
| ISOT Fake News (Ahmed et al., 2017, 2018) | article | fake vs real | University of Victoria ISOT lab | free for research use, cite the papers |
| WELFake (Verma et al., 2021) | article | fake vs real | Zenodo 10.5281/zenodo.4561253 | CC BY 4.0 |
| FakeNewsNet PolitiFact / GossipCop (Shu et al., 2020) | headline | fake vs real | github.com/KaiDMML/FakeNewsNet (`dataset/`) | see repository; titles and URLs only |
| LIAR (Wang, 2017) via LIAR-PLUS (Alhindi et al., 2018) | claim | false vs true | github.com/Tariq60/LIAR-PLUS | see repository |
| Twitter15 / Twitter16 (Ma et al., 2017) | tweet | false vs true | released with Ma et al. (2017, 2018) | Twitter/X terms; source text only |
| PHEME rumour/non-rumour (Zubiaga et al., 2016) | tweet | rumour vs non-rumour | figshare 4010619 | annotations CC BY; tweet content owned by Twitter/X |
| English clickbait (Chakraborty et al., 2016) | headline | clickbait vs not | github.com/bhargaviparanjape/clickbait | see repository |
| BanglaBait (Mahtab et al., 2023) | headline (+ body) | clickbait vs not | Kaggle `motaharmahtab/banglabait-bangla-clickbait-dataset` | see dataset page |
| GloVe 6B 100d (Pennington et al., 2014) | embeddings | – | gensim-data release | PDDL |
| RoBERTa-base (Liu et al., 2019) | encoder | – | spaCy `en_core_web_trf` 3.8.0 | MIT |

## Label harmonisation (important)

Every corpus is mapped to **1 = fake / false / rumour / clickbait**.

* **WELFake: the documentation and the file disagree.** The Zenodo record gives the legend
  `0 = fake, 1 = real`, but the released file codes `1 = fake`: all 6,223 titles ending in
  "- The New York Times" and 21,270 of the 21,289 "(Reuters)" datelines carry label 0, all 6,261
  texts with the fake-site template "Featured image via" carry label 1, and label 1 holds 37,106
  articles, the number of fake articles the same record states (35,028 real).
  The working copy used here was made by following the legend: every label is the inverse of the
  release label, and its `class` column calls the inverted label 1 "Fake". `prep.py` therefore
  inverts the working copy back; with `WELFAKE_RAW` it reads the release as `1 = fake` and does
  not invert. Reading WELFake through its legend makes ISOT↔WELFake transfer look below chance
  while leaving in-domain scores unchanged, which is how the inversion first showed up here.
* LIAR: `pants-fire`, `false` → 1; `mostly-true`, `true` → 0; `half-true` and `barely-true` are
  dropped. Claim-level labels (LIAR) are not the same construct as article-level labels.
* Twitter15/16: `false` → 1, `true` → 0 for the binary view; `unverified` and `non-rumor` are
  dropped. The 169 tweet IDs shared by Twitter15 and Twitter16 are all `unverified`.

## Deduplication

Within each corpus: exact duplicates on normalised text, then MinHash-LSH near-duplicates
(128 permutations, Jaccard ≥ 0.8; word 3-grams for articles, character 5-grams for short texts).
Clusters whose members disagree on the label are removed entirely. Across corpora: near-duplicate
flags (`dup_in`) let every transfer test drop target items that also occur in the source corpus.
Counts are in `results/dedup_report.json` and `results/cross_corpus_overlap.csv`.

## Social-media data and privacy

Only the released text of source posts is used. No user identifiers, follower graphs or
metadata beyond the event name are processed, nothing is re-collected from the platform, and no
tweet text is redistributed. Results are reported in aggregate.

## Zero-shot language model

`src/llm_zeroshot.py` uses Qwen3-0.6B in the 4-bit Q4_K_M GGUF quantisation published by Unsloth
(`unsloth/Qwen3-0.6B-GGUF`, file `Qwen3-0.6B-Q4_K_M.gguf`, 396,705,472 bytes, SHA-256
`ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a`; Apache-2.0). Put it at
`models/Qwen3-0.6B-Q4_K_M.gguf` (not tracked by git). Check the hash before running.
