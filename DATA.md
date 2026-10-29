# Data statement

No dataset is redistributed in this repository. Every corpus is public and is obtained from its
original release. `src/prep.py` expects the layout below under `$DATA_DIR`.

```
$DATA_DIR/
├── isot/ISOT_FINAL_CLEANED.csv                    title, text, label (1 = fake)
├── welfake/WELFake_ULTIMATE_LEAKAGE_CONTROLLED.csv  title, text, label (1 = REAL, see below)
├── fakenewsnet/{politifact,gossipcop}_{fake,real}.csv
├── liar_plus/{train2,val2,test2}.tsv
├── twitter15/{label.txt,source_tweets.txt}
├── twitter16/{label.txt,source_tweets.txt}
├── pheme-rnr-dataset/<event>/{rumours,non-rumours}/<id>/source-tweet/*.json
├── clickbait_en/{clickbait_data.gz,non_clickbait_data.gz}
├── banglabait/{train,dev,test}.csv
└── models/glove-wiki-gigaword-100.gz
```

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

* **WELFake ships `0 = fake, 1 = real`** (Zenodo record). The leakage-controlled file used here
  also carried a `class` column naming label 1 "Fake", which is wrong. We verified the direction
  with signature probes: all 6,223 articles whose title ends in "- The New York Times" have
  label 1, and all texts containing the fake-site template "Featured image via" or titles with
  "BREAKING" have label 0. `prep.py` therefore inverts WELFake. Treating WELFake label 1 as fake
  produces below-chance transfer to ISOT, which is how the inversion first showed up.
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
