#!/usr/bin/env bash
# Reproduce every number, table and figure in the paper.
# Usage:  DATA_DIR=/path/to/data bash run_all.sh
# Runtime on a 2-core CPU, 8 GB RAM: roughly 10-12 hours end to end (no GPU needed).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
step() { echo "== $1  $(date '+%H:%M:%S')"; }

step "1/9 corpora, label harmonisation, deduplication"
python src/prep.py                                         | tee logs/prep.log
python src/label_probe.py                                  | tee logs/label_probe.log
step "2/9 transfer matrices (5 models x 3 regimes x 5 seeds)"
python src/transfer.py --models tfidf_lr,nbsvm,style_gb    | tee logs/transfer_A.log
python src/transfer.py --models tfidf_svm,glove_lr         | tee logs/transfer_B.log
step "3/9 RoBERTa-base representations"
python src/roberta.py embed                                | tee logs/roberta_embed.log
python src/roberta.py matrix                               | tee logs/roberta_matrix.log
step "4/9 mechanism tests"
python src/mechanism.py --tests group,weights,lc,topk,pad,topic,mask,vocab | tee logs/mechanism.log
step "5/9 label-free repairs (leave-one-corpus-out)"
python src/fix.py                                          | tee logs/fix.log
step "6/9 Bengali"
python src/bangla.py                                       | tee logs/bangla.log
step "7/9 aggregation, CIs, decomposition, tests"
python src/analysis.py                                     | tee logs/analysis.log
python src/analysis_extra.py                               | tee logs/analysis_extra.log
step "8/9 figures"
python src/figures.py                                      | tee logs/figures.log
step "9/9 paper numbers, README, claim verification"
python src/paper_numbers.py                                | tee logs/paper_numbers.log
python src/make_readme.py
python src/verify_claims.py                                | tee logs/verify_claims.log
echo "done"
