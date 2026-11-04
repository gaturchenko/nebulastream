#!/usr/bin/env bash

REP=1

# ST & Batch
sudo swapoff -a && \
python3 scripts/run_systests.py \
    --inference-config queries/cwru-bearing/st_batch.yaml \
    --queries cwru-bearing/ablation_cwru.test:02 \
    --data ./datasets/cwru-bearing \
    --repetitions $REP

# Batch + Cache
sudo swapoff -a && \
python3 scripts/run_systests.py \
    --inference-config queries/cwru-bearing/batch_cache.yaml \
    --queries cwru-bearing/ablation_cwru.test:03 \
    --data ./datasets/cwru-bearing \
    --repetitions $REP

# Batch + Dedup & Batch + Cache + Dedup
sudo swapoff -a && \
python3 scripts/run_systests.py \
    --inference-config queries/cwru-bearing/batch_cache_dedup.yaml \
    --queries cwru-bearing/ablation_cwru.test:03 \
    --data ./datasets/cwru-bearing \
    --repetitions $REP

# Process results
./models-benchmarking/venv/bin/python3 scripts/process_results.py \
    --results-dir results \
    --output-csv results/ablation_cwru.csv

rm -rf ./results/*/
