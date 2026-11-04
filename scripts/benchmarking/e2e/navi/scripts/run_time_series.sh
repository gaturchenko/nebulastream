#!/usr/bin/env bash

REP=1
export NES_SARAD_SIZE_200_STRIDE_10_UDF_MODEL=models-benchmarking/models/sarad/sarad_medium.onnx

# Warm-up
/home/gaturchenko/navi/bin/systest \
    --workingDir /home/gaturchenko/navi/working-dir \
    --data /home/gaturchenko/navi/datasets/cwru-bearing \
    --clusterConfig /home/gaturchenko/navi/bin/config/single-node.yaml \
    --log-path /home/gaturchenko/navi/results/_run.log \
    --show-query-performance \
    -t /home/gaturchenko/navi/queries/cwru-bearing/ablation_cwru.test:01 \
    -n 1 \
    --optimizer inference_batch_size=1 \
    -- --worker.default_query_execution.inference.prediction_cache_type=NONE

# UDF
sudo swapoff -a && \
python3 scripts/run_systests.py \
    --inference-config queries/cwru-bearing/udf.yaml \
    --queries cwru-bearing/ablation_cwru.test:01 \
    --data ./datasets/cwru-bearing \
    --repetitions $REP

# ST & Batch
sudo swapoff -a && \
python3 scripts/run_systests.py \
    --inference-config queries/cwru-bearing/st_batch.yaml \
    --queries cwru-bearing/ablation_cwru.test:02 \
    --data ./datasets/cwru-bearing \
    --repetitions $REP

# ST + Cache
sudo swapoff -a && \
python3 scripts/run_systests.py \
    --inference-config queries/cwru-bearing/st_cache.yaml \
    --queries cwru-bearing/ablation_cwru.test:03 \
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
