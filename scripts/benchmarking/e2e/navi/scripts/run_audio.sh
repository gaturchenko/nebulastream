#!/usr/bin/env bash

REP=1
export NES_MHATT_RNN_UDF_MODEL=models-benchmarking/models/mhatt-rnn/u192_l2/saved_model

# Warm-up
#/home/gaturchenko/navi/bin/systest \
#    --workingDir /home/gaturchenko/navi/working-dir \
#    --data /home/gaturchenko/navi/datasets/musan \
#    --clusterConfig /home/gaturchenko/navi/bin/config/single-node.yaml \
#    --log-path /home/gaturchenko/navi/results/_run.log \
#    --show-query-performance \
#    -t /home/gaturchenko/navi/queries/musan/ablation_musan.test:01 \
#    -n 1 \
#    --optimizer post_join_inference_strategy=SEQUENTIAL \
#    -- --worker.default_query_execution.inference.prediction_cache_type=NONE

# UDF
"$HOME/navi/scripts/reset_swap.sh" off && \
python3 scripts/run_systests.py \
    --inference-config queries/musan/udf.yaml \
    --queries musan/ablation_musan.test:01 \
    --data ./datasets/musan \
    --repetitions $REP

# ST
"$HOME/navi/scripts/reset_swap.sh" off && \
python3 scripts/run_systests.py \
    --inference-config queries/musan/st.yaml \
    --queries musan/ablation_musan.test:02 \
    --data ./datasets/musan \
    --repetitions $REP

# Batch
"$HOME/navi/scripts/reset_swap.sh" off && \
python3 scripts/run_systests.py \
    --inference-config queries/musan/batch.yaml \
    --queries musan/ablation_musan.test:02 \
    --data ./datasets/musan \
    --repetitions $REP

# ST + Cache
"$HOME/navi/scripts/reset_swap.sh" off && \
python3 scripts/run_systests.py \
    --inference-config queries/musan/st_cache.yaml \
    --queries musan/ablation_musan.test:03 \
    --data ./datasets/musan \
    --repetitions $REP

# Batch + Dedup
"$HOME/navi/scripts/reset_swap.sh" off && \
python3 scripts/run_systests.py \
    --inference-config queries/musan/batch_dedup.yaml \
    --queries musan/ablation_musan.test:03 \
    --data ./datasets/musan \
    --repetitions $REP


# Batch + Cache & Batch + Cache + Dedup
"$HOME/navi/scripts/reset_swap.sh" off && \
python3 scripts/run_systests.py \
    --inference-config queries/musan/cache_batch_dedup.yaml \
    --queries musan/ablation_musan.test:03 \
    --data ./datasets/musan \
    --repetitions $REP

"$HOME/navi/scripts/reset_swap.sh" on

# Process results
./models-benchmarking/venv/bin/python3 scripts/process_results.py \
    --results-dir results \
    --output-csv results/ablation_musan.csv
rm -rf ./results/*/
