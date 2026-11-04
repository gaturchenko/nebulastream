#!/usr/bin/env bash

export NES_MHATT_RNN_UDF_MODEL=models-benchmarking/models/mhatt-rnn/saved_model

# UDF
"$HOME/navi/scripts/reset_swap.sh" off && \
python3 scripts/run_systests.py \
    --inference-config queries/musan/udf.yaml \
    --queries musan/ablation_musan.test:01 \
    --data ./datasets/musan \
    --repetitions 1

# ST
"$HOME/navi/scripts/reset_swap.sh" off && \
python3 scripts/run_systests.py \
    --inference-config queries/musan/st.yaml \
    --queries musan/ablation_musan.test:02 \
    --data ./datasets/musan \
    --repetitions 1

# Batch
"$HOME/navi/scripts/reset_swap.sh" off && \
python3 scripts/run_systests.py \
    --inference-config queries/musan/batch.yaml \
    --queries musan/ablation_musan.test:02 \
    --data ./datasets/musan \
    --repetitions 1

"$HOME/navi/scripts/reset_swap.sh" on

# Process results
./models-benchmarking/venv/bin/python3 scripts/process_results.py \
    --results-dir results \
    --output-csv results/ablation_musan.csv
rm -rf ./results/*/
