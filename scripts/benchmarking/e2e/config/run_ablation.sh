#!/bin/bash

python3 scripts/benchmarking/e2e/run_systests.py --queries inference/wind_power/ablation.test:02 --inference-config scripts/benchmarking/e2e/config/inference_default.yaml --repetitions 10
python3 scripts/benchmarking/e2e/run_systests.py --queries inference/wind_power/ablation.test:02 --inference-config scripts/benchmarking/e2e/config/inference_cache.yaml --repetitions 10
python3 scripts/benchmarking/e2e/run_systests.py --queries inference/wind_power/ablation.test:02 --inference-config scripts/benchmarking/e2e/config/inference_batch.yaml --repetitions 10
python3 scripts/benchmarking/e2e/run_systests.py --queries inference/wind_power/ablation.test:02 --inference-config scripts/benchmarking/e2e/config/inference_batch_dedup.yaml --repetitions 10
python3 scripts/benchmarking/e2e/run_systests.py --queries inference/wind_power/ablation.test:02 --inference-config scripts/benchmarking/e2e/config/inference_batch_cache.yaml --repetitions 10
python3 scripts/benchmarking/e2e/run_systests.py --queries inference/wind_power/ablation.test:02 --inference-config scripts/benchmarking/e2e/config/inference_batch_cache_dedup.yaml --repetitions 10
