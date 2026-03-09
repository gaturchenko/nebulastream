#!/bin/bash

python3 scripts/benchmarking/e2e/run_systests.py --queries inference/solar_power/ablation.test:01 inference/solar_power/ablation.test:02 --inference-config scripts/benchmarking/e2e/config/inference_default.yaml --repetitions 50
python3 scripts/benchmarking/e2e/run_systests.py --queries inference/solar_power/ablation.test:01 inference/solar_power/ablation.test:02 --inference-config scripts/benchmarking/e2e/config/inference_cache.yaml --repetitions 50
python3 scripts/benchmarking/e2e/run_systests.py --queries inference/solar_power/ablation.test:01 inference/solar_power/ablation.test:02 --inference-config scripts/benchmarking/e2e/config/inference_batch.yaml --repetitions 50
python3 scripts/benchmarking/e2e/run_systests.py --queries inference/solar_power/ablation.test:01 inference/solar_power/ablation.test:02 --inference-config scripts/benchmarking/e2e/config/inference_batch_dedup.yaml --repetitions 50
python3 scripts/benchmarking/e2e/run_systests.py --queries inference/solar_power/ablation.test:01 inference/solar_power/ablation.test:02 --inference-config scripts/benchmarking/e2e/config/inference_batch_cache.yaml --repetitions 50
python3 scripts/benchmarking/e2e/run_systests.py --queries inference/solar_power/ablation.test:01 inference/solar_power/ablation.test:02 --inference-config scripts/benchmarking/e2e/config/inference_batch_cache_dedup.yaml --repetitions 50
