#!/bin/bash

# ST & UDF
python3 scripts/benchmarking/e2e/run_systests.py \
  --inference-config scripts/benchmarking/e2e/config/inference_default.yaml --repetitions 10 \
  --queries \
  inference/solar_power/ablation_solar.test:01 inference/solar_power/ablation_solar.test:02 \
  inference/solar_power/ablation_solar.test:03 inference/solar_power/ablation_solar.test:04 \
  inference/wind_power/ablation_wind.test:01 inference/wind_power/ablation_wind.test:02 \
  inference/wind_power/ablation_wind.test:03 inference/wind_power/ablation_wind.test:04

# ST + Cache
python3 scripts/benchmarking/e2e/run_systests.py \
  --inference-config scripts/benchmarking/e2e/config/solar-power/size-600-stride-60/inference_cache.yaml --repetitions 10 \
  --queries inference/solar_power/ablation_solar.test:01
python3 scripts/benchmarking/e2e/run_systests.py \
  --inference-config scripts/benchmarking/e2e/config/wind-power/size-600-stride-60/inference_cache.yaml --repetitions 10 \
  --queries inference/wind_power/ablation_wind.test:01
python3 scripts/benchmarking/e2e/run_systests.py \
  --inference-config scripts/benchmarking/e2e/config/solar-power/size-300-stride-30/inference_cache.yaml --repetitions 10 \
  --queries inference/solar_power/ablation_solar.test:03
python3 scripts/benchmarking/e2e/run_systests.py \
  --inference-config scripts/benchmarking/e2e/config/wind-power/size-300-stride-30/inference_cache.yaml --repetitions 10 \
  --queries inference/wind_power/ablation_wind.test:03

# Batch
python3 scripts/benchmarking/e2e/run_systests.py \
  --inference-config scripts/benchmarking/e2e/config/solar-power/size-600-stride-60/inference_batch.yaml --repetitions 10 \
  --queries inference/solar_power/ablation_solar.test:01
python3 scripts/benchmarking/e2e/run_systests.py \
  --inference-config scripts/benchmarking/e2e/config/wind-power/size-600-stride-60/inference_batch.yaml --repetitions 10 \
  --queries inference/wind_power/ablation_wind.test:01
python3 scripts/benchmarking/e2e/run_systests.py \
  --inference-config scripts/benchmarking/e2e/config/solar-power/size-300-stride-30/inference_batch.yaml --repetitions 10 \
  --queries inference/solar_power/ablation_solar.test:03
python3 scripts/benchmarking/e2e/run_systests.py \
  --inference-config scripts/benchmarking/e2e/config/wind-power/size-300-stride-30/inference_batch.yaml --repetitions 10 \
  --queries inference/wind_power/ablation_wind.test:03

# Batch + Dedup
python3 scripts/benchmarking/e2e/run_systests.py \
  --inference-config scripts/benchmarking/e2e/config/solar-power/size-600-stride-60/inference_batch_dedup.yaml --repetitions 10 \
  --queries inference/solar_power/ablation_solar.test:01
python3 scripts/benchmarking/e2e/run_systests.py \
  --inference-config scripts/benchmarking/e2e/config/wind-power/size-600-stride-60/inference_batch_dedup.yaml --repetitions 10 \
  --queries inference/wind_power/ablation_wind.test:01
python3 scripts/benchmarking/e2e/run_systests.py \
  --inference-config scripts/benchmarking/e2e/config/solar-power/size-300-stride-30/inference_batch_dedup.yaml --repetitions 10 \
  --queries inference/solar_power/ablation_solar.test:03
python3 scripts/benchmarking/e2e/run_systests.py \
  --inference-config scripts/benchmarking/e2e/config/wind-power/size-300-stride-30/inference_batch_dedup.yaml --repetitions 10 \
  --queries inference/wind_power/ablation_wind.test:03

# Batch + Cache
python3 scripts/benchmarking/e2e/run_systests.py \
  --inference-config scripts/benchmarking/e2e/config/solar-power/size-600-stride-60/inference_batch_cache.yaml --repetitions 10 \
  --queries inference/solar_power/ablation_solar.test:01
python3 scripts/benchmarking/e2e/run_systests.py \
  --inference-config scripts/benchmarking/e2e/config/wind-power/size-600-stride-60/inference_batch_cache.yaml --repetitions 10 \
  --queries inference/wind_power/ablation_wind.test:01
python3 scripts/benchmarking/e2e/run_systests.py \
  --inference-config scripts/benchmarking/e2e/config/solar-power/size-300-stride-30/inference_batch_cache.yaml --repetitions 10 \
  --queries inference/solar_power/ablation_solar.test:03
python3 scripts/benchmarking/e2e/run_systests.py \
  --inference-config scripts/benchmarking/e2e/config/wind-power/size-300-stride-30/inference_batch_cache.yaml --repetitions 10 \
  --queries inference/wind_power/ablation_wind.test:03

# Batch + Cache + Dedup
python3 scripts/benchmarking/e2e/run_systests.py \
  --inference-config scripts/benchmarking/e2e/config/solar-power/size-600-stride-60/inference_batch_cache_dedup.yaml --repetitions 10 \
  --queries inference/solar_power/ablation_solar.test:01
python3 scripts/benchmarking/e2e/run_systests.py \
  --inference-config scripts/benchmarking/e2e/config/wind-power/size-600-stride-60/inference_batch_cache_dedup.yaml --repetitions 10 \
  --queries inference/wind_power/ablation_wind.test:01
python3 scripts/benchmarking/e2e/run_systests.py \
  --inference-config scripts/benchmarking/e2e/config/solar-power/size-300-stride-30/inference_batch_cache_dedup.yaml --repetitions 10 \
  --queries inference/solar_power/ablation_solar.test:03
python3 scripts/benchmarking/e2e/run_systests.py \
  --inference-config scripts/benchmarking/e2e/config/wind-power/size-300-stride-30/inference_batch_cache_dedup.yaml --repetitions 10 \
  --queries inference/wind_power/ablation_wind.test:03
