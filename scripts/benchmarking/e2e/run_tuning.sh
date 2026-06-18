#!/bin/bash

python3 scripts/benchmarking/e2e/run_systests.py --inference-config scripts/benchmarking/e2e/config/tuning.yaml \
  --queries \
  inference/solar_power/ablation_solar.test:01 \
  inference/solar_power/ablation_solar.test:03 \
  inference/wind_power/ablation_wind.test:01 \
  inference/wind_power/ablation_wind.test:03 \
  --repetitions 3
