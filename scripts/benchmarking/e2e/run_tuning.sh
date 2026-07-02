#!/bin/bash

# Uncomment what needs to be tuned (the run can take a long time)

#python3 scripts/benchmarking/e2e/run_systests.py --inference-config scripts/benchmarking/e2e/config/tuning.yaml \
#  --queries \
#  inference/solar_power/ablation_solar.test:01 \
#  inference/solar_power/ablation_solar.test:03 \
#  inference/wind_power/ablation_wind.test:01 \
#  inference/wind_power/ablation_wind.test:03 \
#  --repetitions 3

#python3 scripts/benchmarking/e2e/run_systests.py --inference-config scripts/benchmarking/e2e/config/batch_sizes.yaml \
#  --queries inference/osu_rgb/ablation_osu_rgb.test:01 --repetitions 3

#python3 scripts/benchmarking/e2e/run_systests.py --inference-config scripts/benchmarking/e2e/config/tuning.yaml \
#  --queries inference/ptb-diagnostic-ecg/ablation_ecg.test:01 --repetitions 3
#
#python3 scripts/benchmarking/e2e/run_systests.py --inference-config scripts/benchmarking/e2e/config/rewrite_post_batch.yaml \
#  --queries inference/musan/ablation_musan.test:01 --repetitions 3
#python3 scripts/benchmarking/e2e/run_systests.py --inference-config scripts/benchmarking/e2e/config/no_rewrite.yaml \
#  --queries inference/musan/ablation_musan.test:01 --repetitions 3

#python3 scripts/benchmarking/e2e/run_systests.py --inference-config scripts/benchmarking/e2e/config/cwru/tuning.yaml \
#  --queries inference/cwru_bearing/ablation_cwru.test:01 --repetitions 3
#
#python3 scripts/benchmarking/e2e/run_systests.py --inference-config scripts/benchmarking/e2e/config/cwru/tuning_cache.yaml \
#  --queries inference/cwru_bearing/ablation_cwru.test:02 --repetitions 3
