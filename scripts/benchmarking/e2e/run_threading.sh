#!/bin/bash
# Thread-budget sweep: each test's i-th query splits its budget into (SPE workers x OpenVINO threads), see BUDGET_SPLITS in
# figures/plot_threading.py. The tests set num_streams 1 and each split's inference_num_threads; inference_threading.yaml leaves
# both out, because options passed on the command line must not contradict them (inference_default.yaml's num_streams 0 made
# OpenVINO ignore the thread count). Extra arguments go to run_systests.py, e.g. --results-dir.

python3 scripts/benchmarking/e2e/run_systests.py \
  --queries \
    inference/osu_rgb/threading/nanodet-24.test:01 \
    inference/osu_rgb/threading/nanodet-24.test:02 \
    inference/osu_rgb/threading/nanodet-24.test:03 \
    inference/osu_rgb/threading/nanodet-24.test:04 \
    inference/osu_rgb/threading/nanodet-24.test:05 \
    inference/osu_rgb/threading/nanodet-24.test:06 \
    inference/osu_rgb/threading/nanodet-24.test:07 \
    inference/osu_rgb/threading/nanodet-24.test:08 \
    inference/osu_rgb/threading/nanodet-48.test:01 \
    inference/osu_rgb/threading/nanodet-48.test:02 \
    inference/osu_rgb/threading/nanodet-48.test:03 \
    inference/osu_rgb/threading/nanodet-48.test:04 \
    inference/osu_rgb/threading/nanodet-48.test:05 \
    inference/osu_rgb/threading/nanodet-48.test:06 \
    inference/osu_rgb/threading/nanodet-48.test:07 \
    inference/osu_rgb/threading/nanodet-48.test:08 \
    inference/osu_rgb/threading/nanodet-48.test:09 \
    inference/osu_rgb/threading/nanodet-48.test:10 \
    inference/solar_power/threading/nbeats-1-24.test:01 \
    inference/solar_power/threading/nbeats-1-24.test:02 \
    inference/solar_power/threading/nbeats-1-24.test:03 \
    inference/solar_power/threading/nbeats-1-24.test:04 \
    inference/solar_power/threading/nbeats-1-24.test:05 \
    inference/solar_power/threading/nbeats-1-24.test:06 \
    inference/solar_power/threading/nbeats-1-24.test:07 \
    inference/solar_power/threading/nbeats-1-24.test:08 \
    inference/solar_power/threading/nbeats-32-24.test:01 \
    inference/solar_power/threading/nbeats-32-24.test:02 \
    inference/solar_power/threading/nbeats-32-24.test:03 \
    inference/solar_power/threading/nbeats-32-24.test:04 \
    inference/solar_power/threading/nbeats-32-24.test:05 \
    inference/solar_power/threading/nbeats-32-24.test:06 \
    inference/solar_power/threading/nbeats-32-24.test:07 \
    inference/solar_power/threading/nbeats-32-24.test:08 \
    inference/solar_power/threading/nbeats-1-48.test:01 \
    inference/solar_power/threading/nbeats-1-48.test:02 \
    inference/solar_power/threading/nbeats-1-48.test:03 \
    inference/solar_power/threading/nbeats-1-48.test:04 \
    inference/solar_power/threading/nbeats-1-48.test:05 \
    inference/solar_power/threading/nbeats-1-48.test:06 \
    inference/solar_power/threading/nbeats-1-48.test:07 \
    inference/solar_power/threading/nbeats-1-48.test:08 \
    inference/solar_power/threading/nbeats-1-48.test:09 \
    inference/solar_power/threading/nbeats-1-48.test:10 \
    inference/solar_power/threading/nbeats-32-48.test:01 \
    inference/solar_power/threading/nbeats-32-48.test:02 \
    inference/solar_power/threading/nbeats-32-48.test:03 \
    inference/solar_power/threading/nbeats-32-48.test:04 \
    inference/solar_power/threading/nbeats-32-48.test:05 \
    inference/solar_power/threading/nbeats-32-48.test:06 \
    inference/solar_power/threading/nbeats-32-48.test:07 \
    inference/solar_power/threading/nbeats-32-48.test:08 \
    inference/solar_power/threading/nbeats-32-48.test:09 \
    inference/solar_power/threading/nbeats-32-48.test:10 \
    --inference-config scripts/benchmarking/e2e/config/inference_threading.yaml \
    --repetitions 10 \
    "$@"
