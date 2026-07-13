#!/usr/bin/env bash

# one invocation per hit rate (:01=25%, :02=50%, :03=75%, :04=95%)
python3 scripts/benchmarking/e2e/run_systests.py \
    --queries inference/cache_scope/nbeats300_cache_scope_lat.test:01 \
    --inference-config scripts/benchmarking/e2e/config/cache_scope.yaml \
    --repetitions 3

# no-cache baseline
python3 scripts/benchmarking/e2e/run_systests.py \
    --queries inference/cache_scope/nbeats300_cache_scope_lat.test:02 \
    --inference-config scripts/benchmarking/e2e/config/cache_scope_baseline.yaml \
    --repetitions 3

python3 scripts/benchmarking/e2e/process_cache_scope_results.py
#python3 scripts/benchmarking/e2e/process_cache_scope_results.py --per-record all_latencies.csv   # full distributions
