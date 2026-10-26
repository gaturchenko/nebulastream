#!/bin/bash
# Latency counterpart of run_threading.sh: the same budget splits, each fed at a fixed rate below every split's capacity
# (N-BEATS-1 100000 tuples/s, N-BEATS-32 5000 tuples/s, NanoDet-R101 20 frames/s, see the *-lat.test headers), with a Latency
# sink, so that the splits compare at equal load; capacity stays the unpaced throughput from run_threading.sh. N-BEATS-32 replays
# the first 300000 samples of the solar series, cut here into the build's testdata. Extra arguments go to run_systests.py, e.g.
# --results-dir and --repetitions. Afterwards: process_threading_lat.py --results-dir <dir> --output-csv <csv>.

testdata=cmake-build-release/nes-systests/testdata/large/solar_power
[[ -f "${testdata}/solar-power-300k.csv" ]] || head -n 300000 "${testdata}/solar-power.csv" > "${testdata}/solar-power-300k.csv"
# The Latency sink opens results/latency_timings.csv relative to the systest directory and does not create it.
mkdir -p cmake-build-release/nes-systests/systest/results

queries=()
for test in osu_rgb/threading/nanodet-r101 solar_power/threading/nbeats-1 solar_power/threading/nbeats-32; do
  for i in 01 02 03 04 05 06 07 08; do queries+=("inference/${test}-24-lat.test:${i}"); done
  for i in 01 02 03 04 05 06 07 08 09 10; do queries+=("inference/${test}-48-lat.test:${i}"); done
done

python3 scripts/benchmarking/e2e/run_systests.py \
  --queries "${queries[@]}" \
  --inference-config scripts/benchmarking/e2e/config/inference_threading.yaml \
  "$@"
