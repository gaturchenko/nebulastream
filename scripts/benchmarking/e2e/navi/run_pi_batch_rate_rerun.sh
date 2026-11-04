#!/usr/bin/env bash
# Batch-rate experiments at OpenVINO 1 stream x 4 threads with active cooling, queued behind the post-join rerun
# (run_pi_fan_rerun.sh; stops if that one aborted):
#   1. archive the num_streams 0 x 1 thread results in results/streams0
#   2. the latency sweep (run_batch_rate.sh), 3. the sustainable-throughput ladder (run_batch_throughput.sh), each followed by
#      up to 2 rounds of re-measuring the runs the firmware throttled (move_throttled_runs.py), then processed
# Every run starts below COOLDOWN_C; temperature and flags are sampled every 2 s. Progress: results/run_pi_batch_rate_rerun.log
set -uo pipefail
cd ~/navi
log=results/run_pi_batch_rate_rerun.log
PY=models-benchmarking/venv/bin/python
export SAMPLE_S=2 COOLDOWN_C="${COOLDOWN_C:-60}"
note() { echo "### $* ($(date '+%F %T'))" | tee -a "$log"; }

# Runs a driver, then re-runs it for the repetitions that moved aside because they were throttled, up to 2 rounds.
# measure <name> <results dir> <driver...>
measure() {
    local name="$1" results="$2"
    shift 2
    "$@" > "results/${name}_driver.log" 2>&1
    note "${name}: sweep done"
    for round in 1 2; do
        local moved="results/${name}_throttled/round${round}"
        mkdir -p "$moved"
        "$PY" scripts/move_throttled_runs.py "$results" "$moved" > "$moved/moved.txt"
        local count
        count=$(wc -l < "$moved/moved.txt")
        note "${name}: round ${round}, ${count} throttled runs moved aside"
        [[ "$count" == 0 ]] && return
        "$@" > "results/${name}_driver_refill${round}.log" 2>&1
    done
    count=$("$PY" scripts/move_throttled_runs.py "$results" "results/${name}_throttled/after_round2" | wc -l)
    [[ "$count" == 0 ]] || note "ATTENTION ${name}: ${count} runs still throttled after 2 rounds, moved to results/${name}_throttled/after_round2"
}

: > "$log"
note "waiting for the post-join rerun"
while ! grep -q "Pi fan rerun finished" results/run_pi_fan_rerun.log 2>/dev/null; do
    grep -q "ABORT" results/run_pi_fan_rerun.log 2>/dev/null && { note "ABORT: the post-join rerun aborted"; exit 1; }
    sleep 10
done
while pgrep -f "bin/systest|run_post_join" > /dev/null; do sleep 10; done

for file in queries/batch-rate/nbeats_batch_rate.test queries/batch-rate/nbeats_batch_throughput.test.tmpl; do
    grep -q "openvino_num_streams: \[1\]" "$file" && grep -q "openvino_inference_num_threads: \[4\]" "$file" \
        || { note "ABORT: $file is not set to 1 stream x 4 threads"; exit 1; }
done

# 1. Archive.
mkdir -p results/streams0
for entry in batch_rate batch_rate_driver.log batch_throughput batch_throughput_driver.log batch_throughput_driver_rerun.log \
    batch_throughput_throttled; do
    [[ -e "results/streams0/${entry}" ]] && { note "ABORT: results/streams0/${entry} already exists"; exit 1; }
    [[ -e "results/${entry}" ]] && mv "results/${entry}" results/streams0/
done
note "num_streams 0 results archived in results/streams0; idle $(vcgencmd measure_temp), flags $(vcgencmd get_throttled)"

# 2. Latency sweep.
measure batch_rate results/batch_rate ./scripts/run_batch_rate.sh 3
"$PY" scripts/process_batch_rate.py --repetitions 3 > results/batch_rate_summary.txt 2>&1
note "batch_rate processed"

# 3. Sustainable throughput.
measure batch_throughput results/batch_throughput ./scripts/run_batch_throughput.sh 3
"$PY" scripts/process_batch_throughput.py --repetitions 3 > results/batch_throughput_summary.txt 2>&1
note "Pi batch-rate rerun finished"
