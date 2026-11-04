#!/usr/bin/env bash
# Sensitivity part D: how stable is the profile the optimizer decides on?
#
# Runs the optimizer's profiling job (nes-model-profiler) repeatedly, as a fresh process each time as at a deployment, varying
#   - the measured invocations per batch size, at an idle worker, and
#   - the number of busy cores while profiling, at the default 50 invocations.
# process_profile_stability.py then replays the decisions over the profiles.
#
#   ./scripts/run_profile_stability.sh                      # from ~/navi on the Pi
#   PIN="taskset -c 8-11" CORES="8 9 10" MODELS=... ./run_profile_stability.sh     # on a host shared with other cores
#
# The load is one `yes` per busy core: pure CPU contention, as from another query's worker thread, needing nothing installed.
# MODELS entries are label=path=batch_sizes=dynamic=repetitions. Finished repetitions are skipped, so re-invoking resumes.
set -uo pipefail

PROFILER="${PROFILER:-bin/nes-model-profiler}"
VENV="${VENV:-models-benchmarking/venv}"
RESULTS="${RESULTS:-results/sensitivity/profile}"
ITERATIONS="${ITERATIONS:-5 10 25 50 100}"
LOADS="${LOADS:-1 2 3}"
CORES="${CORES:-0 1 2}"
PIN="${PIN:-}"
COOLDOWN_C="${COOLDOWN_C:-60}"
HAS_VCGENCMD=$(command -v vcgencmd > /dev/null && echo 1 || echo 0)
MODELS="${MODELS:-nbeats=models-benchmarking/models/nbeats/nbeats_size_300_stride_30.onnx=1,2,4,8,16,32,64=false=20 \
sarad=models-benchmarking/models/sarad/sarad_medium.onnx=1,2,4,8=false=10 \
mhatt_rnn=models-benchmarking/models/mhatt-rnn/saved_model=1,2,4=true=10}"

mkdir -p "$RESULTS"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

cooldown() {
    (( HAS_VCGENCMD )) || return 0
    while (( $(vcgencmd measure_temp | sed 's/[^0-9]*\([0-9]*\).*/\1/') >= COOLDOWN_C )); do sleep 5; done
}

load_pids=()
start_load() {
    local count=0
    for core in $CORES; do
        (( count >= $1 )) && break
        taskset -c "$core" yes > /dev/null &
        load_pids+=($!)
        count=$(( count + 1 ))
    done
}
stop_load() {
    (( ${#load_pids[@]} )) && kill "${load_pids[@]}" 2>/dev/null
    wait "${load_pids[@]}" 2>/dev/null
    load_pids=()
}
trap stop_load EXIT

# profile <label> <path> <batch sizes> <dynamic> <iterations> <busy cores> <rep>
profile() {
    local out="${RESULTS}/${1}_i${5}_l${6}_rep${7}.csv"
    [[ -s "$out" ]] && return
    cooldown
    # A repetition hands the profiler its own batch-size order, since the profiler seeds its shuffle with its own repetition.
    local order
    order=$(python3 -c 'import random, sys; s = sys.argv[1].split(","); random.Random(int(sys.argv[2])).shuffle(s); print(",".join(s))' "$3" "$7")
    start_load "$6"
    # shellcheck disable=SC2086
    if $PIN "$PROFILER" \
        --model="$2" --label="$1" --batch_sizes="$order" --repetitions=1 --iterations="$5" \
        --worker.query_engine.number_of_worker_threads=2 \
        --worker.default_query_execution.inference.openvino_inference_num_threads=4 \
        --worker.default_query_execution.inference.openvino_num_streams=1 \
        --worker.default_query_execution.inference.openvino_enable_cpu_pinning=false \
        --worker.default_query_execution.inference.openvino_share_compiled_model=false \
        --worker.default_query_execution.inference.prediction_cache_type=NONE \
        --worker.default_query_execution.inference.use_batch_deduplication="$4" \
        > "${out}.tmp" 2> "${out%.csv}.err"; then
        mv "${out}.tmp" "$out"
    else
        echo "  !! ${out} failed, see ${out%.csv}.err"
        rm -f "${out}.tmp"
    fi
    stop_load
}

started=$(date +%s)
for entry in $MODELS; do
    IFS='=' read -r label path batch_sizes dynamic reps <<< "$entry"
    [[ -e "$path" ]] || { echo "  !! ${label}: no model at ${path}, skipped"; continue; }
    echo "### ${label}: iterations ($(( ($(date +%s) - started) / 60 )) min elapsed)"
    for rep in $(seq 1 "$reps"); do
        for iterations in $ITERATIONS; do
            profile "$label" "$path" "$batch_sizes" "$dynamic" "$iterations" 0 "$rep"
        done
    done
    echo "### ${label}: load ($(( ($(date +%s) - started) / 60 )) min elapsed)"
    for rep in $(seq 1 "$reps"); do
        for load in $LOADS; do
            profile "$label" "$path" "$batch_sizes" "$dynamic" 50 "$load" "$rep"
        done
    done
done
echo "### profile stability finished after $(( ($(date +%s) - started) / 60 )) min"
