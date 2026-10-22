#!/usr/bin/env bash
# Pilot for the post-join batching decision (revision: optimizer with an actual alternative choice).
#
# Post-join inference over k payloads either invokes the model once with a batch of k (BATCHED) or k times with one payload
# (SEQUENTIAL); OPTIMIZER_CHOOSES batches iff k * t(1) / t(k) >= post_join_inference_minimum_speedup (1.1). The end-to-end
# experiment is only worth running if that speedup falls below the threshold for some model and configuration, i.e. if the
# choice is not always BATCHED. This pilot measures t(B) with nes-model-profiler, which runs the optimizer's own profiling job
# (same sessions, same options, same synthetic input), for every model x OpenVINO session setting x static/dynamic batch shape.
#
#   ./scripts/run_post_join_pilot.sh [reps]
#   MODELS="squeezedet=models-benchmarking/models/squeezedet/squeezedet.onnx" SESSIONS="1:4" ./scripts/run_post_join_pilot.sh 3
#   ./scripts/summarize_post_join_pilot.py results/post_join_pilot      # afterwards
#
# Run from ~/navi with bin/nes-model-profiler deployed (scripts/arm-build/submit_arm_build.sh --targets "nes-model-profiler").
# Every repetition of a cell is its own profiler run with its own CSV, and finished ones are skipped, so re-invoking after an
# interruption resumes. Before each repetition the SoC cools below COOLDOWN_C: a profiling run keeps the Pi under full load
# for minutes, and past its 80 C soft limit the firmware caps the clock mid-run, which would skew t(1) against t(k).
set -uo pipefail

REPS="${1:-5}"
MODELS="${MODELS:-nbeats=models-benchmarking/models/nbeats/nbeats_size_300_stride_30.onnx \
mhatt_rnn=models-benchmarking/models/mhatt-rnn/saved_model \
sarad=models-benchmarking/models/sarad/sarad_medium.onnx \
squeezedet=models-benchmarking/models/squeezedet/squeezedet.onnx}"
# OpenVINO sessions as num_streams:inference_num_threads. An explicit num_streams=0 (NES's default, 0:1) makes OpenVINO 2025.3
# rebuild its executor config without the thread count, so inference_num_threads has no effect; what it falls back to depends on
# the build. On x86 (TBB) the zero-stream config is one 1-thread stream run inline on the caller's thread, i.e. one core. On the
# Pi it occupies ~3.3 of 4 cores (SqueezeDet: 81 ms at 0:1 and 0:4, 82 ms at 1:4, 196 ms at 1:1). 0:1 is what deployed queries
# run with; 1:1 and 1:4 pin the intra-op parallelism to one core and to all cores.
SESSIONS="${SESSIONS:-0:1 1:1 1:4}"
# Batch deduplication (or a prediction cache) makes batched sessions accept partially filled batches, i.e. a dynamic batch
# dimension, which OpenVINO may compile into slower kernels; single-tuple sessions stay static either way.
DYNAMIC="${DYNAMIC:-false true}"
# The k a post-join operator sees is its number of payload fields, so non-powers of two matter here.
BATCH_SIZES="${BATCH_SIZES:-1,2,3,4,6,8}"
# Only enters the profile through a shared compiled model; kept at the Pi experiments' setting.
WORKER_THREADS="${WORKER_THREADS:-2}"
SAMPLE_S="${SAMPLE_S:-5}"
# 0 disables the wait.
COOLDOWN_C="${COOLDOWN_C:-65}"
VENV="${VENV:-models-benchmarking/venv}"
PROFILER="${PROFILER:-bin/nes-model-profiler}"
RESULTS="${RESULTS:-results/post_join_pilot}"

[[ -x "$PROFILER" ]] || { echo "No profiler binary at ${PROFILER}" >&2; exit 1; }
mkdir -p "$RESULTS"

# ovc converts the models on import.
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

thermal_sampler() {
    [[ -s "${RESULTS}/thermal.csv" ]] || echo "time_us,temp_c,throttled" > "${RESULTS}/thermal.csv"
    while :; do
        echo "$(( $(date +%s%N) / 1000 )),$(vcgencmd measure_temp | sed 's/[^0-9.]//g'),$(vcgencmd get_throttled | cut -d= -f2)" \
            >> "${RESULTS}/thermal.csv"
        sleep "$SAMPLE_S"
    done
}
cooldown() {
    [[ "$COOLDOWN_C" == 0 ]] && return
    local waited=0
    while (( $(vcgencmd measure_temp | sed 's/[^0-9]*\([0-9]*\).*/\1/') >= COOLDOWN_C )); do
        sleep 5
        waited=$(( waited + 5 ))
    done
    (( waited > 0 )) && echo "  (cooled down for ${waited} s)"
}

# The profiler seeds its batch-size shuffle with the repetition, which is always 1 here, so every repetition hands it its own
# order instead.
shuffled_batch_sizes() {
    python3 -c 'import random, sys; sizes = sys.argv[1].split(","); random.Random(int(sys.argv[2])).shuffle(sizes); print(",".join(sizes))' \
        "$BATCH_SIZES" "$1"
}

thermal_sampler &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null' EXIT

echo "### post-join pilot: reps=$REPS batch_sizes=$BATCH_SIZES sessions(streams:threads)=[$SESSIONS] dynamic=[$DYNAMIC]"
echo "### throttling flags at start: $(vcgencmd get_throttled)"
started=$(date +%s)

for entry in $MODELS; do
    label="${entry%%=*}"
    model="${entry#*=}"
    [[ -e "$model" ]] || { echo "  !! ${label}: no model at ${model}, skipped" >&2; continue; }
    for session in $SESSIONS; do
        streams="${session%%:*}"
        threads="${session#*:}"
        for dynamic in $DYNAMIC; do
            cell="${label}_s${streams}_t${threads}_dyn-${dynamic}"
            # Cells of an earlier driver version ran all their repetitions in one profiler run.
            if [[ -s "${RESULTS}/profile_${cell}.csv" ]]; then
                echo "### ${cell}: done, skipped"
                continue
            fi
            for rep in $(seq 1 "$REPS"); do
                out="${RESULTS}/profile_${cell}_rep${rep}.csv"
                [[ -s "$out" ]] && continue
                cooldown
                echo "### ${cell} rep ${rep}/${REPS}  ($(date +%H:%M:%S), $(( ($(date +%s) - started) / 60 )) min elapsed, $(vcgencmd measure_temp))"
                # Written to a temporary file first, so that an interrupted repetition is re-run rather than counted as done.
                if "$PROFILER" \
                    --model="$model" \
                    --label="$label" \
                    --batch_sizes="$(shuffled_batch_sizes "$rep")" \
                    --repetitions=1 \
                    --worker.query_engine.number_of_worker_threads="$WORKER_THREADS" \
                    --worker.default_query_execution.inference.openvino_inference_num_threads="$threads" \
                    --worker.default_query_execution.inference.openvino_num_streams="$streams" \
                    --worker.default_query_execution.inference.openvino_enable_cpu_pinning=false \
                    --worker.default_query_execution.inference.openvino_share_compiled_model=false \
                    --worker.default_query_execution.inference.prediction_cache_type=NONE \
                    --worker.default_query_execution.inference.use_batch_deduplication="$dynamic" \
                    > "${out}.tmp" 2> "${RESULTS}/profile_${cell}_rep${rep}.err"; then
                    # The summary pairs t(1) and t(k) by repetition, so the CSV carries the driver's repetition.
                    awk -F, -v rep="$rep" 'BEGIN { OFS = "," } NR == 1 { print; next } { $7 = rep; print }' "${out}.tmp" > "$out"
                    rm -f "${out}.tmp"
                else
                    echo "  !! ${cell} rep ${rep} failed, see ${RESULTS}/profile_${cell}_rep${rep}.err" >&2
                fi
            done
        done
    done
done

echo
echo "### pilot finished after $(( ($(date +%s) - started) / 60 )) min; throttling flags: $(vcgencmd get_throttled)"
