#!/usr/bin/env bash
# Sustainable throughput per batch size on the Pi, the throughput side of the source-rate x batch-size experiment (see
# queries/batch-rate/nbeats_batch_throughput.test.tmpl).
#
# Every fixed batch size runs at a ladder of offered rates around its capacity, so the largest rate it still delivers lies
# inside the ladder; the window-only query runs at the top of the ladder, to show where the source and
# the windowing cap the rate. Each run is rate-limited like the latency sweep; only the rate differs.
#
#   ./scripts/run_batch_throughput.sh [reps]
#   BATCH_SIZES="1 2" ./scripts/run_batch_throughput.sh 1      # a subset, e.g. to re-measure
#   ./scripts/process_batch_throughput.py                      # afterwards
#
# Run from ~/navi. Layout it expects (deploy from scripts/benchmarking/e2e/batch_rate/ in the repo):
#   queries/batch-rate/{nbeats_batch_throughput.test.tmpl,batch_rate_fixed.yaml,gen_batch_rate_data.py}
#   scripts/{run_batch_throughput.sh,process_batch_throughput.py,process_batch_rate.py,run_systests_lat.py (with --retries)}
#
# Repetitions are interleaved and run_systests_lat.py fills only missing repetitions, as in run_batch_rate.sh.
set -uo pipefail

REPS="${1:-3}"
# Unset means all; set but empty runs only the window-only query.
BATCH_SIZES="${BATCH_SIZES-1 2 4 8 16 32 64 128}"
WINDOW_ONLY="${WINDOW_ONLY:-1}"
# Seconds of data per run at the offered rate.
DURATION_S="${DURATION_S:-30}"
SAMPLE_S="${SAMPLE_S:-5}"
# The high-rate cells heat the Pi 4 past its 80 C soft limit, where the firmware caps the clock (get_throttled bit 3), so
# every run waits until the SoC has cooled below COOLDOWN_C; 0 disables the wait.
COOLDOWN_C="${COOLDOWN_C:-70}"
# A rate above a batch size's capacity fails deterministically once the backlog exhausts the buffer pool; retrying it
# three times, the runner's default, would only repeat that.
RETRIES="${RETRIES:-1}"
VENV="${VENV:-models-benchmarking/venv}"

# Offered rates [results/s] per batch size, a ~1.25x ladder around each batch size's capacity. Batching amortizes only the
# model invocation: on top of t(B) / B, every result costs ~0.13 ms of windowing, batching and emitting, so the capacity
# levels off near 1 / (0.13 ms + t(B) / B) (trial runs at OpenVINO num_streams 0 x 1 thread: B=1 ~510, 2 ~915, 4 ~1570,
# 8 ~2290 results/s) instead of following the profiled B / t(B). The window-only query sustained 5000 results/s and failed at
# 12500. One stream x 4 threads invokes faster (t(1) 1.66 vs 2.0 ms, t(8) 1.88 vs 2.3 ms), which moves the estimate to ~560,
# ~1010, ~1760 and ~2740 results/s, so B=2, 4 and 8 reach one rate further.
declare -A RATES=(
    [1]="315 400 500 630 800"
    [2]="630 800 1000 1250 1600"
    [4]="1000 1250 1600 2000 2500"
    [8]="1600 2000 2500 3150 4000"
    [16]="2500 3150 4000 5000"
    [32]="3150 4000 5000 6300"
    [64]="3150 4000 5000 6300 8000"
    [128]="3150 4000 5000 6300 8000"
)
WINDOW_ONLY_RATES="${WINDOW_ONLY_RATES:-4000 5000 6300 8000 10000 12500}"

QUERY_DIR=queries/batch-rate
TEMPLATE=nbeats_batch_throughput.test.tmpl
TEST_DIR=queries/batch-throughput
DATA_DIR=datasets/synthetic
# Must match the LatencySink log_path in the template: run_systests_lat.py collects latency_timings.csv from --results-dir.
RESULTS=results/batch_throughput

[[ -f "${QUERY_DIR}/${TEMPLATE}" ]] || { echo "No such template: ${QUERY_DIR}/${TEMPLATE}" >&2; exit 1; }
grep -q "'${RESULTS}/latency_timings.csv'" "${QUERY_DIR}/${TEMPLATE}" \
    || { echo "The LatencySink log_path in ${TEMPLATE} must be ${RESULTS}/latency_timings.csv" >&2; exit 1; }
mkdir -p "$RESULTS" "$TEST_DIR" "$DATA_DIR"

# shellcheck source=/dev/null
source "${VENV}/bin/activate"

# One data file and one .test per rate, each holding DURATION_S seconds at that rate.
prepare_rate() {
    local rate=$1
    local data="${DATA_DIR}/batch_throughput_${rate}.csv"
    local rows=$(( rate * DURATION_S + 75 ))
    if [[ ! -f "$data" || "$(wc -l < "$data")" -ne "$rows" ]]; then
        python3 "${QUERY_DIR}/gen_batch_rate_data.py" --output "$data" --rows "$rows" --step-ms 1
    fi
    sed -e "s|@TUPLE_RATE@|${rate}|" -e "s|@DATA@|$(pwd)/${data}|" "${QUERY_DIR}/${TEMPLATE}" \
        > "${TEST_DIR}/nbeats_throughput_r${rate}.test"
}

sudo -n swapoff -a 2>/dev/null && sudo -n swapon -a 2>/dev/null \
    || echo "WARNING: could not reset swap; results may carry over pressure from the previous run" >&2

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

thermal_sampler &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null' EXIT

# Cells: (batch size, section, rate). The window-only query ignores the batch size and runs under B=1's configuration.
# The grid is written down so that process_batch_throughput.py can list the cells that left no results.
mkdir -p "${RESULTS}/_configs"
echo "batch_size,section,offered_rate" > "${RESULTS}/_grid.csv"
cells=()
for batch_size in $BATCH_SIZES; do
    for rate in ${RATES[$batch_size]}; do
        cells+=("${batch_size} 01 ${rate}")
        echo "${batch_size},01,${rate}" >> "${RESULTS}/_grid.csv"
    done
done
if [[ "$WINDOW_ONLY" == 1 ]]; then
    for rate in $WINDOW_ONLY_RATES; do
        cells+=("1 02 ${rate}")
        echo ",02,${rate}" >> "${RESULTS}/_grid.csv"
    done
fi
for batch_size in $BATCH_SIZES 1; do
    sed "s/^\"optimizer.inference_batch_size\": .*/\"optimizer.inference_batch_size\": [${batch_size}]/" \
        "${QUERY_DIR}/batch_rate_fixed.yaml" > "${RESULTS}/_configs/fixed_${batch_size}.yaml"
done
for rate in $(for cell in "${cells[@]}"; do echo "${cell##* }"; done | sort -un); do
    prepare_rate "$rate"
done

echo "### batch-throughput sweep: reps=$REPS batch sizes=[$BATCH_SIZES] window only=$WINDOW_ONLY ${#cells[@]} cells, ${DURATION_S} s each"
echo "### throttling flags at start: $(vcgencmd get_throttled)"
started=$(date +%s)

for rep in $(seq 1 "$REPS"); do
    log="${RESULTS}/_driver_rep${rep}.log"
    : > "$log"
    for cell in "${cells[@]}"; do
        read -r batch_size section rate <<< "$cell"
        cooldown
        echo "### rep $rep/$REPS  B=${batch_size} :${section} ${rate}/s  ($(date +%H:%M:%S), $(( ($(date +%s) - started) / 60 )) min elapsed)"
        python3 scripts/run_systests_lat.py \
            --inference-config "${RESULTS}/_configs/fixed_${batch_size}.yaml" \
            --queries "batch-throughput/nbeats_throughput_r${rate}.test:${section}" \
            --results-dir "$RESULTS" \
            --repetitions "$rep" \
            --retries "$RETRIES" \
            >> "$log" 2>&1 \
            || echo "  !! B=${batch_size} :${section} ${rate}/s failed every attempt, see ${log}"
    done
done

echo
echo "### sweep finished after $(( ($(date +%s) - started) / 60 )) min; throttling flags: $(vcgencmd get_throttled)"
