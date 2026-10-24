#!/usr/bin/env bash
# Source-rate x batch-size sweep on the Pi (revision experiment; see queries/batch-rate/nbeats_batch_rate.test).
#
# Every rate section of the .test runs under every fixed batch size (batch_rate_fixed.yaml) and under OPTIMIZER_CHOOSES
# (batch_rate_chooses.yaml), so the figure can show where each fixed batch size keeps up and meets the 75 ms SLO and where
# the optimizer's choice falls.
#
#   ./scripts/run_batch_rate.sh [reps]
#   SECTIONS="01 07" CONFIGS=fixed ./scripts/run_batch_rate.sh 1      # a subset, e.g. to re-measure
#   ./scripts/process_batch_rate.py                                    # afterwards
#
# Run from ~/navi. Layout it expects (deploy from scripts/benchmarking/e2e/batch_rate/ in the repo):
#   queries/batch-rate/{nbeats_batch_rate.test,batch_rate_*.yaml,gen_batch_rate_data.py}
#   scripts/{run_batch_rate.sh,process_batch_rate.py,run_systests_lat.py}
#
# Repetitions are interleaved: rep r of every cell runs before rep r+1 of any, so slow drift (thermal state, page cache)
# spreads over all cells instead of biasing whichever ran last. run_systests_lat.py fills only the repetitions a cell is
# missing, so re-invoking this script after an interruption resumes where it stopped.
#
# A background sampler logs the SoC temperature and the throttling flags every SAMPLE_S seconds to thermal.csv;
# process_batch_rate.py joins them to each run, because a throttled run measures the clock governor, not the batch size.
set -uo pipefail

REPS="${1:-3}"
SECTIONS="${SECTIONS:-01 02 03 04 05 06 07}"
CONFIGS="${CONFIGS:-fixed chooses}"
# 60 s of data per run at 1 sample/ms. Cells that fall behind (batch 1 at 1000 results/s) take correspondingly longer.
ROWS="${ROWS:-60000}"
SAMPLE_S="${SAMPLE_S:-5}"
# Each run waits until the SoC has cooled below COOLDOWN_C, so that no run inherits the previous one's heat; 0 disables the wait.
COOLDOWN_C="${COOLDOWN_C:-0}"
VENV="${VENV:-models-benchmarking/venv}"

QUERY_DIR=queries/batch-rate
TEST_FILE=nbeats_batch_rate.test
DATA=datasets/synthetic/batch_rate.csv
# Must match the LatencySink log_path in the .test: run_systests_lat.py collects latency_timings.csv from --results-dir.
RESULTS=results/batch_rate

[[ -f "${QUERY_DIR}/${TEST_FILE}" ]] || { echo "No such .test: ${QUERY_DIR}/${TEST_FILE}" >&2; exit 1; }
grep -q "'${RESULTS}/latency_timings.csv'" "${QUERY_DIR}/${TEST_FILE}" \
    || { echo "The LatencySink log_path in ${TEST_FILE} must be ${RESULTS}/latency_timings.csv" >&2; exit 1; }
mkdir -p "$RESULTS" "$(dirname "$DATA")"

# ovc (model import on every run) and PyYAML (runner) come from the benchmarking venv.
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

if [[ ! -f "$DATA" || "$(wc -l < "$DATA")" -ne "$ROWS" ]]; then
    python3 "${QUERY_DIR}/gen_batch_rate_data.py" --output "$DATA" --rows "$ROWS" --step-ms 1
fi

# Swap left behind by an earlier memory-heavy run inflates the next measurements (see run_systems.sh).
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

# run_systests_lat.py abandons every remaining cell of its invocation once one run fails all its retries. A batch size that
# cannot keep up accumulates window results until the buffer pool runs out and the query fails (batch 1 at 1000 results/s
# after ~30 s), so every (configuration, section) pair gets its own invocation and a failed cell stays a gap in the grid.
# The fixed-batch YAML is split into one YAML per batch size; with the same keys in the same order, the result directory
# names match those of a run over the whole YAML.
variants=()
for config in $CONFIGS; do
    if [[ "$config" == fixed ]]; then
        mkdir -p "${RESULTS}/_configs"
        batch_sizes=$(sed -n 's/^"optimizer.inference_batch_size": \[\(.*\)\]/\1/p' "${QUERY_DIR}/batch_rate_fixed.yaml" | tr -d ,)
        for batch_size in $batch_sizes; do
            sed "s/^\"optimizer.inference_batch_size\": .*/\"optimizer.inference_batch_size\": [${batch_size}]/" \
                "${QUERY_DIR}/batch_rate_fixed.yaml" > "${RESULTS}/_configs/fixed_${batch_size}.yaml"
            variants+=("${RESULTS}/_configs/fixed_${batch_size}.yaml")
        done
    else
        variants+=("${QUERY_DIR}/batch_rate_${config}.yaml")
    fi
done

echo "### batch-rate sweep: reps=$REPS sections=[$SECTIONS] configs=[$CONFIGS] rows=$ROWS"
echo "### throttling flags at start: $(vcgencmd get_throttled)"
started=$(date +%s)

for rep in $(seq 1 "$REPS"); do
    for variant in "${variants[@]}"; do
        name=$(basename "$variant" .yaml)
        log="${RESULTS}/_driver_rep${rep}_${name}.log"
        : > "$log"
        echo "### rep $rep/$REPS  ${name}  ($(date +%H:%M:%S), $(( ($(date +%s) - started) / 60 )) min elapsed)"
        for section in $SECTIONS; do
            cooldown
            python3 scripts/run_systests_lat.py \
                --inference-config "$variant" \
                --queries "batch-rate/${TEST_FILE}:${section}" \
                --results-dir "$RESULTS" \
                --repetitions "$rep" \
                >> "$log" 2>&1 \
                || echo "  !! section ${section} failed every attempt, see ${log}"
        done
    done
done

echo
echo "### sweep finished after $(( ($(date +%s) - started) / 60 )) min; throttling flags: $(vcgencmd get_throttled)"
