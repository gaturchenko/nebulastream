#!/usr/bin/env bash
# Runs the cells of one sensitivity query (gen_sensitivity_tests.py) on the Pi, like run_batch_rate.sh.
#
#   ./scripts/run_sensitivity_sweep.sh <name>     # name: sarad, nbeats_w37, nbeats_w150, nbeats_w300
#
# Run from ~/navi. Repetitions are interleaved, so slow drift spreads over all cells; run_systests_lat.py fills only the
# repetitions a cell is missing, so re-invoking this script resumes. Every (section, config) pair is its own invocation, since a
# cell that falls behind until the buffer pool runs out would otherwise abandon the remaining ones and must stay a gap in the grid.
set -uo pipefail

NAME="$1"
SAMPLE_S="${SAMPLE_S:-2}"
COOLDOWN_C="${COOLDOWN_C:-60}"
VENV="${VENV:-models-benchmarking/venv}"
QUERY_DIR=queries/sensitivity
RESULTS="results/sensitivity/${NAME}"
CELLS="${QUERY_DIR}/cells_${NAME}.txt"

[[ -f "$CELLS" ]] || { echo "No such cell list: ${CELLS}" >&2; exit 1; }
grep -q "'${RESULTS}/latency_timings.csv'" "${QUERY_DIR}/${NAME}.test" \
    || { echo "The sink path of ${NAME}.test must be ${RESULTS}/latency_timings.csv" >&2; exit 1; }
mkdir -p "$RESULTS"

# shellcheck source=/dev/null
source "${VENV}/bin/activate"
scripts/reset_swap.sh || echo "WARNING: could not reset swap" >&2

thermal_sampler() {
    [[ -s "${RESULTS}/thermal.csv" ]] || echo "time_us,temp_c,throttled" > "${RESULTS}/thermal.csv"
    while :; do
        echo "$(( $(date +%s%N) / 1000 )),$(vcgencmd measure_temp | sed 's/[^0-9.]//g'),$(vcgencmd get_throttled | cut -d= -f2)" \
            >> "${RESULTS}/thermal.csv"
        sleep "$SAMPLE_S"
    done
}
cooldown() {
    while (( $(vcgencmd measure_temp | sed 's/[^0-9]*\([0-9]*\).*/\1/') >= COOLDOWN_C )); do sleep 5; done
}
thermal_sampler &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null' EXIT

started=$(date +%s)
max_reps=$(awk '{ if ($3 > max) max = $3 } END { print max }' "$CELLS")
for rep in $(seq 1 "$max_reps"); do
    while read -r section config reps; do
        (( rep > reps )) && continue
        cooldown
        log="${RESULTS}/_driver_${config}_${section}.log"
        echo "### ${NAME} rep ${rep}: section ${section} ${config} ($(date +%H:%M:%S), $(( ($(date +%s) - started) / 60 )) min elapsed)"
        python3 scripts/run_systests_lat.py \
            --inference-config "${QUERY_DIR}/${config}.yaml" \
            --queries "sensitivity/${NAME}.test:${section}" \
            --results-dir "$RESULTS" \
            --repetitions "$rep" \
            >> "$log" 2>&1 \
            || echo "  !! section ${section} ${config} failed every attempt, see ${log}"
    done < "$CELLS"
done
echo "### ${NAME} finished after $(( ($(date +%s) - started) / 60 )) min; throttling flags: $(vcgencmd get_throttled)"
