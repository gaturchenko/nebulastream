#!/usr/bin/env bash
# Post-join batching experiment on the Pi (revision: an optimizer choice compared against its alternatives).
#
# k audio streams are joined and MHAtt-RNN infers on each stream's window; the inference operator over the k payloads runs
# BATCHED (one batch-k invocation) or SEQUENTIAL (k single invocations), forced or chosen by OPTIMIZER_CHOOSES from its
# profile. The forced plans run at every offered rate (window ADVANCE) to measure latency and where each falls behind;
# OPTIMIZER_CHOOSES deterministically deploys one of the two forced plans, so it runs once per k and dedup setting, to record
# its decision and confirm that its performance is that of the plan it chose.
#
#   ./scripts/run_post_join.sh [reps]
#   KS="2 4" ADVANCES_MS="1000 500" ./scripts/run_post_join.sh 1      # a subset
#   ./scripts/process_post_join.py                                    # afterwards
#
# Run from ~/navi. Layout it expects (deploy from scripts/benchmarking/e2e/post_join/ in the repo):
#   queries/post-join/{gen_post_join_tests.py,gen_post_join_data.py}       the .test files are generated into queries/post-join/
#   scripts/{run_post_join.sh,process_post_join.py,process_batch_rate.py,run_systests_lat.py (with --retries)}
#
# Repetitions are interleaved and run_systests_lat.py fills only missing repetitions, as in run_batch_rate.sh.
set -uo pipefail

REPS="${1:-3}"
KS="${KS:-2 3 4 6 8}"
# One joined record per ADVANCE; the generator writes the inference sections in this order.
ADVANCES_MS="${ADVANCES_MS:-1000 500 333 200}"
# Strategy:dedup pairs that run at every rate. Batch deduplication makes the batched sessions dynamic (see the pilot).
CONFIGS="${CONFIGS:-BATCHED:true SEQUENTIAL:true}"
# The static-batch control: both forced plans without deduplication, at these k only.
STATIC_KS="${STATIC_KS:-2 8}"
# OPTIMIZER_CHOOSES runs once per k and dedup setting, at this ADVANCE (must be one of ADVANCES_MS).
CHOOSES_DEDUP="${CHOOSES_DEDUP:-true false}"
CHOOSES_ADVANCE_MS="${CHOOSES_ADVANCE_MS:-1000}"
# Join-only queries at these k and ADVANCEs show where the sources, windows and joins alone cap the rate.
CONTROL_KS="${CONTROL_KS:-8}"
CONTROL_ADVANCES_MS="${CONTROL_ADVANCES_MS:-200}"
# Seconds of audio per run.
DURATION_S="${DURATION_S:-60}"
SAMPLE_S="${SAMPLE_S:-5}"
COOLDOWN_C="${COOLDOWN_C:-70}"
# A plan that falls behind fails deterministically once its backlog exhausts the buffer pool; retrying only repeats that.
RETRIES="${RETRIES:-1}"
VENV="${VENV:-models-benchmarking/venv}"
RECORDINGS="${RECORDINGS:-datasets/musan/musan-us-gov-0242.csv datasets/musan/musan-us-gov-0096.csv datasets/musan/musan-us-gov-0021.csv}"
MODEL="${MODEL:-models-benchmarking/models/mhatt-rnn/saved_model}"

QUERY_DIR=queries/post-join
DATA_DIR=datasets/post_join
# Must match the generator's --latency-log: run_systests_lat.py collects latency_timings.csv from --results-dir.
RESULTS=results/post_join

for script in gen_post_join_tests.py gen_post_join_data.py; do
    [[ -f "${QUERY_DIR}/${script}" ]] || { echo "No ${QUERY_DIR}/${script}" >&2; exit 1; }
done
mkdir -p "$RESULTS/_configs" "$DATA_DIR"

# ovc (model import on every run) and PyYAML (runner) come from the benchmarking venv.
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

section_of() {  # section_of <advance> <list>: 1-based position of <advance> in <list>, empty if absent
    local i=1
    for advance in $2; do
        [[ "$advance" == "$1" ]] && { printf '%02d' "$i"; return; }
        i=$(( i + 1 ))
    done
}
[[ -n "$(section_of "$CHOOSES_ADVANCE_MS" "$ADVANCES_MS")" ]] || { echo "CHOOSES_ADVANCE_MS must be one of ADVANCES_MS" >&2; exit 1; }
number_of_advances=$(wc -w <<< "$ADVANCES_MS")

max_k=$(tr ' ' '\n' <<< "$KS $STATIC_KS $CONTROL_KS" | sort -n | tail -1)
# shellcheck disable=SC2086 # RECORDINGS is a list on purpose
python3 "${QUERY_DIR}/gen_post_join_data.py" --recordings $RECORDINGS --streams "$max_k" --duration-s "$DURATION_S" \
    --output-dir "$DATA_DIR" || exit 1
# shellcheck disable=SC2086
python3 "${QUERY_DIR}/gen_post_join_tests.py" \
    --ks $(tr ' ' '\n' <<< "$KS $STATIC_KS $CONTROL_KS" | sort -nu) \
    --advances-ms $ADVANCES_MS --control-advances-ms $CONTROL_ADVANCES_MS \
    --data-dir "$(pwd)/${DATA_DIR}" --model "$(pwd)/${MODEL}" \
    --latency-log "${RESULTS}/latency_timings.csv" --output-dir "$QUERY_DIR" || exit 1

config_for() {  # config_for <strategy> <dedup>: path of the one-combination YAML
    local path="${RESULTS}/_configs/${1}_dedup-${2}.yaml"
    printf '"optimizer.post_join_inference_strategy": [%s]\n"worker.default_query_execution.inference.use_batch_deduplication": [%s]\n' \
        "$1" "$2" > "$path"
    echo "$path"
}

# Cells: k, strategy, dedup, section. The grid is written down so that process_post_join.py can list cells without results.
echo "k,strategy,dedup,section,advance_ms,query" > "${RESULTS}/_grid.csv"
cells=()
add_cell() {
    cells+=("$1 $2 $3 $4")
    echo "$1,$2,$3,$4,$5,$6" >> "${RESULTS}/_grid.csv"
}
for k in $KS; do
    for config in $CONFIGS; do
        for advance in $ADVANCES_MS; do
            add_cell "$k" "${config%%:*}" "${config#*:}" "$(section_of "$advance" "$ADVANCES_MS")" "$advance" inference
        done
    done
done
for k in $STATIC_KS; do
    for strategy in BATCHED SEQUENTIAL; do
        for advance in $ADVANCES_MS; do
            add_cell "$k" "$strategy" false "$(section_of "$advance" "$ADVANCES_MS")" "$advance" inference
        done
    done
done
for k in $KS; do
    for dedup in $CHOOSES_DEDUP; do
        add_cell "$k" OPTIMIZER_CHOOSES "$dedup" "$(section_of "$CHOOSES_ADVANCE_MS" "$ADVANCES_MS")" "$CHOOSES_ADVANCE_MS" inference
    done
done
# The join-only query has no inference operator, so the strategy is irrelevant; it runs under BATCHED without dedup.
for k in $CONTROL_KS; do
    for advance in $CONTROL_ADVANCES_MS; do
        section=$(printf '%02d' $(( number_of_advances + 10#$(section_of "$advance" "$CONTROL_ADVANCES_MS") )))
        add_cell "$k" BATCHED false "$section" "$advance" join-only
    done
done

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

echo "### post-join sweep: reps=$REPS ks=[$KS] advances=[$ADVANCES_MS] configs=[$CONFIGS] ${#cells[@]} cells, ${DURATION_S} s each"
echo "### throttling flags at start: $(vcgencmd get_throttled)"
started=$(date +%s)

for rep in $(seq 1 "$REPS"); do
    log="${RESULTS}/_driver_rep${rep}.log"
    : > "$log"
    for cell in "${cells[@]}"; do
        read -r k strategy dedup section <<< "$cell"
        cooldown
        echo "### rep $rep/$REPS  k=${k} ${strategy} dedup=${dedup} :${section}  ($(date +%H:%M:%S), $(( ($(date +%s) - started) / 60 )) min elapsed)"
        python3 scripts/run_systests_lat.py \
            --inference-config "$(config_for "$strategy" "$dedup")" \
            --queries "post-join/post_join_k${k}.test:${section}" \
            --results-dir "$RESULTS" \
            --repetitions "$rep" \
            --retries "$RETRIES" \
            >> "$log" 2>&1 \
            || echo "  !! k=${k} ${strategy} dedup=${dedup} :${section} failed every attempt, see ${log}"
    done
done

echo
echo "### sweep finished after $(( ($(date +%s) - started) / 60 )) min; throttling flags: $(vcgencmd get_throttled)"
