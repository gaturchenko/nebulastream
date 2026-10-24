#!/usr/bin/env bash
# Single entry point for the framework comparison on the Pi.
# Runs three serving/streaming systems on the same workload:
#   1. Flink + OpenVINO   (baseline; substitutes the former NES-UDF case)
#   2. NES MODEL_INFERENCE
#   3. TorchServe
# Each writes rows into results/ in a shared comparison schema; the per-run trace
# subdirectories are cleaned at the end, the processed CSVs are kept.
#
# Usage, from the navi root (relative paths below assume CWD=~/navi):
#
#   ./scripts/run_systems.sh [profile]        # profile = waymo (default) | cwru | musan
#
# ============================================================================
# Experiment configuration lives in ONE file per workload: scripts/systems/<profile>.env.
# That file names the model, the dataset, the .test, the TorchServe handler and the Flink
# runner, and carries the input geometry the TorchServe handler needs. To add a workload,
# copy an .env and adjust it -- nothing in this script is workload-specific.
#
# The pieces that cannot be sourced from here are the .test file itself (the RESIZE_IMAGE /
# ARRAY_AGG window geometry and the CREATE MODEL path are NES-side) and the defaults inside
# the Flink runner. Keep all three in sync; each .env says which values must match.
# ============================================================================
set -uo pipefail

# Which arms to run. A change that touches only one system (a sink rewrite, a TorchServe worker
# count) does not need the other two re-measured, and re-running them costs minutes per pass and
# adds run-to-run noise to numbers that cannot have moved.
STAGES="${STAGES:-flink nes torchserve}"
stage_enabled() { [[ " $STAGES " == *" $1 "* ]]; }

PROFILE="${1:-waymo}"
PROFILE_FILE="scripts/systems/${PROFILE}.env"
[[ -f "$PROFILE_FILE" ]] || { echo "No such profile: $PROFILE_FILE" >&2; exit 1; }

# A sweep driver varies one dimension across otherwise identical runs, so the values it exports
# have to survive the profile being sourced -- `source` would overwrite them with the profile's
# defaults. Captured here and reapplied below. TEST_FILE is what carries a regenerated .test (the
# ingestion rate lives in it as MOCK_TUPLE_RATE); RATE is the same number for the Flink runner.
_override_test_file="${TEST_FILE:-}"

# shellcheck source=/dev/null
source "$PROFILE_FILE"

[[ -n "$_override_test_file" ]] && TEST_FILE="$_override_test_file"

: "${WORKLOAD:?profile must set WORKLOAD}"
: "${MODEL_SOURCE:?profile must set MODEL_SOURCE}"
: "${MODEL_IR:?profile must set MODEL_IR}"
: "${TS_MODEL_NAME:?profile must set TS_MODEL_NAME}"
: "${TS_HANDLER:?profile must set TS_HANDLER}"
: "${TS_SERIALIZED:?profile must set TS_SERIALIZED}"
: "${GEOMETRY_JSON:?profile must set GEOMETRY_JSON}"
: "${TEST_FILE:?profile must set TEST_FILE}"
: "${DATA_DIR:?profile must set DATA_DIR}"
: "${ST_CONFIG:?profile must set ST_CONFIG}"
: "${UDF_CONFIG:?profile must set UDF_CONFIG}"
: "${FLINK_RUN:?profile must set FLINK_RUN}"

# TorchServe's load_models=all scans the whole store directory, so a store that also holds the
# model's other artefacts (a SavedModel directory, a .tar) makes it try to load those as models too;
# the bogus workers then crashloop. A profile can point MODEL_STORE at a directory holding only the
# archive this run builds.
MODEL_STORE="${MODEL_STORE:-$(dirname "$TS_SERIALIZED")}"
mkdir -p "$MODEL_STORE"
VENV=./models-benchmarking/venv
TS_GEOMETRY=scripts/torchserve/input_geometry.json
TS_CONFIG=scripts/torchserve/config.properties
# TorchServe's own Prometheus counters, scraped around the run. They are what separates the model
# invocation inside the handler (ts_inference_latency_microseconds) and the time a request waits in
# the worker queue (ts_queue_latency_microseconds) from the HTTP/IPC cost the sink sees -- the cost
# breakdown the revision asks for. Without this the sink's round-trip is one opaque number.
TS_METRICS_URL="${TS_METRICS_URL:-http://127.0.0.1:8082/metrics}"
TS_EXTRA_FILES="${TS_EXTRA_FILES:+${TS_EXTRA_FILES},}${TS_GEOMETRY}"

echo "=== Framework comparison: $WORKLOAD (profile $PROFILE)"

# Release swap before measuring. Swap filled by an earlier run is not reclaimed when that run
# exits, and it measurably contaminates the next one: immediately after the 3-worker CWRU sweep
# left 1.68 GB swapped, a 1-worker pass measured the handler at 138.5 ms with 60.6 ms of queueing;
# the identical pass after swapoff/swapon measured 79.5 ms and 3.8 ms. The ablation scripts have
# always done this (run_audio*.sh); the systems runs did not, which is why their results depended
# on what had run before them.
if command -v sudo >/dev/null 2>&1; then
    sudo swapoff -a 2>/dev/null && sudo swapon -a 2>/dev/null \
        || echo "WARNING: could not reset swap; results may carry over pressure from the previous run" >&2
fi

# Thermal state per arm. The Pi 4's firmware lowers the clock at its 80 C soft limit (vcgencmd get_throttled bits 0-3), and a
# throttled arm measures the clock governor, not the system. Each arm waits until the SoC is below COOLDOWN_C (0 = no wait),
# samples temperature and flags every SAMPLE_S seconds while it runs, and is re-run up to THROTTLE_RETRIES times if any sample
# was flagged. Every attempt is appended to results-thermal/arms.csv, which lies outside results/ like results-reps/; REP_LABEL
# (set by the wrappers) names the repetition there.
SAMPLE_S="${SAMPLE_S:-5}"
COOLDOWN_C="${COOLDOWN_C:-0}"
THROTTLE_RETRIES="${THROTTLE_RETRIES:-0}"
THERMAL_LOG=results-thermal/arms.csv
mkdir -p "$(dirname "$THERMAL_LOG")"
[[ -s "$THERMAL_LOG" ]] || echo "time,workload,test_file,rate,rep_label,arm,attempt,temp_c_max,flagged_samples,kept" > "$THERMAL_LOG"

cooldown() {
    [[ "$COOLDOWN_C" == 0 ]] && return
    local waited=0
    while (( $(vcgencmd measure_temp | sed 's/[^0-9]*\([0-9]*\).*/\1/') >= COOLDOWN_C )); do
        sleep 5
        waited=$(( waited + 5 ))
    done
    (( waited > 0 )) && echo "  (cooled down for ${waited} s)"
}

# run_arm <arm> <cleanup function> <command...>: runs one arm under the thermal rule above. The cleanup function removes a
# throttled attempt's output before the next attempt.
run_arm() {
    local arm="$1" cleanup="$2"
    shift 2
    local attempt samples sampler max flagged
    for (( attempt = 1; attempt <= THROTTLE_RETRIES + 1; attempt++ )); do
        cooldown
        samples=$(mktemp)
        ( while :; do
            echo "$(vcgencmd measure_temp | sed 's/[^0-9.]//g') $(vcgencmd get_throttled | cut -d= -f2)"
            sleep "$SAMPLE_S"
          done ) > "$samples" &
        sampler=$!
        "$@"
        kill "$sampler" 2>/dev/null
        wait "$sampler" 2>/dev/null
        max=$(awk '{ if ($1 > m) m = $1 } END { print m + 0 }' "$samples")
        flagged=0
        while read -r _ flags; do
            [[ -n "$flags" ]] && (( flags & 0xF )) && flagged=$(( flagged + 1 ))
        done < "$samples"
        rm -f "$samples"
        local kept=yes
        (( flagged > 0 && attempt <= THROTTLE_RETRIES )) && kept=no
        [[ "$kept" == yes && "$flagged" -gt 0 ]] && kept=throttled
        echo "$(date +%FT%T),${WORKLOAD},${TEST_FILE},${RATE:-},${REP_LABEL:-},${arm},${attempt},${max},${flagged},${kept}" >> "$THERMAL_LOG"
        echo "  ${arm}: attempt ${attempt}, max ${max} C, ${flagged} throttled samples (${kept})"
        [[ "$kept" != no ]] && return
        "$cleanup"
    done
}
clean_flink() { rm -f "$FLINK_STATS"; }
clean_results_dirs() { rm -rf ./results/*/; }

# Clear stale per-query result directories before measuring. process_results_lat.py scans the whole
# results/ tree, so a directory left behind by an earlier run -- a manual systest invocation, or a
# run interrupted before its trailing cleanup -- is silently folded into this run's numbers and
# cannot be told apart afterwards: the combination label naming the directory is identical.
# Other experiments keep their results under results/ too, and the processors scan the whole tree, so this only runs when every
# directory there is a systems run (its query is a *systems*.test); move anything else out of results/ first.
for dir in ./results/*/; do
    [[ -d "$dir" ]] || continue
    compgen -G "${dir}*systems*.test_*" > /dev/null \
        || { echo "ERROR: $dir is not a systems run; move other experiments out of results/ before running this" >&2; exit 1; }
done
rm -rf ./results/*/
stage_enabled nes && rm -f "results/nes_${WORKLOAD}.csv"
stage_enabled torchserve && rm -f "results/torchserve_${WORKLOAD}.csv"

# Activate the benchmarking venv UP FRONT. It provides `ovc` (needed to load the NES model in the
# MODEL_INFERENCE / CREATE MODEL path AND to build the Flink OpenVINO IR) and pandas (needed by the
# result post-processing). The NES phase fails to bind the model without ovc on PATH, so this must
# precede it.
source "$VENV/bin/activate"

# The TorchServe handler reads the input geometry from input_geometry.json bundled into the .mar
# (workers don't inherit the parent env), so write it here before archiving.
echo "$GEOMETRY_JSON" > "$TS_GEOMETRY"

# Rebuild the OpenVINO IR whenever it is missing or older than the model, so Flink never runs a
# stale one after a model is re-uploaded. Not forced every run: ovc on a large model is itself a
# heavy job, and running it immediately before Flink opens one engine per lane has already once
# left too little memory for the last engine to finish compiling, which stalls the whole job.
if [[ ! -f "$MODEL_IR" || "$MODEL_SOURCE" -nt "$MODEL_IR" ]]; then
    echo "Converting $MODEL_SOURCE -> $MODEL_IR (ovc)"
    ovc "$MODEL_SOURCE" --output_model "$MODEL_IR" --compress_to_fp16=False
else
    echo "Reusing up-to-date IR: $MODEL_IR"
fi

# ---- Flink + OpenVINO baseline (replaces the former NES-UDF case) ----
# Java/Flink job (not a NES systest); manages its own OpenVINO/JNI env and writes a comparison-schema
# row directly to results/flink_openvino_stats.csv. Fresh file so rows don't accumulate. The runner
# carries this workload's window geometry, parallelism and ingestion rate as its defaults; only the
# comparison-key columns and the model are passed in, so the rate stays matched to MOCK_TUPLE_RATE.
# Per-workload, like the NES and TorchServe outputs. The runner defaults to one shared
# flink_openvino_stats.csv, so without this each workload silently overwrites the previous
# workload's Flink row and only the last one survives on disk.
# Absolute: the Flink runner cd's into its own project directory before launching the job, so a
# relative path would land in baselines/flink-openvino-baseline/results/ instead of here.
FLINK_STATS="$PWD/results/flink_${WORKLOAD}.csv"
rm -f "$FLINK_STATS"

# Image workloads also take their frame geometry from the profile. The runner has defaults of its
# own, but they belong to whichever model it was written for, and a stale default is not visible in
# a result -- it surfaces as the model rejecting the tensor ("Expected 28800, but got 359424"), or
# worse, not at all. GEOMETRY_JSON is what the TorchServe handler already uses, so deriving the
# Flink geometry from it keeps one source of truth per workload. Array workloads carry "samples"
# instead, and their runners own the window geometry.
FLINK_GEOMETRY=()
if geometry_wh=$(python3 -c '
import json, sys
g = json.loads(sys.argv[1])
if "width" in g and "height" in g:
    print(g["width"], g["height"])
' "$GEOMETRY_JSON") && [[ -n "$geometry_wh" ]]; then
    read -r geometry_width geometry_height <<<"$geometry_wh"
    FLINK_GEOMETRY=(WIDTH="$geometry_width" HEIGHT="$geometry_height")
    echo "Flink frame geometry from profile: ${geometry_width}x${geometry_height}"
fi

# The runner's own RATE default is the profile's operating point and normally the right one; a
# sweep overrides it, and the SAME number must already be in the .test this run uses, or the three
# systems are measured at different offered loads.
FLINK_RATE=()
[[ -n "${RATE:-}" ]] && FLINK_RATE=(RATE="$RATE")

# Flink's inference lanes are NES's worker threads: taken from the .test this run uses, so the two cannot drift apart.
TEST_PATH="queries/${TEST_FILE}"
FLINK_PARALLELISM=$(sed -n 's/^GlobalConfiguration worker.query_engine.number_of_worker_threads: \[\([0-9]\+\)\].*/\1/p' "$TEST_PATH" | head -1)
[[ -n "$FLINK_PARALLELISM" ]] || { echo "No number_of_worker_threads in $TEST_PATH" >&2; exit 1; }
echo "Flink inference lanes = NES worker threads = $FLINK_PARALLELISM"

if stage_enabled flink; then
    run_arm flink clean_flink env QUERY_NAME="${QUERY_NAME:-flink}" PARAM_NAME=baseline PARAM_VALUE=flink REPETITION=0 \
        MODEL="$MODEL_IR" STATS="$FLINK_STATS" PARALLELISM="$FLINK_PARALLELISM" "${FLINK_GEOMETRY[@]}" "${FLINK_RATE[@]}" \
        "$FLINK_RUN"
fi

# ---- NES MODEL_INFERENCE ----
# Processed straight away, so that a TorchServe attempt can be discarded without touching NES's traces.
if stage_enabled nes; then
    run_arm nes clean_results_dirs python3 scripts/run_systests_lat.py \
        --inference-config "$ST_CONFIG" \
        --queries "${TEST_FILE}:02" \
        --data "$DATA_DIR" \
        --repetitions 1
    python3 scripts/process_results_lat.py \
        --results-dir results \
        --output-csv "results/nes_${WORKLOAD}.csv"
    rm -rf ./results/*/
fi

# ---- TorchServe lane check ----
# The HTTP sink opens `connections` keep-alive sockets, so the client is not the thing limiting how
# many requests can be outstanding. TorchServe's own worker count stays at the config default (1),
# because more workers measurably HURT on this hardware -- see scripts/run_torchserve_workers.sh,
# which swept CWRU at 1/2/3 workers x 3 repetitions:
#
#   workers   handler   e2e P95   e2e P99   peak RSS   swap
#         1   81.9 ms   151.9     181.2       940 MB   275 MB
#         2   83.6 ms   175.9     217.7      1879 MB   319 MB
#         3  116.4 ms   296.5     433.5      2812 MB  1681 MB
#
# Throughput was ~10.3 records/s in every case -- the offered rate -- so none of these is saturated
# and the extra workers had nothing to queue for. What they did buy was contention: the TorchServe
# query still runs NES's worker threads for decoding and windowing, so N workers means 3 + N
# runnable threads on 4 cores, and each worker holds its own compiled copy of the 190 MB model.
#
# So the check here is NOT parity of lane counts. It is only that the sink can keep the server's
# workers busy: connections >= workers. The baseline runs in the configuration that measured best
# for it, which is also the configuration we can defend.
SINK_CONNECTIONS=$(sed -n 's/.*[^0-9]\([0-9]\+\) AS .SINK..connections.*/\1/p' "$TEST_PATH" | head -1)
CONFIG_WORKERS=$(sed -n 's/^default_workers_per_model=\([0-9]\+\).*/\1/p' "$TS_CONFIG" | tail -1)

# TS_WORKERS overrides the server's worker count, for sweeping it against a fixed client.
if [[ -n "${TS_WORKERS:-}" ]]; then
    export TS_DEFAULT_WORKERS_PER_MODEL="$TS_WORKERS"
fi
EFFECTIVE_WORKERS="${TS_WORKERS:-${CONFIG_WORKERS:-1}}"

if [[ -n "$SINK_CONNECTIONS" && "$SINK_CONNECTIONS" -lt "$EFFECTIVE_WORKERS" ]]; then
    echo "ERROR: the sink opens $SINK_CONNECTIONS connection(s) but TorchServe runs $EFFECTIVE_WORKERS worker(s)." >&2
    echo "       The client would be the limiter, which is the artefact this setup exists to avoid." >&2
    deactivate
    exit 1
fi
echo "TorchServe workers = $EFFECTIVE_WORKERS, sink connections = ${SINK_CONNECTIONS:-1}"

# ---- TorchServe ----
if stage_enabled torchserve; then
    # 1. Stop any running/orphaned instance FIRST, so we don't archive a loaded model or
    #    double-start. `|| true` because --stop returns non-zero when nothing is running.
    torchserve --stop 2>/dev/null || true
    sleep 2

    # 2. (Re)build the model archive while the server is down. TS_EXTRA_FILES carries whatever the
    #    serialized file needs beside it -- the geometry always, plus e.g. an IR's .bin.
    torch-model-archiver --model-name "$TS_MODEL_NAME" --version 1.0 \
      --serialized-file "$TS_SERIALIZED" \
      --handler "$TS_HANDLER" \
      --extra-files "$TS_EXTRA_FILES" \
      --export-path "$MODEL_STORE" --force

    # 3. Start ONCE (config.properties auto-loads every .mar in the model store — the store is the
    #    model's own directory, so that's exactly this archive), then WAIT for it to load before
    #    hitting the endpoints — first load on the Pi CPU takes 10-30 s.
    torchserve --start --ncs --ts-config "$TS_CONFIG" --model-store "$MODEL_STORE"

    echo "Waiting for TorchServe to become healthy..."
    healthy=0
    for i in $(seq 1 60); do
      if curl -sf 127.0.0.1:8080/ping 2>/dev/null | grep -q Healthy; then
        echo "TorchServe healthy after ${i}s"
        healthy=1
        break
      fi
      sleep 1
    done
    if [ "$healthy" -ne 1 ]; then
      echo "ERROR: TorchServe did not become healthy in 60s. Check logs/model_log.log." >&2
      torchserve --stop 2>/dev/null || true
      deactivate
      exit 1
    fi

    curl -s 127.0.0.1:8080/ping
    curl -s 127.0.0.1:8082/metrics | head

    run_arm torchserve clean_results_dirs python3 scripts/run_systests_lat.py \
        --inference-config "$UDF_CONFIG" \
        --queries "${TEST_FILE}:03" \
        --data "$DATA_DIR" \
        --torchserve-metrics-url "$TS_METRICS_URL" \
        --repetitions 1
fi

# Process the TorchServe results (NES's were processed after its arm). It runs only if its arm did: scanning for traces that
# were never produced would overwrite a good CSV with an empty one.
if stage_enabled torchserve; then
    python3 scripts/process_torchserve_results.py \
        --results-dir results \
        --output-csv "results/torchserve_${WORKLOAD}.csv"
    torchserve --stop
fi

deactivate
rm -rf ./results/*/

echo
echo "=== $WORKLOAD done. Comparison rows:"
echo "  Flink      : $FLINK_STATS"
echo "  NES        : results/nes_${WORKLOAD}.csv"
echo "  TorchServe : results/torchserve_${WORKLOAD}.csv"
