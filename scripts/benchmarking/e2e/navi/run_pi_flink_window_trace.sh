#!/usr/bin/env bash
# Per-window timeline of the Flink CWRU arm, at the tuned operating point (2 lanes, the 61 s X228
# input, 10 windows/s), to locate where the windows in its end-to-end tail wait.
#
# Runs a COPY of the CWRU runner against target/flink-trace.jar (the baseline plus WindowTrace, which
# is inert unless FLINK_WINDOW_TRACE is set); the production jar and runner are untouched. Per rep:
#   flink_rep<r>.csv   the usual comparison row
#   trace_rep<r>.csv   one row per window: scheduled/arrival/emitted/lane/infer start+end/sink, nanoTime
#   gc_rep<r>.log      JVM GC log, timenanos decorator = the trace's clock
#   vmstat_rep<r>.log  1 s system samples (run queue, CPU split)
# Outputs in ~/navi/results-flink-window-trace, progress in ~/navi/run_pi_flink_window_trace.log.
set -uo pipefail
cd ~/navi

OUT=results-flink-window-trace
REPS=${REPS:-3}
COOLDOWN=60
IDLE_MAX_C=45
BASE=baselines/flink-openvino-baseline
RUNNER=$BASE/scripts/run-pi-cwru-sarad-trace.sh
log=run_pi_flink_window_trace.log
note() { echo "### $* ($(date '+%F %T'))" | tee -a "$log"; }

[[ -f $BASE/target/flink-trace.jar ]] || { echo "Missing $BASE/target/flink-trace.jar" >&2; exit 1; }
sed 's#^JAR=.*#JAR="$PROJECT_DIR/target/flink-trace.jar"#' $BASE/scripts/run-pi-cwru-sarad.sh > "$RUNNER"
chmod +x "$RUNNER"
grep -q 'flink-trace.jar' "$RUNNER" || { echo "Runner copy does not use the trace jar" >&2; exit 1; }
mkdir -p "$OUT"

temp_c() { vcgencmd measure_temp | sed 's/[^0-9.]//g'; }

for rep in $(seq 1 "$REPS"); do
    sleep "$COOLDOWN"
    while (( $(echo "$(temp_c) > $IDLE_MAX_C" | bc) )); do sleep 10; done
    "$HOME/navi/scripts/reset_swap.sh"
    note "rep $rep start at $(temp_c) C"
    vmstat -n 1 > "$OUT/vmstat_rep$rep.log" &
    vmstat_pid=$!
    env PARALLELISM=2 \
        CSV="$PWD/datasets/cwru-bearing/X228_FE_time_60s.csv" \
        STATS="$PWD/$OUT/flink_rep$rep.csv" \
        QUERY_NAME=flink REPETITION="$rep" \
        FLINK_WINDOW_TRACE="$PWD/$OUT/trace_rep$rep.csv" \
        JDK_JAVA_OPTIONS="-Xlog:gc*:file=$PWD/$OUT/gc_rep$rep.log:timenanos,uptime,level,tags" \
        "$RUNNER" > "$OUT/run_rep$rep.log" 2>&1 || note "!! rep $rep: runner failed"
    kill "$vmstat_pid" 2>/dev/null
    note "rep $rep done at $(temp_c) C, throttled=$(vcgencmd get_throttled)"
done
note "all done"
