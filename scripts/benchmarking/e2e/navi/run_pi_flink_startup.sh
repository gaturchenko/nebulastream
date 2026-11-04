#!/usr/bin/env bash
# Flink's end-to-end P95 with and without its start-up, on all three workloads, at the tuned operating points
# (Waymo 1 lane, CWRU 2 lanes, MUSAN 1 lane; the inputs and run_systems.sh path of the tuning finals), 5 repetitions.
#
# The per-window trace showed Flink's CWRU tail is its first ~3 s: both lanes overlap at ~200 ms per inference, which
# exactly matches the offered rate, so the start-up backlog cannot drain. These runs give, from the SAME runs, the
# full-run P95 and the P95 of the results whose input arrived after the first STEADY_AFTER_S seconds.
#
# The production jar is swapped for target/flink-trace.jar (the baseline plus WindowTrace, inert unless
# FLINK_WINDOW_TRACE is set) for the duration and restored on exit; results/ is held aside like the tuning drivers do.
# Outputs: ~/navi/results-flink-startup/<workload>/{flink,trace}_rep<r>.csv, startup_runs.csv (one row per run).
# Progress: ~/navi/run_pi_flink_startup.log
set -uo pipefail
cd ~/navi
log=run_pi_flink_startup.log
PY=models-benchmarking/venv/bin/python
REPS="${REPS:-5}"
STEADY_AFTER_S="${STEADY_AFTER_S:-5}"
OUT=results-flink-startup
TARGET=baselines/flink-openvino-baseline/target
JAR=$TARGET/flink-openvino-baseline-1.0-SNAPSHOT.jar
export SAMPLE_S=2 COOLDOWN_C="${COOLDOWN_C:-60}" THROTTLE_RETRIES="${THROTTLE_RETRIES:-2}"
note() { echo "### $* ($(date '+%F %T'))" | tee -a "$log"; }

: > "$log"
pgrep -f "[b]in/systest|[r]un_pi_.*\.sh|[r]un_systems|[j]ava " | grep -v "^$$\$" > /dev/null \
    && { note "ABORT: an experiment is running"; exit 1; }
[[ -e "$OUT" ]] && { note "ABORT: $OUT already exists"; exit 1; }
[[ -f "$TARGET/flink-trace.jar" ]] || { note "ABORT: no $TARGET/flink-trace.jar"; exit 1; }
[[ -f datasets/musan/musan-us-gov-0242-200s.csv && -f datasets/cwru-bearing/X228_FE_time_60s.csv ]] \
    || { note "ABORT: tuning inputs missing"; exit 1; }
HOLD=results-hold-flink-startup
[[ -e "$HOLD" || -e "$JAR.prod" ]] && { note "ABORT: $HOLD or $JAR.prod already exists"; exit 1; }

restore() {
    [[ -f "$JAR.prod" ]] && mv "$JAR.prod" "$JAR" && note "production jar restored"
    [[ -d "$HOLD" ]] || return
    mv results "flink-startup-scratch-$(date +%Y%m%d-%H%M%S)" && mv "$HOLD" results
    note "results/ restored"
}
mv "$JAR" "$JAR.prod" && cp "$TARGET/flink-trace.jar" "$JAR" || { note "ABORT: could not swap the jar"; restore; exit 1; }
mv results "$HOLD" && mkdir results || { note "ABORT: could not move results/ aside"; restore; exit 1; }
trap restore EXIT
trap 'exit 1' INT TERM

mkdir -p "$OUT"/{waymo,cwru,musan}
for rep in $(seq 1 "$REPS"); do
    for workload in waymo cwru musan; do
        case "$workload" in
            waymo) env=(TEST_FILE=waymo/systems.test FLINK_LANES=1) ;;
            cwru)  env=(TEST_FILE=cwru-bearing/systems.test FLINK_LANES=2 FLINK_CSV="$PWD/datasets/cwru-bearing/X228_FE_time_60s.csv") ;;
            musan) env=(TEST_FILE=musan/systems.test FLINK_LANES=1 FLINK_CSV="$PWD/datasets/musan/musan-us-gov-0242-200s.csv") ;;
        esac
        env "${env[@]}" STAGES=flink REP_LABEL="${workload}_flink_startup_rep${rep}" \
            FLINK_WINDOW_TRACE="$PWD/$OUT/$workload/trace_rep$rep.csv" \
            ./scripts/run_systems.sh "$workload" > "$OUT/$workload/flink_rep$rep.log" 2>&1 \
            || note "!! $workload rep $rep: run_systems.sh failed"
        if [[ -f "results/flink_${workload}.csv" ]]; then
            mv "results/flink_${workload}.csv" "$OUT/$workload/flink_rep$rep.csv"
        else
            note "!! $workload rep $rep: no result"
        fi
        [[ -f "$OUT/$workload/trace_rep$rep.csv" ]] || note "!! $workload rep $rep: no trace"
    done
    note "rep $rep done"
done

"$PY" - "$OUT" "$STEADY_AFTER_S" > "$OUT/startup_runs.csv" <<'PY'
import csv, math, sys
from pathlib import Path
root, after_s = Path(sys.argv[1]), float(sys.argv[2])
def nearest_rank(values, q):  # LatencyStats' percentile, so the full-run value matches the comparison row
    values = sorted(values)
    return values[max(0, min(len(values) - 1, math.ceil(q / 100 * len(values)) - 1))]
print("workload,repetition,results,steady_results,steady_after_s,row_p95_ms,p50_full_ms,p95_full_ms,p99_full_ms,"
      "p50_steady_ms,p95_steady_ms,p99_steady_ms")
for workload in ("waymo", "cwru", "musan"):
    for trace in sorted((root / workload).glob("trace_rep*.csv")):
        rep = trace.stem.removeprefix("trace_rep")
        rows = [r for r in csv.DictReader(open(trace)) if int(r["arrival_ns"]) and int(r["sink_ns"])]
        first = min(int(r["arrival_ns"]) for r in rows)
        e2e = [(int(r["sink_ns"]) - int(r["arrival_ns"])) / 1e6 for r in rows]
        steady = [l for r, l in zip(rows, e2e) if int(r["arrival_ns"]) - first >= after_s * 1e9]
        row = list(csv.DictReader(open(root / workload / f"flink_rep{rep}.csv")))[-1]
        full = [nearest_rank(e2e, q) for q in (50, 95, 99)]
        st = [nearest_rank(steady, q) for q in (50, 95, 99)]
        print(f"{workload},{rep},{len(e2e)},{len(steady)},{after_s:g},{float(row['e2e_latency_ms_p95']):.3f},"
              + ",".join(f"{v:.3f}" for v in full + st))
PY
note "all done: $OUT/startup_runs.csv"
