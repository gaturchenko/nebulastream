#!/usr/bin/env bash
# Systems comparison with every system tuned the same way, at OpenVINO 1 stream x 4 threads per inference lane, for one workload
# (the procedure of run_pi_cwru_tuning.sh, for waymo and musan):
#   1. input long enough for ~600 results per run (Waymo: its 991 frames at 10/s; MUSAN: a 200 s cut of the recording)
#   2. pilot: each system's parallelism knob over the same candidates (NAVI worker threads 1-4, Flink inference lanes 1-4,
#      TorchServe workers 1-3), PILOT_REPS repetitions
#   3. per system, the candidate with the lowest median P95 among those that keep up with the offered rate in every run
#   4. final: the chosen configurations measured afresh, FINAL_REPS repetitions
# A memory guard kills a run's engines once MemAvailable drops below MIN_AVAILABLE_MB (private compiled models can exhaust the
# Pi, which the watchdog answers with a reboot); such a candidate has no result and is not chosen. NES's side of the TorchServe
# query stays as published, with enough HTTP connections for 3 workers.
# results/ is swapped for an empty one and restored afterwards. Outputs: ~/navi/results-tuning-<workload>/{pilot,final}/
# <system>_<knob><n>_rep<r>.csv, selection.txt, memory_kills.txt. Progress: ~/navi/run_pi_systems_tuning.log
#
#   ./run_pi_systems_tuning.sh waymo musan
set -uo pipefail
cd ~/navi
log=run_pi_systems_tuning.log
PY=models-benchmarking/venv/bin/python
PILOT_REPS="${PILOT_REPS:-2}"
FINAL_REPS="${FINAL_REPS:-5}"
MIN_AVAILABLE_MB="${MIN_AVAILABLE_MB:-600}"
export SAMPLE_S=2 COOLDOWN_C="${COOLDOWN_C:-60}" THROTTLE_RETRIES="${THROTTLE_RETRIES:-2}"
note() { echo "### $* ($(date '+%F %T'))" | tee -a "$log"; }

: > "$log"
pgrep -f "[b]in/systest|[r]un_pi_(fan|batch_rate|systems|ablation|ablation_udf)_rerun\.sh|[r]un_pi_cwru_tuning|[r]un_systems|[j]ava " \
    > /dev/null && { note "ABORT: an experiment is running"; exit 1; }

HOLD=results-hold-systems-tuning
[[ -e "$HOLD" ]] && { note "ABORT: $HOLD already exists"; exit 1; }
WATCH=""
restore() {
    [[ -n "$WATCH" ]] && kill "$WATCH" 2>/dev/null
    [[ -d "$HOLD" ]] || return
    mv results "systems-tuning-scratch-$(date +%Y%m%d-%H%M%S)" && mv "$HOLD" results
    note "results/ restored"
}

memory_guard() {  # memory_guard <file>: kills the engines while MemAvailable is below the limit
    while sleep 2; do
        local available
        available=$(awk '/MemAvailable/ { print int($2 / 1024) }' /proc/meminfo)
        if (( available < MIN_AVAILABLE_MB )); then
            pkill -9 -f "[b]in/systest"; pkill -9 -f "[j]ava "; pkill -9 -f "[m]odel_service_worker"; pkill -9 -f "[o]rg.pytorch.serve"
            echo "$(date '+%F %T') killed the engines at ${available} MB available" >> "$1"
            sleep 10
        fi
    done
}

for WORKLOAD in "$@"; do
    case "$WORKLOAD" in
        waymo) BASE=queries/waymo/systems.test; OFFERED=10; PUBLISHED_WORKERS=2; DATA="" ;;
        musan) BASE=queries/musan/systems.test; OFFERED=$(awk 'BEGIN { print 16000 / 5328 }'); PUBLISHED_WORKERS=2
               DATA=datasets/musan/musan-us-gov-0242-200s.csv ;;
        *) note "ABORT: no tuning setup for $WORKLOAD"; exit 1 ;;
    esac
    OUT="results-tuning-${WORKLOAD}"
    [[ -e "$OUT" ]] && { note "ABORT: $OUT already exists"; exit 1; }
    mkdir -p "$OUT/pilot" "$OUT/final"
    DIR=$(dirname "${BASE#queries/}")

    # 1. Input and .test variants: one per NAVI worker count, one for the TorchServe runs.
    if [[ -n "$DATA" && ! -f "$DATA" ]]; then
        awk -F, '$1 < 200000' datasets/musan/musan-us-gov-0242.csv > "$DATA"
    fi
    attach=$(sed -n 's#^ATTACH FILE ##p' "$BASE" | head -1)
    for workers in 1 2 3 4 ts; do
        variant="queries/${DIR}/tuning_w${workers}.test"
        if [[ "$workers" == ts ]]; then
            sed "s#^\( *\)[0-9]\+ AS \`SINK\`.connections#\13 AS \`SINK\`.connections#" "$BASE" > "$variant"
        else
            sed "s#^GlobalConfiguration worker.query_engine.number_of_worker_threads: \[[0-9]\+\]#GlobalConfiguration worker.query_engine.number_of_worker_threads: [${workers}]#" \
                "$BASE" > "$variant"
        fi
        [[ -n "$DATA" ]] && sed -i "s#^ATTACH FILE ${attach}\$#ATTACH FILE $PWD/${DATA}#" "$variant"
    done
    grep -q "3 AS \`SINK\`.connections" "queries/${DIR}/tuning_wts.test" \
        && grep -q "number_of_worker_threads: \[4\]" "queries/${DIR}/tuning_w4.test" \
        || { note "ABORT: could not generate the ${WORKLOAD} .test variants"; exit 1; }
    if [[ -n "$DATA" ]]; then
        grep -q "ATTACH FILE $PWD/${DATA}" "queries/${DIR}/tuning_w1.test" || { note "ABORT: ${WORKLOAD} input not replaced"; exit 1; }
        note "${WORKLOAD}: input $(wc -l < "$DATA") samples, last timestamp $(tail -1 "$DATA" | cut -d, -f1) ms"
    fi

    mv results "$HOLD" && mkdir results || { note "ABORT: could not move results/ aside"; exit 1; }
    trap restore EXIT
    trap 'exit 1' INT TERM
    memory_guard "$OUT/memory_kills.txt" &
    WATCH=$!

    measure() {  # measure <phase> <rep> <system> <knob><n>
        local phase="$1" rep="$2" system="$3" config="$4" n="${4//[!0-9]/}"
        local -a env=(STAGES="$system" REP_LABEL="${WORKLOAD}_tuning_${phase}_${config}_rep${rep}")
        case "$system" in
            nes) env+=(TEST_FILE="${DIR}/tuning_w${n}.test") ;;
            flink) env+=(TEST_FILE="${DIR}/tuning_w${PUBLISHED_WORKERS}.test" FLINK_LANES="$n")
                   [[ -n "$DATA" ]] && env+=(FLINK_CSV="$PWD/$DATA") ;;
            torchserve) env+=(TEST_FILE="${DIR}/tuning_wts.test" TS_WORKERS="$n") ;;
        esac
        env "${env[@]}" ./scripts/run_systems.sh "$WORKLOAD" > "$OUT/$phase/${system}_${config}_rep${rep}.log" 2>&1 \
            || note "!! ${WORKLOAD} $phase $system $config rep $rep: run_systems.sh failed"
        if [[ -f "results/${system}_${WORKLOAD}.csv" ]]; then
            mv "results/${system}_${WORKLOAD}.csv" "$OUT/$phase/${system}_${config}_rep${rep}.csv"
        else
            note "!! ${WORKLOAD} $phase $system $config rep $rep: no result"
        fi
    }

    PILOT=(nes:w1 nes:w2 nes:w3 nes:w4 flink:lanes1 flink:lanes2 flink:lanes3 flink:lanes4
           torchserve:workers1 torchserve:workers2 torchserve:workers3)
    for rep in $(seq 1 "$PILOT_REPS"); do
        for candidate in "${PILOT[@]}"; do
            measure pilot "$rep" "${candidate%%:*}" "${candidate#*:}"
        done
        note "${WORKLOAD}: pilot rep $rep done"
    done

    "$PY" - "$OUT" "$OFFERED" "$PILOT_REPS" > "$OUT/selection.txt" <<'PY'
import csv, statistics, sys
from collections import defaultdict
from pathlib import Path
root, offered, reps = Path(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
def metrics(system, path):
    rows = list(csv.DictReader(open(path)))
    if system == "nes":
        row = next(r for r in rows if r.get("is_inference_pipeline") == "True")
        rate = float(row["sink_latency_count"]) / (float(row["end_to_end_duration_us"]) / 1e6)
        return float(row["p95_sink_latency_us"]) / 1000, rate
    row = rows[-1]
    return float(row["e2e_latency_ms_p95"]), float(row["end_to_end_throughput"])
runs = defaultdict(list)
for path in sorted((root / "pilot").glob("*.csv")):
    system, config, _ = path.stem.split("_")
    runs[(system, config)].append(metrics(system, path))
choice = {}
for (system, config), values in sorted(runs.items()):
    p95 = statistics.median(v[0] for v in values)
    feasible = len(values) == reps and all(v[1] >= 0.95 * offered for v in values)
    verdict = "keeps up" if feasible else ("incomplete" if len(values) < reps else "falls behind")
    print(f"{system:10} {config:9} P95 {p95:7.1f} ms  rates {[round(v[1], 2) for v in values]}  {verdict}")
    if feasible and (system not in choice or p95 < choice[system][1]):
        choice[system] = (config, p95)
for system, (config, p95) in sorted(choice.items()):
    print(f"CHOSEN {system} {config}")
PY
    note "${WORKLOAD}: selection: $(grep CHOSEN "$OUT/selection.txt" | cut -d' ' -f2- | tr '\n' ';')"

    mapfile -t chosen < <(grep "^CHOSEN" "$OUT/selection.txt" | awk '{print $2 ":" $3}')
    if [[ ${#chosen[@]} -eq 0 ]]; then
        note "ATTENTION ${WORKLOAD}: no configuration kept up"
    fi
    for rep in $(seq 1 "$FINAL_REPS"); do
        for candidate in "${chosen[@]}"; do
            measure final "$rep" "${candidate%%:*}" "${candidate#*:}"
        done
        note "${WORKLOAD}: final rep $rep done"
    done

    kill "$WATCH" 2>/dev/null
    WATCH=""
    restore
    trap - EXIT INT TERM
    rm -f queries/"${DIR}"/tuning_w{1,2,3,4,ts}.test
    note "${WORKLOAD}: tuning finished"
done
note "Pi systems tuning finished"
