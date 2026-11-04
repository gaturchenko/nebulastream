#!/usr/bin/env bash
# CWRU (SARAD) systems comparison with every system tuned the same way, at OpenVINO 1 stream x 4 threads per inference lane:
#   1. a 61 s recording (X228 looped 6 times, timestamps shifted), so that a run has ~600 windows and its P95 rests on ~30
#   2. pilot: each system's parallelism knob over the same candidates (NAVI worker threads 1-4, Flink inference lanes 1-3,
#      TorchServe workers 1-3), PILOT_REPS repetitions
#   3. per system, the candidate with the lowest median P95 among those that keep up with the offered rate
#   4. final: the chosen configurations measured afresh, FINAL_REPS repetitions
# Flink stops at 3 lanes: each holds a private compiled SARAD, and 3 already use ~2 GB of the Pi's 3.8 GB.
# results/ is swapped for an empty one (run_systems.sh wipes and scans it) and restored afterwards. Outputs:
# ~/navi/results-cwru-tuning/{pilot,final}/<system>_<knob><n>_rep<r>.csv, selection.txt. Progress: ~/navi/run_pi_cwru_tuning.log
set -uo pipefail
cd ~/navi
log=run_pi_cwru_tuning.log
PY=models-benchmarking/venv/bin/python
PILOT_REPS="${PILOT_REPS:-2}"
FINAL_REPS="${FINAL_REPS:-5}"
export SAMPLE_S=2 COOLDOWN_C="${COOLDOWN_C:-60}" THROTTLE_RETRIES="${THROTTLE_RETRIES:-2}"
OUT=results-cwru-tuning
DATA=datasets/cwru-bearing/X228_FE_time_60s.csv
note() { echo "### $* ($(date '+%F %T'))" | tee -a "$log"; }

: > "$log"
pgrep -f "[b]in/systest|[r]un_pi_(fan|batch_rate|systems|ablation|ablation_udf)_rerun\.sh|[r]un_systems|[j]ava " > /dev/null \
    && { note "ABORT: an experiment is running"; exit 1; }
[[ -e "$OUT" ]] && { note "ABORT: $OUT already exists"; exit 1; }
mkdir -p "$OUT/pilot" "$OUT/final"

# 1. Data and one .test per NAVI worker count (the TorchServe query in them keeps NES's side as in systems.test).
if [[ ! -f "$DATA" ]]; then
    awk -F, -v OFS=, 'FNR == 1 && NR > 1 { copy++ } { print $1 + copy * 10154, $2 }' \
        datasets/cwru-bearing/X228_FE_time.csv datasets/cwru-bearing/X228_FE_time.csv datasets/cwru-bearing/X228_FE_time.csv \
        datasets/cwru-bearing/X228_FE_time.csv datasets/cwru-bearing/X228_FE_time.csv datasets/cwru-bearing/X228_FE_time.csv > "$DATA"
fi
note "data: $(wc -l < "$DATA") samples, last timestamp $(tail -1 "$DATA" | cut -d, -f1) ms"
for workers in 1 2 3 4; do
    sed -e "s#/X228_FE_time.csv\$#/X228_FE_time_60s.csv#" \
        -e "s#^GlobalConfiguration worker.query_engine.number_of_worker_threads: \[3\]#GlobalConfiguration worker.query_engine.number_of_worker_threads: [${workers}]#" \
        queries/cwru-bearing/systems.test > "queries/cwru-bearing/systems60_w${workers}.test"
    grep -q "ATTACH FILE /home/gaturchenko/navi/${DATA}" "queries/cwru-bearing/systems60_w${workers}.test" \
        && grep -q "number_of_worker_threads: \[${workers}\]" "queries/cwru-bearing/systems60_w${workers}.test" \
        || { note "ABORT: could not generate systems60_w${workers}.test"; exit 1; }
done

HOLD=results-hold-cwru-tuning
[[ -e "$HOLD" ]] && { note "ABORT: $HOLD already exists"; exit 1; }
mv results "$HOLD" && mkdir results || { note "ABORT: could not move results/ aside"; exit 1; }
restore() {
    [[ -d "$HOLD" ]] || return
    mv results "cwru-tuning-scratch-$(date +%Y%m%d-%H%M%S)" && mv "$HOLD" results
    note "results/ restored"
}
trap restore EXIT
trap 'exit 1' INT TERM

# measure <phase> <rep> <system> <knob><n>: one run_systems.sh pass of one arm, its comparison CSV kept under $OUT/<phase>.
measure() {
    local phase="$1" rep="$2" system="$3" config="$4" n="${4//[!0-9]/}"
    local -a env=(STAGES="$system" REP_LABEL="cwru_tuning_${phase}_${config}_rep${rep}" TEST_FILE=cwru-bearing/systems60_w3.test)
    case "$system" in
        nes) env+=(TEST_FILE="cwru-bearing/systems60_w${n}.test") ;;
        flink) env+=(FLINK_LANES="$n" FLINK_CSV="$PWD/$DATA") ;;
        torchserve) env+=(TS_WORKERS="$n") ;;
    esac
    env "${env[@]}" ./scripts/run_systems.sh cwru > "$OUT/$phase/${system}_${config}_rep${rep}.log" 2>&1 \
        || note "!! $phase $system $config rep $rep: run_systems.sh failed"
    if [[ -f "results/${system}_cwru.csv" ]]; then
        mv "results/${system}_cwru.csv" "$OUT/$phase/${system}_${config}_rep${rep}.csv"
    else
        note "!! $phase $system $config rep $rep: no result"
    fi
}

PILOT=(nes:w1 nes:w2 nes:w3 nes:w4 flink:lanes1 flink:lanes2 flink:lanes3 torchserve:workers1 torchserve:workers2 torchserve:workers3)
for rep in $(seq 1 "$PILOT_REPS"); do
    for candidate in "${PILOT[@]}"; do
        measure pilot "$rep" "${candidate%%:*}" "${candidate#*:}"
    done
    note "pilot rep $rep done"
done

# 3. Selection: lowest median P95 among the candidates that deliver >= 95% of the offered window rate in every repetition.
"$PY" - "$OUT" > "$OUT/selection.txt" <<'PY'
import csv, statistics, sys
from collections import defaultdict
from pathlib import Path
root = Path(sys.argv[1])
OFFERED = 48000 / 4800  # windows/s: 48 kHz samples, a window every 4800 samples (ADVANCE 100 ms)
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
    keeps_up = all(v[1] >= 0.95 * OFFERED for v in values)
    print(f"{system:10} {config:9} P95 {p95:7.1f} ms  rates {[round(v[1], 2) for v in values]}  {'keeps up' if keeps_up else 'falls behind'}")
    if keeps_up and (system not in choice or p95 < choice[system][1]):
        choice[system] = (config, p95)
for system, (config, p95) in sorted(choice.items()):
    print(f"CHOSEN {system} {config}")
PY
note "selection: $(grep CHOSEN "$OUT/selection.txt" | cut -d' ' -f2- | tr '\n' ';')"

# 4. Final runs of the chosen configurations.
mapfile -t chosen < <(grep "^CHOSEN" "$OUT/selection.txt" | awk '{print $2 ":" $3}')
[[ ${#chosen[@]} -gt 0 ]] || { note "ABORT: no configuration kept up"; exit 1; }
for rep in $(seq 1 "$FINAL_REPS"); do
    for candidate in "${chosen[@]}"; do
        measure final "$rep" "${candidate%%:*}" "${candidate#*:}"
    done
    note "final rep $rep done"
done
rm -f queries/cwru-bearing/systems60_w{1,2,3,4}.test
note "Pi CWRU tuning finished"
