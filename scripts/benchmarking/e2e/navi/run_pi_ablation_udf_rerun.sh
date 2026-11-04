#!/usr/bin/env bash
# Rerun of the ablation's UDF cells with standalone UDFs: each UDF compiles its model with OpenVINO's default configuration, its
# thread count capped at the hardware threads (StandaloneOpenVinoSession), and uses none of the inference operators' runtime or
# session options, standing for a stream processor without an inference operator. The runs replace the UDF runs in
# results-ablation, which move to results-ablation/udf_replaced_<time>/, and ablation_{musan,cwru}.csv are reprocessed.
# Same protocol as run_pi_ablation_rerun.sh: REPS interleaved repetitions, cooldown and swap reset before each run, throttled runs
# re-measured, results/ swapped for an empty one and restored afterwards. Progress: ~/navi/run_pi_ablation_udf_rerun.log
set -uo pipefail
cd ~/navi
log=run_pi_ablation_udf_rerun.log
PY=models-benchmarking/venv/bin/python
REPS="${REPS:-3}"
COOLDOWN_C="${COOLDOWN_C:-60}"
SAMPLE_S=2
OUT=results-ablation
OLD="$OUT/udf_replaced_$(date +%Y%m%d-%H%M%S)"
export NES_MHATT_RNN_UDF_MODEL=models-benchmarking/models/mhatt-rnn/saved_model
export NES_SARAD_SIZE_200_STRIDE_10_UDF_MODEL=models-benchmarking/models/sarad/sarad_medium.onnx
unset NES_UDF_OPENVINO_NUM_STREAMS NES_UDF_OPENVINO_INFERENCE_NUM_THREADS
note() { echo "### $* ($(date '+%F %T'))" | tee -a "$log"; }

declare -A CELL=(
    [musan]="queries/musan/udf.yaml musan/ablation_musan_lat.test:01"
    [cwru]="queries/cwru-bearing/udf.yaml cwru-bearing/ablation_cwru_lat.test:01"
)

: > "$log"
pgrep -f "[b]in/systest|[r]un_pi_(fan|batch_rate|systems|ablation)_rerun\.sh|[r]un_post_join|[r]un_batch|[r]un_systems" > /dev/null \
    && { note "ABORT: an experiment is running"; exit 1; }
grep -q "Model UDF session: OpenVINO chose" bin/systest \
    || { note "ABORT: bin/systest predates the thread-capped standalone UDF sessions; install the new build"; exit 1; }
[[ -e "$OLD" ]] && { note "ABORT: $OLD already exists"; exit 1; }
for workload in musan cwru; do
    compgen -G "$OUT/raw_${workload}/*/ablation_${workload}_lat.test_01" > /dev/null \
        || { note "ABORT: no UDF runs in $OUT/raw_${workload}"; exit 1; }
done

HOLD=results-hold-ablation-udf
[[ -e "$HOLD" ]] && { note "ABORT: $HOLD already exists"; exit 1; }
mv results "$HOLD" && mkdir results || { note "ABORT: could not move results/ aside"; exit 1; }
SAMPLER=""
restore() {
    [[ -n "$SAMPLER" ]] && kill "$SAMPLER" 2>/dev/null
    [[ -d "$HOLD" ]] || return
    mv results "ablation-udf-scratch-$(date +%Y%m%d-%H%M%S)" && mv "$HOLD" results
    note "results/ restored"
}
trap restore EXIT
trap 'exit 1' INT TERM
note "results/ moved aside to $HOLD; idle $(vcgencmd measure_temp), flags $(vcgencmd get_throttled)"

thermal_sampler() {
    echo "time_us,temp_c,throttled" > results/thermal.csv
    while :; do
        echo "$(( $(date +%s%N) / 1000 )),$(vcgencmd measure_temp | sed 's/[^0-9.]//g'),$(vcgencmd get_throttled | cut -d= -f2)" \
            >> results/thermal.csv
        sleep "$SAMPLE_S"
    done
}
cooldown() {
    while (( $(vcgencmd measure_temp | sed 's/[^0-9]*\([0-9]*\).*/\1/') >= COOLDOWN_C )); do sleep 5; done
}
sweep() {  # sweep <workload>: run_systests_lat.py fills only missing repetitions
    local workload="$1" rep config query
    read -r config query <<< "${CELL[$workload]}"
    for rep in $(seq 1 "$REPS"); do
        cooldown
        "$HOME/navi/scripts/reset_swap.sh"
        python3 scripts/run_systests_lat.py --inference-config "$config" --queries "$query" \
            --data "./datasets/${query%%/*}" --repetitions "$rep" >> "systems-logs/ablation_udf_${workload}.log" 2>&1 \
            || note "!! ${workload} UDF rep ${rep} failed, see systems-logs/ablation_udf_${workload}.log"
    done
}

source "${PY%/python}/activate"
mkdir -p systems-logs "$OLD"
for workload in cwru musan; do
    thermal_sampler &
    SAMPLER=$!
    sweep "$workload"
    note "${workload}: UDF sweep done"
    for round in 1 2; do
        moved="$OLD/throttled_${workload}/round${round}"
        mkdir -p "$moved"
        "$PY" scripts/move_throttled_runs.py results "$moved" > "$moved/moved.txt"
        count=$(wc -l < "$moved/moved.txt")
        note "${workload}: round ${round}, ${count} throttled runs moved aside"
        [[ "$count" == 0 ]] && break
        sweep "$workload"
    done
    kill "$SAMPLER" 2>/dev/null
    SAMPLER=""

    # Swap the UDF runs only if every repetition completed: the operator-session ones move to $OLD, the standalone ones take
    # their place, then reprocess.
    completed=$(compgen -G "results/*/ablation_${workload}_lat.test_01/rep-*/latency_timings.csv" | wc -l)
    if (( completed < REPS )); then
        note "ATTENTION ${workload}: only ${completed}/${REPS} UDF runs completed; the previous UDF runs stay in place"
        mkdir -p "$OLD/incomplete_${workload}" && mv results/* "$OLD/incomplete_${workload}/"
        continue
    fi
    for run in "$OUT/raw_${workload}"/*/ablation_"${workload}"_lat.test_01; do
        combination=$(basename "$(dirname "$run")")
        mkdir -p "$OLD/raw_${workload}/${combination}"
        mv "$run" "$OLD/raw_${workload}/${combination}/"
    done
    for run in results/*/ablation_"${workload}"_lat.test_01; do
        combination=$(basename "$(dirname "$run")")
        mkdir -p "$OUT/raw_${workload}/${combination}"
        mv "$run" "$OUT/raw_${workload}/${combination}/"
    done
    mv results/thermal.csv "$OUT/raw_${workload}/thermal_udf.csv"
    cp "$OUT/ablation_${workload}.csv" "$OLD/ablation_${workload}.csv"
    "$PY" scripts/process_results_lat.py --results-dir "$OUT/raw_${workload}" --output-csv "$OUT/ablation_${workload}.csv" \
        > "$OUT/process_${workload}_udf.log" 2>&1 || note "!! processing ${workload} failed, see $OUT/process_${workload}_udf.log"
    rm -rf results/*
    note "${workload}: UDF runs replaced, $OUT/ablation_${workload}.csv reprocessed"
done
note "Pi ablation UDF rerun finished"
