#!/usr/bin/env bash
# Inference-configuration ablation (MUSAN/MHAtt-RNN, CWRU/SARAD) at OpenVINO 1 stream x 4 threads with active cooling, queued
# behind the systems rerun (run_pi_systems_rerun.sh; stops if any earlier rerun aborted). Replaces run_audio_lat.sh and
# run_time_series_lat.sh for the paper's figure:
#   * the same configurations (MUSAN: UDF, ST, Batch; CWRU: UDF, ST x batch 1/12, cache scopes, batch+cache, batch+dedup), now
#     REPS repetitions each, interleaved, every run starting below COOLDOWN_C with swap reset
#   * the UDF baselines run the same session as the inference operators (NES_UDF_OPENVINO_*) and the same model files
#   * temperature and throttle flags are sampled into results/thermal.csv; runs during which the firmware limited the CPU are
#     moved aside and re-measured (up to 2 rounds)
#   * results/ is swapped for an empty one (the processor scans the whole tree) and restored afterwards, also on interruption
# Outputs: results-ablation/ablation_{musan,cwru}.csv (process_results_lat.py), raw runs in results-ablation/raw_<workload>/.
# Progress: ~/navi/run_pi_ablation_rerun.log
set -uo pipefail
cd ~/navi
log=run_pi_ablation_rerun.log
PY=models-benchmarking/venv/bin/python
REPS="${REPS:-3}"
COOLDOWN_C="${COOLDOWN_C:-60}"
SAMPLE_S=2
OUT=results-ablation
export NES_UDF_OPENVINO_NUM_STREAMS=1 NES_UDF_OPENVINO_INFERENCE_NUM_THREADS=4
# The native queries' CREATE MODEL files; the MUSAN UDF used to point at a u192_l2 export that no longer exists.
export NES_MHATT_RNN_UDF_MODEL=models-benchmarking/models/mhatt-rnn/saved_model
export NES_SARAD_SIZE_200_STRIDE_10_UDF_MODEL=models-benchmarking/models/sarad/sarad_medium.onnx
note() { echo "### $* ($(date '+%F %T'))" | tee -a "$log"; }

# <inference config> <query> per cell, the configurations behind ablation_musan_2.csv and ablation_cwru_5.csv.
MUSAN_CELLS=(
    "queries/musan/udf.yaml musan/ablation_musan_lat.test:01"
    "queries/musan/st.yaml musan/ablation_musan_lat.test:02"
    "queries/musan/batch.yaml musan/ablation_musan_lat.test:02"
)
CWRU_CELLS=(
    "queries/cwru-bearing/udf.yaml cwru-bearing/ablation_cwru_lat.test:01"
    "queries/cwru-bearing/st_batch.yaml cwru-bearing/ablation_cwru_lat.test:02"
    "queries/cwru-bearing/st_cache.yaml cwru-bearing/ablation_cwru_lat.test:03"
    "queries/cwru-bearing/batch_cache.yaml cwru-bearing/ablation_cwru_lat.test:03"
    "queries/cwru-bearing/batch_cache_dedup.yaml cwru-bearing/ablation_cwru_lat.test:03"
)

: > "$log"
note "waiting for the systems rerun"
while ! grep -q "Pi systems rerun finished" run_pi_systems_rerun.log 2>/dev/null; do
    grep -qs "ABORT" run_pi_systems_rerun.log results/run_pi_batch_rate_rerun.log results/run_pi_fan_rerun.log \
        && { note "ABORT: an earlier rerun aborted"; exit 1; }
    sleep 10
done
while pgrep -f "[b]in/systest|[r]un_systems|[t]orchserve|[j]ava " > /dev/null; do sleep 10; done

for test in queries/musan/ablation_musan_lat.test queries/cwru-bearing/ablation_cwru_lat.test; do
    grep -q "openvino_num_streams: \[1\]" "$test" && grep -q "openvino_inference_num_threads: \[4\]" "$test" \
        || { note "ABORT: $test is not set to 1 stream x 4 threads"; exit 1; }
done
grep -q NES_UDF_OPENVINO_NUM_STREAMS bin/systest \
    || { note "ABORT: bin/systest predates the UDF session options (NES_UDF_OPENVINO_*); install the new build"; exit 1; }
[[ -e "$OUT" ]] && { note "ABORT: $OUT already exists"; exit 1; }
mkdir -p "$OUT"

HOLD=results-hold-ablation
[[ -e "$HOLD" ]] && { note "ABORT: $HOLD already exists"; exit 1; }
mv results "$HOLD" && mkdir results || { note "ABORT: could not move results/ aside"; exit 1; }
SAMPLER=""
restore() {
    [[ -n "$SAMPLER" ]] && kill "$SAMPLER" 2>/dev/null
    [[ -d "$HOLD" ]] || return
    mv results "ablation-scratch-$(date +%Y%m%d-%H%M%S)" && mv "$HOLD" results
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

# sweep <workload> <cells...>: REPS interleaved repetitions; run_systests_lat.py fills only missing repetitions, so a second
# pass re-measures exactly the runs moved aside.
sweep() {
    local workload="$1" rep cell config query
    shift
    for rep in $(seq 1 "$REPS"); do
        for cell in "$@"; do
            read -r config query <<< "$cell"
            cooldown
            "$HOME/navi/scripts/reset_swap.sh"
            python3 scripts/run_systests_lat.py --inference-config "$config" --queries "$query" \
                --data "./datasets/${query%%/*}" --repetitions "$rep" >> "systems-logs/ablation_${workload}.log" 2>&1 \
                || note "!! ${workload} ${config} ${query} rep ${rep} failed, see systems-logs/ablation_${workload}.log"
        done
    done
}

source "${PY%/python}/activate"
mkdir -p systems-logs
for workload in musan cwru; do
    if [[ "$workload" == musan ]]; then cells=("${MUSAN_CELLS[@]}"); else cells=("${CWRU_CELLS[@]}"); fi
    thermal_sampler &
    SAMPLER=$!
    sweep "$workload" "${cells[@]}"
    note "${workload}: sweep done"
    for round in 1 2; do
        moved="$OUT/throttled_${workload}/round${round}"
        mkdir -p "$moved"
        "$PY" scripts/move_throttled_runs.py results "$moved" > "$moved/moved.txt"
        count=$(wc -l < "$moved/moved.txt")
        note "${workload}: round ${round}, ${count} throttled runs moved aside"
        [[ "$count" == 0 ]] && break
        sweep "$workload" "${cells[@]}"
    done
    count=$("$PY" scripts/move_throttled_runs.py results "$OUT/throttled_${workload}/after_round2" | wc -l)
    [[ "$count" == 0 ]] || note "ATTENTION ${workload}: ${count} runs still throttled after 2 rounds"
    kill "$SAMPLER" 2>/dev/null
    SAMPLER=""
    "$PY" scripts/process_results_lat.py --results-dir results --output-csv "$OUT/ablation_${workload}.csv" \
        > "$OUT/process_${workload}.log" 2>&1 || note "!! processing ${workload} failed, see $OUT/process_${workload}.log"
    mkdir -p "$OUT/raw_${workload}"
    mv results/* "$OUT/raw_${workload}/"
    note "${workload}: processed into $OUT/ablation_${workload}.csv"
done
note "Pi ablation rerun finished"
