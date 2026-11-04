#!/usr/bin/env bash
# Memory check on the Pi: the MUSAN ablation's UDF query (MHAtt-RNN UDF on StandaloneOpenVinoSession, OpenVINO defaults, 2 worker
# threads), then the native ST query as a reference. Samples the systest process's RSS and the system's MemAvailable every second;
# kills systest once MemAvailable drops below MIN_AVAILABLE_MB or after the variant's time limit, so a leak cannot exhaust memory
# and trip the watchdog again.
cd ~/navi
source models-benchmarking/venv/bin/activate
export NES_MHATT_RNN_UDF_MODEL=models-benchmarking/models/mhatt-rnn/saved_model
MIN_AVAILABLE_MB=${MIN_AVAILABLE_MB:-700}
OUT=results/memcheck_udf
mkdir -p "$OUT"
sed "s#'results/latency_timings.csv'#'${OUT}/latency_timings.csv'#" queries/musan/ablation_musan_lat.test > queries/musan/memcheck_udf.test
grep -q "${OUT}/latency_timings.csv" queries/musan/memcheck_udf.test || { echo "### sink path not replaced" >> "$OUT/memcheck.log"; exit 1; }

# variant section max_s optimizer-and-worker options
run() {
    local variant="$1" section="$2" max_s="$3"
    shift 3
    while (( $(vcgencmd measure_temp | sed 's/[^0-9]*\([0-9]*\).*/\1/') >= 60 )); do sleep 5; done
    "$HOME/navi/scripts/reset_swap.sh"
    echo "### $variant start $(date +%T) available $(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 )) MB" >> "$OUT/memcheck.log"
    bin/systest --workingDir ~/navi/working-dir --data ~/navi/datasets/musan --clusterConfig bin/config/single-node.yaml \
        --log-path "$OUT/$variant.log" -t "queries/musan/memcheck_udf.test:${section}" -n 1 "$@" > "$OUT/$variant.out" 2>&1 &
    local pid=$!
    echo "t_s,rss_mb,mem_available_mb,temp_c" > "$OUT/rss_$variant.csv"
    local start rss avail elapsed
    start=$(date +%s)
    while kill -0 "$pid" 2>/dev/null; do
        rss=$(( $(ps -o rss= -p "$pid" 2>/dev/null || echo 0) / 1024 ))
        avail=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 ))
        elapsed=$(( $(date +%s) - start ))
        echo "$elapsed,$rss,$avail,$(vcgencmd measure_temp | sed 's/[^0-9.]//g')" >> "$OUT/rss_$variant.csv"
        if (( avail < MIN_AVAILABLE_MB || elapsed > max_s )); then
            kill -9 "$pid"
            echo "### $variant KILLED: rss ${rss} MB, available ${avail} MB after ${elapsed} s" >> "$OUT/memcheck.log"
        fi
        sleep 1
    done
    wait "$pid"
    echo "### $variant end $(date +%T) rc=$?" >> "$OUT/memcheck.log"
    mv trace_*.json "$OUT/" 2>/dev/null
    [[ -f "$OUT/latency_timings.csv" ]] && mv "$OUT/latency_timings.csv" "$OUT/latency_$variant.csv"
}

run udf 01 180 --optimizer inference_batch_size=1 --optimizer post_join_inference_strategy=SEQUENTIAL \
    -- --worker.default_query_execution.inference.prediction_cache_type=NONE
run st 02 120 --optimizer post_join_inference_strategy=SEQUENTIAL \
    -- --worker.default_query_execution.inference.prediction_cache_type=NONE \
    --worker.default_query_execution.inference.openvino_share_compiled_model=true
rm -f queries/musan/memcheck_udf.test
echo "### done" >> "$OUT/memcheck.log"
