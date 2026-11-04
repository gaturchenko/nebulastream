#!/usr/bin/env bash
# Memory check on the Pi: k=2 post-join SEQUENTIAL (ADVANCE 1000 ms, dedup), first with one OpenVINO thread per call
# (num_streams 1, inference_num_threads 1), then at the deployed setting (num_streams 0). Samples the systest process's own RSS
# and the system's MemAvailable every second; kills systest once MemAvailable drops below MIN_AVAILABLE_MB (the buffer pool alone
# puts ~1.5 GB into its RSS, so an RSS limit cannot tell a leak from the pool) or after MAX_S.
cd ~/navi
source models-benchmarking/venv/bin/activate
MIN_AVAILABLE_MB=${MIN_AVAILABLE_MB:-700}
MAX_S=${MAX_S:-420}
OUT=results/memcheck5
mkdir -p "$OUT"
base=queries/post-join/post_join_k2.test
sed "s#results/post_join/latency_timings.csv#${OUT}/latency_timings.csv#" "$base" > queries/post-join/memcheck_s0t1.test
sed -e "s#results/post_join/latency_timings.csv#${OUT}/latency_timings.csv#" \
    -e "s#number_of_worker_threads: \[2\]#number_of_worker_threads: [1]#" "$base" > queries/post-join/memcheck_s0t1w1.test
grep -h "openvino_num_streams\|openvino_inference_num_threads\|number_of_worker_threads" queries/post-join/memcheck_s*.test >> "$OUT/memcheck.log"
for variant in s0t1w1; do
    while (( $(vcgencmd measure_temp | sed 's/[^0-9]*\([0-9]*\).*/\1/') >= 65 )); do sleep 5; done
    echo "### $variant start $(date +%T)" >> "$OUT/memcheck.log"
    bin/systest --workingDir ~/navi/working-dir --data ~/navi/datasets --clusterConfig bin/config/single-node.yaml \
        --log-path "$OUT/$variant.log" -t "queries/post-join/memcheck_${variant}.test:01" -n 1 \
        --optimizer post_join_inference_strategy=SEQUENTIAL -- --worker.default_query_execution.inference.use_batch_deduplication=true \
        > "$OUT/$variant.out" 2>&1 &
    pid=$!
    echo "t_s,rss_mb,mem_available_mb,temp_c" > "$OUT/rss_$variant.csv"
    start=$(date +%s)
    while kill -0 $pid 2>/dev/null; do
        rss=$(( $(ps -o rss= -p $pid 2>/dev/null || echo 0) / 1024 ))
        avail=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 ))
        elapsed=$(( $(date +%s) - start ))
        echo "$elapsed,$rss,$avail,$(vcgencmd measure_temp | sed 's/[^0-9.]//g')" >> "$OUT/rss_$variant.csv"
        if (( avail < MIN_AVAILABLE_MB || elapsed > MAX_S )); then
            kill -9 $pid
            echo "### $variant KILLED: rss ${rss} MB, available ${avail} MB after ${elapsed} s" >> "$OUT/memcheck.log"
        fi
        sleep 1
    done
    wait $pid; rc=$?
    mv trace_*.json "$OUT/" 2>/dev/null
    [[ -f "$OUT/latency_timings.csv" ]] && mv "$OUT/latency_timings.csv" "$OUT/latency_$variant.csv"
    echo "### $variant end $(date +%T) rc=$rc $(grep -c PASSED "$OUT/$variant.out") passed" >> "$OUT/memcheck.log"
done
echo "### done" >> "$OUT/memcheck.log"
