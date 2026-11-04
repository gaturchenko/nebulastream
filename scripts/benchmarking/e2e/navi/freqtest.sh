#!/usr/bin/env bash
# 1.5 GHz thermal test: k=4 forced SEQUENTIAL at ADVANCE 333 ms (the hottest post-join cell), started below 60 C like the grid,
# with the ARM clock capped at 1.5 GHz through cpufreq; temperature, throttling flags and clock sampled every second.
cd ~/navi
source models-benchmarking/venv/bin/activate
OUT=results/freqtest
mkdir -p "$OUT"
sed "s#results/post_join/latency_timings.csv#${OUT}/latency_timings.csv#" queries/post-join/post_join_k4.test > queries/post-join/freqtest_k4.test
echo 1500000 | sudo -n tee /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq > /dev/null
echo "### max freq $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq)" >> "$OUT/freqtest.log"
while (( $(vcgencmd measure_temp | sed 's/[^0-9]*\([0-9]*\).*/\1/') >= 60 )); do sleep 5; done
echo "### start $(date +%T) $(vcgencmd measure_temp)" >> "$OUT/freqtest.log"
bin/systest --workingDir ~/navi/working-dir --data ~/navi/datasets --clusterConfig bin/config/single-node.yaml \
    --log-path "$OUT/run.log" -t queries/post-join/freqtest_k4.test:03 -n 1 \
    --optimizer post_join_inference_strategy=SEQUENTIAL -- --worker.default_query_execution.inference.use_batch_deduplication=true \
    > "$OUT/run.out" 2>&1 &
pid=$!
echo "t_s,temp_c,throttled,arm_hz,mem_available_mb" > "$OUT/thermal_1s.csv"
start=$(date +%s)
while kill -0 $pid 2>/dev/null; do
    avail=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 ))
    echo "$(( $(date +%s) - start )),$(vcgencmd measure_temp | sed 's/[^0-9.]//g'),$(vcgencmd get_throttled | cut -d= -f2),$(vcgencmd measure_clock arm | cut -d= -f2),$avail" >> "$OUT/thermal_1s.csv"
    (( avail < 700 )) && { kill -9 $pid; echo "### KILLED low memory" >> "$OUT/freqtest.log"; }
    sleep 1
done
wait $pid; rc=$?
mv trace_*.json "$OUT/" 2>/dev/null
echo 1800000 | sudo -n tee /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq > /dev/null
echo "### end $(date +%T) rc=$rc $(grep -c PASSED "$OUT/run.out") passed; max freq restored to $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq)" >> "$OUT/freqtest.log"
