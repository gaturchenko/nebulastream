#!/usr/bin/env bash
# Systems comparison at OpenVINO 1 stream x 4 threads with active cooling, queued behind the batch-rate rerun
# (run_pi_batch_rate_rerun.sh; stops if that one or the post-join rerun aborted):
#   1. archive the num_streams 0 x 1 thread systems results in systems-streams0/
#   2. give the systems runs an empty results/ (run_systems.sh wipes and scans it) and restore the other experiments' results/
#      afterwards, also if this script is interrupted
#   3. the three-workload matrix (3 repetitions), CWRU Flink repetitions 4-6 and the Waymo rate sweep
# run_systems.sh lets every arm cool below COOLDOWN_C and re-runs a throttled arm up to THROTTLE_RETRIES times, logging every
# attempt to results-thermal/arms.csv. Progress: ~/navi/run_pi_systems_rerun.log (outside results/, which is swapped out).
set -uo pipefail
cd ~/navi
log=run_pi_systems_rerun.log
export SAMPLE_S=2 COOLDOWN_C="${COOLDOWN_C:-60}" THROTTLE_RETRIES="${THROTTLE_RETRIES:-2}"
note() { echo "### $* ($(date '+%F %T'))" | tee -a "$log"; }

: > "$log"
note "waiting for the batch-rate rerun"
while ! grep -q "Pi batch-rate rerun finished" results/run_pi_batch_rate_rerun.log 2>/dev/null; do
    grep -qs "ABORT" results/run_pi_batch_rate_rerun.log results/run_pi_fan_rerun.log && { note "ABORT: an earlier rerun aborted"; exit 1; }
    sleep 10
done
while pgrep -f "[b]in/systest|[r]un_batch|[r]un_post_join" > /dev/null; do sleep 10; done

# The deployed files must be the 1 stream x 4 threads versions.
check() { grep -q "$2" "$1" || { note "ABORT: $1 does not contain '$2'"; exit 1; }; }
for test in waymo musan cwru-bearing; do
    check "queries/${test}/systems.test" "openvino_num_streams: \[1\]"
    check "queries/${test}/systems.test" "openvino_inference_num_threads: \[4\]"
done
for handler in sarad mhatt_rnn squeezedet; do
    check "scripts/torchserve/${handler}_handler.py" '_NUM_STREAMS", "1"'
    check "scripts/torchserve/${handler}_handler.py" '_INFERENCE_THREADS", "4"'
done
for runner in waymo-squeezedet cwru-sarad musan-mhatt; do
    check "baselines/flink-openvino-baseline/scripts/run-pi-${runner}.sh" 'STREAMS="${STREAMS:-1}"'
    check "baselines/flink-openvino-baseline/scripts/run-pi-${runner}.sh" 'THREADS="${THREADS:-4}"'
done
grep -q FLINK_OV_SHARED_LANES baselines/flink-openvino-baseline/target/native-libs/libflink_openvino_bridge_jni.so \
    || { note "ABORT: the Flink JNI bridge was not rebuilt with per-lane streams"; exit 1; }
check scripts/run_systems.sh "THROTTLE_RETRIES"

# 1. Archive.
A=systems-streams0
[[ -e "$A" ]] && { note "ABORT: $A already exists"; exit 1; }
mkdir -p "$A"
for entry in results-reps results-sweep results-memory results-workers results-thermal; do
    [[ -e "$entry" ]] && mv "$entry" "$A"/
done
note "num_streams 0 systems results archived in $A"

# 2. Isolate results/.
HOLD=results-hold-systems
[[ -e "$HOLD" ]] && { note "ABORT: $HOLD already exists"; exit 1; }
mv results "$HOLD" && mkdir results || { note "ABORT: could not move results/ aside"; exit 1; }
restore() {
    [[ -d "$HOLD" ]] || return
    mv results "systems-scratch-$(date +%Y%m%d-%H%M%S)" && mv "$HOLD" results
    note "results/ restored"
}
trap restore EXIT
trap 'exit 1' INT TERM
note "results/ moved aside to $HOLD for the systems runs; idle $(vcgencmd measure_temp), flags $(vcgencmd get_throttled)"

# 3. Measurements.
mkdir -p systems-logs
./scripts/run_systems_reps.sh 3 waymo cwru musan > systems-logs/reps.log 2>&1
note "workload matrix done"
FIRST_REP=4 STAGES=flink ./scripts/run_systems_reps.sh 3 cwru > systems-logs/reps_cwru_flink_4-6.log 2>&1
note "CWRU Flink repetitions 4-6 done"
RATES="10 15 20 30 45 65 90" ./scripts/run_systems_rate_sweep.sh waymo 3 > systems-logs/rate_sweep.log 2>&1
note "Waymo rate sweep done"

failures=$(grep -h "!!" systems-logs/*.log | wc -l)
throttled=$(awk -F, '$10 == "throttled"' results-thermal/arms.csv | wc -l)
retried=$(awk -F, '$10 == "no"' results-thermal/arms.csv | wc -l)
note "${failures} failure lines in systems-logs; ${retried} throttled attempts re-run, ${throttled} kept although throttled"
note "Pi systems rerun finished"
