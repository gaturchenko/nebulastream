#!/usr/bin/env bash
# Waymo rate sweep, TorchServe arm only, at the 3 workers its tuning pilot chose (run_pi_systems_tuning.sh) and NES's HTTP sink
# with 3 connections, as in the tuning runs, so that the sweep matches the tuned workload matrix. Same rates, repetitions and
# thermal guards as run_pi_systems_rerun.sh. The earlier TorchServe curve (1 worker, 2 connections) moves to
# results-sweep-torchserve-1worker/; results/ is swapped for an empty one and restored afterwards.
# Progress: ~/navi/run_pi_ts_sweep_workers3.log
set -uo pipefail
cd ~/navi
log=run_pi_ts_sweep_workers3.log
export SAMPLE_S=2 COOLDOWN_C="${COOLDOWN_C:-60}" THROTTLE_RETRIES="${THROTTLE_RETRIES:-2}"
note() { echo "### $* ($(date '+%F %T'))" | tee -a "$log"; }

: > "$log"
pgrep -f "[b]in/systest|[r]un_pi_(fan|batch_rate|systems|ablation|ablation_udf)_rerun\.sh|[r]un_pi_(cwru|systems)_tuning|[r]un_pi_flink_sweep|[r]un_systems|[j]ava |[o]rg.pytorch.serve" \
    > /dev/null && { note "ABORT: an experiment is running"; exit 1; }
grep -q 'SWEEP_TEST_FILE' scripts/run_systems_rate_sweep.sh || { note "ABORT: the sweep script has no SWEEP_TEST_FILE override"; exit 1; }

TEST=queries/waymo/systems_ts3.test
sed "s#^\( *\)[0-9]\+ AS \`SINK\`.connections#\13 AS \`SINK\`.connections#" queries/waymo/systems.test > "$TEST"
grep -q "3 AS \`SINK\`.connections" "$TEST" || { note "ABORT: could not set 3 HTTP connections in $TEST"; exit 1; }

OLD=results-sweep-torchserve-1worker
[[ -e "$OLD" ]] && { note "ABORT: $OLD already exists"; exit 1; }
mkdir -p "$OLD" && mv results-sweep/torchserve_waymo_r*_rep[0-9].csv "$OLD"/ || { note "ABORT: could not archive the old TorchServe sweep"; exit 1; }
note "archived $(ls "$OLD" | wc -l) TorchServe sweep files in $OLD"

HOLD=results-hold-ts-sweep
[[ -e "$HOLD" ]] && { note "ABORT: $HOLD already exists"; exit 1; }
mv results "$HOLD" && mkdir results || { note "ABORT: could not move results/ aside"; exit 1; }
restore() {
    [[ -d "$HOLD" ]] || return
    mv results "ts-sweep-scratch-$(date +%Y%m%d-%H%M%S)" && mv "$HOLD" results
    note "results/ restored"
}
trap restore EXIT
trap 'exit 1' INT TERM

mkdir -p systems-logs
STAGES=torchserve TS_WORKERS=3 SWEEP_TEST_FILE=waymo/systems_ts3.test RATES="10 15 20 30 45 65 90" \
    ./scripts/run_systems_rate_sweep.sh waymo 3 > systems-logs/rate_sweep_ts_workers3.log 2>&1
note "TorchServe rate sweep at 3 workers done: $(ls results-sweep/torchserve_waymo_r*_rep[0-9].csv | wc -l) runs, $(grep -c '!!' systems-logs/rate_sweep_ts_workers3.log) failures"
restore
trap - EXIT INT TERM
rm -f "$TEST"
note "Pi TorchServe sweep finished"
