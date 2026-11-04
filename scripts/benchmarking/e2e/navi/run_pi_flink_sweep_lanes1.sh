#!/usr/bin/env bash
# Waymo rate sweep, Flink arm only, at the 1 inference lane its tuning pilot chose (run_pi_systems_tuning.sh), so that the sweep
# matches the tuned workload matrix; NAVI's sweep already ran at its tuned 2 workers and TorchServe's does not depend on its
# worker count. Same rates, repetitions and thermal guards as run_pi_systems_rerun.sh. The earlier Flink curve (lanes from the
# .test) moves to results-sweep-flink-lanes-from-test/; results/ is swapped for an empty one and restored afterwards.
# Progress: ~/navi/run_pi_flink_sweep_lanes1.log
set -uo pipefail
cd ~/navi
log=run_pi_flink_sweep_lanes1.log
export SAMPLE_S=2 COOLDOWN_C="${COOLDOWN_C:-60}" THROTTLE_RETRIES="${THROTTLE_RETRIES:-2}"
note() { echo "### $* ($(date '+%F %T'))" | tee -a "$log"; }

: > "$log"
pgrep -f "[b]in/systest|[r]un_pi_(fan|batch_rate|systems|ablation|ablation_udf)_rerun\.sh|[r]un_pi_(cwru|systems)_tuning|[r]un_systems|[j]ava " \
    > /dev/null && { note "ABORT: an experiment is running"; exit 1; }
grep -q 'FLINK_LANES' scripts/run_systems.sh || { note "ABORT: run_systems.sh has no FLINK_LANES override"; exit 1; }

OLD=results-sweep-flink-lanes-from-test
[[ -e "$OLD" ]] && { note "ABORT: $OLD already exists"; exit 1; }
mkdir -p "$OLD" && mv results-sweep/flink_waymo_r*_rep[0-9].csv "$OLD"/ || { note "ABORT: could not archive the old Flink sweep"; exit 1; }
note "archived $(ls "$OLD" | wc -l) Flink sweep files in $OLD"

HOLD=results-hold-flink-sweep
[[ -e "$HOLD" ]] && { note "ABORT: $HOLD already exists"; exit 1; }
mv results "$HOLD" && mkdir results || { note "ABORT: could not move results/ aside"; exit 1; }
restore() {
    [[ -d "$HOLD" ]] || return
    mv results "flink-sweep-scratch-$(date +%Y%m%d-%H%M%S)" && mv "$HOLD" results
    note "results/ restored"
}
trap restore EXIT
trap 'exit 1' INT TERM

mkdir -p systems-logs
STAGES=flink FLINK_LANES=1 RATES="10 15 20 30 45 65 90" ./scripts/run_systems_rate_sweep.sh waymo 3 \
    > systems-logs/rate_sweep_flink_lanes1.log 2>&1
note "Flink rate sweep at 1 lane done: $(ls results-sweep/flink_waymo_r*_rep[0-9].csv | wc -l) runs, $(grep -c '!!' systems-logs/rate_sweep_flink_lanes1.log) failures"
restore
trap - EXIT INT TERM
note "Pi Flink sweep finished"
