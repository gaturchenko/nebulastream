#!/usr/bin/env bash
# Sensitivity runs of the implementation selection on the Pi, in sequence: part B (SARAD), part C (window scaling), part D (profile
# stability). Each sweep is followed by up to 2 rounds of re-measuring runs the firmware throttled. Progress: results/sensitivity/run.log
# Re-invoking resumes, since the sweeps skip finished repetitions.
set -uo pipefail
cd ~/navi
log=results/sensitivity/run.log
PY=models-benchmarking/venv/bin/python
export SAMPLE_S=2 COOLDOWN_C=60
note() { echo "### $* ($(date '+%F %T'))" | tee -a "$log"; }

measure() {
    local name="$1" results="results/sensitivity/$1"
    scripts/run_sensitivity_sweep.sh "$name" > "results/sensitivity/${name}_driver.log" 2>&1
    note "${name}: sweep done"
    for round in 1 2; do
        local moved="results/sensitivity/throttled/${name}/round${round}"
        mkdir -p "$moved"
        "$PY" scripts/move_throttled_runs.py "$results" "$moved" > "$moved/moved.txt"
        local count
        count=$(wc -l < "$moved/moved.txt")
        note "${name}: round ${round}, ${count} throttled runs moved aside"
        [[ "$count" == 0 ]] && return
        scripts/run_sensitivity_sweep.sh "$name" > "results/sensitivity/${name}_driver_refill${round}.log" 2>&1
    done
}

pgrep -f "bin/systest|nes-model-profiler" > /dev/null && { note "ABORT: something else is running"; exit 1; }
note "starting; $(vcgencmd measure_temp), flags $(vcgencmd get_throttled)"
measure sarad
note "PART B DONE"
measure nbeats_w37
measure nbeats_w150
measure nbeats_w300
note "PART C DONE"
scripts/run_profile_stability.sh > results/sensitivity/profile_driver.log 2>&1
note "PART D DONE"
note "ALL DONE; flags $(vcgencmd get_throttled)"
