#!/usr/bin/env bash
# Add-on to run_pi_sensitivity.sh: the SARAD slides that bound u_max from above. Waits for the main chain, as both need the Pi alone.
set -uo pipefail
cd ~/navi
log=results/sensitivity/run.log
PY=models-benchmarking/venv/bin/python
export SAMPLE_S=2 COOLDOWN_C=60
note() { echo "### $* ($(date '+%F %T'))" | tee -a "$log"; }
while ! grep -q "ALL DONE" "$log"; do
    pgrep -f run_pi_sensitivity.sh > /dev/null || { note "ABORT knee: the main chain stopped without finishing"; exit 1; }
    sleep 20
done
scripts/run_sensitivity_sweep.sh sarad_knee > results/sensitivity/sarad_knee_driver.log 2>&1
moved=results/sensitivity/throttled/sarad_knee/round1
mkdir -p "$moved"
"$PY" scripts/move_throttled_runs.py results/sensitivity/sarad_knee "$moved" > "$moved/moved.txt"
note "sarad_knee: $(wc -l < "$moved/moved.txt") throttled runs moved aside"
[[ -s "$moved/moved.txt" ]] && scripts/run_sensitivity_sweep.sh sarad_knee > results/sensitivity/sarad_knee_driver_refill.log 2>&1
note "KNEE DONE; flags $(vcgencmd get_throttled)"
