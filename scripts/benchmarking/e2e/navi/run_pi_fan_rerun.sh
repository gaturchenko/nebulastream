#!/usr/bin/env bash
# Pi post-join rerun with active cooling, at OpenVINO 1 stream x 4 threads and the stock 1.8 GHz:
#   1. archive the uncooled results of 2026-10-23
#   2. fan check: stop unless the idle SoC is below IDLE_MAX_C (with the fan it idles at ~31 C and the hottest cell, k=4
#      SEQUENTIAL at ADVANCE 333 ms, peaked at 39.4 C on 2026-10-24; without it, 56-58 C idle and 83.7 C)
#   3. the profiling pilot (all models), 4. experiment A (k=2,4, dedup), as on 2026-10-23
#   5. any run during which the firmware limited the CPU is moved aside and re-measured (up to 2 rounds), then processed
# Temperature and flags are sampled every 2 s. COOLDOWN_C is only a safety net: runs start below it, and with the fan the SoC
# never gets near it, so it costs no time. Progress: results/run_pi_fan_rerun.log
set -uo pipefail
cd ~/navi
log=results/run_pi_fan_rerun.log
PY=models-benchmarking/venv/bin/python
R=results/post_join
export SAMPLE_S=2 COOLDOWN_C="${COOLDOWN_C:-60}"
IDLE_MAX_C="${IDLE_MAX_C:-45}"
note() { echo "### $* ($(date '+%F %T'))" | tee -a "$log"; }

# Prints "<max temp> <flagged samples>" over a thermal.csv; flagged = any of the current-state bits 0-3 (under-voltage,
# frequency capped, throttled, soft temperature limit).
thermal_stats() {
    "$PY" - "$1" <<'PY'
import sys
import pandas as pd
t = pd.read_csv(sys.argv[1])
flags = t["throttled"].map(lambda v: int(str(v), 16) & 0xF)
print(f"{t['temp_c'].max():.1f} {int((flags != 0).sum())}")
PY
}

[[ "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)" == 1800000 ]] || { note "ABORT: max frequency is not 1.8 GHz"; exit 1; }
pgrep -f "bin/systest|run_post_join" > /dev/null && { note "ABORT: an experiment is already running"; exit 1; }

# 1. Archive.
A=results/uncooled_1x4
[[ -e "$A" ]] && { note "ABORT: $A already exists"; exit 1; }
mkdir -p "$A"
mv results/post_join results/post_join_pilot results/post_join_throttled results/streams1_chooses_profiled_default \
    results/post_join_*.log results/post_join_*.txt results/run_pi_post_join.log results/followup_pi_post_join.out "$A"/ 2>/dev/null
note "uncooled results archived in $A; idle $(vcgencmd measure_temp), flags $(vcgencmd get_throttled)"

# 2. Fan check.
idle=$(vcgencmd measure_temp | sed 's/[^0-9]*\([0-9]*\).*/\1/')
(( idle < IDLE_MAX_C )) || { note "ABORT: idle SoC at ${idle} C, not below ${IDLE_MAX_C} C; is the fan running?"; exit 1; }
note "fan check passed: idle ${idle} C"

# 3. Pilot.
COOLDOWN_C="$COOLDOWN_C" ./scripts/run_post_join_pilot.sh 3 > results/post_join_pilot_driver.log 2>&1
"$PY" scripts/summarize_post_join_pilot.py results/post_join_pilot > results/post_join_pilot_summary.txt 2>&1
read -r pilot_max pilot_flagged < <(thermal_stats results/post_join_pilot/thermal.csv)
[[ "$pilot_flagged" == 0 ]] && note "pilot done: max ${pilot_max} C, no flags" \
    || note "ATTENTION pilot done with ${pilot_flagged} flagged samples (max ${pilot_max} C): its profile is not clean"

# 4. Experiment A.
export KS="2 4" ADVANCES_MS="1000 500 333" CONFIGS="BATCHED:true SEQUENTIAL:true" STATIC_KS=" " CHOOSES_DEDUP="true" \
    CHOOSES_ADVANCE_MS=1000 CONTROL_KS=4 CONTROL_ADVANCES_MS=333 DURATION_S=60
./scripts/run_post_join.sh 3 > results/post_join_driver.log 2>&1
"$PY" scripts/process_post_join.py --repetitions 3 > results/post_join_summary.txt 2>&1
note "sweep done"

# 5. Re-measure throttled runs: run_post_join.sh refills exactly the repetitions moved aside.
for round in 1 2; do
    moved="results/post_join_throttled/round${round}"
    mkdir -p "$moved"
    "$PY" - "$R" "$moved" > "$moved/moved.txt" <<'PY'
import shutil, sys
from pathlib import Path
import pandas as pd
root, target_root = Path(sys.argv[1]), Path(sys.argv[2])
runs = pd.read_csv(root / "post_join_runs.csv", dtype={"section": str})
for run in runs[runs["throttled_during_run"] == True].itertuples():
    pattern = f"optimizer.post_join_inference_strategy={run.strategy}__inference.use_batch_deduplication={str(run.dedup).lower()}"
    for rep_dir in root.glob(f"{pattern}/post_join_k{run.k}.test_{str(run.section).zfill(2)}/{run.repetition}"):
        target = target_root / rep_dir.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(rep_dir), str(target))
        print(target)
PY
    count=$(wc -l < "$moved/moved.txt")
    note "round ${round}: ${count} throttled runs moved aside"
    [[ "$count" == 0 ]] && break
    ./scripts/run_post_join.sh 3 > "results/post_join_refill_round${round}.log" 2>&1
    "$PY" scripts/process_post_join.py --repetitions 3 > results/post_join_summary.txt 2>&1
done
remaining=$("$PY" -c "import pandas as pd; r = pd.read_csv('$R/post_join_runs.csv'); print(int((r['throttled_during_run'] == True).sum()))")
[[ "$remaining" == 0 ]] || note "ATTENTION ${remaining} runs still throttled after 2 rounds"
note "Pi fan rerun finished"
