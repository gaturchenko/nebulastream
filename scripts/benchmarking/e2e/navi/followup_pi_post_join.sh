#!/usr/bin/env bash
# After the Pi post-join sweep: install the new systest (its optimizer profiles with the test files' GlobalConfiguration), rerun the
# OPTIMIZER_CHOOSES runs with it, then rerun every run that was throttled mid-run, starting below 55 C, and reprocess.
set -uo pipefail
cd ~/navi
log=results/run_pi_post_join.log
R=results/post_join
while ! grep -q "Pi post-join run finished" "$log"; do sleep 60; done
while pgrep -f "run_post_join.sh|bin/systest" > /dev/null; do sleep 10; done
mv bin/systest bin/systest.streams1-cli && mv bin/systest.new bin/systest && mv bin/BUILD_INFO.txt bin/BUILD_INFO.txt.streams1-cli \
    && mv bin/BUILD_INFO.txt.new bin/BUILD_INFO.txt || { echo "### binary swap FAILED $(date)" >> "$log"; exit 1; }
echo "### follow-up: new systest installed $(date)" >> "$log"
cp results/post_join_summary.txt results/post_join_summary_before_followup.txt
export KS="2 4" ADVANCES_MS="1000 500 333" CHOOSES_DEDUP="true" CHOOSES_ADVANCE_MS=1000 CONTROL_ADVANCES_MS=333 DURATION_S=60
mkdir -p results/streams1_chooses_profiled_default
mv $R/optimizer.post_join_inference_strategy=OPTIMIZER_CHOOSES__* results/streams1_chooses_profiled_default/
cp $R/_grid.csv $R/_grid_full.csv
CONFIGS=" " STATIC_KS=" " CONTROL_KS=" " COOLDOWN_C=60 ./scripts/run_post_join.sh 3 > results/post_join_followup_chooses.log 2>&1
cp $R/_grid_full.csv $R/_grid.csv
echo "### follow-up: OPTIMIZER_CHOOSES rerun done $(date)" >> "$log"
models-benchmarking/venv/bin/python scripts/process_post_join.py --repetitions 3 > results/post_join_summary.txt 2>&1
# Runs throttled mid-run: move their repetitions aside so that the driver refills exactly those.
mkdir -p results/post_join_throttled
models-benchmarking/venv/bin/python - "$R" > results/post_join_throttled/moved.txt <<'PY'
import shutil, sys
from pathlib import Path
import pandas as pd
root = Path(sys.argv[1])
runs = pd.read_csv(root / "post_join_runs.csv", dtype={"section": str})
throttled = runs[runs["throttled_during_run"] == True]
for run in throttled.itertuples():
    pattern = f"optimizer.post_join_inference_strategy={run.strategy}__inference.use_batch_deduplication={str(run.dedup).lower()}"
    for rep_dir in root.glob(f"{pattern}/post_join_k{run.k}.test_{str(run.section).zfill(2)}/{run.repetition}"):
        target = root.parent / "post_join_throttled" / rep_dir.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(rep_dir), str(target))
        print(target)
PY
echo "### follow-up: $(wc -l < results/post_join_throttled/moved.txt) throttled runs moved aside $(date)" >> "$log"
CONFIGS="BATCHED:true SEQUENTIAL:true" STATIC_KS=" " CONTROL_KS=4 COOLDOWN_C=55 ./scripts/run_post_join.sh 3 > results/post_join_followup_throttled.log 2>&1
models-benchmarking/venv/bin/python scripts/process_post_join.py --repetitions 3 > results/post_join_summary.txt 2>&1
echo "### follow-up finished $(date)" >> "$log"
