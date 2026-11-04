#!/usr/bin/env bash
# Pi half of the post-join experiment at OpenVINO 1 stream x 4 threads: the profiling pilot (all models), then the end-to-end
# check of the decision (k=2,4, dedup), then processing.
cd ~/navi
log=results/run_pi_post_join.log
echo "### Pi post-join run started $(date)" > "$log"
./scripts/run_post_join_pilot.sh 3 > results/post_join_pilot_driver.log 2>&1
models-benchmarking/venv/bin/python scripts/summarize_post_join_pilot.py results/post_join_pilot > results/post_join_pilot_summary.txt 2>&1
echo "### pilot done $(date)" >> "$log"
KS="2 4" ADVANCES_MS="1000 500 333" CONFIGS="BATCHED:true SEQUENTIAL:true" STATIC_KS=" " \
CHOOSES_DEDUP="true" CHOOSES_ADVANCE_MS=1000 CONTROL_KS=4 CONTROL_ADVANCES_MS=333 DURATION_S=60 \
    ./scripts/run_post_join.sh 3 > results/post_join_driver.log 2>&1
echo "### sweep done $(date)" >> "$log"
models-benchmarking/venv/bin/python scripts/process_post_join.py --repetitions 3 > results/post_join_summary.txt 2>&1
echo "### Pi post-join run finished $(date)" >> "$log"
