#!/usr/bin/env bash
# Resumes the Pi end-to-end grid after the pilot with a 60 C cooldown threshold (at 1 stream x 4 threads a 60 s run heats the Pi
# by ~13 C, so 70 C starts crossed the 80 C soft limit mid-run). Finished repetitions are skipped by run_systests_lat.py.
cd ~/navi
log=results/run_pi_post_join.log
echo "### resumed with COOLDOWN_C=60 $(date)" >> "$log"
KS="2 4" ADVANCES_MS="1000 500 333" CONFIGS="BATCHED:true SEQUENTIAL:true" STATIC_KS=" " \
CHOOSES_DEDUP="true" CHOOSES_ADVANCE_MS=1000 CONTROL_KS=4 CONTROL_ADVANCES_MS=333 DURATION_S=60 COOLDOWN_C=60 \
    ./scripts/run_post_join.sh 3 >> results/post_join_driver.log 2>&1
echo "### sweep done $(date)" >> "$log"
models-benchmarking/venv/bin/python scripts/process_post_join.py --repetitions 3 > results/post_join_summary.txt 2>&1
echo "### Pi post-join run finished $(date)" >> "$log"
