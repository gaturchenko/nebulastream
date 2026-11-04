#!/usr/bin/env bash
# Re-runs only the `assess` cells of the model memory experiment after the memory assessment was fixed to invoke every session it
# measures once (ModelProfiler.cpp: setUpAndInvoke). The private/shared cells run with inference_memory_strategy=FIXED, which never
# assesses, so their numbers stay valid and are kept.
#
#   ./run_pi_memory_assess_rerun.sh            # 3 repetitions, all workloads
#
# Refuses to run on the binary the old assessment ran on. The old assess rows and their raw files move to
# results/model_memory/assess-setup-only/, so run_model_memory.py (which skips cells that already passed) runs them again.
set -uo pipefail
cd ~/navi

OLD_SYSTEST_MD5=870ea85fa16054762e5a2268bca40319
RESULTS=results/model_memory
ARCHIVE=${RESULTS}/assess-setup-only
LOG=run_pi_memory_assess_rerun.log

if [[ "$(md5sum bin/systest | cut -d' ' -f1)" == "$OLD_SYSTEST_MD5" ]]; then
    echo "bin/systest is still the binary with the setup-only assessment; copy the new build first" >&2
    exit 1
fi
if pgrep -f "bin/systest|run_model_memory.py" >/dev/null; then
    echo "another systest or model memory run is active" >&2
    exit 1
fi

{
    echo "### assess re-run started $(date), systest $(md5sum bin/systest | cut -d' ' -f1), $(grep source_revision bin/BUILD_INFO.txt)"
    if [[ ! -d "$ARCHIVE" ]]; then
        mkdir -p "$ARCHIVE/raw"
        cp -p "${RESULTS}/runs.csv" "${RESULTS}/assess.csv" "${RESULTS}/model_memory.csv" "$ARCHIVE/"
        mv "${RESULTS}"/raw/*_assess_rep* "$ARCHIVE/raw/"
        # Keep every row but the assess ones; the header has no mode "assess", so it stays.
        awk -F, '$6 != "assess"' "$ARCHIVE/runs.csv" > "${RESULTS}/runs.csv"
        rm "${RESULTS}/assess.csv"
        echo "archived $(grep -c ',assess,' "$ARCHIVE/runs.csv") assess runs to $ARCHIVE"
    fi
    models-benchmarking/venv/bin/python scripts/run_model_memory.py 3 --modes assess
    models-benchmarking/venv/bin/python scripts/process_model_memory.py
    echo "### assess re-run finished $(date)"
} >> "$LOG" 2>&1
