#!/usr/bin/env bash
# Repeated framework-comparison runs, for numbers that survive the run-to-run variance a single
# run hides. One CWRU Flink cell alone produced 97 ms and 236 ms inference medians on consecutive
# runs, so a single repetition per cell is not enough to report.
#
#   ./scripts/run_systems_reps.sh [reps] [profile ...]
#
# Each repetition is a full run_systems.sh pass, and its three comparison CSVs are moved into
# results-reps/ as <system>_<workload>_rep<N>.csv. That directory sits OUTSIDE results/, which
# run_systems.sh wipes at the start of every pass.
#
# With SHARE_COMPILED_MODEL=1 only the Flink stage is run, with compiled-model sharing enabled
# (the counterpart of the XXXX-2 openvino_share_compiled_model option), writing
# flink_<workload>_shared_rep<N>.csv. That is the best-effort version of the baseline: without it
# each Flink lane holds its own copy of the model.
set -uo pipefail

REPS="${1:-3}"
shift || true
PROFILES=("$@")
if [[ ${#PROFILES[@]} -eq 0 ]]; then
    PROFILES=(waymo cwru musan)
fi

# Passed straight through to run_systems.sh, and used to decide which CSVs to collect: with
# STAGES=torchserve the other two arms never run, so looking for their output would report a
# missing file for something that was deliberately skipped.
STAGES="${STAGES:-flink nes torchserve}"
SHARE="${SHARE_COMPILED_MODEL:-0}"
REPS_DIR=results-reps
mkdir -p "$REPS_DIR"

echo "### repetitions=$REPS profiles=${PROFILES[*]} stages=[$STAGES] share_compiled_model=$SHARE"
started=$(date +%s)

for rep in $(seq 1 "$REPS"); do
    for profile in "${PROFILES[@]}"; do
        # shellcheck source=/dev/null
        source "scripts/systems/${profile}.env"
        echo
        echo "### rep $rep/$REPS  profile $profile  ($(date +%H:%M:%S))"

        if [[ "$SHARE" == "1" ]]; then
            # Flink only: NES and TorchServe are unaffected by this setting.
            source ./models-benchmarking/venv/bin/activate
            FLINK_OV_SHARE_COMPILED_MODEL=1 \
            QUERY_NAME=flink PARAM_NAME=sharing PARAM_VALUE=shared REPETITION="$rep" \
            MODEL="$MODEL_IR" STATS="$PWD/${REPS_DIR}/flink_${WORKLOAD}_shared_rep${rep}.csv" \
                "$FLINK_RUN" > "/tmp/reps_${WORKLOAD}_shared_rep${rep}.log" 2>&1 \
                || echo "  !! flink (shared) failed, see /tmp/reps_${WORKLOAD}_shared_rep${rep}.log"
            deactivate 2>/dev/null || true
        else
            STAGES="$STAGES" ./scripts/run_systems.sh "$profile" > "/tmp/reps_${WORKLOAD}_rep${rep}.log" 2>&1 \
                || echo "  !! run_systems.sh failed, see /tmp/reps_${WORKLOAD}_rep${rep}.log"
            for system in $STAGES; do
                src="results/${system}_${WORKLOAD}.csv"
                if [[ -f "$src" ]]; then
                    mv "$src" "${REPS_DIR}/${system}_${WORKLOAD}_rep${rep}.csv"
                else
                    echo "  !! missing $src"
                fi
            done
        fi
        echo "  rep $rep $profile done ($(( ($(date +%s) - started) / 60 )) min elapsed)"
    done
done

echo
echo "### all repetitions finished after $(( ($(date +%s) - started) / 60 )) min"
ls -la "$REPS_DIR"
