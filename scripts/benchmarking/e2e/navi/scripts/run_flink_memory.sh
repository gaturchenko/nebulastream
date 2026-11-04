#!/usr/bin/env bash
# Private vs. shared compiled model: what the memory-sharing contribution is actually worth.
#
#   ./scripts/run_flink_memory.sh [reps] [profile ...]
#
# The Flink baseline gives each inference lane its own compiled OpenVINO model by default;
# FLINK_OV_SHARE_COMPILED_MODEL=1 makes the native engine hand every lane one shared compiled model
# with a per-lane infer request. Three repetitions showed that sharing buys NO latency (CWRU e2e P95
# 932 vs 929 ms), so its justification rests entirely on footprint -- which, until this script, we
# had never measured. A paper claim about memory sharing needs a memory number.
#
# Only the Flink arm runs: NES and TorchServe have no equivalent setting.
#
# Peak RSS of the job JVM is the metric. The Flink job runs as a single-JVM MiniCluster, so one
# process holds the heap AND the native OpenVINO allocations, which is exactly the quantity the
# sharing changes. Sampled every 2 s, because engine construction is where the allocation happens
# and it is over in tens of seconds.
set -uo pipefail

REPS="${1:-3}"
shift || true
PROFILES=("$@")
if [[ ${#PROFILES[@]} -eq 0 ]]; then
    PROFILES=(waymo cwru musan)
fi

OUT=results-memory
mkdir -p "$OUT"
summary="$OUT/flink_memory.csv"
echo "mode,workload,rep,peak_rss_mb,min_available_mb,share_log_hits,inference_p50_ms,e2e_p95_ms" > "$summary"

echo "### Flink memory: reps=$REPS profiles=${PROFILES[*]}"
started=$(date +%s)

for mode in private shared; do
    for profile in "${PROFILES[@]}"; do
        # shellcheck source=/dev/null
        source "scripts/systems/${profile}.env"
        for rep in $(seq 1 "$REPS"); do
            echo
            echo "### $mode $WORKLOAD rep $rep/$REPS ($(date +%H:%M:%S))"
            log="/tmp/flinkmem_${mode}_${WORKLOAD}_rep${rep}.log"
            memlog="/tmp/flinkmem_${mode}_${WORKLOAD}_rep${rep}.mem"

            ( while true; do
                avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
                rss=$(ps -eo rss,args 2>/dev/null | awk '/[b]enchmark\.openvino/{s+=$1} END {print s+0}')
                echo "$avail $rss"
                sleep 2
              done ) > "$memlog" 2>&1 &
            sampler=$!

            if [[ "$mode" == "shared" ]]; then
                FLINK_OV_SHARE_COMPILED_MODEL=1 STAGES=flink REP_LABEL="memory_${mode}_rep${rep}" ./scripts/run_systems.sh "$profile" > "$log" 2>&1 \
                    || echo "  !! run failed, see $log"
            else
                STAGES=flink REP_LABEL="memory_${mode}_rep${rep}" ./scripts/run_systems.sh "$profile" > "$log" 2>&1 \
                    || echo "  !! run failed, see $log"
            fi
            kill "$sampler" 2>/dev/null

            # The run is only meaningful if the mode actually engaged: the native engine logs
            # "compiled, shared with later lanes" once and "reusing the shared one" per extra lane.
            # "|| true", not "|| echo 0": grep -c already PRINTS 0 when it matches nothing and
            # exits 1, so an echo on the failure branch appends a second line and the embedded
            # newline splits the CSV row in two.
            hits=$(grep -c "shared with later lanes\|reusing the shared one" "$log" 2>/dev/null || true)
            hits=${hits:-0}
            if [[ "$mode" == "shared" && "$hits" -eq 0 ]]; then
                echo "  !! sharing did NOT engage -- the run measures the private configuration"
            fi
            if [[ "$mode" == "private" && "$hits" -ne 0 ]]; then
                echo "  !! sharing engaged in a private run -- env leaked between runs"
            fi

            inf="" ; e2e=""
            if [[ -f "results/flink_${WORKLOAD}.csv" ]]; then
                inf=$(tail -1 "results/flink_${WORKLOAD}.csv" | awk -F, '{print $14}')
                e2e=$(tail -1 "results/flink_${WORKLOAD}.csv" | awk -F, '{print $11}')
                mv "results/flink_${WORKLOAD}.csv" "$OUT/flink_${WORKLOAD}_${mode}_rep${rep}.csv"
            else
                echo "  !! no Flink result row"
            fi

            # Only samples taken while the job was alive describe it.
            read -r peak minavail <<<"$(awk '$2>0 {if($2>p)p=$2; if(m==0||$1<m)m=$1} END {printf "%.0f %.0f", p/1024, m/1024}' "$memlog")"
            echo "$mode,$WORKLOAD,$rep,$peak,$minavail,$hits,$inf,$e2e" >> "$summary"
            echo "  peak RSS ${peak} MB, min available ${minavail} MB, share-log hits $hits"
        done
    done
done

echo
echo "### finished after $(( ($(date +%s) - started) / 60 )) min"
column -s, -t "$summary"
