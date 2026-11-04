#!/usr/bin/env bash
# Flink + OpenVINO SARAD baseline on the Raspberry Pi (aarch64), CWRU bearing-vibration workload.
#
# Counterpart of the NES "UDF" case in queries/cwru-bearing/systems.test: it reads the same
# "timestamp,value" CSV, aggregates a 200 ms sliding window (ADVANCE 100 ms) of the paced 48 kHz
# sample stream into the model's 9600-sample input, runs SARAD through OpenVINO via the JNI bridge,
# discards the anomaly score and writes a comparison-schema stats row.
# Use it to SUBSTITUTE the NES-UDF row in the CWRU framework comparison.
#
# Window geometry is expressed in SAMPLES and must stay consistent with the .test:
#   RATE=48000 tuples/s, SIZE 200 ms -> 9600 samples (the model input), ADVANCE 100 ms -> 4800 samples.
#
# ADVANCE is 100 ms here, not the 10 ms of the CWRU ablation: SARAD costs ~60 ms per inference on the
# Pi, so a 10 ms advance asks for 100 inferences/s = 6 core-seconds per second from 3 worker threads.
# Every system saturates there and the P95 latency this comparison reports would measure queue
# backlog rather than per-system overhead. At 100 ms the offered load is 10 inferences/s -- the same
# rate as the Waymo comparison -- which all three systems serve at ~20% utilisation.
#
# Fairness parity with the NES side (queries/cwru-bearing/systems.test):
#   * PARALLELISM=3  <-> worker.query_engine.number_of_worker_threads: 3
#   * STREAMS=1, THREADS=4 <-> the NES OpenVINO session: one stream with four inference threads per worker thread
#     (worker.default_query_execution.inference.openvino_num_streams / openvino_inference_num_threads in the .test).
#     With FLINK_OV_SHARE_COMPILED_MODEL=1 the shared model gets one stream per lane (FLINK_OV_SHARED_LANES).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

OV_DIR="${OV_DIR:-/home/gaturchenko/navi/models-benchmarking/venv/lib/python3.13/site-packages/openvino}"
MODEL="${MODEL:-/home/gaturchenko/navi/models-benchmarking/models/sarad/sarad_medium.xml}"
CSV="${CSV:-/home/gaturchenko/navi/datasets/cwru-bearing/X228_FE_time.csv}"
STATS="${STATS:-/home/gaturchenko/navi/results/flink_openvino_stats.csv}"

DEVICE="${DEVICE:-CPU}"
PARALLELISM="${PARALLELISM:-3}"   # Flink parallelism == number of inference lanes; run_systems.sh passes the .test's worker threads
STREAMS="${STREAMS:-1}"           # OpenVINO num_streams, like NES (negative = unset)
THREADS="${THREADS:-4}"           # OpenVINO inference_num_threads per lane, like NES

VALUE_COL="${VALUE_COL:-1}"       # CWRU CSV is "timestamp,value"...
HAS_HEADER="${HAS_HEADER:-false}" # ...with no header row
WINDOW_SAMPLES="${WINDOW_SAMPLES:-9600}"
ADVANCE_SAMPLES="${ADVANCE_SAMPLES:-4800}"
SCALE_DIVISOR="${SCALE_DIVISOR:-0}"  # CWRU values are already float32 g-units: no scaling

QUERY_NAME="${QUERY_NAME:-flink_cwru_sarad}"
PARAM_NAME="${PARAM_NAME:-baseline}"
PARAM_VALUE="${PARAM_VALUE:-flink}"
REPETITION="${REPETITION:-0}"

# Fixed ingestion rate (tuples/s) matching the NES/TorchServe TCP source MOCK_TUPLE_RATE (CWRU=48000).
RATE="${RATE:-48000}"

NATIVE_DIR="$PROJECT_DIR/target/native-libs"
JAR="$PROJECT_DIR/target/flink-trace.jar"

[[ -f "$NATIVE_DIR/libflink_openvino_bridge_jni.so" ]] || { echo "Missing native lib; run scripts/build-native-pi.sh"; exit 1; }
[[ -f "$JAR" ]] || { echo "Missing app jar: $JAR"; exit 1; }
[[ -f "$PROJECT_DIR/cp.txt" ]] || { echo "Missing classpath file: $PROJECT_DIR/cp.txt"; exit 1; }
[[ -f "$CSV" ]] || { echo "Missing dataset: $CSV"; exit 1; }

# Convert ONNX -> OpenVINO IR on first use (needs ovc from the venv).
if [[ ! -f "$MODEL" ]]; then
  ONNX="${MODEL%.*}.onnx"
  [[ -f "$ONNX" ]] || { echo "Missing model (no XML and no ONNX): $MODEL / $ONNX"; exit 1; }
  echo "Converting $ONNX -> $MODEL with ovc"
  VENV_ROOT="${OV_DIR%%/lib/*}"
  ( source "$VENV_ROOT/bin/activate" 2>/dev/null || true
    ovc "$ONNX" --output_model "$MODEL" --compress_to_fp16=False )
fi

mkdir -p "$(dirname "$STATS")"
cd "$PROJECT_DIR"
export FLINK_OV_SHARED_LANES="${FLINK_OV_SHARED_LANES:-$PARALLELISM}"
export LD_LIBRARY_PATH="$OV_DIR/libs:$NATIVE_DIR:${LD_LIBRARY_PATH:-}"

echo "Flink OpenVINO SARAD: model=$MODEL device=$DEVICE parallelism=$PARALLELISM streams=$STREAMS threads=$THREADS window=${WINDOW_SAMPLES}/${ADVANCE_SAMPLES} rate=$RATE"
java \
  -Djava.library.path="$NATIVE_DIR" \
  -cp "$JAR:$(cat "$PROJECT_DIR/cp.txt")" \
  benchmark.openvino.OpenVinoCsvWindowedInferenceJob \
  "$MODEL" "$CSV" "$DEVICE" "$PARALLELISM" "$STREAMS" "$THREADS" \
  "$VALUE_COL" "$HAS_HEADER" "$WINDOW_SAMPLES" "$ADVANCE_SAMPLES" "$SCALE_DIVISOR" \
  false "$STATS" "$QUERY_NAME" "$PARAM_NAME" "$PARAM_VALUE" "$REPETITION" "$RATE"

echo "Wrote stats: $STATS"
