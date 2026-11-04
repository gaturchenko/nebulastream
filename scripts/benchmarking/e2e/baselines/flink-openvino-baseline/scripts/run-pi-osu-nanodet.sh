#!/usr/bin/env bash
# Flink + OpenVINO NanoDet baseline on the Raspberry Pi (aarch64).
#
# This is the Flink counterpart of the NES "UDF" object-detection case on the OSU-RGB dataset:
# it reads the same single-column base64 CSV, decodes+resizes to 192x192, runs NanoDet through
# OpenVINO (via the JNI bridge), discards detections, and writes latency/throughput stats.
# Use it to SUBSTITUTE the NES-UDF row in the OSU-RGB framework comparison.
#
# Prereqs on the Pi (see build-native-pi.sh):
#   * target/native-libs/libflink_openvino_bridge_jni.so   (built on the Pi)
#   * target/flink-openvino-baseline-1.0-SNAPSHOT.jar       (copied; arch-independent)
#   * cp.txt + the ~/.m2 dependency jars it references       (copied)
#   * the OpenVINO pip wheel (provides libopenvino + the ARM CPU plugin at runtime)
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

OV_DIR="${OV_DIR:-/home/gaturchenko/navi/models-benchmarking/venv/lib/python3.13/site-packages/openvino}"
MODEL="${MODEL:-/home/gaturchenko/navi/models-benchmarking/models/nanodet/nanodet.xml}"
CSV="${CSV:-/home/gaturchenko/navi/datasets/osu-rgb/otcbvs_osu_rgb.csv}"
STATS="${STATS:-/home/gaturchenko/navi/results/flink_openvino_stats.csv}"

DEVICE="${DEVICE:-CPU}"
PARALLELISM="${PARALLELISM:-1}"   # Flink parallelism == number of inference lanes
STREAMS="${STREAMS:-1}"           # OpenVINO num_streams
THREADS="${THREADS:-1}"           # OpenVINO inference_num_threads
WIDTH="${WIDTH:-192}"
HEIGHT="${HEIGHT:-192}"
BASE64_COL="${BASE64_COL:-0}"     # OSU-RGB CSV is single-column base64...
HAS_HEADER="${HAS_HEADER:-false}" # ...with no header row

# Comparison-key columns written into the stats row so it slots into the NES/TorchServe table
# (substituting the NES-UDF case). Override to match the labels you compare against.
QUERY_NAME="${QUERY_NAME:-flink_osu_rgb}"
PARAM_NAME="${PARAM_NAME:-baseline}"
PARAM_VALUE="${PARAM_VALUE:-flink}"
REPETITION="${REPETITION:-0}"

# Fixed ingestion rate (tuples/s) to match the NES/TorchServe TCP source MOCK_TUPLE_RATE (OSU-RGB=30),
# so all three systems see the same offered load. Set RATE=0 for an unbounded (backpressure) source.
RATE="${RATE:-30}"

NATIVE_DIR="$PROJECT_DIR/target/native-libs"
JAR="$PROJECT_DIR/target/flink-openvino-baseline-1.0-SNAPSHOT.jar"

[[ -f "$NATIVE_DIR/libflink_openvino_bridge_jni.so" ]] || { echo "Missing native lib; run scripts/build-native-pi.sh"; exit 1; }
[[ -f "$JAR" ]] || { echo "Missing app jar: $JAR"; exit 1; }
[[ -f "$PROJECT_DIR/cp.txt" ]] || { echo "Missing classpath file: $PROJECT_DIR/cp.txt"; exit 1; }

# Convert ONNX -> OpenVINO IR on first use (needs ovc from the venv).
if [[ ! -f "$MODEL" ]]; then
  ONNX="${MODEL%.*}.onnx"
  [[ -f "$ONNX" ]] || { echo "Missing model (no XML and no ONNX): $MODEL / $ONNX"; exit 1; }
  echo "Converting $ONNX -> $MODEL with ovc"
  VENV_ROOT="${OV_DIR%%/lib/*}"   # .../venv/lib/pythonX/site-packages/openvino -> .../venv
  ( source "$VENV_ROOT/bin/activate" 2>/dev/null || true
    ovc "$ONNX" --output_model "$MODEL" --compress_to_fp16=False )
fi

mkdir -p "$(dirname "$STATS")"
cd "$PROJECT_DIR"
export LD_LIBRARY_PATH="$OV_DIR/libs:$NATIVE_DIR:${LD_LIBRARY_PATH:-}"

echo "Flink OpenVINO NanoDet: model=$MODEL device=$DEVICE parallelism=$PARALLELISM streams=$STREAMS threads=$THREADS ${WIDTH}x${HEIGHT}"
java \
  -Djava.library.path="$NATIVE_DIR" \
  -cp "$JAR:$(cat "$PROJECT_DIR/cp.txt")" \
  benchmark.openvino.OpenVinoCsvObjectDetectionJob \
  "$MODEL" "$CSV" "$DEVICE" "$PARALLELISM" "$STREAMS" "$THREADS" \
  "$BASE64_COL" "$HAS_HEADER" "$WIDTH" "$HEIGHT" false "$STATS" \
  "$QUERY_NAME" "$PARAM_NAME" "$PARAM_VALUE" "$REPETITION" "$RATE"

echo "Wrote stats: $STATS"
