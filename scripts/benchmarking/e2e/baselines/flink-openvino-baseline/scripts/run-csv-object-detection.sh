#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Your current submodule location:
# nebulastream/scripts/benchmarking/e2e/baselines/flink-openvino-baseline
NEBULASTREAM_DIR="${NEBULASTREAM_DIR:-$(realpath "$PROJECT_DIR/../../../../..")}"

VCPKG_TARGET_TRIPLET="${VCPKG_TARGET_TRIPLET:-x64-linux-none-local}"
VCPKG_INSTALLED_DIR="${VCPKG_INSTALLED_DIR:-$NEBULASTREAM_DIR/cmake-build-release/vcpkg_installed}"
VCPKG_PREFIX="$VCPKG_INSTALLED_DIR/$VCPKG_TARGET_TRIPLET"

NATIVE_DIR="$PROJECT_DIR/target/native-libs"
NATIVE_BUILD_DIR="$PROJECT_DIR/target/native-build"

MODELS_BASE_DIR="${MODELS_BASE_DIR:-$NEBULASTREAM_DIR/nes-systests/testdata/model}"
DATASETS_BASE_DIR="${DATASETS_BASE_DIR:-$NEBULASTREAM_DIR/nes-systests/testdata/large}"

MODEL_REL_PATH="${1:?Usage: $0 <model-rel-path> <csv-rel-path> [device] [flink-parallelism] [ov-streams] [ov-threads] [base64-col] [has-header] [width] [height] [print-results] [stats-csv-path]}"
CSV_REL_PATH="${2:?Usage: $0 <model-rel-path> <csv-rel-path> [device] [flink-parallelism] [ov-streams] [ov-threads] [base64-col] [has-header] [width] [height] [print-results] [stats-csv-path]}"

MODEL_PATH="$MODELS_BASE_DIR/$MODEL_REL_PATH"
CSV_PATH="$DATASETS_BASE_DIR/$CSV_REL_PATH"

DEVICE="${3:-CPU}"
FLINK_PARALLELISM="${4:-1}"
OPENVINO_STREAMS="${5:-1}"
OPENVINO_THREADS="${6:-1}"

BASE64_COLUMN="${7:-1}"
HAS_HEADER="${8:-true}"

WIDTH="${9:-320}"
HEIGHT="${10:-320}"
PRINT_RESULTS="${11:-false}"
STATS_CSV_PATH="${12:-$PROJECT_DIR/target/openvino-object-detection-stats.csv}"

if [[ ! -f "$NATIVE_DIR/libflink_openvino_bridge_jni.so" ]]; then
  echo "Missing native JNI library. Run:"
  echo "  ./scripts/build-with-vcpkg.sh"
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  ONNX_PATH="${MODEL_PATH%.*}.onnx"

  if [[ ! -f "$ONNX_PATH" ]]; then
    echo "Missing OpenVINO model and matching ONNX model:"
    echo "  XML:  $MODEL_PATH"
    echo "  ONNX: $ONNX_PATH"
    exit 1
  fi

  if ! command -v ovc >/dev/null 2>&1; then
    echo "Found ONNX model but 'ovc' is not available on PATH:"
    echo "  $ONNX_PATH"
    exit 1
  fi

  echo "Missing OpenVINO XML model, converting ONNX with ovc:"
  echo "  ONNX: $ONNX_PATH"
  echo "  XML:  $MODEL_PATH"
  ovc --compress_to_fp16=False "$ONNX_PATH" --output_model "$MODEL_PATH"
fi

if [[ ! -f "$CSV_PATH" ]]; then
  echo "Missing CSV dataset:"
  echo "  $CSV_PATH"
  exit 1
fi

mvn -q dependency:build-classpath -Dmdep.outputFile=cp.txt

export LD_LIBRARY_PATH="$NATIVE_DIR:$NATIVE_BUILD_DIR:$VCPKG_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "Using:"
echo "  NEBULASTREAM_DIR=$NEBULASTREAM_DIR"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  CSV_PATH=$CSV_PATH"
echo "  DEVICE=$DEVICE"
echo "  FLINK_PARALLELISM=$FLINK_PARALLELISM"
echo "  OPENVINO_STREAMS=$OPENVINO_STREAMS"
echo "  OPENVINO_THREADS=$OPENVINO_THREADS"
echo "  BASE64_COLUMN=$BASE64_COLUMN"
echo "  HAS_HEADER=$HAS_HEADER"
echo "  WIDTH=$WIDTH"
echo "  HEIGHT=$HEIGHT"
echo "  PRINT_RESULTS=$PRINT_RESULTS"
echo "  STATS_CSV_PATH=$STATS_CSV_PATH"

java \
  -Djava.library.path="$NATIVE_DIR:$NATIVE_BUILD_DIR" \
  -cp "$PROJECT_DIR/target/flink-openvino-baseline-1.0-SNAPSHOT.jar:$(cat cp.txt)" \
  benchmark.openvino.OpenVinoCsvObjectDetectionJob \
  "$MODEL_PATH" \
  "$CSV_PATH" \
  "$DEVICE" \
  "$FLINK_PARALLELISM" \
  "$OPENVINO_STREAMS" \
  "$OPENVINO_THREADS" \
  "$BASE64_COLUMN" \
  "$HAS_HEADER" \
  "$WIDTH" \
  "$HEIGHT" \
  "$PRINT_RESULTS" \
  "$STATS_CSV_PATH"
