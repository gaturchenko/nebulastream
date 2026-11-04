#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Default layout:
# nebulastream/
#   nes-systests/testdata/model/
#   scripts/benchmarking/e2e/baselines/flink-openvino-baseline/
#
# From flink-openvino-baseline, NebulaStream root is ../../../../..
NEBULASTREAM_DIR="${NEBULASTREAM_DIR:-$(realpath "$PROJECT_DIR/../../../../..")}"

VCPKG_TARGET_TRIPLET="${VCPKG_TARGET_TRIPLET:-x64-linux-none-local}"
VCPKG_INSTALLED_DIR="${VCPKG_INSTALLED_DIR:-$NEBULASTREAM_DIR/cmake-build-release/vcpkg_installed}"
VCPKG_PREFIX="$VCPKG_INSTALLED_DIR/$VCPKG_TARGET_TRIPLET"

NATIVE_DIR="$PROJECT_DIR/target/native-libs"
NATIVE_BUILD_DIR="$PROJECT_DIR/target/native-build"

MODELS_BASE_DIR="${MODELS_BASE_DIR:-$NEBULASTREAM_DIR/nes-systests/testdata/model}"

MODEL_REL_PATH="${1:-nanodet.xml}"
MODEL_PATH="$MODELS_BASE_DIR/$MODEL_REL_PATH"

DEVICE="${2:-CPU}"
COUNT="${3:-1000}"
FLINK_PARALLELISM="${4:-1}"
OPENVINO_STREAMS="${5:-1}"
OPENVINO_THREADS="${6:-1}"

if [[ ! -f "$NATIVE_DIR/libflink_openvino_bridge_jni.so" ]]; then
  echo "Missing native JNI library. Run:"
  echo "  ./scripts/build-with-vcpkg.sh"
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing model:"
  echo "  $MODEL_PATH"
  echo
  echo "Base model directory:"
  echo "  $MODELS_BASE_DIR"
  echo
  echo "Pass the model path relative to the base directory, for example:"
  echo "  ./scripts/run-local.sh nanodet/nanodet.xml CPU 1000 1 1 1"
  echo
  echo "Or override MODELS_BASE_DIR:"
  echo "  MODELS_BASE_DIR=/path/to/models ./scripts/run-local.sh nanodet.xml CPU 1000 1 1 1"
  exit 1
fi

mvn -q dependency:build-classpath -Dmdep.outputFile=cp.txt

export LD_LIBRARY_PATH="$NATIVE_DIR:$NATIVE_BUILD_DIR:$VCPKG_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "Using:"
echo "  NEBULASTREAM_DIR=$NEBULASTREAM_DIR"
echo "  MODELS_BASE_DIR=$MODELS_BASE_DIR"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  DEVICE=$DEVICE"
echo "  COUNT=$COUNT"
echo "  FLINK_PARALLELISM=$FLINK_PARALLELISM"
echo "  OPENVINO_STREAMS=$OPENVINO_STREAMS"
echo "  OPENVINO_THREADS=$OPENVINO_THREADS"

java \
  -Djava.library.path="$NATIVE_DIR:$NATIVE_BUILD_DIR" \
  -cp "$PROJECT_DIR/target/flink-openvino-baseline-1.0-SNAPSHOT.jar:$(cat cp.txt)" \
  benchmark.openvino.OpenVinoFlinkJob \
  "$MODEL_PATH" \
  "$DEVICE" \
  "$COUNT" \
  "$FLINK_PARALLELISM" \
  "$OPENVINO_STREAMS" \
  "$OPENVINO_THREADS"
