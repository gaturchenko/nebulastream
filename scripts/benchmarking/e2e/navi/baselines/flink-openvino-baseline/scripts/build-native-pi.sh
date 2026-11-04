#!/usr/bin/env bash
# Build the OpenVINO JNI bridge (libflink_openvino_bridge_jni.so) on the Raspberry Pi (aarch64).
#
# Unlike build-with-vcpkg.sh (x86, links a vcpkg OpenVINO tree), the Pi has no vcpkg OpenVINO
# and no JDK/cmake. It DOES have:
#   * g++            (Debian arm64)
#   * the OpenVINO aarch64 runtime + C++ headers + libs shipped inside the pip `openvino` wheel
#     (libopenvino.so.2530, libopenvino_arm_cpu_plugin.so, include/openvino/openvino.hpp).
# So we skip cmake entirely and compile the three tiny bridge sources directly with g++,
# linking against the wheel's libopenvino by its exact soname and baking an rpath so the .so
# resolves it at runtime. JNI headers are copied in from a JDK (jni/ dir) because the Pi has
# only a JRE. This builds the SAME 3 sources cmake would, into one combined shared object.
#
# Cost: ~25 s, no OpenVINO build. Re-run whenever native/*.cpp changes.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# OpenVINO C++ SDK bundled in the pip wheel. Override OV_DIR if the venv path differs.
OV_DIR="${OV_DIR:-/home/gaturchenko/navi/models-benchmarking/venv/lib/python3.13/site-packages/openvino}"
OV_LIB_SONAME="${OV_LIB_SONAME:-libopenvino.so.2530}"

NATIVE_SRC="$PROJECT_DIR/native"
JNI_INC="$PROJECT_DIR/jni"
OUT_DIR="$PROJECT_DIR/target/native-libs"
OUT_SO="$OUT_DIR/libflink_openvino_bridge_jni.so"

for f in openvino.hpp; do
  [[ -f "$OV_DIR/include/openvino/$f" ]] || { echo "Missing OpenVINO header: $OV_DIR/include/openvino/$f"; exit 1; }
done
[[ -f "$OV_DIR/libs/$OV_LIB_SONAME" ]] || { echo "Missing OpenVINO lib: $OV_DIR/libs/$OV_LIB_SONAME"; exit 1; }
[[ -f "$JNI_INC/jni.h" ]] || { echo "Missing jni.h under $JNI_INC (copy from a JDK include/)"; exit 1; }

mkdir -p "$OUT_DIR"
echo "Building $OUT_SO against $OV_DIR/libs/$OV_LIB_SONAME"

g++ -std=c++20 -O2 -fPIC -shared \
  -I"$NATIVE_SRC" \
  -I"$OV_DIR/include" \
  -I"$JNI_INC" -I"$JNI_INC/linux" \
  "$NATIVE_SRC/OpenVinoEngine.cpp" \
  "$NATIVE_SRC/OpenVinoC.cpp" \
  "$NATIVE_SRC/OpenVinoJNI.cpp" \
  -L"$OV_DIR/libs" -l:"$OV_LIB_SONAME" \
  -Wl,-rpath,"$OV_DIR/libs" \
  -o "$OUT_SO"

echo "Built: $OUT_SO"
nm -D "$OUT_SO" | grep -c Java_benchmark_openvino_NativeOpenVinoBridge | xargs echo "Exported JNI symbols:"
