#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Default: this repo is located at:
# nebulastream/baselines/flink-openvino-baseline
NEBULASTREAM_DIR="${NEBULASTREAM_DIR:-$(realpath "$PROJECT_DIR/../../../../..")}"
echo "$NEBULASTREAM_DIR"

VCPKG_ROOT="${VCPKG_ROOT:-$NEBULASTREAM_DIR/vcpkg-repository}"
VCPKG_TARGET_TRIPLET="${VCPKG_TARGET_TRIPLET:-x64-linux-none-local}"
VCPKG_INSTALLED_DIR="${VCPKG_INSTALLED_DIR:-$NEBULASTREAM_DIR/cmake-build-release/vcpkg_installed}"

VCPKG_PREFIX="$VCPKG_INSTALLED_DIR/$VCPKG_TARGET_TRIPLET"

if [[ ! -f "$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" ]]; then
  echo "Missing vcpkg toolchain:"
  echo "  $VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
  exit 1
fi

if [[ ! -f "$VCPKG_PREFIX/share/openvino/OpenVINOConfig.cmake" ]]; then
  echo "Missing OpenVINO package config:"
  echo "  $VCPKG_PREFIX/share/openvino/OpenVINOConfig.cmake"
  echo
  echo "Make sure NebulaStream has already been configured/built with the same triplet."
  exit 1
fi

echo "Using:"
echo "  VCPKG_ROOT=$VCPKG_ROOT"
echo "  VCPKG_TARGET_TRIPLET=$VCPKG_TARGET_TRIPLET"
echo "  VCPKG_INSTALLED_DIR=$VCPKG_INSTALLED_DIR"
echo "  VCPKG_PREFIX=$VCPKG_PREFIX"

mvn clean package \
  -Dvcpkg.root="$VCPKG_ROOT" \
  -Dvcpkg.triplet="$VCPKG_TARGET_TRIPLET" \
  -Dvcpkg.installed.dir="$VCPKG_INSTALLED_DIR"
