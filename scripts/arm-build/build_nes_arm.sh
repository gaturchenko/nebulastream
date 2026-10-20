#!/usr/bin/env bash
# Builds the NebulaStream release binaries for the Raspberry Pi on an arm64 machine.
#
# Runs inside an ubuntu:24.04 container on a GCP Batch VM, but works on any aarch64 Linux machine that has internet access:
#   NES_SOURCE_URI=gs://.../source.tar.zst NES_ARTIFACT_URI=gs://.../ ./build_nes_arm.sh
#
# Why ubuntu:24.04: the Pi runs Raspberry Pi OS Trixie (glibc 2.41, libstdc++ from GCC 14). Ubuntu 24.04 has glibc 2.39 and the
# GCC 14 standard library, so its binaries run on the Pi (a program may use an older glibc than the host, never a newer one),
# while still providing the C++23 library the code needs. It is also the base of the project's own development image, so the
# toolchain matches what the x86 builds use.
#
# The Pi is an ARMv8.0-A machine while the builder is an Axion (ARMv8.2+). Both NebulaStream (cmake/HostIdentification.cmake plus
# the arm64 branch in CMakeLists.txt) and the vcpkg dependencies (vcpkg/custom-triplets/bits/arch/arm64.cmake) pin -march=armv8-a,
# so nothing emits instructions the Pi lacks. Do not set NES_BUILD_NATIVE=ON here.
set -euo pipefail

: "${NES_SOURCE_URI:?Set NES_SOURCE_URI to the gs:// URI of the source tarball}"
: "${NES_ARTIFACT_URI:?Set NES_ARTIFACT_URI to the gs:// prefix the build artifacts are uploaded to}"
NES_VCPKG_CACHE_URI="${NES_VCPKG_CACHE_URI:-}"
NES_LLVM_VERSION="${NES_LLVM_VERSION:-21-with-fix-173075}"
NES_TRIPLET="${NES_TRIPLET:-arm64-linux-none-local}"
NES_BUILD_TYPE="${NES_BUILD_TYPE:-Release}"
NES_LOG_LEVEL="${NES_LOG_LEVEL:-ERROR}"
NES_TARGETS="${NES_TARGETS:-systest nes-single-node-worker}"
NES_LLVM_TOOLCHAIN_VERSION="${NES_LLVM_TOOLCHAIN_VERSION:-19}"
# The stdlib token of the prebuilt LLVM asset. It is the name of the standard library itself, which is not the "local" that
# the matching vcpkg triplet uses for the same choice (see the STDLIB handling in docker/dependency/Dependency.dockerfile).
NES_LLVM_STDLIB="${NES_LLVM_STDLIB:-libstdcxx}"
NES_CMAKE_VERSION="${NES_CMAKE_VERSION:-3.31.11-0kitware1ubuntu24.04.1}"
NES_MOLD_VERSION="${NES_MOLD_VERSION:-2.37.1}"
WORK_DIR="${WORK_DIR:-/work}"
JOBS="${JOBS:-$(nproc)}"

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "This script builds the Raspberry Pi binaries and must run on an aarch64 machine, but this one is $(uname -m)" >&2
    exit 1
fi

SRC_DIR="${WORK_DIR}/src"
BUILD_DIR="${WORK_DIR}/build"
CACHE_DIR="${WORK_DIR}/vcpkg-cache"
CLANG_DIR="${WORK_DIR}/clang"
mkdir -p "${WORK_DIR}" "${CACHE_DIR}"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# Fingerprint of this script, so a log makes it obvious which version actually ran after a fix was uploaded.
log "build_nes_arm.sh sha256 $(sha256sum "${BASH_SOURCE[0]}" | cut -c1-12)"

# The VM and its disk are gone the moment the task fails, taking every log vcpkg and CMake wrote with them. Print the reports that
# name the actual error into the job log, and upload the rest next to the artifacts, so a failure can be diagnosed without
# re-running the build blind.
report_failure() {
    local status=$?
    if [[ ${status} -eq 0 ]]; then
        return 0
    fi
    log "Failed with status ${status}; collecting diagnostics"

    local issue_body="${BUILD_DIR}/vcpkg_installed/vcpkg/issue_body.md"
    if [[ -f "${issue_body}" ]]; then
        echo "================ vcpkg issue_body.md (the failing port's own logs) ================"
        head -c 200000 "${issue_body}"
        echo "================ end of issue_body.md ================"
    fi

    local collected="${WORK_DIR}/diagnostics"
    mkdir -p "${collected}"
    cp "${issue_body}" "${collected}/" 2>/dev/null || true
    cp "${BUILD_DIR}/vcpkg-manifest-install.log" "${collected}/" 2>/dev/null || true
    cp "${BUILD_DIR}/CMakeFiles/CMakeConfigureLog.yaml" "${collected}/" 2>/dev/null || true
    find "${SRC_DIR}/vcpkg-repository/buildtrees" -maxdepth 2 -name "*.log" -print0 2>/dev/null \
        | tar --null -T - -czf "${collected}/buildtree-logs.tar.gz" 2>/dev/null || true

    if [[ -n "${NES_ARTIFACT_URI}" ]]; then
        tar -czf "${WORK_DIR}/diagnostics.tar.gz" -C "${collected}" . 2>/dev/null || true
        gcloud storage cp "${WORK_DIR}/diagnostics.tar.gz" \
            "${NES_ARTIFACT_URI%/}/diagnostics-$(date -u +%Y%m%d-%H%M%S).tar.gz" || true
    fi
}
trap report_failure EXIT

###############################################################################
log "Installing the build toolchain"
###############################################################################
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
    ca-certificates curl wget gpg git zip unzip tar zstd \
    build-essential g++-14 make pkg-config ninja-build ccache \
    libssl-dev openssl linux-libc-dev \
    python3 python3-venv python3-pip \
    openjdk-21-jre-headless

# The LLVM toolchain the project builds with, from the same source its development image uses.
curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /etc/apt/keyrings/llvm-snapshot.gpg
chmod a+r /etc/apt/keyrings/llvm-snapshot.gpg
CODENAME="$(. /etc/os-release && echo "${VERSION_CODENAME}")"
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/llvm-snapshot.gpg] http://apt.llvm.org/${CODENAME}/ llvm-toolchain-${CODENAME}-${NES_LLVM_TOOLCHAIN_VERSION} main" \
    > /etc/apt/sources.list.d/llvm-snapshot.list
apt-get update -y
# clang-tools provides clang-scan-deps. The toolchain file compiles everything as C++23, which makes CMake scan for modules, and
# a missing scanner turns every try_compile into a failure - FindThreads then reports no threads at all.
apt-get install -y --no-install-recommends \
    "clang-${NES_LLVM_TOOLCHAIN_VERSION}" "libclang-rt-${NES_LLVM_TOOLCHAIN_VERSION}-dev" \
    "clang-tools-${NES_LLVM_TOOLCHAIN_VERSION}"

# vcpkg runs the system cmake and ninja on arm64 (VCPKG_FORCE_SYSTEM_BINARIES below), so the cmake has to be as new as the one the
# project's images pin rather than the one Ubuntu ships.
curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor -o /etc/apt/keyrings/kitware-archive-keyring.gpg
echo "deb [signed-by=/etc/apt/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ ${CODENAME} main" \
    > /etc/apt/sources.list.d/kitware.list
apt-get update -y
apt-get install -y --no-install-recommends "cmake=${NES_CMAKE_VERSION}" "cmake-data=${NES_CMAKE_VERSION}"

# mold links the multi-hundred-megabyte binaries far faster than ld; cmake/UseMold.cmake picks it up when it is on the PATH.
wget -q "https://github.com/rui314/mold/releases/download/v${NES_MOLD_VERSION}/mold-${NES_MOLD_VERSION}-$(uname -m)-linux.tar.gz"
tar -xf "mold-${NES_MOLD_VERSION}-$(uname -m)-linux.tar.gz"
cp -r "mold-${NES_MOLD_VERSION}-$(uname -m)-linux/"* /usr
rm -rf "mold-${NES_MOLD_VERSION}-$(uname -m)-linux"*

# The Rust bindings of nes-network are built through Corrosion, which needs cargo on the PATH.
curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
export PATH="${HOME}/.cargo/bin:${PATH}"

# vcpkg/custom-triplets/toolchains/local.cmake locates the compiler as clang++-$ENV{LLVM_TOOLCHAIN_VERSION}, which the project's
# development image sets. Nothing else sets it here.
export LLVM_TOOLCHAIN_VERSION="${NES_LLVM_TOOLCHAIN_VERSION}"

# Only needed to move the source, the dependency cache and the artifacts in and out of Cloud Storage.
if ! command -v gcloud >/dev/null; then
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        > /etc/apt/sources.list.d/google-cloud-sdk.list
    apt-get update -y
    apt-get install -y --no-install-recommends google-cloud-cli
fi

###############################################################################
log "Fetching the sources from ${NES_SOURCE_URI}"
###############################################################################
mkdir -p "${SRC_DIR}"
gcloud storage cp "${NES_SOURCE_URI}" "${WORK_DIR}/source.tar.zst"
tar --zstd -xf "${WORK_DIR}/source.tar.zst" -C "${SRC_DIR}"
rm -f "${WORK_DIR}/source.tar.zst"

# The arm64 chainload toolchain has to name CMAKE_SYSTEM_PROCESSOR, because setting CMAKE_SYSTEM_NAME next to it turns off
# CMake's own detection. Ports that branch on an empty processor take their x86 path: nautilus then misses its ARM guard and
# requires an LLVM matching the clang version, which the prebuilt LLVM 21 is not. Fail here rather than 15 minutes into vcpkg.
if ! grep -q "CMAKE_SYSTEM_PROCESSOR aarch64" "${SRC_DIR}/vcpkg/custom-triplets/toolchains/local.cmake"; then
    echo "The uploaded source predates the arm64 CMAKE_SYSTEM_PROCESSOR fix in vcpkg/custom-triplets/toolchains/." >&2
    echo "Re-run submit_arm_build.sh to package the current working tree." >&2
    exit 9
fi

###############################################################################
log "Fetching the prebuilt LLVM/MLIR toolchain for arm64"
###############################################################################
# Nautilus' MLIR backend compiles against this prebuilt LLVM rather than a vcpkg port, exactly as docker/dependency/Dependency.dockerfile does.
LLVM_TARBALL="nes-llvm-${NES_LLVM_VERSION}-arm64-none-${NES_LLVM_STDLIB}.tar.zstd"
if [[ -n "${NES_VCPKG_CACHE_URI}" ]] && gcloud storage cp "${NES_VCPKG_CACHE_URI}/${LLVM_TARBALL}" "${WORK_DIR}/llvm.tar.zstd" 2>/dev/null; then
    log "Reusing the cached LLVM toolchain"
else
    LLVM_URL="https://github.com/nebulastream/clang-binaries/releases/download/vmlir-${NES_LLVM_VERSION}/${LLVM_TARBALL}"
    if ! wget -q -O "${WORK_DIR}/llvm.tar.zstd" "${LLVM_URL}"; then
        echo "Could not download ${LLVM_URL}." >&2
        echo "The assets published under that tag are:" >&2
        wget -q -O - "https://api.github.com/repos/nebulastream/clang-binaries/releases/tags/vmlir-${NES_LLVM_VERSION}" \
            | grep -o '"name": "nes-llvm[^"]*"' >&2 || true
        echo "Set NES_LLVM_VERSION or NES_LLVM_STDLIB in this script to one of them." >&2
        exit 1
    fi
    if [[ -n "${NES_VCPKG_CACHE_URI}" ]]; then
        gcloud storage cp "${WORK_DIR}/llvm.tar.zstd" "${NES_VCPKG_CACHE_URI}/${LLVM_TARBALL}" || true
    fi
fi
mkdir -p "${CLANG_DIR}"
zstd --decompress "${WORK_DIR}/llvm.tar.zstd" --stdout | tar -x -C "${WORK_DIR}"
rm -f "${WORK_DIR}/llvm.tar.zstd"
if [[ ! -d "${CLANG_DIR}/lib/cmake/mlir" ]]; then
    echo "The LLVM archive did not unpack to ${CLANG_DIR} as expected; nes-nautilus needs MLIR from there" >&2
    exit 1
fi
export CMAKE_PREFIX_PATH="${CLANG_DIR}:${CMAKE_PREFIX_PATH:-}"

###############################################################################
log "Preparing vcpkg"
###############################################################################
# The dependencies are built from the manifest in the source tree, which carries this fork's own ports (OpenVINO, IREE, libpng,
# nautilus), so the upstream prebuilt dependency image cannot be used.
# vcpkg has no prebuilt tools for arm64 and uses the system cmake, ninja and curl instead. Exported before bootstrapping so that
# building the vcpkg tool itself honours it too.
export VCPKG_FORCE_SYSTEM_BINARIES=1

if [[ ! -d "${SRC_DIR}/vcpkg-repository" ]]; then
    # A full clone, not a shallow one: vcpkg resolves the manifest's builtin-baseline and every port tree through git history, and
    # refuses to work in a shallow repository.
    git clone https://github.com/microsoft/vcpkg.git "${SRC_DIR}/vcpkg-repository"
    "${SRC_DIR}/vcpkg-repository/bootstrap-vcpkg.sh" --disableMetrics
fi
# A binary cache makes a second run cheap, which matters because the builder is a Spot VM that can be reclaimed mid-build.
if [[ -n "${NES_VCPKG_CACHE_URI}" ]]; then
    log "Restoring the vcpkg binary cache from ${NES_VCPKG_CACHE_URI}/binaries"
    gcloud storage rsync -r "${NES_VCPKG_CACHE_URI}/binaries" "${CACHE_DIR}" 2>/dev/null || true
fi
export VCPKG_BINARY_SOURCES="clear;files,${CACHE_DIR},readwrite"

###############################################################################
log "Configuring"
###############################################################################
# The dependencies are built during configure, which is the long pole of the whole job.
#
# USE_LOCAL_MLIR=ON makes the build consume the prebuilt LLVM/MLIR fetched above. Left off, the project enables the manifest's
# mlir feature instead, which pulls in the llvm port and compiles LLVM 21.1.8 from source: hours of work, and a different compiler
# than the patched build the x86 side uses.
#
# cmake/ImportDependencies.cmake derives the vcpkg triplet itself and ignores a VCPKG_TARGET_TRIPLET passed on the command line:
# it appends "libcxx" or, with USE_LIBCXX_IF_AVAILABLE=OFF, "local". The Raspberry Pi has libstdc++ but no libc++ runtime, and the
# prebuilt LLVM downloaded above is the libstdc++ build, so the standard library has to be the system one on both sides.
cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE="${NES_BUILD_TYPE}" \
    -DNES_LOG_LEVEL="${NES_LOG_LEVEL}" \
    -DCMAKE_C_COMPILER="clang-${NES_LLVM_TOOLCHAIN_VERSION}" \
    -DCMAKE_CXX_COMPILER="clang++-${NES_LLVM_TOOLCHAIN_VERSION}" \
    -DCMAKE_TOOLCHAIN_FILE="${SRC_DIR}/vcpkg-repository/scripts/buildsystems/vcpkg.cmake" \
    -DVCPKG_OVERLAY_TRIPLETS="${SRC_DIR}/vcpkg/custom-triplets" \
    -DVCPKG_OVERLAY_PORTS="${SRC_DIR}/vcpkg/vcpkg-registry/ports" \
    -DUSE_LIBCXX_IF_AVAILABLE=OFF \
    -DUSE_LOCAL_MLIR=ON \
    -DNES_ENABLES_TESTS=ON \
    -DNES_ENABLE_PRECOMPILED_HEADERS=OFF \
    -DNES_BUILD_NATIVE=OFF

# The triplet is only settled inside configure, so confirm the dependencies landed where the libstdc++ build expects them before
# spending the next hour compiling against them.
if [[ ! -d "${BUILD_DIR}/vcpkg_installed/${NES_TRIPLET}" ]]; then
    echo "Expected the dependencies under vcpkg_installed/${NES_TRIPLET}, but found:" >&2
    ls "${BUILD_DIR}/vcpkg_installed" >&2 || true
    exit 1
fi

# Push the dependency cache before the long compile so that a reclaimed Spot VM does not throw the dependencies away with it.
if [[ -n "${NES_VCPKG_CACHE_URI}" ]]; then
    log "Saving the vcpkg binary cache"
    gcloud storage rsync -r "${CACHE_DIR}" "${NES_VCPKG_CACHE_URI}/binaries" || true
fi

###############################################################################
log "Building ${NES_TARGETS}"
###############################################################################
# shellcheck disable=SC2086 # NES_TARGETS is a list of target names on purpose
cmake --build "${BUILD_DIR}" -j "${JOBS}" --target ${NES_TARGETS}

###############################################################################
log "Packaging the artifacts"
###############################################################################
STAGE_DIR="${WORK_DIR}/stage"
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}/bin"

for target in ${NES_TARGETS}; do
    binary="$(find "${BUILD_DIR}" -type f -executable -name "${target}" -print -quit)"
    if [[ -z "${binary}" ]]; then
        echo "Target ${target} produced no binary" >&2
        exit 1
    fi
    cp "${binary}" "${STAGE_DIR}/bin/"
done

# Everything except the Rust proc-macro helpers links statically into the binaries, but ship any plugin that did come out shared.
while IFS= read -r library; do
    cp "${library}" "${STAGE_DIR}/bin/"
done < <(find "${BUILD_DIR}" -name "*.so" -not -path "*/cargo/*" -not -path "*/_deps/*")

{
    echo "built_at:        $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "builder:         $(uname -srm), $(nproc) cpus"
    echo "build_type:      ${NES_BUILD_TYPE}"
    echo "triplet:         ${NES_TRIPLET}"
    echo "targets:         ${NES_TARGETS}"
    if [[ -f "${SRC_DIR}/GIT_REVISION" ]]; then
        echo "source_revision: $(cat "${SRC_DIR}/GIT_REVISION")"
    fi
    echo "clang:           $(clang-${NES_LLVM_TOOLCHAIN_VERSION} --version | head -1)"
    echo
    echo "# Highest glibc and libstdc++ symbol versions required; the Pi must offer at least these."
    for binary in "${STAGE_DIR}"/bin/*; do
        [[ -x "${binary}" ]] || continue
        echo "$(basename "${binary}"): \
glibc $(objdump -T "${binary}" 2>/dev/null | grep -o 'GLIBC_[0-9.]*' | sort -V | tail -1 || echo n/a), \
libstdc++ $(objdump -T "${binary}" 2>/dev/null | grep -o 'GLIBCXX_[0-9.]*' | sort -V | tail -1 || echo n/a)"
    done
    echo
    echo "# Shared libraries the Pi has to provide"
    ldd "${STAGE_DIR}/bin/systest" 2>/dev/null || true
} > "${STAGE_DIR}/BUILD_INFO.txt"

cat "${STAGE_DIR}/BUILD_INFO.txt"

ARTIFACT_NAME="nes-arm64-$(date -u +%Y%m%d-%H%M%S).tar.zst"
tar --zstd -cf "${WORK_DIR}/${ARTIFACT_NAME}" -C "${STAGE_DIR}" .
gcloud storage cp "${WORK_DIR}/${ARTIFACT_NAME}" "${NES_ARTIFACT_URI%/}/${ARTIFACT_NAME}"
gcloud storage cp "${STAGE_DIR}/BUILD_INFO.txt" "${NES_ARTIFACT_URI%/}/BUILD_INFO.txt"

if [[ -n "${NES_VCPKG_CACHE_URI}" ]]; then
    gcloud storage rsync -r "${CACHE_DIR}" "${NES_VCPKG_CACHE_URI}/binaries" || true
fi

log "Done: ${NES_ARTIFACT_URI%/}/${ARTIFACT_NAME}"
