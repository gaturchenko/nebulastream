#!/usr/bin/env bash
# Prepares a GCP Batch job that builds the Raspberry Pi binaries, and prints the command that submits it.
#
# It packages the current working tree (including uncommitted changes), uploads it next to the build script in Cloud Storage and
# renders the job definition. It does not submit anything unless --submit is passed, so that no credits are spent by accident.
#
# Usage:
#   scripts/arm-build/submit_arm_build.sh --bucket gs://my-bucket
#   scripts/arm-build/submit_arm_build.sh --bucket gs://my-bucket --submit
#   scripts/arm-build/submit_arm_build.sh --bucket gs://my-bucket --log-level INFO --submit   # keep optimizer/plan INFO logs
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/scripts/arm-build"

BUCKET=""
JOB_NAME=""
LOCATION="europe-west3"
PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
NETWORK=""
SUBNETWORK=""
MACHINE_TYPE="n4a-standard-16"
PROVISIONING_MODEL="SPOT"
DISK_GB="250"
CPU_MILLI="16000"
MEMORY_MIB="65536"
MAX_RUN_DURATION="21600s"
TARGETS="systest nes-single-node-worker"
BUILD_TYPE="Release"
# Compile-time NES_LOG_LEVEL; a Release build compiles everything below ERROR out, including the optimizer's decisions.
LOG_LEVEL="ERROR"
OUT_DIR="${TMPDIR:-/tmp}"
SUBMIT="false"

usage() { sed -n '2,12p' "${BASH_SOURCE[0]}"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bucket) BUCKET="${2%/}"; shift 2 ;;
        --job-name) JOB_NAME="$2"; shift 2 ;;
        --location) LOCATION="$2"; shift 2 ;;
        --project) PROJECT="$2"; shift 2 ;;
        --network) NETWORK="$2"; shift 2 ;;
        --subnetwork) SUBNETWORK="$2"; shift 2 ;;
        --machine-type) MACHINE_TYPE="$2"; shift 2 ;;
        --provisioning-model) PROVISIONING_MODEL="$2"; shift 2 ;;
        --disk-gb) DISK_GB="$2"; shift 2 ;;
        --max-run-duration) MAX_RUN_DURATION="$2"; shift 2 ;;
        --targets) TARGETS="$2"; shift 2 ;;
        --build-type) BUILD_TYPE="$2"; shift 2 ;;
        --log-level) LOG_LEVEL="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --submit) SUBMIT="true"; shift ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown argument: $1" >&2; usage 1 ;;
    esac
done

[[ -n "${BUCKET}" ]] || { echo "--bucket is required, e.g. --bucket gs://nes-arm-builds" >&2; exit 1; }
[[ "${BUCKET}" == gs://* ]] || { echo "--bucket must start with gs://" >&2; exit 1; }
[[ -n "${PROJECT}" ]] || { echo "No project configured; pass --project" >&2; exit 1; }

: "${NETWORK:=projects/${PROJECT}/global/networks/arm-build-vpc}"
: "${SUBNETWORK:=projects/${PROJECT}/regions/${LOCATION}/subnetworks/arm-build-subnet}"

STAMP="$(date -u +%Y%m%d-%H%M%S)"
REVISION="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
DIRTY=""
if ! git -C "${REPO_ROOT}" diff --quiet 2>/dev/null || [[ -n "$(git -C "${REPO_ROOT}" ls-files --others --exclude-standard)" ]]; then
    DIRTY="-dirty"
fi
: "${JOB_NAME:=nes-arm-build-${STAMP}}"

PREFIX="${BUCKET}/arm-build/${STAMP}"
SOURCE_URI="${PREFIX}/source.tar.zst"
BUILD_SCRIPT_URI="${PREFIX}/build_nes_arm.sh"
ARTIFACT_URI="${PREFIX}"
JOB_JSON_URI="${PREFIX}/job.json"
VCPKG_CACHE_URI="${BUCKET}/vcpkg-cache"

###############################################################################
# Package the working tree: tracked files plus anything untracked that is not ignored, so that uncommitted work is built too.
###############################################################################
TARBALL="${OUT_DIR}/nes-source-${STAMP}.tar.zst"
echo "Packaging ${BRANCH}@${REVISION}${DIRTY} -> ${TARBALL}"
echo "${BRANCH}@${REVISION}${DIRTY}" > "${REPO_ROOT}/GIT_REVISION"
trap 'rm -f "${REPO_ROOT}/GIT_REVISION"' EXIT
# GIT_REVISION is written into the tree first so that it is picked up as an untracked file and travels with the sources.
(cd "${REPO_ROOT}" && git ls-files -co --exclude-standard -z | tar --null --zstd -cf "${TARBALL}" -T -)
echo "Packaged $(du -h "${TARBALL}" | cut -f1)"

###############################################################################
# Render the job definition.
###############################################################################
# The build script is fetched through the Cloud Storage JSON API, whose object path has to be URL-encoded.
BUCKET_NAME="${BUCKET#gs://}"
SCRIPT_OBJECT="arm-build/${STAMP}/build_nes_arm.sh"
SCRIPT_OBJECT_ENCODED="${SCRIPT_OBJECT//\//%2F}"
BUILD_SCRIPT_URL="https://storage.googleapis.com/storage/v1/b/${BUCKET_NAME}/o/${SCRIPT_OBJECT_ENCODED}?alt=media"

JOB_JSON="${OUT_DIR}/${JOB_NAME}.json"
sed \
    -e "s|@BUILD_SCRIPT_URL@|${BUILD_SCRIPT_URL}|g" \
    -e "s|@SOURCE_URI@|${SOURCE_URI}|g" \
    -e "s|@ARTIFACT_URI@|${ARTIFACT_URI}|g" \
    -e "s|@VCPKG_CACHE_URI@|${VCPKG_CACHE_URI}|g" \
    -e "s|@TARGETS@|${TARGETS}|g" \
    -e "s|@BUILD_TYPE@|${BUILD_TYPE}|g" \
    -e "s|@LOG_LEVEL@|${LOG_LEVEL}|g" \
    -e "s|@JOBS@|$((CPU_MILLI / 1000))|g" \
    -e "s|@CPU_MILLI@|${CPU_MILLI}|g" \
    -e "s|@MEMORY_MIB@|${MEMORY_MIB}|g" \
    -e "s|@MAX_RUN_DURATION@|${MAX_RUN_DURATION}|g" \
    -e "s|@MACHINE_TYPE@|${MACHINE_TYPE}|g" \
    -e "s|@PROVISIONING_MODEL@|${PROVISIONING_MODEL}|g" \
    -e "s|@DISK_GB@|${DISK_GB}|g" \
    "${SCRIPT_DIR}/batch-job.json.tmpl" > "${JOB_JSON}"
python3 -c "import json,sys; json.load(open(sys.argv[1]))" "${JOB_JSON}"
echo "Rendered ${JOB_JSON}"

###############################################################################
# Upload the inputs, then either submit or print the command.
###############################################################################
UPLOAD_CMDS=(
    "gcloud storage cp '${TARBALL}' '${SOURCE_URI}'"
    "gcloud storage cp '${SCRIPT_DIR}/build_nes_arm.sh' '${BUILD_SCRIPT_URI}'"
    "gcloud storage cp '${JOB_JSON}' '${JOB_JSON_URI}'"
)
SUBMIT_CMD="gcloud batch jobs submit ${JOB_NAME} \\
    --location=${LOCATION} \\
    --config=${JOB_JSON} \\
    --network=${NETWORK} \\
    --subnetwork=${SUBNETWORK} \\
    --no-external-ip-address"

if [[ "${SUBMIT}" == "true" ]]; then
    for cmd in "${UPLOAD_CMDS[@]}"; do
        echo "+ ${cmd}"
        eval "${cmd}"
    done
    echo "+ ${SUBMIT_CMD}"
    eval "${SUBMIT_CMD}"
else
    cat <<INSTRUCTIONS

Nothing was uploaded or submitted. Run these when you are ready:

$(printf '%s\n' "${UPLOAD_CMDS[@]}")

${SUBMIT_CMD}

The job definition is read from the local filesystem, so to submit from Cloud Shell or another machine, fetch it there first:
  gcloud storage cp ${JOB_JSON_URI} job.json
  gcloud batch jobs submit ${JOB_NAME} --location=${LOCATION} --config=job.json \\
      --network=${NETWORK} --subnetwork=${SUBNETWORK} --no-external-ip-address

Follow it with:
  gcloud batch jobs describe ${JOB_NAME} --location=${LOCATION}
  gcloud beta batch jobs tail ${JOB_NAME} --location=${LOCATION}

Collect the result with:
  gcloud storage ls ${ARTIFACT_URI}/
  gcloud storage cp '${ARTIFACT_URI}/nes-arm64-*.tar.zst' .
INSTRUCTIONS
fi
