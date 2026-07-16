#!/usr/bin/env bash
# Single entry point for the framework comparison on the Pi.
# Runs three serving/streaming systems on the same workload:
#   1. Flink + OpenVINO   (baseline; substitutes the former NES-UDF case)
#   2. NES MODEL_INFERENCE
#   3. TorchServe
# Each writes rows into results/ in a shared comparison schema; the per-run trace
# subdirectories are cleaned at the end, the processed CSVs are kept.
#
# Run from the navi root (relative paths below assume CWD=~/navi).

# ============================================================================
# Experiment configuration — the SINGLE place to change the model / input size.
# When swapping a model or a variant with a different input size, edit
# ONLY the values below (and, in the .test file, the RESIZE_IMAGE(W, H) calls and
# the CREATE MODEL path, which are NES-side and can't be sourced from here — keep
# them in sync with IMAGE_WIDTH/IMAGE_HEIGHT and MODEL_ONNX).
# Everything else — the TorchServe handler geometry, the Flink preprocessing size,
# and the OpenVINO IR conversion — is derived from these.
# ============================================================================
MODEL_ONNX="${MODEL_ONNX:-/home/gaturchenko/navi/models-benchmarking/models/squeezedet/squeezedet-192x624.onnx}"
IMAGE_WIDTH="${IMAGE_WIDTH:-624}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-192}"
INGEST_RATE="${INGEST_RATE:-10}"                       # must equal MOCK_TUPLE_RATE in the .test source
TEST_FILE="${TEST_FILE:-waymo/systems.test}"           # NES query test file, relative to queries/
DATA_DIR="${DATA_DIR:-./datasets/waymo}"               # NES systest --data (dataset root)
INFERENCE_CONFIG_DIR="${INFERENCE_CONFIG_DIR:-queries/osu-rgb}"  # generic st.yaml / udf.yaml sweeps

# TorchServe endpoint name + handler. TS_MODEL_NAME must match the .test Http sink `model`
# (e.g. 'squeezedet' in waymo/systems.test); the handler must match the model's input layout
# (squeezedet_handler.py = NHWC, nanodet_handler.py = NCHW).
TS_MODEL_NAME="${TS_MODEL_NAME:-squeezedet}"
TS_HANDLER="${TS_HANDLER:-scripts/torchserve/squeezedet_handler.py}"

# Flink baseline runner + fairness knobs. FLINK_PARALLELISM matches the NES
# worker.query_engine.number_of_worker_threads in the .test (2); FLINK_STREAMS=0 /
# FLINK_THREADS=1 mirror NES's OpenVINO compile options num_streams(0),
# inference_num_threads(1) — 0 is passed as an EXPLICIT property value (needs the
# rebuilt JNI lib, see baselines/.../scripts/build-native-pi.sh).
FLINK_RUN="${FLINK_RUN:-baselines/flink-openvino-baseline/scripts/run-pi-waymo-squeezedet.sh}"
FLINK_PARALLELISM="${FLINK_PARALLELISM:-2}"
FLINK_STREAMS="${FLINK_STREAMS:-0}"
FLINK_THREADS="${FLINK_THREADS:-1}"

# The Flink baseline reads frames straight from a CSV — it does not parse the .test, so give it the
# same file the .test ATTACHes and tell it which column holds the base64 image. Use an ABSOLUTE path
# (the Flink run script cd's into its own dir). BASE64_COL is 0-indexed:
#   OSU-RGB  = single column of base64            -> 0
#   WAYMO    = "timestamp,base64jpeg" (2 columns) -> 1
DATASET_CSV="${DATASET_CSV:-/home/gaturchenko/navi/datasets/waymo/camera_5.csv}"
BASE64_COL="${BASE64_COL:-1}"

# Derived paths / names — no need to edit.
MODEL_XML="${MODEL_ONNX%.onnx}.xml"
MODEL_STORE="$(dirname "$MODEL_ONNX")"
VENV=./models-benchmarking/venv
TS_GEOMETRY=scripts/torchserve/input_geometry.json

# Activate the benchmarking venv UP FRONT. It provides `ovc` (needed to load the NES model in the
# MODEL_INFERENCE / CREATE MODEL path AND to build the Flink OpenVINO IR) and pandas (needed by the
# result post-processing). The NES phase fails to bind the model without ovc on PATH, so this must
# precede it.
source "$VENV/bin/activate"

# The TorchServe handler reads the input geometry from input_geometry.json bundled into the .mar
# (workers don't inherit the parent env), so write it here before archiving.
printf '{"width": %s, "height": %s}\n' "$IMAGE_WIDTH" "$IMAGE_HEIGHT" > "$TS_GEOMETRY"

# Regenerate the OpenVINO IR from the (possibly newly uploaded) ONNX so Flink never runs a stale
# model. Forced each run because ovc only auto-converts when the XML is missing.
echo "Converting $MODEL_ONNX -> $MODEL_XML (ovc)"
ovc "$MODEL_ONNX" --output_model "$MODEL_XML" --compress_to_fp16=False

# ---- Flink + OpenVINO baseline (replaces the former NES-UDF case) ----
# Java/Flink job (not a NES systest); manages its own OpenVINO/JNI env and writes a comparison-schema
# row directly to results/flink_openvino_stats.csv. Fresh file so rows don't accumulate. The source is
# paced at INGEST_RATE to match the NES/TorchServe TCP MOCK_TUPLE_RATE (fair offered load).
rm -f results/flink_openvino_stats.csv
QUERY_NAME="${QUERY_NAME:-flink}" PARAM_NAME=baseline PARAM_VALUE=flink REPETITION=0 \
    PARALLELISM="$FLINK_PARALLELISM" STREAMS="$FLINK_STREAMS" THREADS="$FLINK_THREADS" \
    MODEL="$MODEL_XML" WIDTH="$IMAGE_WIDTH" HEIGHT="$IMAGE_HEIGHT" RATE="$INGEST_RATE" \
    CSV="$DATASET_CSV" BASE64_COL="$BASE64_COL" \
    "$FLINK_RUN"

# ---- NES MODEL_INFERENCE ----
python3 scripts/run_systests_lat.py \
    --inference-config "$INFERENCE_CONFIG_DIR/st.yaml" \
    --queries "${TEST_FILE}:02" \
    --data "$DATA_DIR" \
    --repetitions 1

# ---- TorchServe ----
TS_CONFIG=scripts/torchserve/config.properties

# 1. Stop any running/orphaned instance FIRST, so we don't archive a loaded model or
#    double-start. `|| true` because --stop returns non-zero when nothing is running.
torchserve --stop 2>/dev/null || true
sleep 2

# 2. (Re)build the model archive while the server is down.
torch-model-archiver --model-name "$TS_MODEL_NAME" --version 1.0 \
  --serialized-file "$MODEL_ONNX" \
  --handler "$TS_HANDLER" \
  --extra-files "$TS_GEOMETRY" \
  --export-path "$MODEL_STORE" --force

# 3. Start ONCE (config.properties auto-loads every .mar in the model store — the store is the
#    model's own directory, so that's exactly this archive), then WAIT for it to load before
#    hitting the endpoints — first load on the Pi CPU takes 10-30 s.
#    The handler reads the input geometry from the input_geometry.json bundled into the .mar.
torchserve --start --ncs --ts-config "$TS_CONFIG" --model-store "$MODEL_STORE"

echo "Waiting for TorchServe to become healthy..."
healthy=0
for i in $(seq 1 60); do
  if curl -sf 127.0.0.1:8080/ping 2>/dev/null | grep -q Healthy; then
    echo "TorchServe healthy after ${i}s"
    healthy=1
    break
  fi
  sleep 1
done
if [ "$healthy" -ne 1 ]; then
  echo "ERROR: TorchServe did not become healthy in 60s. Check logs/model_log.log." >&2
  torchserve --stop 2>/dev/null || true
  deactivate
  exit 1
fi

curl -s 127.0.0.1:8080/ping
curl -s 127.0.0.1:8082/metrics | head

python3 scripts/run_systests_lat.py \
    --inference-config "$INFERENCE_CONFIG_DIR/udf.yaml" \
    --queries "${TEST_FILE}:03" \
    --data "$DATA_DIR" \
    --repetitions 1

# Process results. NES traces are the per-query JSON under results/<combo>/... — scan the results
# DIRECTORY (not the TorchServe timings file, which holds no NES traces).
python3 scripts/process_results_lat.py \
    --results-dir results \
    --output-csv results/nes_torchserve_osu_rgb.csv

python3 scripts/process_torchserve_results.py \
    --results-dir results \
    --output-csv results/torchserve_osu_rgb.csv

torchserve --stop

deactivate
rm -rf ./results/*/
