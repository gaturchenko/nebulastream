#!/usr/bin/env bash
# Flink + OpenVINO MHATT-RNN baseline on the Raspberry Pi (aarch64), MUSAN audio workload.
#
# Counterpart of the NES "UDF" case in queries/musan/systems.test: it reads the same
# "timestamp,audioSample" CSV, aggregates a 1 s sliding window (ADVANCE 333 ms) of the paced 16 kHz
# sample stream into the model's 16000-sample input, scales int16 samples to [-1, 1], runs MHATT-RNN
# through OpenVINO via the JNI bridge, discards the keyword probabilities and writes a
# comparison-schema stats row.
# Use it to SUBSTITUTE the NES-UDF row in the MUSAN framework comparison.
#
# ONE audio stream, matching the single-source systems.test: the three-source join of the ablation
# experiment is not part of the framework comparison.
#
# Window geometry is expressed in SAMPLES and must stay consistent with the .test:
#   RATE=16000 tuples/s, SIZE 1 sec -> 16000 samples (the model input), ADVANCE 333 ms -> 5328 samples.
#
# Fairness parity with the NES side (queries/musan/systems.test):
#   * PARALLELISM=2  <-> worker.query_engine.number_of_worker_threads: 2
#   * STREAMS=1, THREADS=4 <-> the NES OpenVINO session: one stream with four inference threads per worker thread
#     (worker.default_query_execution.inference.openvino_num_streams / openvino_inference_num_threads in the .test).
#     With FLINK_OV_SHARE_COMPILED_MODEL=1 the shared model gets one stream per lane (FLINK_OV_SHARED_LANES).
#
# The model is the TensorFlow SavedModel's OpenVINO IR (saved_model.xml). NES loads the SavedModel
# through the same OpenVINO frontend, so both systems execute the same graph.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

OV_DIR="${OV_DIR:-/home/gaturchenko/navi/models-benchmarking/venv/lib/python3.13/site-packages/openvino}"
MODEL="${MODEL:-/home/gaturchenko/navi/models-benchmarking/models/mhatt-rnn/saved_model.xml}"
# The 60 s subset, matching the ATTACH in queries/musan/systems.test. The full capture is 583 s;
# both systems must read the same file or the comparison is meaningless.
CSV="${CSV:-/home/gaturchenko/navi/datasets/musan/musan-us-gov-0242-60s.csv}"
STATS="${STATS:-/home/gaturchenko/navi/results/flink_openvino_stats.csv}"

DEVICE="${DEVICE:-CPU}"
PARALLELISM="${PARALLELISM:-2}"   # Flink parallelism == number of inference lanes; run_systems.sh passes the .test's worker threads
STREAMS="${STREAMS:-1}"           # OpenVINO num_streams, like NES (negative = unset)
THREADS="${THREADS:-4}"           # OpenVINO inference_num_threads per lane, like NES

VALUE_COL="${VALUE_COL:-1}"       # MUSAN CSV is "timestamp,audioSample"...
HAS_HEADER="${HAS_HEADER:-false}" # ...with no header row
WINDOW_SAMPLES="${WINDOW_SAMPLES:-16000}"
ADVANCE_SAMPLES="${ADVANCE_SAMPLES:-5328}"
SCALE_DIVISOR="${SCALE_DIVISOR:-32767}"  # int16 -> [-1, 1], as the .test's audioSample / 32767.0

QUERY_NAME="${QUERY_NAME:-flink_musan_mhatt_rnn}"
PARAM_NAME="${PARAM_NAME:-baseline}"
PARAM_VALUE="${PARAM_VALUE:-flink}"
REPETITION="${REPETITION:-0}"

# Fixed ingestion rate (tuples/s) matching the NES/TorchServe TCP source MOCK_TUPLE_RATE (MUSAN=16000).
RATE="${RATE:-16000}"

NATIVE_DIR="$PROJECT_DIR/target/native-libs"
JAR="$PROJECT_DIR/target/flink-openvino-baseline-1.0-SNAPSHOT.jar"

[[ -f "$NATIVE_DIR/libflink_openvino_bridge_jni.so" ]] || { echo "Missing native lib; run scripts/build-native-pi.sh"; exit 1; }
[[ -f "$JAR" ]] || { echo "Missing app jar: $JAR"; exit 1; }
[[ -f "$PROJECT_DIR/cp.txt" ]] || { echo "Missing classpath file: $PROJECT_DIR/cp.txt"; exit 1; }
[[ -f "$CSV" ]] || { echo "Missing dataset: $CSV"; exit 1; }

# Convert the SavedModel -> OpenVINO IR on first use (needs ovc from the venv). OpenVINO reads
# TensorFlow SavedModels with its own frontend, so no TensorFlow install is involved.
if [[ ! -f "$MODEL" ]]; then
  SAVED_MODEL="$(dirname "$MODEL")/saved_model"
  [[ -d "$SAVED_MODEL" ]] || { echo "Missing model (no XML and no SavedModel dir): $MODEL / $SAVED_MODEL"; exit 1; }
  echo "Converting $SAVED_MODEL -> $MODEL with ovc"
  VENV_ROOT="${OV_DIR%%/lib/*}"
  ( source "$VENV_ROOT/bin/activate" 2>/dev/null || true
    ovc "$SAVED_MODEL" --output_model "$MODEL" --compress_to_fp16=False )
fi

mkdir -p "$(dirname "$STATS")"
cd "$PROJECT_DIR"
export FLINK_OV_SHARED_LANES="${FLINK_OV_SHARED_LANES:-$PARALLELISM}"
export LD_LIBRARY_PATH="$OV_DIR/libs:$NATIVE_DIR:${LD_LIBRARY_PATH:-}"

echo "Flink OpenVINO MHATT-RNN: model=$MODEL device=$DEVICE parallelism=$PARALLELISM streams=$STREAMS threads=$THREADS window=${WINDOW_SAMPLES}/${ADVANCE_SAMPLES} rate=$RATE"
java \
  -Djava.library.path="$NATIVE_DIR" \
  -cp "$JAR:$(cat "$PROJECT_DIR/cp.txt")" \
  benchmark.openvino.OpenVinoCsvWindowedInferenceJob \
  "$MODEL" "$CSV" "$DEVICE" "$PARALLELISM" "$STREAMS" "$THREADS" \
  "$VALUE_COL" "$HAS_HEADER" "$WINDOW_SAMPLES" "$ADVANCE_SAMPLES" "$SCALE_DIVISOR" \
  false "$STATS" "$QUERY_NAME" "$PARAM_NAME" "$PARAM_VALUE" "$REPETITION" "$RATE"

echo "Wrote stats: $STATS"
