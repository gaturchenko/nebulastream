# flink-openvino-baseline

Flink + OpenVINO object-detection baseline. Reads a base64-frame CSV, decodes/resizes each
frame, runs the model through OpenVINO (native JNI bridge), discards detections, and writes
latency/throughput stats. On OSU-RGB with NanoDet it is the Flink counterpart of the NES-UDF
case — use it to substitute the NES-UDF row in the framework comparison.

The Java job is model-agnostic (it reads the input/output tensor sizes from the model), so
NanoDet needs **no code change** — just pass its input size (`192 192`) and point at the model.

## x86 (dev) build

    ./scripts/build-with-vcpkg.sh                     # builds native libs against a vcpkg OpenVINO
    ./scripts/run-csv-object-detection.sh <model-rel> <csv-rel> CPU 1 1 1 0 false 192 192

## Raspberry Pi (aarch64) deployment

The Pi has no vcpkg OpenVINO, no JDK, and no cmake — but it has `g++`, a JRE, and a full
OpenVINO aarch64 C++ SDK bundled inside the pip `openvino` wheel (headers + libs + arm CPU
plugin). So the native bridge is compiled directly with `g++` against the wheel; no OpenVINO
build and no cmake are involved (~25 s).

What gets copied to the Pi under `~/navi/baselines/flink-openvino-baseline`:

  * `native/` + `jni/` (jni.h/jni_md.h from any JDK — the ABI is arch-independent on 64-bit Linux)
  * `target/flink-openvino-baseline-1.0-SNAPSHOT.jar` (arch-independent; not recompiled)
  * `cp.txt` + the `~/.m2` dependency jars it lists (same `/home/gaturchenko/.m2` path on both hosts)

Then, on the Pi:

    ./scripts/build-native-pi.sh          # g++ -> target/native-libs/libflink_openvino_bridge_jni.so
    ./scripts/run-pi-osu-nanodet.sh       # converts nanodet.onnx->IR (ovc) on first run, then runs

`run-pi-osu-nanodet.sh` defaults to the single-thread OSU-RGB NanoDet config
(`parallelism=1 streams=1 threads=1`, base64 col 0, no header, 192x192) and writes
`~/navi/results/flink_openvino_stats.csv`. Override via env vars (`PARALLELISM`, `THREADS`,
`STREAMS`, `MODEL`, `CSV`, `STATS`, `RATE`, and the comparison labels `QUERY_NAME`, `PARAM_NAME`,
`PARAM_VALUE`, `REPETITION`).

### Fair ingestion rate

For the comparison to be fair, the Flink source is paced at a fixed rate (`RATE`, tuples/s)
matching the NES/TorchServe TCP source `MOCK_TUPLE_RATE` (OSU-RGB = 30) — otherwise Flink's
`FileSource` would be a backpressure-driven firehose while the other systems emit at a controlled
rate. Pacing is cumulative (same scheme as the NES `TCPDataServer` SendRateLimiter), applied in a
single ingestion lane (parser pinned to parallelism 1, mirroring the one NES source), and the
ingestion timestamp is stamped after pacing. `RATE=0` restores an unbounded source.

### Stats columns

The writer emits **one wide row per run**, aligned with the NES / TorchServe comparison tables
(`process_results.py` / `process_torchserve_results.py`) so a Flink row drops straight into the
framework comparison, substituting the NES-UDF case:

`query_name, inference_config_param_name, inference_config_param_value, repetition, flink_records,
end_to_end_duration_us, end_to_end_throughput, end_to_end_latency_us, e2e_latency_ms_{mean,p50,p95,p99},
inference_latency_ms_{mean,p50,p95,p99}`

`end_to_end_duration_us` (wall-clock span from the first record's arrival to the last record's
output), `end_to_end_throughput` (records/s) and `end_to_end_latency_us` mirror the NES columns of
the same name; `e2e_latency_ms_*` mirror the TorchServe `e2e_latency_ms_*` (parse+preprocess+infer);
`inference_latency_ms_*` is the pure-OpenVINO `predict` distribution. Repeated runs append rows.
Set the label env vars to whatever key columns you group/plot by.

Note the preprocessing here is RGB, scaled to [0,1], NCHW — a standard normalization, not
byte-identical to the NES `RESIZE_IMAGE` path, but equivalent in compute for a framework-level
latency/throughput comparison.
