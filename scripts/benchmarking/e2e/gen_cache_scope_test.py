#!/usr/bin/env python3
"""Generates the cache-scope latency benchmark systest for nbeats_size_300_stride_30.onnx."""

NUM_INPUTS = 75
RECORDS = 1_000_000
HOTSET = 64
CACHE_ENTRIES = 512
EMIT_RATE = 100_000
KEY_SEED = 1
HIT_RATES = [25, 50, 75, 95]

fields = [f"f{i:02d}" for i in range(1, NUM_INPUTS + 1)]


def source_block(pct: int) -> str:
    name = f"cacheStreamH{pct}"
    logical_fields = ", ".join(f"{f} FLOAT32 NOT NULL" for f in fields)
    # one CACHE_HOTSET entry per column; identical schedule, per-column valueOffset makes each record a ramp
    schema = ", ".join(f"CACHE_HOTSET FLOAT32 {RECORDS} {pct} {HOTSET} {KEY_SEED} {i}" for i in range(NUM_INPUTS))
    return f"""CREATE LOGICAL SOURCE {name}({logical_fields});
CREATE PHYSICAL SOURCE FOR {name} TYPE Generator SET(
       'ALL' as `SOURCE`.STOP_GENERATOR_WHEN_SEQUENCE_FINISHES,
       1 AS `SOURCE`.SEED,
       'FIXED' AS `SOURCE`.GENERATOR_RATE_TYPE,
       'emit_rate {EMIT_RATE}' AS `SOURCE`.GENERATOR_RATE_CONFIG,
       '{schema}' AS `SOURCE`.GENERATOR_SCHEMA
);
"""


def sink_block(tag: str) -> str:
    # All sinks share one log_path: run_systests.py collects exactly
    # 'results/latency_timings.csv' (relative to the systest cwd) into each rep dir,
    # and only one query section runs per invocation, so the paths never collide.
    return f"""CREATE SINK nbeatsLatency{tag}(prediction VARSIZED NOT NULL, ingestion_time UINT64 NOT NULL) TYPE Latency SET(
       'results/latency_timings.csv' AS `SINK`.log_path,
       'ingestion_time' AS `SINK`.ingest_field
);
"""


def query_block(tag: str, source: str, config_lines: list[str], comment: str) -> str:
    inner_fields = ",\n            ".join(fields)
    cfg = ("\n".join(config_lines) + "\n") if config_lines else ""
    return f"""{comment}
{cfg}SELECT prediction, ingestion_time
FROM MODEL_INFERENCE(nbeats300, (
        SELECT
            {inner_fields},
            CURRENT_TIME() AS ingestion_time
        FROM {source}
    )
)
INTO nbeatsLatency{tag};
----
"""


hit_rate_expectation = {
    pct: (RECORDS * pct // 100 - HOTSET, RECORDS - (RECORDS * pct // 100 - HOTSET)) for pct in HIT_RATES
}
expect_lines = "\n".join(
    f"#   {pct}% hot -> {hits} hits / {misses} misses (once warm, LFU/LRU/SECOND_CHANCE)"
    for pct, (hits, misses) in hit_rate_expectation.items()
)

header = f"""# name: inference/cache_scope/nbeats300_cache_scope_lat.test
# description: Thread-local vs. global prediction cache latency benchmark on nbeats_size_300_stride_30.onnx
#              ({NUM_INPUTS} FLOAT32 inputs fed directly, no windowing, so the CACHE_HOTSET workload controls
#              the cache hit rate deterministically).
# groups: [Inference, CacheScope]
#
# Workload: CACHE_HOTSET with a hotset of {HOTSET} keys; hot accesses round-robin, cold keys never repeat.
# A GLOBAL cache pays the {HOTSET} warm-up misses once; THREAD_LOCAL caches pay them once per worker thread.
# Expected steady-state totals over {RECORDS} records (single consumer):
{expect_lines}
# FIFO cannot keep a hotset resident under cold traffic; its hit rate is lower (workload is still identical
# for both scopes, so the scope comparison remains valid).
# The cache capacity ({CACHE_ENTRIES}) keeps the hotset resident for LRU/SECOND_CHANCE down to 25% hot
# (needs roughly hotset * 100 / hotPercent = {HOTSET * 4} entries).
#
# Run with scripts/benchmarking/e2e/run_systests.py, which sweeps threads/policy/scope from a YAML,
# runs one query section (= one hit rate) per systest invocation, and collects the Latency sink's
# 'results/latency_timings.csv' plus the run log into results/<combo>/<query>/rep-NN/:
#   python3 scripts/benchmarking/e2e/run_systests.py \\
#       --queries inference/cache_scope/nbeats300_cache_scope_lat.test:01 \\
#       --inference-config scripts/benchmarking/e2e/config/cache_scope.yaml --repetitions 3
# Sections: :01 = 25% hot, :02 = 50% hot, :03 = 75% hot, :04 = 95% hot.
# One-time setup per build tree: mkdir -p cmake-build-release/nes-systests/systest/results
# (the sink opens its CSV relative to the systest cwd and does not create the directory).
# No SEQUENTIAL_EXECUTION: it would chain the sections as dependencies and break single-section
# invocation; the runner executes one section per invocation, so there is no concurrency anyway.
# Consolidate with scripts/benchmarking/e2e/process_cache_scope_results.py.
# Deliberately NO Configuration matrix lines here: in-file combinations would override the runner's sweep.
#
# Requires ovc 2025.3 on PATH for the OpenVINO model import.
# Knobs: records/rate are baked into GENERATOR_SCHEMA ({RECORDS} records at {EMIT_RATE} tuples/s -> ~{RECORDS // EMIT_RATE} s per run).

GlobalConfiguration worker.default_query_execution.inference.openvino_inference_num_threads: [1]
GlobalConfiguration worker.default_query_execution.operator_buffer_size: [131072]
# 16384 x 128 KiB = 2 GiB; raise on machines with more memory (the server ablation tests use 524288)
GlobalConfiguration worker.number_of_buffers_in_global_buffer_manager: [16384]

"""

blocks = [header]
for pct in HIT_RATES:
    blocks.append(source_block(pct))
blocks.append(f"""
CREATE MODEL nbeats300 ('model/power/pretrained/nbeats/nbeats_size_300_stride_30.onnx')
INPUT ({", ".join(f"{f} FLOAT32" for f in fields)})
OUTPUT (prediction VARSIZED);

""")
for pct in HIT_RATES:
    blocks.append(sink_block(f"H{pct}"))

for idx, pct in enumerate(HIT_RATES, start=1):
    blocks.append(query_block(f"H{pct}", f"cacheStreamH{pct}", [], f"\n# :{idx:02d} -- {pct}% hot accesses (cache config comes from the runner's YAML sweep)"))

out = "".join(blocks)
path = "/home/gaturchenko/CLionProjects/nebulastream/nes-systests/inference/cache_scope/nbeats300_cache_scope_lat.test"
import os
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as fh:
    fh.write(out)
print(f"wrote {path} ({len(out)} bytes)")
