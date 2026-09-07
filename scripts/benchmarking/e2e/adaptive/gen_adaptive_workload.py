#!/usr/bin/env python3
"""Generate the systest files for the adaptive-inference-optimizer characterization.

The experiment asks how each configuration knob (batch size, prediction cache, batch
deduplication, model precision) responds to the two environment factors the optimizer has to
react to: the *ingestion rate* and the *fraction of duplicate records*.

Rate and duplicate ratio are properties of the source, so they cannot be swept by
run_systests.py (which only sweeps worker/optimizer parameters). They are therefore baked into
the generated systest: one Generator physical source per (rate, duplicate structure, duplicate
percentage) point, and one query section per (source, model precision) pair. The knobs proper
stay in the YAML sweeps under config/, so the runner's combination labels carry them.

Two duplicate *structures* are generated, because caching and deduplication react to different
kinds of repetition:

  CACHE_HOTSET  a fixed hotset of `--hotset-size` keys is revisited over the whole stream while
                cold keys never repeat. Duplicates are spread out in time, so a prediction cache
                captures them, but a single batch rarely contains the same key twice.
  CACHE_GROUPED every key is emitted as one contiguous run (a miss followed by its repetitions).
                Duplicates are adjacent, so within-batch deduplication captures them even with a
                small batch, and a cache of any capacity captures them too.

Outputs (into --out-dir and --manifest-dir):
  nbeats300_adaptive.test      the (rate x duplicates x precision) grid
  nbeats300_adaptive_cal.test  calibration sections: unbounded rate, used to locate saturation
  adaptive_manifest.json       section number -> workload parameters, for the results processor
  sections.env                 shell fragment with the section lists the driver script needs

Regenerate after Phase 0 with the calibrated --rates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Dict, List, Tuple

DEFAULT_MODEL = "model/power/pretrained/nbeats/nbeats_size_300_stride_30.onnx"
DEFAULT_INT8_MODEL = "model/power/pretrained/nbeats/nbeats_size_300_stride_30_int8.onnx"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--rates",
        type=int,
        nargs="+",
        default=[10_000, 25_000, 50_000, 100_000],
        help="Offered ingestion rates in tuples/s. Set these from the Phase 0 calibration so that "
             "the grid straddles the saturation point of the baseline configuration.",
    )
    parser.add_argument("--duration-s", type=float, default=20.0, help="Target duration of one run (records = rate * duration).")
    parser.add_argument("--hotset-dups", type=int, nargs="+", default=[0, 50, 90, 99], help="Duplicate percentages for CACHE_HOTSET.")
    parser.add_argument("--grouped-dups", type=int, nargs="+", default=[50, 90], help="Duplicate percentages for CACHE_GROUPED.")
    parser.add_argument("--hotset-size", type=int, default=64, help="Number of recurring keys in the CACHE_HOTSET working set.")
    parser.add_argument("--key-seed", type=int, default=1)
    parser.add_argument(
        "--sources-per-point",
        type=int,
        default=1,
        help="Physical Generator sources per operating point, each offering rate/N. One source emits at "
             "most one buffer per flush interval, so a single source cannot offer more than about "
             "tuples-per-buffer / flush-interval (~175k tuples/s at 1 MiB and 10 ms) - well below the "
             "capacity of a batched configuration on a many-core node. Each source gets a disjoint key "
             "range, so the duplicate percentage is preserved while the working set becomes N x hotset.",
    )
    parser.add_argument("--num-inputs", type=int, default=75, help="Number of FLOAT32 model input columns (nbeats300 takes 75).")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument(
        "--int8-model-path",
        default=None,
        help="Defaults to --model-path with an _int8 suffix, which is where quantize_nbeats.py "
             "writes it. Both paths are resolved against the systest testdata directory, whose "
             "layout differs between checkouts (model/power/... vs model/solar_power/...), so "
             "pass --model-path to match the tree you are running on.",
    )
    parser.add_argument("--backend", default="OpenVINO", help="Inference backend. INT8 requires OpenVINO; IREE rejects non-f32 graphs.")
    parser.add_argument("--flush-interval-ms", type=int, default=10, help="Generator flush interval; caps how many tuples reach one buffer.")
    parser.add_argument(
        "--operator-buffer-size",
        type=int,
        default=1_048_576,
        help="Tuple buffer size in bytes. Bounds the achievable batch size: a batch is flushed at "
             "every buffer boundary, so effective batch <= buffer capacity in tuples.",
    )
    parser.add_argument("--buffers", type=int, default=32_768, help="worker.number_of_buffers_in_global_buffer_manager.")
    parser.add_argument("--page-size", type=int, default=16_384)
    parser.add_argument("--calibration-records", type=int, default=500_000, help="Record count for the unbounded-rate calibration sections.")
    parser.add_argument("--calibration-dups", type=int, nargs="+", default=[0, 90], help="Duplicate percentages of the calibration sections.")
    parser.add_argument("--out-dir", type=Path, default=root / "nes-systests" / "inference" / "adaptive")
    parser.add_argument("--manifest-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--corner-rates",
        type=int,
        nargs="+",
        default=None,
        help="Rates used for the Phase 2 interaction grid (default: lowest and highest of --rates).",
    )
    parser.add_argument(
        "--corner-dups",
        type=int,
        nargs="+",
        default=None,
        help="CACHE_HOTSET duplicate percentages used for the Phase 2 interaction grid "
             "(default: lowest and highest of --hotset-dups).",
    )
    return parser.parse_args()


def derive_int8_path(model_path: str) -> str:
    path = PurePosixPath(model_path)
    return str(path.with_name(path.stem + "_int8" + path.suffix))


def field_names(count: int) -> List[str]:
    return [f"f{i:02d}" for i in range(1, count + 1)]


def logical_source(name: str, fields: List[str]) -> str:
    columns = ", ".join(f"{f} FLOAT32 NOT NULL" for f in fields)
    return f"CREATE LOGICAL SOURCE {name}({columns});\n"


def generator_schema(structure: str, records: int, dup: int, hotset: int, key_seed: int, count: int) -> str:
    """One generator field per model input column.

    All columns share the key schedule and differ only by valueOffset, so a repeated key repeats
    the whole record - which is what both the prediction cache and the batch deduplication key on.
    """
    if structure == "hotset":
        return ", ".join(f"CACHE_HOTSET FLOAT32 {records} {dup} {hotset} {key_seed} {i}" for i in range(count))
    return ", ".join(f"CACHE_GROUPED FLOAT32 {records} {dup} {key_seed} {i}" for i in range(count))


def physical_source(name: str, rate: int, schema: str, flush_ms: int) -> str:
    return f"""CREATE PHYSICAL SOURCE FOR {name} TYPE Generator SET(
       'ALL' as `SOURCE`.STOP_GENERATOR_WHEN_SEQUENCE_FINISHES,
       1 AS `SOURCE`.SEED,
       'FIXED' AS `SOURCE`.GENERATOR_RATE_TYPE,
       'emit_rate {rate}' AS `SOURCE`.GENERATOR_RATE_CONFIG,
       {flush_ms} AS `SOURCE`.FLUSH_INTERVAL_MS,
       '{schema}' AS `SOURCE`.GENERATOR_SCHEMA
);
"""


def model_block(name: str, path: str, backend: str, fields: List[str]) -> str:
    inputs = ", ".join(f"{f} FLOAT32" for f in fields)
    return f"""CREATE MODEL {name} ('{path}' BACKEND {backend})
INPUT ({inputs})
OUTPUT (prediction VARSIZED);

"""


def sink_block(name: str) -> str:
    # Every section writes the same file: run_systests.py runs one section per invocation and
    # collects 'results/latency_timings.csv' (relative to the systest cwd) into the rep dir.
    return f"""CREATE SINK {name}(prediction VARSIZED NOT NULL, ingestion_time UINT64 NOT NULL) TYPE Latency SET(
       'results/latency_timings.csv' AS `SINK`.log_path,
       'ingestion_time' AS `SINK`.ingest_field
);
"""


def query_block(comment: str, model: str, source: str, sink: str, fields: List[str]) -> str:
    inner = ",\n            ".join(fields)
    return f"""{comment}
SELECT prediction, ingestion_time
FROM MODEL_INFERENCE({model}, (
        SELECT
            {inner},
            CURRENT_TIME() AS ingestion_time
        FROM {source}
    )
)
INTO {sink};
----
"""


def global_config(args: argparse.Namespace) -> str:
    return f"""GlobalConfiguration enable_event_trace: [true]
GlobalConfiguration worker.default_query_execution.operator_buffer_size: [{args.operator_buffer_size}]
GlobalConfiguration worker.default_query_execution.page_size: [{args.page_size}]
GlobalConfiguration worker.number_of_buffers_in_global_buffer_manager: [{args.buffers}]
GlobalConfiguration worker.default_query_execution.inference.openvino_inference_num_threads: [1]

"""


def build_test(
    args: argparse.Namespace,
    points: List[Dict[str, object]],
    header: str,
    test_name: str,
    source_only_first: bool = False,
) -> Tuple[str, List[Dict[str, object]]]:
    """Render one .test file.

    `points` describe the sources (rate / duplicate structure / duplicate percentage / records).
    Every point yields two query sections, one per model precision. With `source_only_first`, an
    extra section :01 reads the first source into a Void sink without any inference operator; that
    is the generator's own throughput ceiling.
    """
    fields = field_names(args.num_inputs)
    blocks: List[str] = [header, global_config(args)]

    shards = max(1, args.sources_per_point)
    for point in points:
        source = str(point["source"])
        blocks.append(logical_source(source, fields))
        records_per_source = max(1, int(point["records"]) // shards)
        rate_per_source = max(1, int(point["rate"]) // shards)
        for shard in range(shards):
            # Disjoint key ranges keep the duplicate percentage of each shard - and therefore of the
            # merged stream - equal to the configured one; only the working set grows by N.
            key_seed = args.key_seed + shard * (records_per_source + args.hotset_size + 1)
            schema = generator_schema(
                str(point["structure"]),
                records_per_source,
                int(point["dup"]),
                args.hotset_size,
                key_seed,
                args.num_inputs,
            )
            blocks.append(physical_source(source, rate_per_source, schema, args.flush_interval_ms))
        blocks.append("\n")

    blocks.append(model_block("nbeats300", args.model_path, args.backend, fields))
    blocks.append(model_block("nbeats300q", args.int8_model_path, args.backend, fields))

    sections: List[Dict[str, object]] = []
    sink_blocks: List[str] = []
    query_blocks: List[str] = []
    section_no = 0

    def record(point: Dict[str, object], precision: str) -> None:
        sections.append(
            {
                "test": test_name,
                "section": f"{section_no:02d}",
                "query_name": f"{test_name}_{section_no:02d}",
                "offered_rate": point["rate"],
                "structure": point["structure"],
                "duplicate_percent": point["dup"],
                "precision": precision,
                "records": point["records"],
                "hotset_size": (args.hotset_size * max(1, args.sources_per_point))
            if point["structure"] == "hotset"
            else None,
                "source": point["source"],
            }
        )

    if source_only_first:
        section_no += 1
        sink_blocks.append("CREATE SINK snkCalVoid(f01 FLOAT32 NOT NULL) TYPE Void;\n")
        first = points[0]
        # `SELECT f01` alone keeps the qualified name (SCALHOT0$F01) and fails to bind to the
        # sink's unqualified `f01`; the alias renames it, which is how the systests project raw
        # source fields into a sink.
        query_blocks.append(
            f"\n# :{section_no:02d} -- generator ceiling: no inference operator, "
            f"{first['records']} records\n"
            f"SELECT f01 AS f01 FROM {first['source']} INTO snkCalVoid;\n----\n"
        )
        record(first, "none")

    for point in points:
        for precision, model in (("fp32", "nbeats300"), ("int8", "nbeats300q")):
            section_no += 1
            sink = f"snk{point['source']}{precision.capitalize()}"
            sink_blocks.append(sink_block(sink))
            comment = (
                f"\n# :{section_no:02d} -- rate={point['rate']} tuples/s, structure={point['structure']}, "
                f"duplicates={point['dup']}%, precision={precision}, records={point['records']}"
            )
            query_blocks.append(query_block(comment, model, str(point["source"]), sink, fields))
            record(point, precision)

    blocks.extend(sink_blocks)
    blocks.extend(query_blocks)
    return "".join(blocks), sections


def main() -> int:
    args = parse_args()
    if args.int8_model_path is None:
        args.int8_model_path = derive_int8_path(args.model_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- main grid -------------------------------------------------------------------------
    points: List[Dict[str, object]] = []
    for rate_index, rate in enumerate(args.rates, start=1):
        records = int(rate * args.duration_s)
        for dup in args.hotset_dups:
            points.append(
                {"source": f"sHot{dup}R{rate_index}", "rate": rate, "structure": "hotset", "dup": dup, "records": records}
            )
        for dup in args.grouped_dups:
            points.append(
                {"source": f"sGrp{dup}R{rate_index}", "rate": rate, "structure": "grouped", "dup": dup, "records": records}
            )

    main_header = f"""# name: inference/adaptive/nbeats300_adaptive.test
# description: Knob-response characterization for the adaptive inference optimizer.
#              One section per (ingestion rate, duplicate structure, duplicate percentage, model
#              precision); the knobs (batch size, prediction cache, batch deduplication) come from
#              the YAML sweeps in scripts/benchmarking/e2e/adaptive/config/.
# groups: [Inference, Adaptive]
#
# GENERATED FILE - edit gen_adaptive_workload.py and regenerate.
#
# Rates: {args.rates} tuples/s; each section emits rate * {args.duration_s}s records.
# Duplicates: CACHE_HOTSET {args.hotset_dups}% (hotset of {args.hotset_size} keys, spread over the
# stream) and CACHE_GROUPED {args.grouped_dups}% (contiguous runs; mean run length = 1/(1-p)).
#
# A batch is flushed at every tuple-buffer boundary, so the configured batch size is an upper
# bound: effective batch = min(configured, tuples per buffer, rate * flush interval). The
# {args.operator_buffer_size} B buffer and the {args.flush_interval_ms} ms flush interval therefore
# take part in the batch-size response; process_adaptive_results.py reports the effective batch
# size measured from the event trace.
#
# No Configuration matrix lines: in-file combinations would override the runner's sweep.
# Requires ovc 2025.3 on PATH for the OpenVINO model import, and the INT8 model produced by
# quantize_nbeats.py at '{args.int8_model_path}'.
# One-time setup per build tree: mkdir -p cmake-build-release/nes-systests/systest/results

"""
    main_test, main_sections = build_test(args, points, main_header, "nbeats300_adaptive.test")

    # ---- calibration -----------------------------------------------------------------------
    # An "unbounded" fixed rate: the generator is asked for far more tuples than it can make, so
    # the run measures the ceiling of the source and of each configuration rather than a target.
    unbounded = 1_000_000_000
    cal_points: List[Dict[str, object]] = [
        {
            "source": f"sCalHot{dup}",
            "rate": unbounded,
            "structure": "hotset",
            "dup": dup,
            "records": args.calibration_records,
        }
        for dup in args.calibration_dups
    ]
    cal_header = f"""# name: inference/adaptive/nbeats300_adaptive_cal.test
# description: Phase 0 calibration for the adaptive-optimizer characterization: unbounded offered
#              rate, so each run reports the maximum sustainable throughput of the configuration
#              under test. Use the results to place the --rates grid of the main test around
#              saturation (e.g. 0.25x, 0.5x, 1.0x, 1.5x of the baseline's peak).
# groups: [Inference, Adaptive]
#
# GENERATED FILE - edit gen_adaptive_workload.py and regenerate.
#
# The first section has no inference operator at all: it measures the Generator's own ceiling with
# {args.num_inputs} FLOAT32 columns, which is the upper bound on every offered rate in the main
# test. If a main-test rate exceeds it, that operating point is generator-bound, not system-bound.

"""
    cal_test, cal_sections = build_test(
        args, cal_points, cal_header, "nbeats300_adaptive_cal.test", source_only_first=True
    )

    main_path = args.out_dir / "nbeats300_adaptive.test"
    cal_path = args.out_dir / "nbeats300_adaptive_cal.test"
    main_path.write_text(main_test, encoding="utf-8")
    cal_path.write_text(cal_test, encoding="utf-8")

    # ---- manifest + shell fragment ---------------------------------------------------------
    manifest = {
        "rates": args.rates,
        "duration_s": args.duration_s,
        "hotset_size": args.hotset_size,
        "num_inputs": args.num_inputs,
        "operator_buffer_size": args.operator_buffer_size,
        "flush_interval_ms": args.flush_interval_ms,
        "sources_per_point": args.sources_per_point,
        "model_path": args.model_path,
        "int8_model_path": args.int8_model_path,
        "sections": main_sections + cal_sections,
    }
    manifest_path = args.manifest_dir / "adaptive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def spec(section: Dict[str, object], test_file: str) -> str:
        return f"inference/adaptive/{test_file}:{section['section']}"

    corner_rates = args.corner_rates or [min(args.rates), max(args.rates)]
    corner_dups = args.corner_dups or [min(args.hotset_dups), max(args.hotset_dups)]

    def is_corner(section: Dict[str, object]) -> bool:
        return (
            section["structure"] == "hotset"
            and section["offered_rate"] in corner_rates
            and section["duplicate_percent"] in corner_dups
        )

    lines = [
        "# GENERATED by gen_adaptive_workload.py - sourced by run_adaptive_screening.sh",
        f'ADAPTIVE_TEST="inference/adaptive/nbeats300_adaptive.test"',
        f'ADAPTIVE_CAL_TEST="inference/adaptive/nbeats300_adaptive_cal.test"',
    ]

    def emit(var: str, specs: List[str]) -> None:
        lines.append(f'{var}="{" ".join(specs)}"')

    emit("SECTIONS_CAL_SOURCE_ONLY", [spec(cal_sections[0], "nbeats300_adaptive_cal.test")])
    emit(
        "SECTIONS_CAL_FP32",
        [spec(s, "nbeats300_adaptive_cal.test") for s in cal_sections if s["precision"] == "fp32"],
    )
    emit(
        "SECTIONS_CAL_INT8",
        [spec(s, "nbeats300_adaptive_cal.test") for s in cal_sections if s["precision"] == "int8"],
    )
    emit("SECTIONS_FP32", [spec(s, "nbeats300_adaptive.test") for s in main_sections if s["precision"] == "fp32"])
    emit("SECTIONS_INT8", [spec(s, "nbeats300_adaptive.test") for s in main_sections if s["precision"] == "int8"])
    emit(
        "SECTIONS_CORNERS_FP32",
        [spec(s, "nbeats300_adaptive.test") for s in main_sections if s["precision"] == "fp32" and is_corner(s)],
    )
    emit(
        "SECTIONS_CORNERS_INT8",
        [spec(s, "nbeats300_adaptive.test") for s in main_sections if s["precision"] == "int8" and is_corner(s)],
    )
    sections_env = args.manifest_dir / "sections.env"
    sections_env.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {main_path} ({len(main_sections)} sections, {len(points)} sources)")
    print(f"wrote {cal_path} ({len(cal_sections)} sections)")
    print(f"wrote {manifest_path}")
    print(f"wrote {sections_env}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
