#!/usr/bin/env python3
"""Consolidate the adaptive-inference screening runs into one tidy CSV.

Walks the tree written by run_systests.py

    <results-dir>/<combination>/<query_name>/rep-NN/{*.json, latency_timings.csv, _run.log, ...}

and emits one row per (configuration, operating point, repetition) with the knobs and the two
environment factors as proper columns, so the response of each knob to ingestion rate and duplicate
ratio can be read off directly (and fed to the optimizer's cost model).

Columns of interest:
    offered_rate, duplicate_percent, structure        the environment the optimizer must react to
    batch_size, cache_type, cache_entries, dedup, precision, worker_threads   the knobs
    sink_throughput, sustained_ratio                  did the configuration keep up with the rate
    effective_batch_size                              tuples per inference task, from the trace;
                                                      the configured batch size is only an upper
                                                      bound (batches flush at buffer boundaries)
    latency_us_p50/p95/p99, latency_us_mean/max       per-record latency from the LatencySink
    cache_hits, cache_misses, observed_hit_rate       from the operator's own log line
    source_saturated                                  the generator could not keep up, so the
                                                      offered rate was not actually offered
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pandas as pd
except ImportError:  # pragma: no cover - runtime environment dependent
    print("pandas is required. Install it with: pip install pandas", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from process_results import (  # noqa: E402  (path set above)
    parse_inference_config,
    parse_inference_pipeline_id,
    parse_pipeline_aggregate_metric_rows,
)

# The calibration sections offer an unreachable rate on purpose, so "the generator could not keep
# up" is their intended state, not a warning-worthy finding.
UNBOUNDED_RATE_THRESHOLD = 1e8
CACHE_STATS_PATTERN = re.compile(r"InferModelPhysicalOperator cache hits=(\d+), misses=(\d+)")
SOURCE_SATURATION_PATTERN = re.compile(r"Can not produce all required tuples in the flushInterval")
KNOB_COLUMNS = {
    "optimizer.inference_batch_size": "batch_size",
    "batch_size": "batch_size",
    "prediction_cache_type": "cache_type",
    "number_of_entries_prediction_cache": "cache_entries",
    "use_batch_deduplication": "dedup",
    "worker.query_engine.number_of_worker_threads": "worker_threads",
}
DEFAULT_KNOBS = {"batch_size": "1", "cache_type": "NONE", "cache_entries": "", "dedup": "false", "worker_threads": ""}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=here / "results")
    parser.add_argument("--manifest", type=Path, default=here / "adaptive_manifest.json")
    parser.add_argument("--output-csv", type=Path, default=None, help="Default: <results-dir>/adaptive_results.csv")
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.1,
        help="Fraction of the leading records dropped from the latency statistics; they carry model "
             "compilation and cache warm-up (default: 0.1).",
    )
    parser.add_argument(
        "--sample-latency",
        type=int,
        default=0,
        help="If > 0, also write <output>_latency_sample.csv with this many sampled records per run, "
             "for plotting latency distributions.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> Dict[str, Dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(section["query_name"]): section for section in payload["sections"]}


def knobs_from_label(label: str) -> Dict[str, str]:
    parts = parse_inference_config(label)
    names = parts["param_name"].split("__") if parts["param_name"] else []
    values = parts["param_value"].split("__") if parts["param_value"] else []
    knobs = dict(DEFAULT_KNOBS)
    for name, value in zip(names, values):
        column = KNOB_COLUMNS.get(name)
        if column is not None:
            knobs[column] = value
    if knobs["cache_type"].upper() == "NONE":
        knobs["cache_entries"] = ""
    if knobs["batch_size"] == "1":
        knobs["dedup"] = "false"
    return knobs


def expected_unique_fraction(structure: str, duplicate_percent: float, batch: Optional[float], hotset: Optional[float]) -> Optional[float]:
    """Fraction of a batch that deduplication still has to send to the model.

    Both generator workloads are deterministic, so this is exact rather than probabilistic, and it
    is the quantity a cost model would use to decide whether deduplication pays:

      CACHE_GROUPED  keys arrive in contiguous runs of 1/(1-d) records, so a batch of B records
                     spans B(1-d) runs plus at most one partial run at each boundary.
      CACHE_HOTSET   hot accesses cycle round-robin through `hotset` keys, so a batch containing
                     d*B hot accesses touches min(d*B, hotset) distinct hot keys; the cold ones
                     never repeat.

    The gap between this and the measured speed-up is the deduplication machinery's own cost.
    """
    if batch is None or batch <= 0:
        return None
    d = float(duplicate_percent) / 100.0
    b = float(batch)
    if structure == "grouped":
        return min(1.0, (1.0 - d) + 1.0 / b)
    if structure == "hotset":
        if hotset is None:
            return None
        return min(1.0, (min(d * b, float(hotset)) + (1.0 - d) * b) / b)
    return None


def read_log(rep_dir: Path) -> Dict[str, object]:
    result: Dict[str, object] = {
        "cache_hits": None,
        "cache_misses": None,
        "source_saturated": False,
        "source_saturation_events": 0,
        "inference_pipeline_id": None,
    }
    logs = sorted(rep_dir.glob("*.log"))
    for log_path in logs:
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        hits = 0
        misses = 0
        found = False
        for match in CACHE_STATS_PATTERN.finditer(text):
            hits += int(match.group(1))
            misses += int(match.group(2))
            found = True
        if found:
            result["cache_hits"] = hits
            result["cache_misses"] = misses
        saturation = len(SOURCE_SATURATION_PATTERN.findall(text))
        if saturation:
            result["source_saturation_events"] = int(result["source_saturation_events"]) + saturation
            result["source_saturated"] = True
        pipeline_id = parse_inference_pipeline_id(log_path)
        if pipeline_id is not None:
            result["inference_pipeline_id"] = pipeline_id
    return result


def read_trace(rep_dir: Path, inference_pipeline_id: Optional[int]) -> Dict[str, object]:
    """Pipeline-level counters for this run.

    `pipeline_*` describe the model-inference pipeline. `ingested_tuples` is the largest tuple count
    of any pipeline in the trace, used as a rough stand-in for what the query read.

    Caveat, and it matters: the batching operator hands the inference operator a buffer whose
    payload is an `EmittedBatch{batchId}` and whose numberOfTuples is *batch metadata*, not scanned
    tuples - a different contract from a normal emit/scan. So these counters are a screening signal,
    not proof. The authoritative per-record counter is the prediction cache's hits+misses, logged by
    the operator itself (`cache_hits`/`cache_misses` below): it is incremented inside the scan loop
    and never reads buffer metadata. When the two disagree, believe the cache counter.
    See verify_batch_delivery.test for a standalone check against a Checksum sink.
    """
    empty: Dict[str, object] = {
        "pipeline_tuples": None,
        "pipeline_task_count": None,
        "pipeline_task_duration_us": None,
        "tuples_per_task": None,
        "effective_batch_size": None,
        "per_record_task_us": None,
        "ingested_tuples": None,
        "tuple_completion_ratio": None,
        "trace_throughput": None,
    }
    for json_path in sorted(rep_dir.glob("*.json")):
        if json_path.name == "systest-performance.json":
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        events = payload.get("traceEvents")
        if not isinstance(events, list):
            continue
        rows = parse_pipeline_aggregate_metric_rows(events)
        if not rows:
            continue

        out = dict(empty)
        tuple_counts = [float(row["pipeline_tuples"]) for row in rows if row.get("pipeline_tuples")]
        if tuple_counts:
            out["ingested_tuples"] = max(tuple_counts)
        # Wall-clock span of the busiest pipeline, so a source-only run (no inference operator, no
        # LatencySink CSV) still yields a throughput number.
        spans = [
            (float(row["pipeline_tuples"]), float(row["pipeline_task_span_us"]))
            for row in rows
            if row.get("pipeline_tuples") and row.get("pipeline_task_span_us")
        ]
        if spans:
            tuples, span = max(spans, key=lambda item: item[0])
            if span > 0:
                out["trace_throughput"] = tuples / span * 1e6

        for row in rows:
            if inference_pipeline_id is None or str(row.get("pipeline_id")) != str(inference_pipeline_id):
                continue
            tuples = row.get("pipeline_tuples")
            tasks = row.get("pipeline_task_count")
            duration = row.get("pipeline_task_duration_us")
            out["pipeline_tuples"] = tuples
            out["pipeline_task_count"] = tasks
            out["pipeline_task_duration_us"] = duration
            if tuples and tasks:
                out["tuples_per_task"] = float(tuples) / float(tasks)
            if tuples and duration:
                out["per_record_task_us"] = float(duration) / float(tuples)
            if tuples and out["ingested_tuples"]:
                out["tuple_completion_ratio"] = float(tuples) / float(out["ingested_tuples"])
            break
        return out
    return empty


def read_latency(rep_dir: Path, warmup_fraction: float, sample: int) -> Dict[str, object]:
    empty: Dict[str, object] = {
        "records": None,
        "sink_throughput": None,
        "sink_throughput_whole_run": None,
        "latency_us_mean": None,
        "latency_us_p50": None,
        "latency_us_p95": None,
        "latency_us_p99": None,
        "latency_us_max": None,
        "_sample": None,
    }
    csv_path = rep_dir / "latency_timings.csv"
    if not csv_path.exists():
        return empty
    try:
        frame = pd.read_csv(csv_path, usecols=["recv_ts_us", "latency_us"], dtype="float64")
    except (OSError, ValueError) as exc:
        print(f"Could not read {csv_path}: {exc}", file=sys.stderr)
        return empty
    if frame.empty:
        return empty

    frame = frame.sort_values("recv_ts_us")
    total = len(frame)
    # Both throughput and latency are measured over the post-warm-up window. The leading records
    # carry per-worker-thread model compilation and cache warm-up, and those fixed costs are a large
    # fraction of a short run - an INT8 model compiles ~2.6x slower than its FP32 counterpart, which
    # is enough to invert a throughput comparison on a one-second run.
    warm = frame.iloc[int(total * max(0.0, min(0.9, warmup_fraction))) :]
    span_us = float(frame["recv_ts_us"].iloc[-1] - frame["recv_ts_us"].iloc[0])
    warm_span_us = float(warm["recv_ts_us"].iloc[-1] - warm["recv_ts_us"].iloc[0]) if len(warm) > 1 else 0.0
    latency = warm["latency_us"]
    result: Dict[str, object] = {
        "records": total,
        "sink_throughput": (len(warm) / warm_span_us * 1e6) if warm_span_us > 0 else None,
        "sink_throughput_whole_run": (total / span_us * 1e6) if span_us > 0 else None,
        "latency_us_mean": float(latency.mean()),
        "latency_us_p50": float(latency.quantile(0.50)),
        "latency_us_p95": float(latency.quantile(0.95)),
        "latency_us_p99": float(latency.quantile(0.99)),
        "latency_us_max": float(latency.max()),
        "_sample": latency.sample(min(sample, len(latency)), random_state=0).to_numpy() if sample > 0 else None,
    }
    return result


def read_query_duration(rep_dir: Path) -> Optional[float]:
    path = rep_dir / "systest-performance.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("end_to_end_duration_us")
    return float(value) if isinstance(value, (int, float)) else None


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        return 1
    manifest = load_manifest(args.manifest)

    rows: List[Dict[str, object]] = []
    samples: List[pd.DataFrame] = []

    for combo_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        knobs = knobs_from_label(combo_dir.name)
        for query_dir in sorted(path for path in combo_dir.iterdir() if path.is_dir()):
            section = manifest.get(query_dir.name)
            if section is None:
                print(f"No manifest entry for query '{query_dir.name}' - skipping", file=sys.stderr)
                continue
            for rep_dir in sorted(path for path in query_dir.iterdir() if path.is_dir()):
                log = read_log(rep_dir)
                trace = read_trace(rep_dir, log["inference_pipeline_id"])
                latency = read_latency(rep_dir, args.warmup_fraction, args.sample_latency)
                sample = latency.pop("_sample")

                row: Dict[str, object] = {
                    "combination": combo_dir.name,
                    "query_name": query_dir.name,
                    "repetition": rep_dir.name,
                    "offered_rate": section["offered_rate"],
                    "structure": section["structure"],
                    "duplicate_percent": section["duplicate_percent"],
                    "hotset_size": section["hotset_size"],
                    "precision": section["precision"],
                    "records_configured": section["records"],
                    **knobs,
                    **trace,
                    **latency,
                    "query_duration_us": read_query_duration(rep_dir),
                    "cache_hits": log["cache_hits"],
                    "cache_misses": log["cache_misses"],
                    "source_saturated": log["source_saturated"],
                    "source_saturation_events": log["source_saturation_events"],
                }

                hits = row["cache_hits"]
                misses = row["cache_misses"]
                if hits is not None and misses is not None and (hits + misses) > 0:
                    row["observed_hit_rate"] = hits / (hits + misses)
                else:
                    row["observed_hit_rate"] = None

                # A batch is flushed at every buffer boundary, so the model sees
                # min(configured batch, tuples in the task) records per call.
                tuples_per_task = row.get("tuples_per_task")
                try:
                    configured_batch = float(knobs["batch_size"])
                except (TypeError, ValueError):
                    configured_batch = None
                if tuples_per_task and configured_batch:
                    row["effective_batch_size"] = min(configured_batch, float(tuples_per_task))

                delivered = row.get("records")
                configured_records = section["records"]
                row["records_delivered_ratio"] = (
                    float(delivered) / float(configured_records)
                    if delivered and configured_records
                    else None
                )

                row["expected_unique_fraction"] = expected_unique_fraction(
                    str(section["structure"]),
                    float(section["duplicate_percent"]),
                    row.get("effective_batch_size"),
                    section["hotset_size"],
                )

                # "Did it keep up?" must be judged over the WHOLE run: a configuration that falls
                # behind and then drains its backlog shows a post-warm-up rate above the offered one,
                # so the trimmed figure reports >1.0 for a run that never kept up. The trimmed figure
                # is the right one for comparing steady-state speed, and is reported separately.
                offered = float(section["offered_rate"])
                whole = row.get("sink_throughput_whole_run")
                steady = row.get("sink_throughput")
                unbounded = offered >= UNBOUNDED_RATE_THRESHOLD
                row["sustained_ratio"] = (whole / offered) if (whole and not unbounded) else None
                row["steady_state_ratio"] = (steady / offered) if (steady and not unbounded) else None
                rows.append(row)

                if sample is not None and len(sample):
                    samples.append(
                        pd.DataFrame(
                            {
                                "combination": combo_dir.name,
                                "query_name": query_dir.name,
                                "repetition": rep_dir.name,
                                "offered_rate": section["offered_rate"],
                                "duplicate_percent": section["duplicate_percent"],
                                "structure": section["structure"],
                                "precision": section["precision"],
                                "latency_us": sample,
                            }
                        )
                    )

    if not rows:
        print("No result data found.", file=sys.stderr)
        return 1

    frame = pd.DataFrame(rows)
    ordered = [
        "offered_rate", "duplicate_percent", "structure", "hotset_size", "precision",
        "batch_size", "cache_type", "cache_entries", "dedup", "worker_threads",
        "repetition", "records", "records_configured",
        "sink_throughput", "sink_throughput_whole_run", "trace_throughput", "sustained_ratio", "steady_state_ratio",
        "effective_batch_size", "tuples_per_task", "per_record_task_us",
        "ingested_tuples", "tuple_completion_ratio", "records_delivered_ratio",
        "latency_us_mean", "latency_us_p50", "latency_us_p95", "latency_us_p99", "latency_us_max",
        "cache_hits", "cache_misses", "observed_hit_rate", "expected_unique_fraction",
        "source_saturated", "source_saturation_events",
        "pipeline_tuples", "pipeline_task_count", "pipeline_task_duration_us",
        "query_duration_us", "combination", "query_name",
    ]
    frame = frame[[column for column in ordered if column in frame.columns]]
    frame = frame.sort_values(
        ["precision", "structure", "duplicate_percent", "offered_rate", "batch_size", "cache_type", "dedup", "repetition"]
    )

    output = args.output_csv or (results_dir / "adaptive_results.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} row(s) to {output}")

    if samples:
        sample_path = output.with_name(output.stem + "_latency_sample.csv")
        pd.concat(samples, ignore_index=True).to_csv(sample_path, index=False)
        print(f"Wrote latency samples to {sample_path}")

    # Only a *bounded* offered rate can be missed; the calibration sections are meant to saturate.
    bounded = frame[frame["offered_rate"] < UNBOUNDED_RATE_THRESHOLD]
    saturated = bounded[bounded["source_saturated"] == True]  # noqa: E712 - explicit for pandas
    if not saturated.empty:
        rates = sorted(set(saturated["offered_rate"]))
        print(
            f"WARNING: the generator could not keep up in {len(saturated)} run(s) at rate(s) {rates}. "
            "Those operating points are source-bound: lower the rate or the flush interval before "
            "drawing conclusions from them.",
            file=sys.stderr,
        )

    if "tuple_completion_ratio" in frame.columns:
        dropped = frame[frame["tuple_completion_ratio"].notna() & (frame["tuple_completion_ratio"] < 0.95)]
        if not dropped.empty:
            worst = dropped.nsmallest(1, "tuple_completion_ratio").iloc[0]
            print(
                f"WARNING: {len(dropped)} run(s) delivered fewer tuples to the model than the query "
                f"ingested (worst: {worst['tuple_completion_ratio'] * 100:.1f}% at batch "
                f"{worst['batch_size']}, {worst['worker_threads']} threads). Those runs measured a "
                "truncated workload; their throughput and latency are not comparable with runs that "
                "processed everything.",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
