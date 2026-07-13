#!/usr/bin/env python3
"""Run nes-systests .test queries across inference config combinations."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request
from typing import Dict, Iterable, List, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - runtime environment dependent
    print("PyYAML is required. Install it with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

REPETITIONS_DEFAULT = 5
QUERY_RETRIES = 3
INFERENCE_CONFIG_PREFIX = "worker.default_query_execution.inference."
INFERENCE_CONFIG_LABEL_PREFIX = "inference."
BATCH_SIZE_KEY = f"{INFERENCE_CONFIG_PREFIX}batch_size"
OPTIMIZER_BATCH_SIZE_KEY = "optimizer.inference_batch_size"
OPTIMIZER_BATCH_SIZE_ARG = "inference_batch_size"
OPTIMIZER_REWRITE_POST_JOIN_BATCH_INFERENCE_KEY = "optimizer.rewrite_post_join_batch_inference"
OPTIMIZER_REWRITE_POST_JOIN_BATCH_INFERENCE_ARG = "rewrite_post_join_batch_inference"
USE_BATCH_DEDUPLICATION_KEY = f"{INFERENCE_CONFIG_PREFIX}use_batch_deduplication"
PREDICTION_CACHE_TYPE_KEY = f"{INFERENCE_CONFIG_PREFIX}prediction_cache_type"
PREDICTION_CACHE_ENTRIES_KEY = f"{INFERENCE_CONFIG_PREFIX}number_of_entries_prediction_cache"
PREDICTION_CACHE_NONE = "NONE"
DEFAULT_USE_BATCH_DEDUPLICATION = False
SYSTEST_PERFORMANCE_FILE = "systest-performance.json"
RUN_LOG_SCRATCH = "_run.log"
# Per-record timing CSV written by the HttpSink (TorchServe baseline). Its log_path in the
# .test must point here, i.e. 'results/torchserve_timings.csv' relative to the systest CWD.
TORCHSERVE_TIMINGS_FILE = "torchserve_timings.csv"
# Per-record latency CSV written by the LatencySink. Its log_path in the .test must point
# here, i.e. 'results/latency_timings.csv' relative to the systest CWD.
LATENCY_TIMINGS_FILE = "latency_timings.csv"
TORCHSERVE_METRICS_START_FILE = "torchserve_metrics_start.prom"
TORCHSERVE_METRICS_END_FILE = "torchserve_metrics_end.prom"
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
QUERY_PERFORMANCE_PATTERN = re.compile(
    r"\bPASSED\s+in\s+([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)s\b"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_yaml(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data


def split_query(query: str) -> Tuple[str, str]:
    if ":" in query:
        left, right = query.rsplit(":", 1)
        if left.endswith(".test"):
            return left, f":{right}"
    return query, ""


def resolve_query(query: str, root: Path) -> str:
    path_part, suffix = split_query(query)
    candidate = Path(path_part)
    if not candidate.is_absolute():
        candidate = root / "nes-systests" / path_part
    if not candidate.exists():
        raise FileNotFoundError(f"Query file not found: {candidate}")
    return f"{candidate}{suffix}"


def normalize_values(values: object) -> List[object]:
    if isinstance(values, list):
        return values
    return [values]


def normalize_inference_config(config: Dict[str, object]) -> Dict[str, object]:
    normalized: Dict[str, object] = {}

    for key, value in config.items():
        if key in ("batch_size", OPTIMIZER_BATCH_SIZE_ARG, OPTIMIZER_BATCH_SIZE_KEY):
            key = BATCH_SIZE_KEY
        elif key in (
                OPTIMIZER_REWRITE_POST_JOIN_BATCH_INFERENCE_ARG,
                OPTIMIZER_REWRITE_POST_JOIN_BATCH_INFERENCE_KEY,
        ):
            key = OPTIMIZER_REWRITE_POST_JOIN_BATCH_INFERENCE_KEY
        elif key == "use_batch_deduplication":
            key = USE_BATCH_DEDUPLICATION_KEY
        normalized[key] = value

    if USE_BATCH_DEDUPLICATION_KEY not in normalized:
        normalized[USE_BATCH_DEDUPLICATION_KEY] = DEFAULT_USE_BATCH_DEDUPLICATION

    return normalized


def normalize_combination(combo: Dict[str, object]) -> Dict[str, object]:
    normalized = dict(combo)

    if normalized.get(BATCH_SIZE_KEY) == 1:
        normalized.pop(USE_BATCH_DEDUPLICATION_KEY, None)

    cache_type = normalized.get(PREDICTION_CACHE_TYPE_KEY)
    if isinstance(cache_type, str) and cache_type.upper() == PREDICTION_CACHE_NONE:
        normalized.pop(PREDICTION_CACHE_ENTRIES_KEY, None)

    return normalized


def expand_combinations(config: Dict[str, object]) -> Iterable[Dict[str, object]]:
    if not config:
        yield {}
        return
    keys = list(config.keys())
    values = [normalize_values(config[key]) for key in keys]
    seen = set()
    for combination in itertools.product(*values):
        normalized_combo = normalize_combination(dict(zip(keys, combination)))
        combo_key = tuple((key, format_value(value)) for key, value in normalized_combo.items())
        if combo_key in seen:
            continue
        seen.add(combo_key)
        yield normalized_combo


def format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def sanitize_name(text: str) -> str:
    sanitized = text.replace(os.sep, "_")
    if os.altsep:
        sanitized = sanitized.replace(os.altsep, "_")
    sanitized = sanitized.replace(":", "_")
    sanitized = sanitized.replace(" ", "_")
    return sanitized


def missing_repetitions(query_dir: Path, repetitions: int) -> List[int]:
    missing: List[int] = []
    for repetition in range(1, repetitions + 1):
        rep_dir = query_dir / f"rep-{repetition:02d}"
        if not rep_dir.is_dir():
            missing.append(repetition)
    return missing


def combination_name(combo: Dict[str, object]) -> str:
    if not combo:
        return "default"
    parts: List[str] = []
    for key, value in combo.items():
        label_key = key
        if label_key.startswith(INFERENCE_CONFIG_PREFIX):
            label_key = INFERENCE_CONFIG_LABEL_PREFIX + label_key[len(INFERENCE_CONFIG_PREFIX) :]
        parts.append(f"{label_key}={format_value(value)}")
    return sanitize_name("__".join(parts))


def build_command(
        systest_path: Path,
        query: str,
        params: Dict[str, object],
        *,
        log_path: Path,
) -> List[str]:
    optimizer_params: Dict[str, object] = {}
    worker_params: Dict[str, object] = {}

    for key, value in params.items():
        if key == BATCH_SIZE_KEY:
            optimizer_params[OPTIMIZER_BATCH_SIZE_ARG] = value
        elif key == OPTIMIZER_BATCH_SIZE_KEY:
            optimizer_params[OPTIMIZER_BATCH_SIZE_ARG] = value
        elif key == OPTIMIZER_REWRITE_POST_JOIN_BATCH_INFERENCE_KEY:
            optimizer_params[OPTIMIZER_REWRITE_POST_JOIN_BATCH_INFERENCE_ARG] = value
        else:
            worker_params[key] = value

    cmd = [
        str(systest_path),
        "--log-path", str(log_path),
        "--show-query-performance",
        "-t", query,
        "-n", "1",
    ]
    for key, value in optimizer_params.items():
        cmd.extend(["--optimizer", f"{key}={format_value(value)}"])

    cmd.append("--")
    for key, value in worker_params.items():
        cmd.append(f"--{key}={format_value(value)}")
    return cmd


def parse_query_performance_duration_us(output: str) -> float | None:
    normalized = ANSI_ESCAPE_PATTERN.sub("", output)
    matches = QUERY_PERFORMANCE_PATTERN.findall(normalized)
    if not matches:
        return None
    return sum(float(match) for match in matches) * 1_000_000.0


def write_query_performance(rep_dir: Path, duration_us: float) -> None:
    payload = {
        "end_to_end_duration_us": duration_us,
        "source": "--show-query-performance",
    }
    performance_path = rep_dir / SYSTEST_PERFORMANCE_FILE
    performance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def safe_move(src: Path, dest_dir: Path) -> None:
    dest = dest_dir / src.name
    if dest.exists():
        stem = src.stem
        suffix = src.suffix
        counter = 1
        while True:
            candidate = dest_dir / f"{stem}__{counter}{suffix}"
            if not candidate.exists():
                dest = candidate
                break
            counter += 1
    shutil.move(str(src), str(dest))


def scrape_url(url: str) -> str | None:
    """Fetch a URL's text body (e.g. TorchServe's :8082/metrics), or None on failure."""
    try:
        with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310 - operator-provided local URL
            return response.read().decode("utf-8", errors="ignore")
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"TorchServe metrics scrape failed ({url}): {exc}", file=sys.stderr)
        return None


def snapshot_files(directory: Path, pattern: str) -> Dict[Path, float]:
    if not directory.exists():
        return {}
    snapshot: Dict[Path, float] = {}
    for path in directory.glob(pattern):
        try:
            snapshot[path] = path.stat().st_mtime
        except FileNotFoundError:
            continue
    return snapshot


def collect_artifacts(
        targets: List[Tuple[Path, str]],
        dest_dir: Path,
        before: List[Dict[Path, float]],
) -> None:
    """Move files that appeared/changed since `before` into dest_dir.

    `targets` is a list of (directory, glob) pairs, aligned with `before`.
    """
    for (directory, pattern), before_snapshot in zip(targets, before):
        after = snapshot_files(directory, pattern)
        for path, mtime in after.items():
            if path not in before_snapshot or mtime > before_snapshot[path]:
                safe_move(path, dest_dir)


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_results = root / "scripts" / "benchmarking" / "e2e" / "results"
    default_systest = root / "cmake-build-release" / "nes-systests" / "systest" / "systest"

    parser = argparse.ArgumentParser(
        description="Run nes-systests queries across inference config combinations.",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        required=True,
        help="Query specs, e.g. inference/benchmark/solar-power/SolarPower.test:01",
    )
    parser.add_argument(
        "--inference-config",
        required=True,
        type=Path,
        help="YAML config with parameter lists.",
    )
    parser.add_argument(
        "--systest-path",
        type=Path,
        default=default_systest,
        help=f"Path to systest binary (default: {default_systest}).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results,
        help=f"Directory to write results (default: {default_results}).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=REPETITIONS_DEFAULT,
        help=f"Number of repetitions per combination (default: {REPETITIONS_DEFAULT}).",
    )
    parser.add_argument(
        "--torchserve-metrics-url",
        type=str,
        default=None,
        help="If set, scrape this TorchServe Prometheus URL (e.g. http://127.0.0.1:8082/metrics) "
             "before and after each run, storing the snapshots in the rep dir for "
             "process_torchserve_results.py to diff.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing systest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()

    systest_path = args.systest_path
    if not systest_path.exists():
        print(f"systest binary not found: {systest_path}", file=sys.stderr)
        return 1

    inference_config = normalize_inference_config(load_yaml(args.inference_config))

    try:
        resolved_queries = [resolve_query(query, root) for query in args.queries]
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    results_base = args.results_dir
    if not results_base.is_absolute():
        results_base = root / results_base
    results_base = results_base.resolve()
    if results_base.exists() and not results_base.is_dir():
        print(f"Results path exists and is not a directory: {results_base}", file=sys.stderr)
        return 1
    results_base.mkdir(parents=True, exist_ok=True)
    # systest writes its log here during the run; each rep preserves it as <rep_dir>/_run.log so the
    # pipelined plan dump (PipeliningPhase) survives for process_results.py to identify the
    # model-inference pipeline.
    log_scratch = results_base / RUN_LOG_SCRATCH

    query_specs = []
    for query in resolved_queries:
        query_path, query_suffix = split_query(query)
        query_name = sanitize_name(Path(query_path).name + query_suffix)
        query_specs.append((query, query_name))

    all_combinations = list(expand_combinations(inference_config))
    pending_combinations = []
    for combo in all_combinations:
        combo_label = combination_name(combo)
        combo_dir = results_base / combo_label
        if combo_dir.exists() and not combo_dir.is_dir():
            print(f"Combination results path exists and is not a directory: {combo_dir}", file=sys.stderr)
            return 1

        combo_complete = True
        for _, query_name in query_specs:
            if missing_repetitions(combo_dir / query_name, args.repetitions):
                combo_complete = False
                break
        if not combo_complete:
            pending_combinations.append(combo)

    skipped_count = len(all_combinations) - len(pending_combinations)
    if skipped_count > 0:
        print(f"Skipping {skipped_count} already processed combination(s) from {results_base}.")
    if not pending_combinations:
        print("All combinations are already processed. Nothing to run.")
        return 0

    systest_dir = root / "cmake-build-release" / "nes-systests" / "systest"
    build_dir = root / "cmake-build-release" / "nes-systests"

    # Where systest writes artifacts (subprocess cwd = systest_dir):
    #   trace_*.json / systest JSON     -> systest_dir
    #   stray logs                      -> build_dir / systest_dir (the run log itself goes to --log-path)
    #   torchserve_timings.csv          -> systest_dir/results (HttpSink log_path 'results/...' relative to cwd)
    #   latency_timings.csv             -> systest_dir/results (LatencySink log_path 'results/...' relative to cwd)
    artifact_targets: List[Tuple[Path, str]] = [
        (systest_dir, "*.json"),
        (build_dir, "*.log"),
        (systest_dir, "*.log"),
        (systest_dir / "results", TORCHSERVE_TIMINGS_FILE),
        (systest_dir / "results", LATENCY_TIMINGS_FILE),
    ]

    for combo in pending_combinations:
        combo_label = combination_name(combo)
        combo_dir = results_base / combo_label

        params = dict(combo)

        for query, query_name in query_specs:
            query_dir = combo_dir / query_name
            missing_query_repetitions = missing_repetitions(query_dir, args.repetitions)
            if not missing_query_repetitions:
                print(f"Skipping already processed query: {combo_label}/{query_name}")
                continue

            for repetition in missing_query_repetitions:
                rep_dir = query_dir / f"rep-{repetition:02d}"

                cmd = build_command(systest_path, query, params, log_path=log_scratch)

                print(" ".join(cmd))
                if args.dry_run:
                    continue

                total_attempts = QUERY_RETRIES + 1
                result: subprocess.CompletedProcess[str] | None = None
                before_snapshots: List[Dict[Path, float]] | None = None
                duration_us: float | None = None
                metrics_start_text: str | None = None
                metrics_end_text: str | None = None
                for attempt in range(1, total_attempts + 1):
                    before_snapshots = [
                        snapshot_files(directory, pattern) for directory, pattern in artifact_targets
                    ]

                    # Snapshot TorchServe's cumulative counters just before the run; diffed
                    # against the post-run snapshot to get per-rep inference/queue latency.
                    if args.torchserve_metrics_url:
                        metrics_start_text = scrape_url(args.torchserve_metrics_url)

                    result = subprocess.run(
                        cmd,
                        check=False,
                        cwd=systest_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    if args.torchserve_metrics_url:
                        metrics_end_text = scrape_url(args.torchserve_metrics_url)
                    if result.stdout:
                        print(result.stdout, end="")
                    if result.returncode == 0:
                        duration_us = parse_query_performance_duration_us(result.stdout)
                        break

                    if attempt < total_attempts:
                        print(
                            f"systest failed with exit code {result.returncode} "
                            f"(attempt {attempt}/{total_attempts}). Retrying...",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"systest failed with exit code {result.returncode} "
                            f"after {QUERY_RETRIES} retries. Terminating run.",
                            file=sys.stderr,
                        )
                        return result.returncode

                if result is None or result.returncode != 0 or before_snapshots is None:
                    print("systest run ended without a successful attempt.", file=sys.stderr)
                    return 1

                rep_dir.mkdir(parents=True, exist_ok=True)

                collect_artifacts(artifact_targets, rep_dir, before_snapshots)

                if log_scratch.exists():
                    safe_move(log_scratch, rep_dir)

                if args.torchserve_metrics_url:
                    if metrics_start_text is not None:
                        (rep_dir / TORCHSERVE_METRICS_START_FILE).write_text(metrics_start_text, encoding="utf-8")
                    if metrics_end_text is not None:
                        (rep_dir / TORCHSERVE_METRICS_END_FILE).write_text(metrics_end_text, encoding="utf-8")

                if duration_us is None:
                    print(
                        "Could not parse --show-query-performance runtime from systest output; "
                        f"no {SYSTEST_PERFORMANCE_FILE} written for {rep_dir}.",
                        file=sys.stderr,
                    )
                else:
                    write_query_performance(rep_dir, duration_us)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
