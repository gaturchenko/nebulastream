#!/usr/bin/env python3
"""Run nes-systests .test queries across inference config combinations."""

from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, Iterable, List, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - runtime environment dependent
    print("PyYAML is required. Install it with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

REPETITIONS_DEFAULT = 5


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


def expand_combinations(config: Dict[str, object]) -> Iterable[Dict[str, object]]:
    if not config:
        yield {}
        return
    keys = list(config.keys())
    values = [normalize_values(config[key]) for key in keys]
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


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


def combination_name(combo: Dict[str, object]) -> str:
    if not combo:
        return "default"
    parts = [f"{key}={format_value(value)}" for key, value in combo.items()]
    return sanitize_name("__".join(parts))


def build_command(
    systest_path: Path,
    query: str,
    params: Dict[str, object],
) -> List[str]:
    cmd = [str(systest_path), "-t", query, "--"]
    for key, value in params.items():
        cmd.append(f"--{key}={format_value(value)}")
    return cmd


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
    systest_dir: Path,
    build_dir: Path,
    dest_dir: Path,
    json_before: Dict[Path, float],
    log_before_build: Dict[Path, float],
    log_before_systest: Dict[Path, float],
) -> None:
    json_after = snapshot_files(systest_dir, "*.json")
    log_after_build = snapshot_files(build_dir, "*.log")
    log_after_systest = snapshot_files(systest_dir, "*.log")

    def changed_files(after: Dict[Path, float], before: Dict[Path, float]) -> List[Path]:
        changed = []
        for path, mtime in after.items():
            if path not in before or mtime > before[path]:
                changed.append(path)
        return changed

    for path in (
        changed_files(json_after, json_before)
        + changed_files(log_after_build, log_before_build)
        + changed_files(log_after_systest, log_before_systest)
    ):
        safe_move(path, dest_dir)


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_config = root / "scripts" / "benchmarking" / "e2e" / "config" / "nes_default.yaml"
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
        "--default-overrides",
        type=Path,
        help="Optional YAML file to override nes_default.yaml parameters.",
    )
    parser.add_argument(
        "--default-config",
        type=Path,
        default=default_config,
        help=f"Default config path (default: {default_config}).",
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

    inference_config = load_yaml(args.inference_config)
    # default_config = load_yaml(args.default_config)
    # if args.default_overrides:
    #     overrides = load_yaml(args.default_overrides)
    #     default_config.update(overrides)

    try:
        resolved_queries = [resolve_query(query, root) for query in args.queries]
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    results_base = args.results_dir
    if not results_base.is_absolute():
        results_base = root / results_base
    results_base = results_base.resolve()
    results_base_created = False

    systest_dir = root / "cmake-build-release" / "nes-systests" / "systest"
    build_dir = root / "cmake-build-release" / "nes-systests"

    for combo in expand_combinations(inference_config):
        combo_label = combination_name(combo)
        combo_dir = results_base / combo_label
        if combo_dir.exists():
            print(
                f"Results directory already exists: {combo_dir}. Remove it or choose --results-dir.",
                file=sys.stderr,
            )
            return 1

        params = dict()
        # params = dict(default_config)
        params.update(combo)

        for query in resolved_queries:
            query_path, query_suffix = split_query(query)
            query_name = sanitize_name(Path(query_path).name + query_suffix)
            query_dir = combo_dir / query_name

            for repetition in range(1, args.repetitions + 1):
                rep_dir = query_dir / f"rep-{repetition:02d}"

                json_before = snapshot_files(systest_dir, "*.json")
                log_before_build = snapshot_files(build_dir, "*.log")
                log_before_systest = snapshot_files(systest_dir, "*.log")
                cmd = build_command(systest_path, query, params)

                print(" ".join(cmd))
                if args.dry_run:
                    continue

                result = subprocess.run(cmd, check=False, cwd=systest_dir)
                if result.returncode != 0:
                    print(
                        f"systest failed with exit code {result.returncode}.",
                        file=sys.stderr,
                    )
                    return result.returncode

                if not results_base_created:
                    results_base.mkdir(parents=True, exist_ok=True)
                    results_base_created = True
                if not combo_dir.exists():
                    combo_dir.mkdir(parents=True, exist_ok=False)
                if not query_dir.exists():
                    query_dir.mkdir()
                if not rep_dir.exists():
                    rep_dir.mkdir()

                collect_artifacts(
                    systest_dir,
                    build_dir,
                    rep_dir,
                    json_before,
                    log_before_build,
                    log_before_systest,
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
