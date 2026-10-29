#!/usr/bin/env python3
"""Model memory on the Pi (revision §8): what an inference operator's runtime sessions cost, with one and with two models.

Every inference operator sets up one runtime session per worker thread when its pipeline starts, each with a private compiled
model or all of them sharing one. This measures that cost two ways, for each evaluated model and for one query running two:

  assess   the optimizer's own assessment (inference_memory_strategy=OPTIMIZER_CHOOSES): the worker sets the sessions up once
           more, apart from the query, and reports the resident-set growth of the first session and of each additional one,
           private and shared. Parsed from the systest log.
  private  the query as deployed with a private compiled model per session: the process's resident set while it runs, minus
  shared   that of the same query without a model (the workload's baseline section), which carries the same source, buffer
           pool and preprocessing. The sessions are set up before the first tuple, so the difference is theirs.

Each run is one systest process with one query, so nothing carries over between runs. The worker has 4 threads (the Pi's cores)
and OpenVINO 1 stream x 4 threads, as in every other Pi experiment; the queries invoke their model with single tuples.

    ./scripts/run_model_memory.py [reps]                       # all cells, 3 repetitions by default
    ./scripts/run_model_memory.py 1 --workloads waymo --modes assess private
    ./scripts/process_model_memory.py                          # afterwards

Run from ~/navi with the benchmarking venv's Python. Layout it expects (deploy from scripts/benchmarking/e2e/model_memory/):
    queries/model-memory/*.test.tmpl        the .test files are rendered next to them
    scripts/{run_model_memory.py,process_model_memory.py,reset_swap.sh (from scripts/benchmarking/e2e/)}
    models-benchmarking/models/{squeezedet,segmenter,sarad,mhatt-rnn,nbeats,nanodet}/...   (see MODELS)
    datasets/{waymo,cwru-bearing,musan,synthetic,osu-rgb}/...                              (see DATASETS)

Results go to results/model_memory: runs.csv (one row per run), assess.csv (one row per assessed model), models.csv (model
sizes on disk) and raw/ (each run's log, output and resident-set samples). Repetitions are interleaved, and a run that already
passed is not repeated, so an interrupted sweep resumes where it stopped.
"""

import argparse
import csv
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Short inputs: the sessions are set up before the first tuple, so the data only has to keep the query running long enough to
# sample its resident set. Name -> (source file under datasets/, lines, seconds of data at the template's rate).
DATASETS = {
    "waymo": ("waymo/camera_5.csv", 80, 16),
    "cwru": ("cwru-bearing/X228_FE_time.csv", 480_000, 10),
    "musan": ("musan/musan-us-gov-0242.csv", 160_000, 10),
    "series": ("synthetic/batch_rate.csv", 10_000, 10),
    "osu": ("osu-rgb/otcbvs_osu_rgb.csv", 12, 12),
}

# Model -> path under models-benchmarking/models, as in the templates.
MODELS = {
    "squeezedet": "squeezedet/squeezedet.onnx",
    "segmenter": "segmenter/segmenter.onnx",
    "sarad": "sarad/sarad_medium.onnx",
    "mhatt_rnn": "mhatt-rnn/saved_model",
    "nbeats300": "nbeats/nbeats_size_300_stride_30.onnx",
    "nanodet": "nanodet/nanodet-plus-r101.onnx",
}

# Workload -> (template, dataset, sections). A section is (number, cell name, models it runs); section 1 is the baseline.
WORKLOADS = {
    "waymo": ("waymo.test.tmpl", "waymo", [
        (1, "waymo_baseline", []),
        (2, "squeezedet", ["squeezedet"]),
        (3, "segmenter", ["segmenter"]),
        (4, "squeezedet+segmenter", ["squeezedet", "segmenter"]),
    ]),
    "cwru": ("cwru.test.tmpl", "cwru", [(1, "cwru_baseline", []), (2, "sarad", ["sarad"])]),
    "musan": ("musan.test.tmpl", "musan", [(1, "musan_baseline", []), (2, "mhatt_rnn", ["mhatt_rnn"])]),
    "nbeats": ("nbeats.test.tmpl", "series", [(1, "series_baseline", []), (2, "nbeats300", ["nbeats300"])]),
    "osu": ("osu.test.tmpl", "osu", [(1, "osu_baseline", []), (2, "nanodet", ["nanodet"])]),
}

MODES = ["private", "shared", "assess"]
SHARE_OPTION = "worker.default_query_execution.inference.openvino_share_compiled_model"

# The optimizer's log lines (InferenceImplementationSelector.cpp); whitespace is matched loosely in case the logger wraps.
ASSESSMENT = re.compile(
    r"Inference memory for model '(?P<model>[^']+)' on worker '[^']+': first session (?P<first>\d+) B, each additional session "
    r"(?P<private>\d+) B private or (?P<shared>\d+) B sharing the compiled model; (?P<sessions>\d+) sessions need "
    r"(?P<private_total>\d+) B with a private compiled model each or (?P<shared_total>\d+) B sharing one".replace(" ", r"\s+"))
DEPLOYMENT = re.compile(r"Deploying model '(?P<model>[^']+)' on worker '[^']+' with one (?P<deployment>shared|private) compiled"
                        .replace(" ", r"\s+"))

RUN_FIELDS = ["rep", "workload", "section", "cell", "models", "mode", "rc", "passed", "timed_out", "wall_s", "data_s",
              "samples", "steady_rss_mb", "peak_rss_mb", "max_swap_mb", "mem_available_start_mb"]
ASSESS_FIELDS = ["rep", "workload", "cell", "model", "first_session_b", "additional_private_b", "additional_shared_b",
                 "sessions", "private_total_b", "shared_total_b", "deployed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reps", nargs="?", type=int, default=3)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="deployment root (default: current directory)")
    parser.add_argument("--workloads", nargs="+", choices=list(WORKLOADS), default=list(WORKLOADS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--venv", default="models-benchmarking/venv",
                        help="venv whose bin/ goes first on systest's PATH, for ovc (the MHAtt-RNN import)")
    parser.add_argument("--sample-s", type=float, default=0.2)
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--no-swap-reset", action="store_true", help="leave swap alone (e.g. off the Pi)")
    parser.add_argument("--dry-run", action="store_true", help="print the commands only")
    return parser.parse_args()


def prepare(root: Path, workloads: list[str]) -> None:
    """Writes the short datasets and renders the templates."""
    data_dir = root / "datasets" / "model_memory"
    data_dir.mkdir(parents=True, exist_ok=True)
    for workload in workloads:
        template, dataset, _ = WORKLOADS[workload]
        source, lines, _ = DATASETS[dataset]
        target = data_dir / f"{dataset}.csv"
        if not target.exists():
            with open(root / "datasets" / source) as src, open(target, "w") as dst:
                for _, line in zip(range(lines), src):
                    dst.write(line)
        tmpl = root / "queries" / "model-memory" / template
        tmpl.with_suffix("").write_text(tmpl.read_text().replace("@ROOT@", str(root)))


def model_bytes(path: Path) -> int:
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return path.stat().st_size


def write_models(root: Path, results: Path) -> None:
    with open(results / "models.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "path", "bytes"])
        for model, rel in MODELS.items():
            path = root / "models-benchmarking" / "models" / rel
            writer.writerow([model, rel, model_bytes(path) if path.exists() else ""])


def command(root: Path, test: Path, section: int, mode: str | None, log: Path) -> list[str]:
    cmd = [str(root / "bin" / "systest"),
           "--workingDir", str(root / "working-dir"),
           "--data", str(root / "datasets"),
           "--clusterConfig", str(root / "bin" / "config" / "single-node.yaml"),
           "--log-path", str(log),
           "-t", f"{test}:{section:02d}",
           "-n", "1"]
    if mode == "assess":
        cmd += ["--optimizer", "inference_memory_strategy=OPTIMIZER_CHOOSES", "--"]
    elif mode in ("private", "shared"):
        cmd += ["--optimizer", "inference_memory_strategy=FIXED", "--",
                f"--{SHARE_OPTION}={'true' if mode == 'shared' else 'false'}"]
    else:
        cmd += ["--"]
    return cmd


def status_kb(pid: int) -> dict[str, int] | None:
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    return {key: int(value.split()[0]) for key, value in (line.split(":", 1) for line in text.splitlines())
            if key in ("VmRSS", "VmHWM", "VmSwap")}


def mem_available_mb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return -1


def other_experiment_running() -> bool:
    """A systest process by name: matching "bin/systest" in command lines would also match any shell whose command mentions that
    path, including the one that started this script. The JVM and TorchServe only show in their command lines."""
    by_name = subprocess.run(["pgrep", "-x", "systest"], capture_output=True).returncode == 0
    by_command = subprocess.run(["pgrep", "-f", r"java .*flink-openvino-baseline.*\.jar|java .*org\.pytorch\.serve"],
                                capture_output=True).returncode == 0
    return by_name or by_command


def wait_for_idle_host() -> None:
    """Another experiment's systest, JVM or TorchServe would share the memory being measured."""
    waited = 0
    while other_experiment_running():
        if waited % 300 == 0:
            print(f"  waiting for another experiment to finish ({waited} s)", flush=True)
        time.sleep(10)
        waited += 10


def run(cmd: list[str], out: Path, samples_path: Path, env: dict, sample_s: float, timeout_s: float):
    started = time.monotonic()
    samples = []
    timed_out = False
    with open(out, "w") as stdout:
        proc = subprocess.Popen(cmd, stdout=stdout, stderr=subprocess.STDOUT, env=env)
        while proc.poll() is None:
            status = status_kb(proc.pid)
            if status:
                samples.append((time.monotonic() - started, status.get("VmRSS", 0), status.get("VmHWM", 0),
                                status.get("VmSwap", 0)))
            if time.monotonic() - started > timeout_s:
                proc.kill()
                proc.wait()
                timed_out = True
                break
            time.sleep(sample_s)
    wall = time.monotonic() - started
    with open(samples_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "rss_kb", "hwm_kb", "swap_kb"])
        writer.writerows((f"{t:.2f}", rss, hwm, swap) for t, rss, hwm, swap in samples)
    return proc.returncode, timed_out, wall, samples


def steady_rss_mb(samples, wall: float, data_s: float) -> float | None:
    """Median resident set over the second half of the data, ending two seconds before the process exits (a large model's
    teardown starts freeing a second or more before the exit). By then the sessions are set up and the small buffer pool is fully
    in use, so the resident set has levelled off."""
    window = [rss for t, rss, _, _ in samples if wall - data_s / 2 - 2 <= t <= wall - 2]
    return round(statistics.median(window) / 1024, 1) if window else None


def done(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with open(path) as f:
        return {(row["rep"], row["cell"], row["mode"]) for row in csv.DictReader(f) if row["passed"] == "1"}


def append(path: Path, fields: list[str], row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    results = root / "results" / "model_memory"
    raw = results / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (root / "working-dir").mkdir(exist_ok=True)
    env = dict(os.environ, PATH=f"{root / args.venv / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}")
    if not args.dry_run and not shutil.which("ovc", path=env["PATH"]):
        print(f"No ovc on PATH; the MHAtt-RNN import needs it (--venv {args.venv})", file=sys.stderr)
        return 1

    prepare(root, args.workloads)
    write_models(root, results)
    if not args.dry_run and not args.no_swap_reset:
        # Pages swapped out by an earlier experiment would come back during a run and inflate its resident set.
        if subprocess.run([str(root / "scripts" / "reset_swap.sh")]).returncode != 0:
            print("WARNING: could not reset swap", file=sys.stderr)

    cells = []
    for workload in args.workloads:
        _, dataset, sections = WORKLOADS[workload]
        for section, cell, models in sections:
            for mode in (["baseline"] if not models else args.modes):
                cells.append((workload, dataset, section, cell, models, mode))
    finished = done(results / "runs.csv")
    total = len(cells) * args.reps
    print(f"### model memory: {args.reps} reps x {len(cells)} cells, {sum(1 for _ in finished)} runs already passed", flush=True)

    count = 0
    for rep in range(1, args.reps + 1):
        for workload, dataset, section, cell, models, mode in cells:
            count += 1
            if (str(rep), cell, mode) in finished:
                continue
            test = root / "queries" / "model-memory" / WORKLOADS[workload][0].removesuffix(".tmpl")
            stem = f"{cell.replace('+', '_')}_{mode}_rep{rep}"
            log = raw / f"{stem}.log"
            cmd = command(root, test, section, None if mode == "baseline" else mode, log)
            print(f"### [{count}/{total}] rep {rep} {cell} {mode} ({time.strftime('%H:%M:%S')})", flush=True)
            if args.dry_run:
                print("   ", " ".join(cmd))
                continue
            wait_for_idle_host()
            available = mem_available_mb()
            rc, timed_out, wall, samples = run(cmd, raw / f"{stem}.out", raw / f"{stem}.rss.csv", env, args.sample_s,
                                               args.timeout_s)
            output = (raw / f"{stem}.out").read_text(errors="replace")
            passed = rc == 0 and not timed_out and "PASSED" in output
            data_s = DATASETS[dataset][2]
            steady = steady_rss_mb(samples, wall, data_s)
            peak = round(max((hwm for _, _, hwm, _ in samples), default=0) / 1024, 1)
            swap = round(max((s for _, _, _, s in samples), default=0) / 1024, 1)
            append(results / "runs.csv", RUN_FIELDS, {
                "rep": rep, "workload": workload, "section": section, "cell": cell, "models": "+".join(models), "mode": mode,
                "rc": rc, "passed": int(passed), "timed_out": int(timed_out), "wall_s": round(wall, 1), "data_s": data_s,
                "samples": len(samples), "steady_rss_mb": steady, "peak_rss_mb": peak, "max_swap_mb": swap,
                "mem_available_start_mb": available})
            print(f"    rc={rc} passed={int(passed)} wall={wall:.0f} s steady={steady} MB peak={peak} MB swap={swap} MB",
                  flush=True)

            if mode == "assess" and log.exists():
                text = log.read_text(errors="replace")
                deployed = {m["model"]: m["deployment"] for m in DEPLOYMENT.finditer(text)}
                assessed = list(ASSESSMENT.finditer(text))
                for m in assessed:
                    append(results / "assess.csv", ASSESS_FIELDS, {
                        "rep": rep, "workload": workload, "cell": cell, "model": m["model"],
                        "first_session_b": m["first"], "additional_private_b": m["private"],
                        "additional_shared_b": m["shared"], "sessions": m["sessions"],
                        "private_total_b": m["private_total"], "shared_total_b": m["shared_total"],
                        "deployed": deployed.get(m["model"], "")})
                if len(assessed) != len(models):
                    print(f"    !! {len(assessed)} memory assessments in the log for {len(models)} model(s)", flush=True)
    print(f"### model memory finished ({time.strftime('%H:%M:%S')})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
