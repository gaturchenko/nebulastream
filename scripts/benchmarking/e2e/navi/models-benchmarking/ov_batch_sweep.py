#!/usr/bin/env python3
"""
ov_batch_sweep.py — OpenVINO batch-size scaling sweep, faithful to the NebulaStream (NES) setup.

Purpose
-------
Measure how per-item latency / throughput scale as batch size grows from 1..8, driving the
OpenVINO runtime API *exactly* the way NES does, so the numbers reflect what the "squash N
chained inference operators into one batch operator" optimization would actually see.

NES execution model (verified against OpenVINO 2025.3 source):
  - Each inference operator is thread-local: one compiled model + one infer request per NES
    worker thread. Concurrency across streams comes from NES, not from OpenVINO.
  - NUM_STREAMS=0  -> CPU plugin skips its stream auto-scheduler (plugin.cpp get_performance_streams
    else-branch) and dispatches inference INLINE on the calling thread via CPUStreamsExecutor::Defer.
    No dedicated OpenVINO stream/host threads are spawned.
  - IMPORTANT quirk: with NUM_STREAMS=0 the plugin rebuilds the stream-executor config from the
    stream count alone, so INFERENCE_NUM_THREADS is DROPPED (it always reads back as 0). Execution
    is single-threaded anyway because the inline Defer path is bound to the calling thread. So the
    NES "0 streams, 1 thread" pair is really "0 streams" doing the work; the thread count is a no-op.
  - Because execution is single-threaded, this sweep cleanly isolates the *arithmetic-intensity*
    benefit of batching (SIMD occupancy, weight cache reuse, GEMV->GEMM) from any multi-threading
    or OpenVINO async-stream effects. That is precisely the effect the batch operator must justify.

Defaults therefore mirror NES: streams=0, threads=1 (ignored but set for intent), pinning off,
synchronous infer, a single infer request.

Usage
-----
  python ov_batch_sweep.py --model path/to/model.xml            # or .onnx
  python ov_batch_sweep.py --model m.onnx --batches 1-8 --shape 3,224,224 --csv seg.csv
  python ov_batch_sweep.py --model kws.xml --streams 1 --threads 1 --csv proxy.csv   # streams=1 proxy

Output: a per-batch table on stdout plus a CSV. Feed the CSV to plot_batch_sweep.py.
"""

import argparse
import csv
import statistics
import sys
import threading
import time
from typing import Dict, List, Optional

import numpy as np

try:
    import openvino as ov
    from openvino import properties
    from openvino.properties import hint as hints
except Exception as exc:  # pragma: no cover
    print(f"ERROR: could not import openvino: {exc}", file=sys.stderr)
    sys.exit(1)

try:
    import psutil

    _PROC = psutil.Process()
except Exception:  # psutil is optional; thread counting degrades gracefully
    psutil = None
    _PROC = None


def parse_batches(spec: str) -> List[int]:
    """Accept '1-8', '1,2,4,8', or a single int."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def parse_shape(spec: Optional[str]) -> Optional[List[int]]:
    if not spec:
        return None
    return [int(x) for x in spec.replace("x", ",").split(",") if x.strip()]


def build_static_shapes(model, batch: int, batch_dim: int, override: Optional[List[int]]) -> Dict[str, list]:
    """Return {input_name: [dims...]} with the batch dim set to `batch`.

    - If `override` is given it is the per-input shape WITHOUT the batch dim (e.g. 3,224,224),
      and the batch is inserted at position `batch_dim`.
    - Otherwise the model's own partial shape is used; every non-batch dim must already be static.
    """
    shapes: Dict[str, list] = {}
    for inp in model.inputs:
        name = inp.get_any_name()
        if override is not None:
            dims = list(override)
            dims.insert(batch_dim, batch)
        else:
            ps = inp.get_partial_shape()
            dims = []
            for i, d in enumerate(ps):
                if i == batch_dim:
                    dims.append(batch)
                elif d.is_static:
                    dims.append(d.get_length())
                else:
                    raise ValueError(
                        f"Input '{name}' has a dynamic non-batch dim at position {i}. "
                        f"Pass --shape (dims without batch), e.g. --shape 3,224,224."
                    )
        shapes[name] = dims
    return shapes


def make_inputs(compiled) -> Dict[str, np.ndarray]:
    """Random inputs matching each compiled-model input's shape and element type."""
    feed = {}
    for port in compiled.inputs:
        shape = list(port.get_shape())
        ov_dtype = port.get_element_type()
        np_dtype = ov_dtype.to_dtype() if hasattr(ov_dtype, "to_dtype") else np.float32
        if np.issubdtype(np_dtype, np.floating):
            data = np.random.rand(*shape).astype(np_dtype)
        else:
            data = np.random.randint(0, 8, size=shape).astype(np_dtype)
        feed[port.get_any_name()] = data
    return feed


class ThreadSampler:
    """Background sampler recording the max OS-thread count of this process during a window."""

    def __init__(self, interval_s: float = 0.002):
        self.interval = interval_s
        self._stop = threading.Event()
        self.max_threads = 0
        self._t: Optional[threading.Thread] = None

    def _run(self):
        while not self._stop.is_set():
            if _PROC is not None:
                self.max_threads = max(self.max_threads, _PROC.num_threads())
            self._stop.wait(self.interval)

    def __enter__(self):
        if _PROC is not None:
            self.max_threads = _PROC.num_threads()
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t is not None:
            self._t.join()


def run_batch(core, model_path, batch, args, baseline_per_item):
    core_model = core.read_model(model_path)
    shapes = build_static_shapes(core_model, batch, args.batch_dim, parse_shape(args.shape))
    core_model.reshape(shapes)

    cfg = {
        properties.num_streams: args.streams,
        properties.inference_num_threads: args.threads,
        hints.enable_cpu_pinning: args.pin,
    }
    if args.hint != "none":
        cfg[hints.performance_mode] = {
            "latency": hints.PerformanceMode.LATENCY,
            "throughput": hints.PerformanceMode.THROUGHPUT,
        }[args.hint]

    threads_before = _PROC.num_threads() if _PROC else -1
    compiled = core.compile_model(core_model, args.device, cfg)
    resolved_streams = compiled.get_property(properties.num_streams)
    resolved_threads = compiled.get_property(properties.inference_num_threads)

    request = compiled.create_infer_request()
    feed = make_inputs(compiled)

    for _ in range(args.warmup):
        request.infer(feed)

    # Correctness guard: confirm the model actually processed `batch` items. A model exported with a
    # hardcoded batch (or a Reshape op that pins the batch dim) can silently collapse the batch to 1
    # while reshape() succeeds and latency looks great — a false "batching win". Catch it here.
    out_shape = list(request.get_output_tensor(0).shape)
    out_batch = out_shape[args.batch_dim] if len(out_shape) > args.batch_dim else None
    batch_ok = (out_batch == batch)

    latencies_ms: List[float] = []
    with ThreadSampler() as sampler:
        for _ in range(args.niter):
            t0 = time.perf_counter_ns()
            request.infer(feed)
            latencies_ms.append((time.perf_counter_ns() - t0) / 1e6)
    max_threads = sampler.max_threads

    mean_lat = statistics.fmean(latencies_ms)
    p50 = statistics.median(latencies_ms)
    p90 = sorted(latencies_ms)[int(0.90 * (len(latencies_ms) - 1))]
    per_item = mean_lat / batch
    throughput = batch * 1000.0 / mean_lat  # items/s
    speedup = (baseline_per_item / per_item) if baseline_per_item else 1.0

    return {
        "batch": batch,
        "out_batch": out_batch,
        "batch_ok": batch_ok,
        "streams_set": args.streams,
        "threads_set": args.threads,
        "streams_resolved": resolved_streams,
        "threads_resolved": resolved_threads,
        "os_threads_before_compile": threads_before,
        "os_threads_max_during_run": max_threads,
        "mean_latency_ms": round(mean_lat, 4),
        "p50_latency_ms": round(p50, 4),
        "p90_latency_ms": round(p90, 4),
        "per_item_latency_ms": round(per_item, 4),
        "throughput_items_s": round(throughput, 2),
        "per_item_speedup_vs_b1": round(speedup, 3),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Path to OpenVINO IR (.xml) or ONNX (.onnx) model")
    ap.add_argument("--batches", default="1-8", help="Batch sizes: '1-8', '1,2,4,8', or a single int (default 1-8)")
    ap.add_argument("--shape", default=None,
                    help="Per-input shape WITHOUT batch dim, e.g. '3,224,224'. Needed if the model has "
                         "dynamic non-batch dims. Applied to every input.")
    ap.add_argument("--batch-dim", type=int, default=0, help="Index of the batch dimension (default 0)")
    ap.add_argument("--device", default="CPU", help="Target device (default CPU)")
    # NES-faithful defaults:
    ap.add_argument("--streams", type=int, default=0, help="NUM_STREAMS (NES uses 0; default 0)")
    ap.add_argument("--threads", type=int, default=1,
                    help="INFERENCE_NUM_THREADS (default 1; note: ignored by the plugin when streams=0)")
    ap.add_argument("--pin", action="store_true", help="Enable CPU pinning (default off, matching NES)")
    ap.add_argument("--hint", choices=["none", "latency", "throughput"], default="none",
                    help="Performance hint. 'none' (default) lets explicit streams/threads apply, like NES.")
    ap.add_argument("--niter", type=int, default=300, help="Timed iterations per batch (default 300)")
    ap.add_argument("--warmup", type=int, default=50, help="Warmup iterations per batch (default 50)")
    ap.add_argument("--csv", default=None, help="Write results to this CSV path")
    args = ap.parse_args()

    core = ov.Core()
    batches = parse_batches(args.batches)

    print(f"Model         : {args.model}")
    print(f"Device        : {args.device}")
    print(f"OpenVINO      : {ov.__version__}")
    print(f"Config        : NUM_STREAMS={args.streams}  INFERENCE_NUM_THREADS={args.threads}  "
          f"CPU_PINNING={'on' if args.pin else 'off'}  hint={args.hint}")
    print(f"Batches       : {batches}   niter={args.niter} warmup={args.warmup}")
    print("-" * 118)

    rows = []
    baseline_per_item = None
    for b in batches:
        row = run_batch(core, args.model, b, args, baseline_per_item)
        if baseline_per_item is None:
            baseline_per_item = row["per_item_latency_ms"]
            row["per_item_speedup_vs_b1"] = 1.0
        rows.append(row)
        flag = "" if row["batch_ok"] else f"  <== BATCH COLLAPSED! out_batch={row['out_batch']} (results INVALID)"
        print(
            f"B={row['batch']:<2}  "
            f"lat={row['mean_latency_ms']:8.3f}ms  "
            f"p90={row['p90_latency_ms']:8.3f}ms  "
            f"per_item={row['per_item_latency_ms']:8.4f}ms  "
            f"thr={row['throughput_items_s']:9.1f} it/s  "
            f"speedup={row['per_item_speedup_vs_b1']:5.2f}x  "
            f"[streams->{row['streams_resolved']} threads->{row['threads_resolved']} "
            f"os_thr_max={row['os_threads_max_during_run']}]{flag}"
        )

    if any(not r["batch_ok"] for r in rows):
        print("-" * 118)
        print("!!! WARNING: the model did NOT process the full batch for one or more sizes (batch collapsed\n"
              "    to a smaller dim). Speedup/throughput numbers above are MEANINGLESS. The model has a\n"
              "    hardcoded/pinned batch dimension — re-export it with a dynamic batch (input dim0 = -1).")

    if rows and rows[0]["streams_resolved"] == 0 and rows[0]["threads_resolved"] == 0:
        print("-" * 118)
        print("NOTE: streams resolved to 0 -> inline single-threaded execution on the calling thread "
              "(the NES path).\n      INFERENCE_NUM_THREADS reads back as 0 by design in this path; "
              "execution is single-threaded regardless.")

    # Interpretation hint
    if len(rows) >= 2:
        first, last = rows[0], rows[-1]
        gain = first["per_item_latency_ms"] / last["per_item_latency_ms"]
        verdict = ("STRONG batching benefit" if gain >= 1.5 else
                   "MODEST batching benefit" if gain >= 1.15 else
                   "NEGLIGIBLE / linear scaling (batching not worth it for this model)")
        print("-" * 118)
        print(f"Per-item latency B{first['batch']} -> B{last['batch']}: "
              f"{first['per_item_latency_ms']:.4f} -> {last['per_item_latency_ms']:.4f} ms  "
              f"({gain:.2f}x)  => {verdict}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
