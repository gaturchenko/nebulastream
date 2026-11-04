#!/usr/bin/env python3
"""Profile per-invocation latency of an ONNX model across batch sizes with OpenVINO.

Mirrors nes-inference's ModelProfiler (reshape to batch B, synthetic floats in [0, 1), warm-up
invocations, median of the measured ones) for builds whose log level compiles out the optimizer's
own profile lines.
"""

import argparse
import statistics
import time

import numpy as np
import openvino as ov


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    core = ov.Core()
    model = core.read_model(args.model)
    sample_shape = list(model.input(0).get_partial_shape().get_min_shape())[1:]
    print(f"input {model.input(0).get_any_name()} per-sample shape {sample_shape}, {args.threads} thread(s)")
    print(f"{'batch':>6} {'median_us':>11} {'p95_us':>9} {'us/record':>10} {'records/s':>10}")

    rng = np.random.default_rng(42)
    batch_size = 1
    while batch_size <= args.max_batch_size:
        model = core.read_model(args.model)
        model.reshape([batch_size, *sample_shape])
        compiled = core.compile_model(
            model, "CPU", {"INFERENCE_NUM_THREADS": args.threads, "NUM_STREAMS": 1, "PERFORMANCE_HINT": "LATENCY"}
        )
        request = compiled.create_infer_request()
        data = rng.random((batch_size, *sample_shape), dtype=np.float32)
        for _ in range(args.warmup):
            request.infer({0: data})
        latencies = []
        for _ in range(args.iterations):
            start = time.perf_counter_ns()
            request.infer({0: data})
            latencies.append((time.perf_counter_ns() - start) / 1000)
        median = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[-1]
        print(f"{batch_size:>6} {median:>11.1f} {p95:>9.1f} {median / batch_size:>10.1f} {batch_size * 1e6 / median:>10.0f}")
        batch_size *= 2


if __name__ == "__main__":
    main()
