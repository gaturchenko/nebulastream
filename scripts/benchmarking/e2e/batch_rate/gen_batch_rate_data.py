#!/usr/bin/env python3
"""Write the synthetic series for the source-rate x batch-size micro-benchmark (nbeats_batch_rate.test).

One sample every --step-ms of event time, so a window of 75 * step-ms holds exactly the 75 values
nbeats_size_300_stride_30.onnx consumes. The values are a noisy daily-like sine; their content does
not matter for the benchmark, only that the model sees realistic floats.
"""

import argparse
import math
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--step-ms", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in range(args.rows):
            value = 0.5 + 0.4 * math.sin(2 * math.pi * row / 900) + rng.gauss(0, 0.05)
            handle.write(f"{row * args.step_ms},{value:.6f}\n")


if __name__ == "__main__":
    main()
