#!/usr/bin/env python3
"""Write a CWRU-like stream of arbitrary length for sarad.test, which needs more than the 10 s of X228.

Repeats the vibration values of a CWRU file with 48 samples per millisecond of event time, so that a 200 ms window holds exactly
the 9600 samples of SARAD. The invocation cost of the model does not depend on the values; they only have to be realistic.
"""

import argparse
from pathlib import Path

SAMPLES_PER_MS = 48


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=60 * 48_000)
    args = parser.parse_args()

    values = [line.rstrip("\n").split(",")[1] for line in args.source.open(encoding="utf-8")]
    with args.output.open("w", encoding="utf-8") as handle:
        for row in range(args.rows):
            handle.write(f"{row // SAMPLES_PER_MS},{values[row % len(values)]}\n")


if __name__ == "__main__":
    main()
