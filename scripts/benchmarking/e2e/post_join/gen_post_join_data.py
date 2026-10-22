#!/usr/bin/env python3
"""Cut k distinct audio streams for the post-join experiment from the MUSAN recordings.

Stream i replays recording i mod R from an offset of (i div R) * --offset-step-s seconds, with its timestamps rebased to
start at 0, so all streams cover the same event time while no two of them carry the same samples. Distinct payloads
matter: with batch deduplication enabled (which the dynamic-batch configurations need), identical streams would collapse
into one inference per joined record and make post-join batching win trivially.

    ./gen_post_join_data.py --recordings datasets/musan/musan-us-gov-{0242,0096,0021}.csv \
        --streams 8 --duration-s 60 --output-dir datasets/post_join
"""

import argparse
import sys
from pathlib import Path

SAMPLES_PER_MS = 16


def cut(recording: Path, offset_ms: int, duration_ms: int, output: Path) -> int:
    """Copies the rows of `recording` with timestamps in [offset_ms, offset_ms + duration_ms), rebased to 0."""
    rows = 0
    with recording.open() as source, output.open("w") as sink:
        for line in source:
            timestamp, sample = line.rstrip("\n").split(",")
            ms = int(timestamp)
            if ms < offset_ms:
                continue
            if ms >= offset_ms + duration_ms:
                break
            sink.write(f"{ms - offset_ms},{sample}\n")
            rows += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recordings", type=Path, nargs="+", required=True, help="MUSAN CSVs (timestamp_ms,sample).")
    parser.add_argument("--streams", type=int, default=8, help="Number of streams to cut, i.e. the largest k.")
    parser.add_argument("--duration-s", type=int, default=60, help="Event time every stream covers.")
    parser.add_argument("--offset-step-s", type=int, default=120, help="Offset between streams cut from one recording.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    duration_ms = args.duration_s * 1000
    for i in range(args.streams):
        recording = args.recordings[i % len(args.recordings)]
        offset_ms = (i // len(args.recordings)) * args.offset_step_s * 1000
        output = args.output_dir / f"stream{i}.csv"
        expected = duration_ms * SAMPLES_PER_MS
        if output.exists() and sum(1 for _ in output.open()) == expected:
            print(f"{output}: up to date")
            continue
        rows = cut(recording, offset_ms, duration_ms, output)
        if rows != expected:
            sys.exit(f"{output}: {recording.name} from {offset_ms} ms has {rows} rows, expected {expected}; is it long enough?")
        print(f"{output}: {recording.name} from {offset_ms / 1000:.0f} s, {rows} rows")


if __name__ == "__main__":
    main()
