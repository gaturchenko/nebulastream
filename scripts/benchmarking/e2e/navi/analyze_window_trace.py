#!/usr/bin/env python3
"""Where the Flink CWRU tail waits: per-window decomposition of trace_rep<r>.csv (+ GC log, vmstat)."""

import csv
import re
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results-flink-window-trace")
MS = 1e6
PARTS = ["source_lag", "build", "lane_wait", "inference", "to_sink"]


def load(path):
    with open(path) as f:
        rows = [{k: int(v) for k, v in r.items()} for r in csv.DictReader(f)]
    rows = [r for r in rows if r["sink_ns"] and r["infer_start_ns"]]
    t = {k: np.array([r[k] for r in rows], dtype=np.int64) for k in rows[0]}
    return t


def gc_pauses(path):
    pauses = []
    if not path.exists():
        return pauses
    pat = re.compile(r"\[(\d+)ns\].*\[gc\s*\] GC\(\d+\) (Pause [^0-9]*?).* ([0-9.]+)ms$")
    for line in path.read_text().splitlines():
        m = pat.search(line)
        if m:
            end = int(m.group(1))
            pauses.append((end - float(m.group(3)) * MS, end, m.group(2).strip(), float(m.group(3))))
    return pauses


def vmstat_runqueue(path):
    if not path.exists():
        return None
    rows = [l.split() for l in path.read_text().splitlines() if l.strip() and l.split()[0].isdigit()]
    return np.array([int(r[0]) for r in rows]), np.array([int(r[12]) + int(r[13]) for r in rows])


def lane_of(t, i):
    return int(t["lane"][i])


def report(rep):
    t = load(OUT / f"trace_rep{rep}.csv")
    n = len(t["seq"])
    parts = {
        "source_lag": t["arrival_ns"] - t["scheduled_ns"],   # not in e2e: the clock starts at arrival
        "build": t["emitted_ns"] - t["arrival_ns"],
        "lane_wait": t["infer_start_ns"] - t["emitted_ns"],
        "inference": t["infer_end_ns"] - t["infer_start_ns"],
        "to_sink": t["sink_ns"] - t["infer_end_ns"],
    }
    parts = {k: v / MS for k, v in parts.items()}
    e2e = (t["sink_ns"] - t["arrival_ns"]) / MS
    p50, p95, p99 = np.percentile(e2e, [50, 95, 99])
    tail = e2e >= p95

    print(f"\n=== rep {rep}: {n} windows, e2e P50 {p50:.1f}  P95 {p95:.1f}  P99 {p99:.1f} ms")
    print(f"{'part [ms]':<12}{'P50':>8}{'P95':>8}{'max':>8}{'tail mean':>11}{'share of tail e2e':>19}")
    for k in PARTS:
        v = parts[k]
        share = "" if k == "source_lag" else f"{v[tail].mean() / e2e[tail].mean():>18.0%}"
        print(f"{k:<12}{np.percentile(v, 50):>8.1f}{np.percentile(v, 95):>8.1f}{v.max():>8.1f}{v[tail].mean():>11.1f}{share}")

    since = (t["arrival_ns"] - t["arrival_ns"].min()) / 1e9
    for skip in (5, 10):
        steady = e2e[since >= skip]
        q = np.percentile(steady, [50, 95, 99])
        print(f"e2e without the first {skip:>2} s: n={len(steady)}  P50 {q[0]:.1f}  P95 {q[1]:.1f}  P99 {q[2]:.1f} ms")
    print("first 3.5 s, one line per window (t s, lane, lane_wait, inference, e2e ms):")
    first = np.argsort(t["arrival_ns"])[: int((since < 3.5).sum())]
    print("  " + "  ".join(f"{since[i]:.1f}/{lane_of(t, i)}/{parts['lane_wait'][i]:.0f}/{parts['inference'][i]:.0f}/{e2e[i]:.0f}"
                           for i in first))

    # Lane mechanics. For each window that waited, was its lane busy with an earlier window, and was
    # the OTHER lane free at that moment (round-robin pinned it behind a busy lane)?
    lane = t["lane"]
    lanes = sorted(set(lane.tolist()))
    print(f"windows per lane: {[int((lane == l).sum()) for l in lanes]}")
    order = np.argsort(t["emitted_ns"])
    busy_until = {l: 0 for l in lanes}
    waited, other_idle = 0, 0
    for i in order:
        if parts["lane_wait"][i] > 5:
            waited += 1
            others = [l for l in lanes if l != lane[i]]
            if any(busy_until[o] <= t["emitted_ns"][i] for o in others):
                other_idle += 1
        busy_until[lane[i]] = max(busy_until[lane[i]], t["infer_end_ns"][i])
    print(f"windows waiting >5 ms for their lane: {waited} ({waited / n:.1%}); of those, another lane was idle at "
          f"emit time: {other_idle}")

    # Inference slowdown from overlap: fraction of each inference during which another lane was also inferring.
    starts, ends = t["infer_start_ns"], t["infer_end_ns"]
    overlap = np.zeros(n)
    for i in range(n):
        o = np.clip(np.minimum(ends, ends[i]) - np.maximum(starts, starts[i]), 0, None)
        o[i] = 0
        overlap[i] = o.sum() / max(ends[i] - starts[i], 1)
    alone, shared = parts["inference"][overlap < 0.1], parts["inference"][overlap >= 0.5]
    print(f"inference alone (<10% overlap): n={len(alone)} P50 {np.median(alone):.1f} P95 {np.percentile(alone, 95):.1f}"
          if len(alone) else "inference alone: none")
    print(f"inference overlapped (>=50%):   n={len(shared)} P50 {np.median(shared):.1f} P95 {np.percentile(shared, 95):.1f}"
          if len(shared) else "inference overlapped: none")

    # Emission regularity: windows are due every 100 ms; bursts show up as short gaps.
    gaps = np.diff(np.sort(t["emitted_ns"])) / MS
    print(f"inter-emit gap: P1 {np.percentile(gaps, 1):.1f}  P50 {np.median(gaps):.1f}  P99 {np.percentile(gaps, 99):.1f} ms"
          f"  (<50 ms: {(gaps < 50).sum()})")

    # Tail clustering: do the tail windows come in runs of consecutive windows?
    tail_seq = np.sort(t["seq"][tail])
    runs = np.split(tail_seq, np.where(np.diff(tail_seq) > 1)[0] + 1)
    print(f"tail windows: {tail.sum()} in {len(runs)} runs, longest {max(len(r) for r in runs)}; "
          f"run starts (s): {[float(round((t['arrival_ns'][t['seq'] == r[0]][0] - t['arrival_ns'].min()) / 1e9, 1)) for r in runs][:12]}")

    # Per-window inference slow-downs that precede the tail: the largest inference in the 5 windows before each tail run.
    pauses = gc_pauses(OUT / f"gc_rep{rep}.log")
    if pauses:
        in_run = [p for p in pauses if t["arrival_ns"].min() <= p[1] <= t["sink_ns"].max()]
        hit = sum(1 for i in np.where(tail)[0]
                  if any(p[0] < t["sink_ns"][i] and p[1] > t["arrival_ns"][i] for p in in_run))
        dur = np.array([p[3] for p in in_run]) if in_run else np.array([0.0])
        print(f"GC pauses during the run: {len(in_run)}, total {dur.sum():.0f} ms, max {dur.max():.1f} ms; "
              f"tail windows overlapping a pause: {hit}/{tail.sum()}")
    else:
        print("GC pauses: none logged")

    vm = vmstat_runqueue(OUT / f"vmstat_rep{rep}.log")
    if vm is not None:
        rq, busy = vm
        print(f"vmstat: run queue mean {rq.mean():.1f} max {rq.max()}, CPU busy mean {busy.mean():.0f}%")


for rep in sorted(int(p.stem.removeprefix("trace_rep")) for p in OUT.glob("trace_rep*.csv")):
    report(rep)
