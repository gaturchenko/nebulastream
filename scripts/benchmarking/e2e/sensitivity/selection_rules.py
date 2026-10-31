"""The decision rules of InferenceImplementationSelector.cpp as pure functions, for replaying them offline.

Each decision is a pure function of the profile and a few parameters, so it can be re-evaluated for other parameter values
without deploying a query. replay_selection.py and the parts of this experiment that measure every alternative use these
functions, and check them against the choices the optimizer logged, so a change in the C++ rules shows up as a failed check.

All times are in the same unit; the batch-size rule does not care which.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class BatchChoice:
    batch_size: int
    # admissible: keeps up and meets the SLO; latency-fallback: keeps up, misses the SLO; overload: nothing keeps up.
    branch: str


def largest_candidate(size: float, stride: float, max_batch_size: int = 256) -> int:
    """(size + stride - 1) // stride in integer nanoseconds in the C++; the fill time alone must stay below the window size."""
    return max(1, min(max_batch_size, int((int(size * 1e6) + int(stride * 1e6) - 1) // int(stride * 1e6))))


def candidate_batch_sizes(size: float, stride: float, max_batch_size: int = 256) -> list[int]:
    limit, batch_size, sizes = largest_candidate(size, stride, max_batch_size), 1, []
    while batch_size <= limit:
        sizes.append(batch_size)
        batch_size *= 2
    return sizes


def select_batch_size(
    invocation: Mapping[int, float], size: float, stride: float, max_utilization: float = 0.85, slo_fraction: float = 0.9
) -> BatchChoice:
    """invocation maps every candidate batch size to its profiled t(B). Ties go to the smaller batch, as in the C++."""
    batch_sizes = sorted(invocation)
    throughput = {batch: batch / invocation[batch] for batch in batch_sizes}
    latency = {batch: stride * (batch - 1) + invocation[batch] for batch in batch_sizes}
    keeps_up = [batch for batch in batch_sizes if invocation[batch] / (stride * batch) <= max_utilization]
    admissible = [batch for batch in keeps_up if latency[batch] <= size * slo_fraction]

    def best(candidates: list[int], key) -> int:
        selected = candidates[0]
        for batch in candidates[1:]:
            if key(batch) > key(selected):
                selected = batch
        return selected

    if admissible:
        return BatchChoice(best(admissible, lambda batch: throughput[batch]), "admissible")
    if keeps_up:
        return BatchChoice(best(keeps_up, lambda batch: -latency[batch]), "latency-fallback")
    return BatchChoice(best(batch_sizes, lambda batch: throughput[batch]), "overload")


def select_post_join(speedup: float, minimum_speedup: float = 1.1) -> str:
    return "BATCHED" if speedup >= minimum_speedup else "SEQUENTIAL"


def select_deployment(private_bytes: float, shared_bytes: float, budget_bytes: float) -> str:
    """Without a prediction cache: private copies if they fit, then one shared model, then whichever is smaller."""
    if private_bytes <= budget_bytes:
        return "private"
    if shared_bytes <= budget_bytes:
        return "shared"
    return "private" if private_bytes < shared_bytes else "shared"
