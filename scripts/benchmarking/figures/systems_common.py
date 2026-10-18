"""Shared loading for the revision's systems-comparison figures.

The three systems keep one colour each across every figure in this set, so a reader who learns the
mapping once carries it between them. The hues are the first three slots of the validated
categorical palette: the previous figure used matplotlib's tab10, whose orange and green are
indistinguishable under protanopia (OKLab Delta E 0.7), which put two of the three bars at risk for
colour-blind readers.
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "data" / "systems"

NES, FLINK, TORCHSERVE = "nes", "flink", "torchserve"
LABEL = {NES: "NAVI", FLINK: "Flink UDF", TORCHSERVE: "TorchServe"}
COLOR = {NES: "#2a78d6", FLINK: "#eb6834", TORCHSERVE: "#1baf7a"}
ORDER = [NES, FLINK, TORCHSERVE]

WORKLOADS = ["waymo", "cwru", "musan"]
WORKLOAD_TITLE = {
    "waymo": r"Waymo / SqueezeDet",
    "cwru": r"CWRU / SARAD",
    "musan": r"MUSAN / MHAtt-RNN",
}
# The offered load each workload's published operating point runs at, for the panel subtitles.
WORKLOAD_RATE = {"waymo": "10 frames/s", "cwru": "10 windows/s", "musan": "3 windows/s"}


def load(name: str) -> pd.DataFrame:
    """One of the aggregate CSVs in data/systems/.

    The figures read aggregates only. The per-repetition runs behind them are not kept in this
    tree; aggregate_systems_data.py reduces them, and its docstring says where they live.
    """
    return pd.read_csv(DATA / name)
