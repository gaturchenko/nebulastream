"""Sensitivity of the implementation selection: one figure per input of Table V.

Produces sensitivity-f.pdf, sensitivity-umax.pdf, sensitivity-error.pdf: per rate or slide, the batch size the optimizer selects at
each value of the input, for N-BEATS and SARAD. A hatched cell is unsafe, i.e., it misses the latency bound or falls behind although
an alternative would not. sensitivity-sigma.pdf: per post-join configuration, the speedup each deployment predicts against the
measured one; a deployment batches if its prediction reaches sigma. Above the cells, each figure draws the table's two ranges.

Data: data/sensitivity/{batch_size_sweep,replay_summary,knee_sweep}.csv
      data/post_join/post_join_{runs,decisions}_{x86,pi}.csv
      (replay_selection.py and replay_knee.py; see the README.)
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import load_csv, save, setup_style

HATCH = "//////"
HATCH_LINEWIDTH = 0.4
DARK_CELL_FROM = 16  # batch sizes from here on get white labels
SAFE_COLOR = "0.25"
UNCHANGED_COLOR = "0.6"
BATCHED_COLOR = "#2a78d6"  # as in plot_post_join.py
SEQUENTIAL_COLOR = "#1baf7a"
TIE_COLOR = "0.55"
INPUTS = {
    "slo_fraction": ("sensitivity-f.pdf", r"$f$ (scheduling delay)", False, 0.9),
    "max_utilization": ("sensitivity-umax.pdf", r"$u_{\max}$ (queueing margin)", False, 0.85),
    "error": ("sensitivity-error.pdf", r"Error of $t(B)$ (factor)", True, 1.0),
}


def batch_color(batch_size: int) -> tuple:
    return plt.get_cmap("Blues")(0.22 + 0.63 * math.log2(batch_size) / 6)


def load_nbeats(parameter: str) -> tuple[pd.DataFrame, tuple, tuple]:
    sweep = load_csv("sensitivity", "batch_size_sweep.csv")
    sweep = sweep[sweep["parameter"] == parameter]
    frame = pd.DataFrame({"value": sweep["value"], "row": sweep["offered_rate"], "selected": sweep["selected"].astype(str),
                          "safe": sweep["feasible"]})
    ranges = load_csv("sensitivity", "replay_summary.csv").set_index("parameter").loc[parameter]
    return frame, (ranges["unchanged_from"], ranges["unchanged_to"]), (ranges["feasible_from"], ranges["feasible_to"])


def load_sarad(parameter: str) -> tuple[pd.DataFrame, tuple, tuple, set]:
    sweep = load_csv("sensitivity", "knee_sweep.csv")
    sweep = sweep[sweep["parameter"] == parameter]
    default = INPUTS[parameter][3]
    pivot = sweep.pivot(index="value", columns="slide_ms", values="selected")
    same = (pivot == pivot.loc[default]).all(axis=1)
    safe = sweep.groupby("value")["safe"].all()
    frame = pd.DataFrame({"value": sweep["value"], "row": sweep["slide_ms"], "selected": sweep["selected"], "safe": sweep["safe"]})
    return (frame, (same[same].index.min(), same[same].index.max()), (safe[safe].index.min(), safe[safe].index.max()),
            set(sweep.loc[~sweep["any_feasible"], "slide_ms"]))


def cell_edges(values: np.ndarray, log: bool) -> np.ndarray:
    points = np.log(values) if log else values
    middle = (points[:-1] + points[1:]) / 2
    edges = np.concatenate([[points[0]], middle, [points[-1]]])
    return np.exp(edges) if log else edges


def draw_ranges(ax, y_unchanged: float, y_safe: float, unchanged: tuple, safe: tuple) -> None:
    for y, (low, high), color in ((y_safe, safe, SAFE_COLOR), (y_unchanged, unchanged, UNCHANGED_COLOR)):
        ax.plot([low, high], [y, y], color=color, linewidth=3.5, solid_capstyle="butt", zorder=5)


def draw_cells(ax, frame: pd.DataFrame, rows: list, labels: list, log: bool, default: float, unchanged: tuple, safe: tuple,
               cell_size: float, tick_size: float) -> None:
    MIN_LABEL_WIDTH = 0.045  # of the axis width
    values = np.sort(frame["value"].unique())
    edges = cell_edges(values, log)
    ax.set_xscale("log" if log else "linear")
    ax.set_xlim(edges[0], edges[-1])
    to_axis = ax.transAxes.inverted().transform
    for y, row in enumerate(rows):
        cells = frame[frame["row"] == row].sort_values("value")
        state = list(zip(cells["selected"], cells["safe"]))
        start = 0
        for i in range(1, len(state) + 1):
            if i < len(state) and state[i] == state[start]:
                continue
            selected, is_safe = state[start]
            batches = [int(batch) for batch in selected.split()]
            color = batch_color(max(batches))
            left, right = edges[start], edges[i]
            if is_safe:
                ax.barh(y, right - left, left=left, height=0.92, color=color, linewidth=0, zorder=1)
            else:
                ax.barh(y, right - left, left=left, height=0.92, facecolor="white", edgecolor=color, hatch=HATCH, linewidth=0, zorder=1)
            width = ax.transAxes.inverted().transform(ax.transData.transform((right, y)))[0] \
                - to_axis(ax.transData.transform((left, y)))[0]
            if width >= MIN_LABEL_WIDTH * len(selected.split()):
                center = math.sqrt(left * right) if log else (left + right) / 2
                ink = "white" if is_safe and max(batches) >= DARK_CELL_FROM else "0.1"
                box = {"facecolor": color if is_safe else "white", "edgecolor": "none", "pad": 0.6}
                ax.text(center, y, "/".join(str(batch) for batch in batches), ha="center", va="center", fontsize=cell_size,
                        color=ink, bbox=box, zorder=5)
            start = i
    y_unchanged, y_safe = len(rows) + 0.1, len(rows) + 0.9
    draw_ranges(ax, y_unchanged, y_safe, unchanged, safe)
    ax.axvline(default, color="0.1", linestyle="--", linewidth=0.9, zorder=4)
    ax.set_yticks(list(range(len(rows))) + [y_unchanged, y_safe], labels + ["Unchanged", "Safe"])
    ax.set_ylim(-0.55, len(rows) + 1.45)
    ax.tick_params(axis="both", labelsize=tick_size, length=2)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def plot_input(parameter: str) -> None:
    FIGSIZE = (7.0, 2.35)
    TICK_SIZE = 8
    LABEL_SIZE = 8.5
    CELL_SIZE = 7

    name, label, log, default = INPUTS[parameter]
    nbeats, nbeats_unchanged, nbeats_safe = load_nbeats(parameter)
    sarad, sarad_unchanged, sarad_safe, infeasible = load_sarad(parameter)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    fig.canvas.draw()

    rates = sorted(nbeats["row"].unique())
    draw_cells(axes[0], nbeats, rates, [f"{rate:,.0f}/s" for rate in rates], log, default, nbeats_unchanged, nbeats_safe,
               CELL_SIZE, TICK_SIZE)
    slides = sorted(sarad["row"].unique(), reverse=True)
    draw_cells(axes[1], sarad, slides, [f"{slide:.0f} ms" + ("*" if slide in infeasible else "") for slide in slides], log, default,
               sarad_unchanged, sarad_safe, CELL_SIZE, TICK_SIZE)
    for ax, title in zip(axes, ("N-BEATS, by offered rate", "SARAD, by slide (* no batch size is feasible)")):
        ax.set_title(title, fontsize=LABEL_SIZE, pad=3)
        ax.set_xlabel(label, fontsize=LABEL_SIZE, labelpad=1.5)
    if log:
        for ax, ticks in zip(axes, ([0.1, 0.3, 1, 3, 10, 30], [0.25, 0.5, 1, 2, 4])):
            ax.set_xticks(ticks, [f"$\\times${tick:g}" for tick in ticks])
            ax.xaxis.set_minor_formatter(lambda _x, _pos: "")
    fig.legend(
        handles=[Patch(facecolor=batch_color(16), label="Selected batch size (number)"),
                 Patch(facecolor="white", edgecolor=batch_color(16), hatch=HATCH, linewidth=0, label="Unsafe choice"),
                 Line2D([0], [0], color="0.1", linestyle="--", linewidth=0.9, label="Default")],
        loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=3, frameon=False, fontsize=TICK_SIZE, handlelength=1.6)
    fig.tight_layout(pad=0.3, w_pad=1.2)
    save(fig, name)
    plt.close(fig)


def load_deployments() -> pd.DataFrame:
    frames = []
    for platform, hardware in (("x86", "Server"), ("pi", "Pi")):
        runs = load_csv("post_join", f"post_join_runs_{platform}.csv")
        deployed = runs[(runs["strategy"] == "OPTIMIZER_CHOOSES") & runs["predicted_speedup"].notna()]
        decisions = load_csv("post_join", f"post_join_decisions_{platform}.csv")
        keep = ["k", "dedup", "measured_speedup", "measured_better", "correct"]
        frames.append(deployed[["k", "dedup", "predicted_speedup"]].merge(decisions[keep], on=["k", "dedup"], how="left").assign(hardware=hardware))
    return pd.concat(frames, ignore_index=True)


def plot_sigma() -> None:
    FIGSIZE = (5.2, 2.75)
    TICK_SIZE = 8
    LABEL_SIZE = 8.5
    TIE_SHARE = 0.02  # as in replay_selection.py
    DEFAULT = 1.1

    deployments = load_deployments()
    speedups = deployments["predicted_speedup"]
    unchanged = (math.ceil(speedups[speedups < DEFAULT].max() * 100) / 100, math.floor(speedups[speedups >= DEFAULT].min() * 100) / 100)
    ranges = load_csv("sensitivity", "replay_summary.csv").set_index("parameter").loc["minimum_speedup_all_deployments_correct"]
    safe = (ranges["feasible_from"], ranges["feasible_to"])

    configurations = deployments[["hardware", "k", "dedup"]].drop_duplicates().sort_values(
        ["hardware", "k", "dedup"], ascending=[True, False, True], ignore_index=True)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    labels = []
    for y, config in configurations.iterrows():
        cells = deployments[(deployments["hardware"] == config["hardware"]) & (deployments["k"] == config["k"]) & (deployments["dedup"] == config["dedup"])]
        measured = cells["measured_speedup"].iloc[0]
        if pd.isna(measured):
            style = {"marker": "o", "facecolors": "none", "edgecolors": TIE_COLOR}
        elif abs(measured - 1) < TIE_SHARE:
            style = {"marker": "D", "facecolors": "none", "edgecolors": TIE_COLOR}
        elif cells["measured_better"].iloc[0] == "BATCHED":
            style = {"marker": "o", "facecolors": BATCHED_COLOR, "edgecolors": "white"}
        else:
            style = {"marker": "s", "facecolors": SEQUENTIAL_COLOR, "edgecolors": "white"}
        ax.scatter(cells["predicted_speedup"], [y] * len(cells), s=24, linewidths=0.6, alpha=0.85, zorder=3, **style)
        ax.plot([measured], [y], marker="|", markersize=8, markeredgewidth=1.3, color="0.1", linestyle="none", zorder=2)
        batching = "dedup." if config["dedup"] else "fixed"
        labels.append(f"{config['hardware']}, $k={config['k']}$, {batching}")
    count = len(configurations)
    y_unchanged, y_safe = count + 0.1, count + 0.9
    draw_ranges(ax, y_unchanged, y_safe, unchanged, safe)
    ax.axvline(DEFAULT, color="0.1", linestyle="--", linewidth=0.9, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(0.68, 3.2)
    ticks = [0.75, 1, 1.5, 2, 3]
    ax.set_xticks(ticks, [f"{tick:g}" for tick in ticks])
    ax.xaxis.set_minor_formatter(lambda _x, _pos: "")
    ax.set_yticks(list(range(count)) + [y_unchanged, y_safe], labels + ["Unchanged", "Safe"])
    ax.set_ylim(-0.6, count + 1.45)
    ax.grid(axis="y", color="0.92", linewidth=0.6, zorder=0)
    ax.tick_params(axis="both", labelsize=TICK_SIZE, length=2)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(r"$\sigma$ (minimum speedup) / speedup of batching", fontsize=LABEL_SIZE, labelpad=1.5)
    fig.legend(
        handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=BATCHED_COLOR, markeredgecolor="white", markersize=5, label="Predicted, batching is faster"),
                 Line2D([0], [0], marker="s", color="none", markerfacecolor=SEQUENTIAL_COLOR, markeredgecolor="white", markersize=5, label="Predicted, sequential is faster"),
                 Line2D([0], [0], marker="D", color="none", markerfacecolor="none", markeredgecolor=TIE_COLOR, markersize=4.5, label="Predicted, both tie"),
                 Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=TIE_COLOR, markersize=5, label="Predicted, plans not measured"),
                 Line2D([0], [0], marker="|", color="none", markeredgecolor="0.1", markersize=8, markeredgewidth=1.3, label="Measured"),
                 Line2D([0], [0], color="0.1", linestyle="--", linewidth=0.9, label="Default")],
        loc="upper center", bbox_to_anchor=(0.55, 1.14), ncol=3, frameon=False, fontsize=TICK_SIZE - 0.5, handlelength=1.4, columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    save(fig, "sensitivity-sigma.pdf")
    plt.close(fig)


if __name__ == "__main__":
    setup_style()
    plt.rcParams["hatch.linewidth"] = HATCH_LINEWIDTH
    for parameter in INPUTS:
        plot_input(parameter)
    plot_sigma()
