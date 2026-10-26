"""Threading microbenchmark: SPE workers vs. OpenVINO threads trade-off.

Each split of a thread budget into (SPE workers x OpenVINO threads) is one point: its capacity against its latency at a fixed
load below every split's capacity.
  - ts-threading.pdf: N-BEATS-1 / N-BEATS-32, one panel per model x budget.
  - od-threading.pdf: object detection with NanoDet-Plus on a ResNet-101 backbone, heavy enough that inference bounds the
    query (with the original ShuffleNetV2 NanoDet, reading the frames does), one panel per budget.

Capacity: results per second of the unpaced sweep (run_threading.sh on so015, 10 repetitions, mean), i.e. the inference
pipeline's output tuples over the query's duration. Data: data/threading/results_1.csv (N-BEATS),
data/threading/results_nanodet_r101.csv (NanoDet-R101).
Latency: the paced sweep (run_threading_lat.sh on so015, 5 repetitions, median of the per-run percentiles), from the moment the
input was sent to the result reaching the sink, at N-BEATS-1 6667, N-BEATS-32 333 forecasts/s and NanoDet-R101 20 frames/s.
The marker is the P95, the line runs back to the P50. Data: data/threading/threading_lat.csv (process_threading_lat.py).

Each query name encodes model and total thread budget ("nbeats-1-48.test_05", "nbeats-1-48-lat.test_05"); the test index
selects one (SPE workers x OpenVINO threads) split of that budget.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from common import eng_formatter, load_csv, save, setup_style

# The i-th test in a sweep file maps to the i-th (SPE workers, OpenVINO threads)
# split of the file's thread budget.
BUDGET_SPLITS = {
    24: [(24, 1), (12, 2), (8, 3), (6, 4), (4, 6), (3, 8), (2, 12), (1, 24)],
    48: [(48, 1), (24, 2), (16, 3), (12, 4), (8, 6), (6, 8), (4, 12), (3, 16), (2, 24), (1, 48)],
    96: [(48, 2), (32, 3), (24, 4), (16, 6), (12, 8), (8, 12), (6, 16), (4, 24), (3, 32), (2, 48)],
}

MODEL_LABELS = {
    "mlp": "MLP",
    "nanodet": "NanoDet",
    "nanodet-r101": "NanoDet-R101",
    "dlinear": "DLinear",
    "nbeats": "N-BEATS",
    "nbeats-1": "N-BEATS-1",
    "nbeats-32": "N-BEATS-32",
}

# The paced sweep's offered load, in results per second.
PACED_LOAD = {"N-BEATS-1": 100_000 / 15, "N-BEATS-32": 5_000 / 15, "NanoDet-R101": 20}

THREAD_CMAP = "viridis_r"


def parse_query_name(query_name: str):
    """'nbeats-1-48.test_05' or 'nbeats-1-48-lat.test_05' -> ('N-BEATS-1', 48, workers=8, ov_threads=6)."""
    stem, _, test = query_name.partition(".test_")
    prefix, _, budget = stem.removesuffix("-lat").rpartition("-")
    budget = int(budget)
    workers, ov_threads = BUDGET_SPLITS[budget][int(test) - 1]
    return MODEL_LABELS[prefix], budget, workers, ov_threads


def with_split(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["query_name"].map(parse_query_name)
    df[["Model", "budget", "nes_workers", "openvino_threads"]] = pd.DataFrame(parsed.tolist(), index=df.index)
    return df


def load_points(capacity_csv: str, pipeline_id: int) -> pd.DataFrame:
    """One row per (model, budget, split): mean capacity and median latency percentiles."""
    capacity = load_csv("threading", capacity_csv)
    capacity = capacity[capacity["pipeline_id"] == pipeline_id].copy()
    capacity["capacity"] = capacity["pipeline_tuples"] * 1_000_000 / capacity["end_to_end_duration_us"]
    capacity = with_split(capacity).groupby(["Model", "budget", "nes_workers", "openvino_threads"], as_index=False)[
        "capacity"].mean()

    latency = with_split(load_csv("threading", "threading_lat.csv"))
    latency = latency.groupby(["Model", "budget", "nes_workers", "openvino_threads"], as_index=False)[
        ["p50_ms", "p95_ms"]].median()
    return capacity.merge(latency, on=["Model", "budget", "nes_workers", "openvino_threads"], validate="one_to_one")


def thread_colorbar(fig, axes, values, label_fs, tick_fs):
    """Shared log-scaled SPE-thread-count colorbar right of the panels (call after the layout is final)."""
    norm = mpl.colors.LogNorm(vmin=min(values), vmax=max(values))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=THREAD_CMAP)
    sm.set_array([])
    last = axes[-1].get_position()
    cax = fig.add_axes((last.x1 + 0.015, last.y0, 0.011, last.height))
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("SPE threads", fontsize=label_fs)
    cbar.set_ticks(values)
    cbar.set_ticklabels([str(v) for v in values])
    cbar.ax.tick_params(labelsize=tick_fs)
    cbar.minorticks_off()
    return norm


def draw_panel(ax, points: pd.DataFrame, load: float, norm, marker_size: float) -> None:
    """Capacity vs. latency at the paced load; the marker is the P95, the line runs back to the P50."""
    cmap = mpl.colormaps[THREAD_CMAP]
    for row in points.sort_values("nes_workers").itertuples():
        color = cmap(norm(row.nes_workers))
        ax.plot([row.p50_ms, row.p95_ms], [row.capacity, row.capacity], color=color, linewidth=1.6,
                solid_capstyle="round", zorder=2)
        ax.scatter(row.p95_ms, row.capacity, s=marker_size, color=color, edgecolor="0.25", linewidth=0.4, zorder=3)
    ax.axhline(load, color="0.35", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, points["p95_ms"].max() * 1.08)
    ax.grid(color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def legend_handles(marker_size: float) -> list:
    return [
        Line2D([], [], color="0.4", marker="o", linestyle="none", markersize=marker_size ** 0.5, markeredgecolor="0.25",
               label="P95 latency"),
        Line2D([], [], color="0.4", linewidth=1.6, label="P50 to P95"),
        Line2D([], [], color="0.35", linestyle="--", linewidth=1.0, label="offered load"),
    ]


def plot_ts_threading(points: pd.DataFrame) -> None:
    """Time series (N-BEATS-1/-32): capacity vs. latency per thread split."""
    TITLE_FS = 14
    LABEL_FS = 17
    TICK_FS = 13
    LEGEND_FS = 13
    CBAR_LABEL_FS = 14
    CBAR_TICK_FS = 12
    FIG_SIZE = (16.0, 4.3)
    MARKER_SIZE = 115
    SUBPLOTS_ADJUST = dict(left=0.06, right=0.90, bottom=0.17, top=0.80, wspace=0.30)
    MODEL_ORDER = ["N-BEATS-1", "N-BEATS-32"]
    BUDGET_ORDER = [24, 48]

    setup_style()
    thread_values = sorted(points.loc[points["Model"].isin(MODEL_ORDER), "nes_workers"].unique())
    norm = mpl.colors.LogNorm(vmin=min(thread_values), vmax=max(thread_values))

    fig, axes = plt.subplots(1, len(MODEL_ORDER) * len(BUDGET_ORDER), figsize=FIG_SIZE)
    panels = [(model, budget) for model in MODEL_ORDER for budget in BUDGET_ORDER]
    for ax, (model, budget) in zip(axes, panels):
        draw_panel(ax, points[(points["Model"] == model) & (points["budget"] == budget)], PACED_LOAD[model], norm,
                   MARKER_SIZE)
        ax.set_title(f"{model}, {budget} threads", fontsize=TITLE_FS, pad=8)
        ax.set_xlabel("Latency [ms]", fontsize=LABEL_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.yaxis.set_major_formatter(eng_formatter(decimals=0))
    axes[0].set_ylabel("Capacity [forecasts/s]", fontsize=LABEL_FS)

    fig.subplots_adjust(**SUBPLOTS_ADJUST)
    thread_colorbar(fig, axes, thread_values, label_fs=CBAR_LABEL_FS, tick_fs=CBAR_TICK_FS)
    x_center = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
    fig.legend(handles=legend_handles(MARKER_SIZE), loc="upper center", bbox_to_anchor=(x_center, 1.0), ncol=3,
               frameon=False, fontsize=LEGEND_FS)

    save(fig, "ts-threading.pdf")


def plot_od_threading(points: pd.DataFrame) -> None:
    """Object detection (NanoDet-Plus, ResNet-101 backbone): capacity vs. latency per thread split."""
    TITLE_FS = 16
    LABEL_FS = 18
    TICK_FS = 16
    LEGEND_FS = 15
    CBAR_LABEL_FS = 14
    CBAR_TICK_FS = 12
    FIG_SIZE = (12.0, 4.6)
    MARKER_SIZE = 115
    SUBPLOTS_ADJUST = dict(left=0.08, right=0.90, bottom=0.16, top=0.80, wspace=0.12)
    BUDGET_ORDER = [24, 48]

    setup_style()
    sub = points[points["Model"] == "NanoDet-R101"]
    thread_values = sorted(sub["nes_workers"].unique())
    norm = mpl.colors.LogNorm(vmin=min(thread_values), vmax=max(thread_values))

    fig, axes = plt.subplots(1, len(BUDGET_ORDER), figsize=FIG_SIZE, sharex=True, sharey=True)
    for ax, budget in zip(axes, BUDGET_ORDER):
        draw_panel(ax, sub[sub["budget"] == budget], PACED_LOAD["NanoDet-R101"], norm, MARKER_SIZE)
        ax.set_title(f"{budget} threads", fontsize=TITLE_FS, pad=8)
        ax.set_xlabel("Latency [ms]", fontsize=LABEL_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS)
    axes[0].set_ylabel("Capacity [frames/s]", fontsize=LABEL_FS)

    fig.subplots_adjust(**SUBPLOTS_ADJUST)
    thread_colorbar(fig, axes, thread_values, label_fs=CBAR_LABEL_FS, tick_fs=CBAR_TICK_FS)
    x_center = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
    fig.legend(handles=legend_handles(MARKER_SIZE), loc="upper center", bbox_to_anchor=(x_center, 1.0), ncol=3,
               frameon=False, fontsize=LEGEND_FS)

    save(fig, "od-threading.pdf")


if __name__ == "__main__":
    plot_ts_threading(load_points("results_1.csv", pipeline_id=4))
    plot_od_threading(load_points("results_nanodet_r101.csv", pipeline_id=5))
