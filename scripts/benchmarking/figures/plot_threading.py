"""Threading microbenchmark: SPE workers vs. OpenVINO threads trade-off.

Produces two figures from one thread-budget sweep (results_1.csv):
  - ts-threading.pdf: N-BEATS-1 / N-BEATS-32 latency-throughput scatter,
    colored by SPE thread count, faceted by model x budget and by
    inference-pipeline vs. end-to-end measurement level.
  - od-threading.pdf: the same view for NanoDet object detection,
    faceted by budget only.

Data: data/threading/results_1.csv (run_threading.sh sweep). Each query name
encodes model and total thread budget ("nbeats-1-48.test_05"); the test index
selects one (SPE workers x OpenVINO threads) split of that budget.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

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
    "dlinear": "DLinear",
    "nbeats": "N-BEATS",
    "nbeats-1": "N-BEATS-1",
    "nbeats-32": "N-BEATS-32",
}

THREAD_CMAP = "viridis_r"


def parse_query_name(query_name: str):
    """'nbeats-1-48.test_05' -> ('N-BEATS-1', 48, workers=8, ov_threads=6)."""
    stem, _, test = query_name.partition(".test_")
    prefix, _, budget = stem.rpartition("-")
    budget = int(budget)
    workers, ov_threads = BUDGET_SPLITS[budget][int(test) - 1]
    return MODEL_LABELS[prefix], budget, workers, ov_threads


def load_data() -> pd.DataFrame:
    df = load_csv("threading", "results_1.csv")
    parsed = df["query_name"].map(parse_query_name)
    df[["Model", "budget", "nes_workers", "openvino_threads"]] = pd.DataFrame(
        parsed.tolist(), index=df.index
    )
    df["latency_ms"] = df["latency_us"] / 1_000
    df["end_to_end_latency_ms"] = df["end_to_end_latency_us"] / 1_000
    df["Config"] = (
        df["nes_workers"].astype(str) + " SPE × " + df["openvino_threads"].astype(str) + " OV"
    )
    return df


def stack_measurement_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate over repetitions and stack the inference-pipeline and
    end-to-end views into one long frame with latency_value/throughput_value."""
    agg = (
        df.groupby(
            ["pipeline_id", "Model", "nes_workers", "openvino_threads", "Config", "budget"],
            as_index=False,
        )
        .agg(
            throughput_mean=("throughput", "mean"),
            latency_mean=("latency_ms", "mean"),
            throughput_e2e_mean=("end_to_end_throughput", "mean"),
            latency_e2e_mean=("end_to_end_latency_ms", "mean"),
        )
    )
    pipeline = agg.assign(
        metric_label="Inference pipeline",
        latency_value=agg["latency_mean"],
        throughput_value=agg["throughput_mean"],
    )
    e2e = agg.assign(
        metric_label="End-to-end",
        latency_value=agg["latency_e2e_mean"],
        throughput_value=agg["throughput_e2e_mean"],
    )
    return pd.concat([pipeline, e2e], ignore_index=True)


def thread_colorbar(fig, axes, values, fraction, pad, label_fs, tick_fs):
    """Shared log-scaled SPE-thread-count colorbar on the figure's right edge."""
    norm = mpl.colors.LogNorm(vmin=min(values), vmax=max(values))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=THREAD_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=fraction, pad=pad)
    cbar.set_label("SPE threads", fontsize=label_fs)
    cbar.set_ticks(values)
    cbar.set_ticklabels([str(v) for v in values])
    cbar.ax.tick_params(labelsize=tick_fs)
    cbar.minorticks_off()
    return norm


def plot_ts_threading(df: pd.DataFrame) -> None:
    """Time series (N-BEATS-1/-32): latency vs. throughput per thread split."""
    TITLE_FS = 14
    LABEL_FS = 17
    TICK_FS = 13
    CBAR_LABEL_FS = 14
    CBAR_TICK_FS = 12
    ROW_HEADER_FS = 16
    FACET_HEIGHT = 3.2
    FACET_ASPECT = 1.12
    MARKER_SIZE = 115
    SUBPLOTS_ADJUST = dict(left=0.07, right=0.90, bottom=0.11, top=0.85, hspace=0.42, wspace=0.30)
    MODEL_ORDER = ["N-BEATS-1", "N-BEATS-32"]
    BUDGET_ORDER = [24, 48]
    PIPELINE_ID = 4

    setup_style()

    sub = df[(df["Model"].isin(MODEL_ORDER)) & (df["pipeline_id"] == PIPELINE_ID)].copy()
    # Inference throughput normalized to CPU time: tuples per summed task busy time.
    sub["throughput"] = sub["end_to_end_tuples"] * 1_000_000 / sub["pipeline_task_duration_us"]
    plot_df = stack_measurement_levels(sub)

    plot_df["ModelBudgetLabel"] = (
        plot_df["Model"] + ", " + plot_df["budget"].astype(str) + " threads"
    )
    col_order = [f"{m}, {b} threads" for m in MODEL_ORDER for b in BUDGET_ORDER]
    thread_values = sorted(plot_df["nes_workers"].unique())
    norm = mpl.colors.LogNorm(vmin=min(thread_values), vmax=max(thread_values))

    # Same whitegrid look as od-threading, matching the figures in the paper.
    with sns.axes_style("whitegrid"):
        g = sns.relplot(
            data=plot_df,
            x="latency_value",
            y="throughput_value",
            hue="nes_workers",
            palette=THREAD_CMAP,
            hue_norm=norm,
            marker="o",
            col="ModelBudgetLabel",
            col_order=col_order,
            row="metric_label",
            row_order=["Inference pipeline", "End-to-end"],
            kind="scatter",
            s=MARKER_SIZE,
            edgecolor="0.25",
            linewidth=0.4,
            facet_kws={"sharex": False, "sharey": False},
            height=FACET_HEIGHT,
            aspect=FACET_ASPECT,
            legend=False,
        )

    g.set_axis_labels("Latency [ms]", "Throughput [tuples/s]")
    for ax in g.axes.flat:
        ax.set_title("")
        ax.xaxis.label.set_fontsize(LABEL_FS)
        ax.yaxis.label.set_fontsize(LABEL_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.yaxis.set_major_formatter(eng_formatter(decimals=0))
    for j, col_name in enumerate(col_order):
        g.axes[0, j].set_title(col_name, fontsize=TITLE_FS, pad=8)

    fig = g.figure
    thread_colorbar(fig, g.axes, thread_values, fraction=0.028, pad=0.015,
                    label_fs=CBAR_LABEL_FS, tick_fs=CBAR_TICK_FS)
    fig.subplots_adjust(**SUBPLOTS_ADJUST)

    # Bold measurement-level headers centered above each facet row.
    x_center = (g.axes[0, 0].get_position().x0 + g.axes[0, -1].get_position().x1) / 2
    for axes_row, label, offset in (
        (g.axes[0, :], "Inference pipeline", 0.055),
        (g.axes[1, :], "End-to-end", 0.035),
    ):
        y = max(ax.get_position().y1 for ax in axes_row) + offset
        fig.text(x_center, y, label, ha="center", va="bottom",
                 fontsize=ROW_HEADER_FS, fontweight="bold")

    save(fig, "ts-threading.pdf")


def plot_od_threading(df: pd.DataFrame) -> None:
    """Object detection (NanoDet): latency vs. throughput per thread split."""
    TITLE_FS = 16
    LABEL_FS = 18
    TICK_FS = 16
    CBAR_LABEL_FS = 14
    CBAR_TICK_FS = 12
    FACET_HEIGHT = 3.4
    FACET_ASPECT = 1.65
    MARKER_SIZE = 115
    SUBPLOTS_ADJUST = dict(left=0.08, right=0.90, bottom=0.10, top=0.94, hspace=0.35, wspace=0.25)
    BUDGET_ORDER = [24, 48]
    PIPELINE_ID = 5

    setup_style()

    sub = df[
        (df["Model"] == "NanoDet")
        & (df["pipeline_id"] == PIPELINE_ID)
        & (df["budget"].isin(BUDGET_ORDER))
    ]
    plot_df = stack_measurement_levels(sub)
    thread_values = sorted(plot_df["nes_workers"].unique())
    norm = mpl.colors.LogNorm(vmin=min(thread_values), vmax=max(thread_values))

    # The published figure uses seaborn's whitegrid look (unlike ts-threading).
    with sns.axes_style("whitegrid"):
        g = sns.relplot(
            data=plot_df,
            x="latency_value",
            y="throughput_value",
            hue="nes_workers",
            palette=THREAD_CMAP,
            hue_norm=norm,
            marker="o",
            col="budget",
            col_order=BUDGET_ORDER,
            row="metric_label",
            row_order=["Inference pipeline", "End-to-end"],
            kind="scatter",
            s=MARKER_SIZE,
            edgecolor="0.25",
            linewidth=0.4,
            facet_kws={"sharex": False, "sharey": True},
            height=FACET_HEIGHT,
            aspect=FACET_ASPECT,
            legend=False,
        )

    g.set_titles(row_template="{row_name}", col_template="{col_name} threads")
    g.set_axis_labels("Latency [ms]", "Throughput [tuples/s]")
    for ax in g.axes.flat:
        ax.title.set_fontsize(TITLE_FS)
        ax.xaxis.label.set_fontsize(LABEL_FS)
        ax.yaxis.label.set_fontsize(LABEL_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS)

    fig = g.figure
    thread_colorbar(fig, g.axes, thread_values, fraction=0.035, pad=0.01,
                    label_fs=CBAR_LABEL_FS, tick_fs=CBAR_TICK_FS)
    fig.subplots_adjust(**SUBPLOTS_ADJUST)

    save(fig, "od-threading.pdf")


if __name__ == "__main__":
    df = load_data()
    plot_ts_threading(df)
    plot_od_threading(df)
