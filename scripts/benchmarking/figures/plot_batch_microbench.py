"""Batching microbenchmark: hardware metrics vs. inference batch size.

Produces batch-microbench.pdf: a 2x2 line-plot grid (throughput, IPC, L1/LLC
misses per tuple) over batch sizes 1-4096 for the three N-BEATS model sizes,
with the single-P-core FP32 roofline overlaid on the throughput facet.

Data: data/batching/BatchInferenceMicrobenchmark2.csv
      (BatchInferenceMicrobenchmark, x86 dev machine).
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import eng_formatter, load_csv, save, setup_style

MODEL_LABELS = {
    "nbeats-small.onnx": "Small (3.6MB/0.9M)",
    "nbeats-medium.onnx": "Medium (7.3MB/1.8M)",
    "nbeats-large.onnx": "Large (14.6MB/3.6M)",
}
MODEL_ORDER = list(MODEL_LABELS.values())

MODEL_PARAMS = {
    "Small (3.6MB/0.9M)": 904_262,
    "Medium (7.3MB/1.8M)": 1_810_394,
    "Large (14.6MB/3.6M)": 3_622_658,
}

# Facet metrics in display order: CSV column -> facet title.
METRICS = {
    "throughput_records_per_second": "Throughput [tuples/s]",
    "ipc": "IPC",
    "l1_misses_per_record": "L1 misses / tuple",
    "llc_misses_per_record": "LLC misses / tuple",
}

BATCH_TICKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_csv("batching", "BatchInferenceMicrobenchmark2.csv")
    df = df[df["status"] == "ok"].copy()
    df["model_size"] = pd.Categorical(
        df["model"].map(MODEL_LABELS), categories=MODEL_ORDER, ordered=True
    )

    long_df = df.melt(
        id_vars=["model", "model_size", "backend", "batch_size", "run"],
        value_vars=[c for c in METRICS if c in df.columns],
        var_name="metric",
        value_name="value",
    )
    long_df["metric_label"] = long_df["metric"].map(METRICS)
    return df, long_df


def peak_records_per_second(df: pd.DataFrame) -> dict[str, float]:
    """Single-P-core FP32 roofline from the measured effective frequency.

    AVX2/FMA: 8 fp32 lanes per 256-bit vector x 2 FLOPs per FMA x 2 FMA units
    per cycle; a model forward pass costs ~2 FLOPs per parameter.
    """
    fp32_flops_per_cycle = 8 * 2 * 2
    peak_flops = df["ghz"].median() * 1e9 * fp32_flops_per_cycle
    return {model: peak_flops / (2 * params) for model, params in MODEL_PARAMS.items()}


def plot_batch_microbench(long_df: pd.DataFrame, peaks: dict[str, float]) -> None:
    # Drawn at the size it is printed at, one column of the paper (3.5 in), so every font size is the size on the page. The
    # paper's captions are 8 pt; nothing here goes below 7 pt.
    FIG_SIZE = (3.5, 2.35)
    TITLE_FS = 8
    LABEL_FS = 8.5
    TICK_FS = 7.4
    LEGEND_FS = 7.5
    LINE_WIDTH = 1.0
    MARKER_SIZE = 2.5
    SUBPLOTS_ADJUST = dict(left=0.11, right=0.98, bottom=0.255, top=0.91, hspace=0.36, wspace=0.28)
    TICKS = [1, 4, 16, 64, 256, 1024, 4096]  # every other power of two: all of them do not fit half a column
    PEAK_HEADROOM = 1.12  # y-limit factor so the rooflines stay inside

    setup_style()
    palette = dict(zip(MODEL_ORDER, sns.color_palette("deep", n_colors=len(MODEL_ORDER))))
    fig, grid = plt.subplots(2, 2, figsize=FIG_SIZE, sharex=True)
    max_throughput = long_df.loc[long_df["metric"] == "throughput_records_per_second", "value"].max()

    for ax, (metric, title) in zip(grid.ravel(), METRICS.items()):
        data = long_df[long_df["metric"] == metric]
        for model in MODEL_ORDER:
            series = data[data["model_size"] == model].groupby("batch_size")["value"].mean()
            ax.plot(series.index, series.to_numpy(), color=palette[model], linewidth=LINE_WIDTH, marker="o",
                    markersize=MARKER_SIZE, label=model)
        ax.set_title(title, fontsize=TITLE_FS, pad=3)
        ax.set_xscale("log")
        ax.set_xticks(TICKS)
        ax.set_xticklabels([str(t) for t in TICKS])
        ax.minorticks_off()
        ax.tick_params(axis="both", labelsize=TICK_FS, pad=2, length=2.5)
        ax.yaxis.set_major_formatter(eng_formatter(decimals=1))  # 0 would print a 1.5k tick as 2k
        ax.grid(True, which="major", alpha=0.25, linewidth=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0)
        ax.margins(x=0.05)
        if metric == "throughput_records_per_second":
            ax.set_ylim(0, PEAK_HEADROOM * max(max(peaks.values()), max_throughput))
            for model, peak in peaks.items():
                ax.axhline(y=peak, linestyle="--", linewidth=0.8, color=palette[model], alpha=0.8, zorder=1)
    for ax in grid[-1]:
        ax.set_xlabel("Batch size", fontsize=LABEL_FS, labelpad=2)

    handles, labels = grid[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=LEGEND_FS,
               handlelength=1.4, columnspacing=0.9, handletextpad=0.4)
    fig.subplots_adjust(**SUBPLOTS_ADJUST)
    save(fig, "batch-microbench.pdf")


if __name__ == "__main__":
    df, long_df = load_data()
    plot_batch_microbench(long_df, peak_records_per_second(df))
