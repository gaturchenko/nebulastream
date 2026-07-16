"""Batching microbenchmark: hardware metrics vs. inference batch size.

Produces batch-microbench.pdf: a 2x2 line-plot grid (throughput, IPC, L1/LLC
misses per tuple) over batch sizes 1-4096 for the three N-BEATS model sizes,
with the single-P-core FP32 roofline overlaid on the throughput facet.

Data: data/batching/BatchInferenceMicrobenchmark2.csv
      (BatchInferenceMicrobenchmark, x86 dev machine).
"""

import pandas as pd
import seaborn as sns

from common import load_csv, save, setup_style

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
    FACET_HEIGHT = 3.4
    FACET_ASPECT = 1.65
    SEABORN_CONTEXT = "talk"  # seaborn font-scale preset for all text
    LEGEND_ANCHOR = (0.425, -0.02)
    SUBPLOTS_ADJUST = dict(bottom=0.2, hspace=0.45, wspace=0.25)
    PEAK_HEADROOM = 1.12  # y-limit factor so the roofline labels stay inside

    sns.set_theme(style="whitegrid", context=SEABORN_CONTEXT)
    setup_style()

    g = sns.relplot(
        data=long_df,
        x="batch_size",
        y="value",
        hue="model_size",
        hue_order=MODEL_ORDER,
        col="metric_label",
        col_wrap=2,
        kind="line",
        marker="o",
        legend="auto",
        facet_kws={"sharex": True, "sharey": False},
        height=FACET_HEIGHT,
        aspect=FACET_ASPECT,
    )

    g.set(xscale="log")
    g.set_axis_labels("Batch size", "Value")
    g.set_titles("{col_name}")

    palette = dict(zip(MODEL_ORDER, sns.color_palette(n_colors=len(MODEL_ORDER))))
    max_throughput = long_df.loc[
        long_df["metric_label"].str.contains("Throughput"), "value"
    ].max()

    for ax in g.axes.flat:
        ax.set_xticks(BATCH_TICKS)
        ax.set_xticklabels([str(t) for t in BATCH_TICKS], rotation=45)
        # Show x tick labels on every facet despite the shared x-axis.
        ax.tick_params(axis="x", which="major", labelbottom=True, bottom=True)
        ax.minorticks_off()
        ax.grid(False)
        ax.grid(True, which="major", axis="x", alpha=0.25)
        ax.grid(True, which="major", axis="y", alpha=0.20)
        ax.set_ylim(0)
        ax.margins(x=0.08)

        if "Throughput" in ax.get_title():
            ax.set_ylim(0, PEAK_HEADROOM * max(max(peaks.values()), max_throughput))
            for model, peak in peaks.items():
                ax.axhline(
                    y=peak,
                    linestyle="--",
                    linewidth=1.4,
                    color=palette[model],
                    alpha=0.8,
                    zorder=1,
                )

    sns.move_legend(
        g,
        "lower center",
        bbox_to_anchor=LEGEND_ANCHOR,
        ncol=3,
        frameon=False,
        title=None,
    )
    g.figure.subplots_adjust(**SUBPLOTS_ADJUST)

    # No bbox_inches="tight": preserves the aspect ratio reserved for the legend.
    save(g.figure, "batch-microbench.pdf", tight=False, pad_inches=0.02)


if __name__ == "__main__":
    df, long_df = load_data()
    plot_batch_microbench(long_df, peak_records_per_second(df))
