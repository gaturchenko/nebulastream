"""Optimization ablation on the Pi: UDF baseline vs. ST/Batch/Cache/Dedup.

Produces two figures with the same layout (P50/P95 prediction latency of the
best end-to-end-throughput configuration per setting on the left, inference
pipeline throughput of the best configuration with speedup-over-UDF labels on
the right):
  - e2e-audio.pdf: MUSAN audio / MHAtt-RNN (log latency axis, 1 s window)
  - e2e-ts.pdf:    CWRU vibration / SARAD (linear latency axis, 200 ms window,
                   with the prediction cache split into Local/Global scope)

Data: data/ablation/ablation_musan.csv, data/ablation/ablation_cwru.csv
      (run_systests.py sweeps over inference configurations).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import eng_formatter, load_csv, save, setup_style

# Metrics that must be present and numeric for a row to count as a candidate.
METRICS = [
    "throughput", "end_to_end_throughput",
    "latency_us", "end_to_end_latency_us",
    "p50_sink_latency_us", "p95_sink_latency_us", "p99_sink_latency_us",
]

P50_COLOR = "#1F77B4"
P95_COLOR = "#FF7F0E"

# Both figures are drawn at the size they are printed at, one column of the paper (3.5 in), so every font size is the size on the
# page. The paper's captions are 8 pt; nothing here goes below 7 pt.
COLUMN_WIDTH = 3.5


def classify_settings(df: pd.DataFrame, batch_from_optimizer: bool,
                      split_cache_scope: bool) -> pd.DataFrame:
    """Label every run with its ablation setting (UDF / ST / Batch / +Cache / +Dedup)."""
    df = df.copy()
    udf = df["query_name"].str.contains("01")
    if batch_from_optimizer:
        if "optimizer.post_join_inference_strategy" in df:
            batch = df["optimizer.post_join_inference_strategy"].astype(str).str.upper() != "SEQUENTIAL"
        else:  # runs recorded before the post-join inference strategy replaced the rewrite flag
            batch = df["optimizer.rewrite_post_join_batch_inference"] != False  # noqa: E712
    else:
        batch = df["batch_size"] != 1
    cache_type = df["prediction_cache_type"].fillna("NONE").astype(str).str.strip().str.upper()
    cache = cache_type.replace("", "NONE") != "NONE"
    dedup = (
        df["use_batch_deduplication"].astype(str).str.strip().str.lower()
        .isin(["true", "1", "yes", "y"])
    )

    df["Setting"] = np.select(
        [
            udf,
            batch & cache & dedup,
            batch & cache & ~dedup,
            batch & ~cache & dedup,
            batch & ~cache & ~dedup,
            ~batch & cache,
            ~batch & ~cache,
        ],
        [
            "UDF",
            "Batch+Cache+Dedup",
            "Batch+Cache",
            "Batch+Dedup",
            "Batch",
            "ST+Cache",
            "ST",
        ],
        default="Unknown",
    )

    if split_cache_scope:
        scope = df["prediction_cache_scope"].fillna("").replace(
            {"GLOBAL": " Global", "THREAD_LOCAL": " Local"}
        )
        df["Setting"] = df["Setting"] + scope
    return df


def keep_inference_pipelines(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the flagged inference pipeline of each run; the UDF setting has no
    flag, so keep its heaviest pipeline (max average task duration) instead."""
    df = df.copy()
    df["is_inference_pipeline"] = df["is_inference_pipeline"].fillna(False).astype(bool)
    udf = df["Setting"].eq("UDF")
    udf_heaviest = (
        df[udf]
        .groupby(["query_name", "repetition"])["pipeline_avg_task_duration_us"]
        .transform("max")
    )
    udf_flag = pd.Series(False, index=df.index)
    udf_flag.loc[udf] = df.loc[udf, "pipeline_avg_task_duration_us"].eq(udf_heaviest)
    return df[df["is_inference_pipeline"] | udf_flag].copy()


def load_workload(csv_name: str, batch_from_optimizer: bool, split_cache_scope: bool,
                  config_cols: list[str]) -> pd.DataFrame:
    """Best configuration per setting: inference throughput of the highest-
    throughput config, P50/P95 sink latency [ms] of the best-end-to-end config."""
    df = load_csv("ablation", csv_name)
    df = classify_settings(df, batch_from_optimizer, split_cache_scope)
    df = keep_inference_pipelines(df)

    # Inference throughput = predictions per CPU-second: tuples / summed task busy
    # time. The numerator is tuples (predictions), NOT task count -- batching folds
    # many windows into one task, so a task rate would penalize batching.
    df["throughput"] = df["pipeline_tuples"] * 1_000_000 / df["pipeline_task_duration_us"]

    for col in METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=METRICS)

    # Median over repetitions per (setting, configuration).
    candidates = (
        df.groupby(["Setting"] + config_cols, dropna=False)
        .agg(
            throughput=("throughput", "median"),
            end_to_end_throughput=("end_to_end_throughput", "median"),
            p50_sink_latency_us=("p50_sink_latency_us", "median"),
            p95_sink_latency_us=("p95_sink_latency_us", "median"),
        )
        .reset_index()
    )

    best_e2e = candidates.loc[
        candidates.groupby("Setting")["end_to_end_throughput"].idxmax()
    ].set_index("Setting")
    return pd.DataFrame({
        "throughput": candidates.groupby("Setting")["throughput"].max(),
        "p50_ms": best_e2e["p50_sink_latency_us"] / 1_000,
        "p95_ms": best_e2e["p95_sink_latency_us"] / 1_000,
    })


def draw_latency_throughput(best: pd.DataFrame, *, order, figsize, label_fs, tick_fs,
                            speedup_fs, legend_fs, bar_width, window_ms, window_label,
                            log_latency, log_bottom, adjust, out_name, tick_labels=None,
                            rotate_ticks=None) -> None:
    """Shared two-panel drawing: P50/P95 latency bars + throughput bars with
    speedup-over-UDF labels. All knobs come from the per-chart wrappers."""
    order = [s for s in order if s in best.index]
    x = np.arange(len(order))
    thr_vals = best["throughput"].reindex(order).to_numpy()
    p50_vals = best["p50_ms"].reindex(order).to_numpy()
    p95_vals = best["p95_ms"].reindex(order).to_numpy()

    fig, (ax_lat, ax_thr) = plt.subplots(1, 2, figsize=figsize)

    # Right panel: inference throughput with speedup over the UDF baseline.
    bars_thr = ax_thr.bar(x, thr_vals, color=P50_COLOR, edgecolor="0.25", linewidth=0.3)
    ax_thr.set_ylabel("Throughput\n[windows/s]", fontsize=label_fs, labelpad=2)  # one line is taller than the panel
    ax_thr.yaxis.set_major_formatter(eng_formatter(decimals=1))

    udf_thr = best["throughput"].get("UDF", np.nan)
    labels = []
    for v in thr_vals:
        if np.isfinite(v) and np.isfinite(udf_thr) and udf_thr > 0:
            r = v / udf_thr
            labels.append(f"x{r:.0f}" if r >= 10 else "x" + f"{r:.1f}".rstrip("0").rstrip("."))
        else:
            labels.append("")
    ax_thr.bar_label(bars_thr, labels=labels, padding=1, fontsize=speedup_fs)
    ax_thr.set_ylim(top=np.nanmax(thr_vals) * 1.2)  # headroom for the labels

    # Left panel: P50 vs. P95 prediction latency with the window-size budget line.
    bars_p50 = ax_lat.bar(x - bar_width / 2, p50_vals, bar_width, label="P50",
                          color=P50_COLOR, edgecolor="0.25", linewidth=0.3)
    bars_p95 = ax_lat.bar(x + bar_width / 2, p95_vals, bar_width, label="P95",
                          color=P95_COLOR, edgecolor="0.25", linewidth=0.3)
    window_line = ax_lat.axhline(window_ms, linestyle="--", linewidth=0.8,
                                 color="black", label=window_label)
    ax_lat.set_ylabel("Latency [ms]", fontsize=label_fs, labelpad=2)

    if log_latency:
        # A log axis cannot contain 0: keep the bottom above the smallest bar but
        # below the window line so both stay legible.
        ax_lat.set_yscale("log")
        ax_lat.set_ylim(bottom=log_bottom)
        ax_lat.yaxis.set_major_formatter(eng_formatter(decimals=1))
    else:
        ax_lat.set_ylim(bottom=0)

    # tick_labels optionally abbreviates the setting names (spelled out in the caption).
    names = [tick_labels.get(setting, setting) for setting in order] if tick_labels else order
    for ax in (ax_lat, ax_thr):
        ax.set_xticks(x)
        rotate = rotate_ticks if rotate_ticks is not None else max(len(name) for name in names) > 6
        if rotate:
            ax.set_xticklabels(names, rotation=45, ha="right", rotation_mode="anchor", fontsize=tick_fs)
        else:  # short names fit level
            ax.set_xticklabels(names, fontsize=tick_fs)
        ax.tick_params(axis="both", labelsize=tick_fs, pad=2, length=2.5)

    # One legend above the latency panel, clear of the bars whatever their height; the window size is named there, so it
    # needs no tick of its own.
    fig.legend(handles=[bars_p50, bars_p95, window_line], loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               frameon=False, fontsize=legend_fs, handlelength=1.5, columnspacing=1.0, handletextpad=0.4,
               borderaxespad=0.1)

    fig.subplots_adjust(**adjust)
    save(fig, out_name)


def plot_e2e_audio(best: pd.DataFrame) -> None:
    """MUSAN / MHAtt-RNN ablation (log latency axis, 1 s sliding window)."""
    FIGSIZE = (COLUMN_WIDTH, 1.35)
    LABEL_FS = 8.5
    TICK_FS = 7
    SPEEDUP_FS = 7
    LEGEND_FS = 7.5
    BAR_WIDTH = 0.4
    ADJUST = dict(left=0.12, right=0.99, bottom=0.15, top=0.83, wspace=0.5)
    WINDOW_MS = 1_000
    LOG_BOTTOM = 100  # keeps the window line clear of the axis bottom
    ORDER = ["UDF", "ST", "ST+Cache", "Batch", "Batch+Dedup",
             "Batch+Cache", "Batch+Cache+Dedup"]

    draw_latency_throughput(
        best,
        order=ORDER,
        figsize=FIGSIZE,
        label_fs=LABEL_FS,
        tick_fs=TICK_FS,
        speedup_fs=SPEEDUP_FS,
        legend_fs=LEGEND_FS,
        bar_width=BAR_WIDTH,
        window_ms=WINDOW_MS,
        window_label="Sliding window size (1,000 ms)",
        log_latency=True,
        log_bottom=LOG_BOTTOM,
        adjust=ADJUST,
        out_name="e2e-audio.pdf",
    )


def plot_e2e_ts(best: pd.DataFrame) -> None:
    """CWRU / SARAD ablation (linear latency axis, 200 ms sliding window)."""
    FIGSIZE = (COLUMN_WIDTH, 1.6)
    LABEL_FS = 8.5
    TICK_FS = 7
    SPEEDUP_FS = 7
    LEGEND_FS = 7.5
    BAR_WIDTH = 0.4
    ADJUST = dict(left=0.13, right=0.99, bottom=0.3, top=0.855, wspace=0.5)
    WINDOW_MS = 200
    ORDER = ["UDF", "ST", "ST+Cache Local", "ST+Cache Global", "Batch",
             "Batch+Dedup", "Batch+Cache", "Batch+Cache+Dedup"]
    # Abbreviated so that the labels need little height: ST = single-tuple, B = batch, C = cache (L = thread-local,
    # G = global), D = deduplication. The paper's caption spells them out.
    TICK_LABELS = {"ST+Cache Local": "STC$_L$", "ST+Cache Global": "STC$_G$", "Batch": "B",
                   "Batch+Dedup": "BD", "Batch+Cache": "BC", "Batch+Cache+Dedup": "BCD"}

    draw_latency_throughput(
        best,
        order=ORDER,
        figsize=FIGSIZE,
        label_fs=LABEL_FS,
        tick_fs=TICK_FS,
        speedup_fs=SPEEDUP_FS,
        legend_fs=LEGEND_FS,
        bar_width=BAR_WIDTH,
        window_ms=WINDOW_MS,
        window_label="Sliding window size (200 ms)",
        log_latency=False,
        log_bottom=None,
        adjust=ADJUST,
        out_name="e2e-ts.pdf",
        tick_labels=TICK_LABELS,
        rotate_ticks=True,  # eight bars per panel: even the abbreviations do not fit level
    )


if __name__ == "__main__":
    setup_style()
    plot_e2e_audio(load_workload(
        "ablation_musan.csv",
        batch_from_optimizer=True,
        split_cache_scope=False,
        config_cols=["batch_size", "prediction_cache_type", "use_batch_deduplication"],
    ))
    plot_e2e_ts(load_workload(
        "ablation_cwru.csv",
        batch_from_optimizer=False,
        split_cache_scope=True,
        config_cols=["batch_size", "prediction_cache_type",
                     "number_of_entries_prediction_cache", "use_batch_deduplication"],
    ))
