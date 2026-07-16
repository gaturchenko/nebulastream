"""Optimization ablation on the Pi: UDF baseline vs. ST/Batch/Cache/Dedup.

Produces two figures with the same layout (P50/P95 prediction latency of the
best end-to-end-throughput configuration per setting on the left, inference
pipeline throughput of the best configuration with speedup-over-UDF labels on
the right):
  - e2e-audio.pdf: MUSAN audio / MHAtt-RNN (log latency axis, 1 s window)
  - e2e-ts.pdf:    CWRU vibration / SARAD (linear latency axis, 200 ms window,
                   with the prediction cache split into Local/Global scope)

Data: data/ablation/ablation_musan_2.csv, data/ablation/ablation_cwru_5.csv
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


def classify_settings(df: pd.DataFrame, batch_from_optimizer: bool,
                      split_cache_scope: bool) -> pd.DataFrame:
    """Label every run with its ablation setting (UDF / ST / Batch / +Cache / +Dedup)."""
    df = df.copy()
    udf = df["query_name"].str.contains("01")
    if batch_from_optimizer:
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
                            log_latency, log_bottom, legend_anchors, out_name) -> None:
    """Shared two-panel drawing: P50/P95 latency bars + throughput bars with
    speedup-over-UDF labels. All knobs come from the per-chart wrappers."""
    order = [s for s in order if s in best.index]
    x = np.arange(len(order))
    thr_vals = best["throughput"].reindex(order).to_numpy()
    p50_vals = best["p50_ms"].reindex(order).to_numpy()
    p95_vals = best["p95_ms"].reindex(order).to_numpy()

    fig, (ax_lat, ax_thr) = plt.subplots(1, 2, figsize=figsize)

    # Right panel: inference throughput with speedup over the UDF baseline.
    bars_thr = ax_thr.bar(x, thr_vals, color=P50_COLOR, edgecolor="0.25", linewidth=0.4)
    ax_thr.set_ylabel("Throughput [windows/s]", fontsize=label_fs)
    ax_thr.yaxis.set_major_formatter(eng_formatter(decimals=1))

    udf_thr = best["throughput"].get("UDF", np.nan)
    labels = []
    for v in thr_vals:
        if np.isfinite(v) and np.isfinite(udf_thr) and udf_thr > 0:
            r = v / udf_thr
            labels.append(f"x{r:.0f}" if r >= 10 else "x" + f"{r:.1f}".rstrip("0").rstrip("."))
        else:
            labels.append("")
    ax_thr.bar_label(bars_thr, labels=labels, padding=2, fontsize=speedup_fs)
    ax_thr.set_ylim(top=np.nanmax(thr_vals) * 1.12)  # headroom for the labels

    # Left panel: P50 vs. P95 prediction latency with the window-size budget line.
    bars_p50 = ax_lat.bar(x - bar_width / 2, p50_vals, bar_width, label="P50",
                          color=P50_COLOR, edgecolor="0.25", linewidth=0.4)
    bars_p95 = ax_lat.bar(x + bar_width / 2, p95_vals, bar_width, label="P95",
                          color=P95_COLOR, edgecolor="0.25", linewidth=0.4)
    window_line = ax_lat.axhline(window_ms, linestyle="--", linewidth=1,
                                 color="black", label=window_label)
    ax_lat.set_ylabel("Latency [ms]", fontsize=label_fs)

    if log_latency:
        # A log axis cannot contain 0: keep the bottom above the smallest bar but
        # below the window line so both stay legible.
        ax_lat.set_yscale("log")
        ax_lat.set_ylim(bottom=log_bottom)
        ax_lat.yaxis.set_major_formatter(eng_formatter(decimals=1))
    else:
        ax_lat.set_ylim(bottom=0)
        # Force 0 and the window-size line onto the y-axis alongside the auto ticks.
        top = ax_lat.get_ylim()[1]
        ax_lat.set_yticks(sorted({0, window_ms} | {t for t in ax_lat.get_yticks() if 0 < t <= top}))

    for ax in (ax_lat, ax_thr):
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=30, ha="right", fontsize=tick_fs)
        ax.tick_params(axis="y", labelsize=tick_fs)

    # Two stacked legends top-right: the wide window-size row above the
    # narrow percentile row. Nudge the anchors if they touch.
    leg_window = ax_lat.legend(handles=[window_line], loc="upper right",
                               bbox_to_anchor=legend_anchors[0], frameon=False,
                               fontsize=legend_fs)
    ax_lat.add_artist(leg_window)
    ax_lat.legend(handles=[bars_p50, bars_p95], loc="upper right",
                  bbox_to_anchor=legend_anchors[1], frameon=False, fontsize=legend_fs)

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.28, top=0.90, wspace=0.28)
    save(fig, out_name)


def plot_e2e_audio(best: pd.DataFrame) -> None:
    """MUSAN / MHAtt-RNN ablation (log latency axis, 1 s sliding window)."""
    FIGSIZE = (11, 4.2)
    LABEL_FS = 17
    TICK_FS = 13
    SPEEDUP_FS = 11
    LEGEND_FS = 11
    BAR_WIDTH = 0.4
    WINDOW_MS = 1_000
    LOG_BOTTOM = 100  # keeps the window line clear of the axis bottom
    ORDER = ["UDF", "ST", "ST+Cache", "Batch", "Batch+Dedup",
             "Batch+Cache", "Batch+Cache+Dedup"]
    LEGEND_ANCHORS = ((1.0, 1.03), (1.0, 0.95))

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
        legend_anchors=LEGEND_ANCHORS,
        out_name="e2e-audio.pdf",
    )


def plot_e2e_ts(best: pd.DataFrame) -> None:
    """CWRU / SARAD ablation (linear latency axis, 200 ms sliding window)."""
    FIGSIZE = (11, 4.2)
    LABEL_FS = 17
    TICK_FS = 13
    SPEEDUP_FS = 11
    LEGEND_FS = 11
    BAR_WIDTH = 0.4
    WINDOW_MS = 200
    ORDER = ["UDF", "ST", "ST+Cache Local", "ST+Cache Global", "Batch",
             "Batch+Dedup", "Batch+Cache", "Batch+Cache+Dedup"]
    LEGEND_ANCHORS = ((1.0, 1.0), (1.0, 0.92))

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
        legend_anchors=LEGEND_ANCHORS,
        out_name="e2e-ts.pdf",
    )


if __name__ == "__main__":
    setup_style()
    plot_e2e_audio(load_workload(
        "ablation_musan_2.csv",
        batch_from_optimizer=True,
        split_cache_scope=False,
        config_cols=["batch_size", "prediction_cache_type", "use_batch_deduplication"],
    ))
    plot_e2e_ts(load_workload(
        "ablation_cwru_5.csv",
        batch_from_optimizer=False,
        split_cache_scope=True,
        config_cols=["batch_size", "prediction_cache_type",
                     "number_of_entries_prediction_cache", "use_batch_deduplication"],
    ))
