"""Prediction-cache scope microbenchmark: Local vs. Global thread scaling.

Produces caching-microbench.pdf: throughput speedup over each scope's own
single-thread throughput as thread count grows, one line per key size
(sequential color ramp), Local as solid circles vs. Global as dashed squares,
with the ideal linear-scaling diagonal. The Global (shared, mutex-protected)
cache stays flat and drops below 1x for small keys; Local tracks the ideal.

Data: data/cache/cache_scope_keysize.csv
      (PredictionCacheScopeMicrobenchmark key-size sweep).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from common import load_csv, save, setup_style

SCOPE_STYLE = {
    "Local": dict(marker="o", linestyle="-"),
    "Global": dict(marker="s", linestyle="--"),
}


def load_data() -> pd.DataFrame:
    """Average over repetitions and add per-scope thread-scaling speedup.

    The benchmark CSV labels the scopes private/shared; rename to Local/Global
    for consistency with the implementation.
    """
    df = load_csv("cache", "cache_scope_keysize.csv")
    df["scope"] = df["scope"].map({"private": "Local", "shared": "Global"})
    agg = (
        df.groupby(["policy", "scope", "key_bytes", "threads", "hit_percent"], as_index=False)
        .agg(
            ns_per_op=("ns_per_op_mean", "mean"),
            throughput=("throughput_ops_per_second", "mean"),
            observed_hit_rate=("observed_hit_rate", "mean"),
        )
    )
    base = (
        agg[agg["threads"] == 1]
        .rename(columns={"throughput": "throughput_1t"})
        [["policy", "scope", "key_bytes", "hit_percent", "throughput_1t"]]
    )
    agg = agg.merge(base, on=["policy", "scope", "key_bytes", "hit_percent"])
    agg["speedup"] = agg["throughput"] / agg["throughput_1t"]
    return agg


def fmt_bytes(b: int) -> str:
    if b >= 1 << 20:
        return rf"{b // (1 << 20)}\,MB"
    if b >= 1 << 10:
        return rf"{b // (1 << 10)}\,kB"
    return rf"{b}\,B"


def plot_cache_scaling(agg: pd.DataFrame) -> None:
    FIGSIZE = (5.0, 3.0)
    LABEL_FS = 15
    TICK_FS = 12
    LEGEND_FS = 8
    LEGEND_TITLE_FS = 10
    MARKER_SIZE = 8
    HIT_PERCENT = 50
    Y_TICKS = [0.1, 0.25, 1, 4, 16]
    RAMP = (0.4, 0.95)  # Blues colormap range for the key-size ramp

    setup_style()

    key_levels = sorted(agg["key_bytes"].unique())
    ramp = plt.cm.Blues(np.linspace(*RAMP, len(key_levels)))
    thread_ticks = sorted(agg["threads"].unique())

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(thread_ticks, thread_ticks, color="red", linewidth=2.0, linestyle=":")
    for key, color in zip(key_levels, ramp):
        for scope, style in SCOPE_STYLE.items():
            s = agg[
                (agg["scope"] == scope)
                & (agg["hit_percent"] == HIT_PERCENT)
                & (agg["key_bytes"] == key)
            ].sort_values("threads")
            ax.plot(s["threads"], s["speedup"], color=color,
                    markersize=MARKER_SIZE, linewidth=1.1, **style)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(thread_ticks)
    ax.set_xticklabels(thread_ticks, fontsize=TICK_FS)
    ax.set_yticks(Y_TICKS)
    ax.set_yticklabels([str(t) for t in Y_TICKS], fontsize=TICK_FS)
    ax.set_xlabel("Threads", fontsize=LABEL_FS)
    ax.set_ylabel(r"Throughput Scaling [x]", fontsize=LABEL_FS)
    ax.grid(alpha=0.25, linewidth=0.5, which="major")

    scope_handles = [
        Line2D([], [], color="0.2", linewidth=1.1, markersize=5, label=scope, **style)
        for scope, style in SCOPE_STYLE.items()
    ] + [Line2D([], [], color="red", linewidth=1.1, linestyle=":", label="Ideal")]
    key_handles = [
        Line2D([], [], color=c, linewidth=2.2, label=fmt_bytes(int(k)))
        for k, c in zip(key_levels, ramp)
    ]
    legend_scopes = ax.legend(
        handles=scope_handles, loc="upper left", fontsize=LEGEND_FS,
        framealpha=0.95, edgecolor="none", bbox_to_anchor=(0.0, 0.3),
    )
    ax.add_artist(legend_scopes)
    ax.legend(
        handles=key_handles, loc="upper left", fontsize=LEGEND_FS,
        framealpha=0.95, edgecolor="none", ncol=2,
        title="Key Size", title_fontsize=LEGEND_TITLE_FS, bbox_to_anchor=(0.0, 1.0),
    )

    fig.tight_layout()
    save(fig, "caching-microbench.pdf", pad_inches=0.1)


if __name__ == "__main__":
    plot_cache_scaling(load_data())
