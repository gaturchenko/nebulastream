"""Systems comparison on OSU-RGB object detection (Raspberry Pi).

Produces e2e-systems.pdf: P95 end-to-end latency of NAVI vs. the Flink+OpenVINO
UDF baseline vs. TorchServe external serving, with the 10-FPS source frame
interval as the real-time budget line and per-bar slowdown factors over NAVI.

Data: data/systems/{nes_torchserve_osu_rgb_1,flink_openvino_stats_1,torchserve_osu_rgb_1}.csv
      (outputs of navi scripts/run_systems.sh + process_*results.py).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from common import load_csv, save, setup_style


def load_latencies() -> tuple[list[str], list[float]]:
    """P95 end-to-end latency [ms] per system, NAVI first (slowdown baseline)."""
    df_nes = load_csv("systems", "nes_torchserve_osu_rgb_1.csv")
    df_flink = load_csv("systems", "flink_openvino_stats_1.csv")
    df_torch = load_csv("systems", "torchserve_osu_rgb_1.csv")

    def row(df, suffix):
        return df[df["query_name"].str.endswith(suffix)].iloc[0]

    return ["NAVI", "Flink UDF", "TorchServe"], [
        row(df_nes, "test_02")["p95_sink_latency_us"] / 1_000.0,
        row(df_flink, "flink")["e2e_latency_ms_p95"],
        row(df_torch, "test_03")["e2e_latency_ms_p95"],
    ]


def plot_e2e_systems(systems: list[str], latencies_ms: list[float]) -> None:
    FIGSIZE = (3.0, 2.8)
    LABEL_FS = 15
    TICK_FS = 12
    SLOWDOWN_FS = 9.5
    LEGEND_FS = 9
    BAR_WIDTH = 0.6
    FRAME_INTERVAL_MS = 100  # 10 FPS source frame rate
    HEADROOM = 1.18  # y-limit factor above the tallest bar

    setup_style()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(systems))
    colors = sns.color_palette("tab10", n_colors=len(systems))
    bars = ax.bar(
        x, latencies_ms, width=BAR_WIDTH, color=colors,
        edgecolor="black", linewidth=0.8,
    )

    # Slowdown factor over NAVI above each bar ("x2" = twice NAVI's latency).
    baseline = latencies_ms[0]
    labels = []
    for v in latencies_ms:
        r = v / baseline
        labels.append(f"x{r:.0f}" if r >= 10 else "x" + f"{r:.1f}".rstrip("0").rstrip("."))
    ax.bar_label(bars, labels=labels, padding=-1, fontsize=SLOWDOWN_FS)

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=LABEL_FS, rotation=15)
    ax.set_ylabel("Latency P95 [ms]", fontsize=LABEL_FS)
    ax.set_ylim(0, max(latencies_ms) * HEADROOM)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    ax.axhline(
        FRAME_INTERVAL_MS, linestyle="--", linewidth=1, color="black",
        label="Source frame rate (10 FPS)",
    )
    ax.legend(
        loc="upper right", frameon=False, fontsize=LEGEND_FS,
        bbox_to_anchor=(1.0, 1.05),
    )

    fig.tight_layout()
    save(fig, "e2e-systems.pdf", pad_inches=0.1)


if __name__ == "__main__":
    plot_e2e_systems(*load_latencies())
