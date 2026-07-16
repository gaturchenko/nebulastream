"""Waveforms of the three MUSAN US-gov speech recordings used as sources.

Produces musan-audio.pdf: three stacked full-recording waveforms (no axes
ticks, file name as panel title), rendered from the peak-preserving min/max
envelopes in data/musan/ (see prepare_musan_waveforms.py for how those are
derived from the raw WAVs).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import load_csv, save, setup_style
from prepare_musan_waveforms import SELECTED_FILES


def load_envelopes() -> list[tuple[str, pd.DataFrame]]:
    return [
        (name, load_csv("musan", f"{name.removesuffix('.wav')}_envelope.csv"))
        for name in SELECTED_FILES
    ]


def plot_musan_audio(envelopes: list[tuple[str, pd.DataFrame]]) -> None:
    FIGSIZE = (3, 3)

    setup_style()

    fig, axes = plt.subplots(len(envelopes), 1, figsize=FIGSIZE)
    for ax, (name, env) in zip(axes, envelopes):
        # NaN-separated vertical min/max segments: visually identical to
        # plotting every sample at this figure size.
        x = np.repeat(env["sample"].to_numpy(), 3).astype(float)
        y = np.empty_like(x)
        y[0::3] = env["min"]
        y[1::3] = env["max"]
        y[2::3] = np.nan
        x[2::3] = np.nan
        ax.plot(x, y)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    save(fig, "musan-audio.pdf")


if __name__ == "__main__":
    plot_musan_audio(load_envelopes())
