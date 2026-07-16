"""Regenerate data/musan/*_envelope.csv from the raw MUSAN WAV files.

The musan-audio figure shows the full ~10-minute waveforms of the three MUSAN
US-gov speech recordings used as benchmark sources. Committing the WAVs
(~19 MB each) for a 3x3-inch figure is wasteful: at that size matplotlib can
only render the per-pixel-column amplitude range anyway. This script reduces
each file to a peak-preserving per-bin min/max envelope (visually identical,
~100 KB total), which plot_musan_audio.py renders.

Usage: python prepare_musan_waveforms.py /path/to/MUSAN/raw
"""

import sys
import wave
from pathlib import Path

import numpy as np

from common import DATA_DIR

# Ranked by longest quiet run (see the MUSAN preprocessing notebook); the
# figure shows them in this order.
SELECTED_FILES = ["speech-us-gov-0242.wav", "speech-us-gov-0096.wav", "speech-us-gov-0021.wav"]

SAMPLE_RATE = 16_000
BINS = 3_000  # >> pixel columns of the rendered figure


def load_wav(path: Path) -> np.ndarray:
    """16 kHz mono PCM16 -> float32 in [-1, 1), truncated to whole seconds
    (the benchmark segmenter drops the final partial second)."""
    with wave.open(str(path)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (SAMPLE_RATE, 1, 2)
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    pcm = pcm[: len(pcm) - len(pcm) % SAMPLE_RATE]
    return pcm.astype(np.float32) / 32768.0


def minmax_envelope(samples: np.ndarray, bins: int) -> np.ndarray:
    """(bins, 3) array of [sample_index_of_bin_center, min, max]."""
    bin_size = int(np.ceil(len(samples) / bins))
    num_bins = int(np.ceil(len(samples) / bin_size))
    padded = np.pad(samples, (0, num_bins * bin_size - len(samples)), mode="edge")
    frames = padded.reshape(num_bins, bin_size)
    centers = np.arange(num_bins) * bin_size + bin_size / 2
    return np.column_stack((centers, frames.min(axis=1), frames.max(axis=1)))


if __name__ == "__main__":
    raw_dir = Path(sys.argv[1])
    out_dir = DATA_DIR / "musan"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SELECTED_FILES:
        env = minmax_envelope(load_wav(raw_dir / name), BINS)
        out = out_dir / f"{Path(name).stem}_envelope.csv"
        np.savetxt(out, env, fmt="%.0f,%.6f,%.6f", header="sample,min,max", comments="")
        print(f"wrote {out}")
