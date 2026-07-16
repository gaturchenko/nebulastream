"""Shared helpers for the paper figure scripts.

Every plot_*.py script in this directory follows the same shape: a load/prepare
step reading from data/, one function per chart (with the tweakable font and
figure-size constants at the top of its body), and a __main__ block that renders
the chart(s) into output/.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.ticker as mtick
import pandas as pd

FIGURES_DIR = Path(__file__).resolve().parent
DATA_DIR = FIGURES_DIR / "data"
OUTPUT_DIR = FIGURES_DIR / "output"


def setup_style() -> None:
    """Render all text with LaTeX using the paper's Libertine font."""
    mpl.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"""
            \usepackage{libertine}
            \usepackage[libertine]{newtxmath}
        """,
    })


def load_csv(*relative_parts: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR.joinpath(*relative_parts))


def eng_formatter(decimals: int = 0) -> mtick.FuncFormatter:
    """Axis formatter rendering 1500 as '1.5k', 2e6 as '2M', etc."""

    def fmt(x, _pos):
        for unit, factor in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
            if abs(x) >= factor:
                s = f"{x / factor:.{decimals}f}"
                if decimals:
                    s = s.rstrip("0").rstrip(".")
                return f"{s}{unit}"
        return f"{x:.0f}"

    return mtick.FuncFormatter(fmt)


def save(fig, name: str, tight: bool = True, pad_inches: float = 0.03) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / name
    fig.savefig(
        out,
        format="pdf",
        bbox_inches="tight" if tight else None,
        pad_inches=pad_inches,
    )
    print(f"wrote {out}")
