"""Shared plotting style, so every figure in results/ reads as one system.

Sentiment is a *polarity* scale (negative -> neutral -> positive), so the colors
are a diverging pair with a neutral midpoint rather than three arbitrary
categorical hues: red and blue read as opposites, and gray reads as "neither".

The two poles were validated for colour-vision deficiency (worst adjacent pair
CVD Delta-E 21.6, normal vision 32.3 - both well clear of the 8 / 15 floors).
Bars are additionally direct-labelled with their values, so a reader never has to
rely on hue alone to identify a class.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
FIGURES_DIR: Path = PROJECT_ROOT / "results" / "figures"

# --- diverging palette: red <-> gray <-> blue -------------------------------- #
NEGATIVE_COLOR: str = "#e34948"
NEUTRAL_COLOR: str = "#898781"
POSITIVE_COLOR: str = "#2a78d6"

CLASS_COLORS: dict[str, str] = {
    "negative": NEGATIVE_COLOR,
    "neutral": NEUTRAL_COLOR,
    "positive": POSITIVE_COLOR,
}
CLASS_ORDER: list[str] = ["negative", "neutral", "positive"]

# --- chrome and ink: recessive, never competing with the data ---------------- #
SURFACE: str = "#fcfcfb"
INK_PRIMARY: str = "#0b0b0b"
INK_SECONDARY: str = "#52514e"
INK_MUTED: str = "#898781"
GRIDLINE: str = "#e1e0d9"

# A second, sequential hue for non-class series (e.g. model comparison bars).
SEQ_BLUE: list[str] = ["#cde2fb", "#86b6ef", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]


def use_project_style() -> None:
    """Apply the project's matplotlib defaults. Call once per notebook/script."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "figure.dpi": 110,
            # Recessive chrome: the grid supports the data, it does not compete.
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.edgecolor": GRIDLINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            # Ink: text wears text tokens, never a series color.
            "text.color": INK_PRIMARY,
            "axes.labelcolor": INK_SECONDARY,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
        }
    )


def colors_for(labels: list[str]) -> list[str]:
    """Map class names to their fixed colors (never cycled, never by rank)."""
    return [CLASS_COLORS[label] for label in labels]


def label_bars(
    ax: plt.Axes,
    bars,
    values: list[float],
    fmt: str = "{:,.0f}",
    pad: float = 0.01,
) -> None:
    """Direct-label bars, so identity/value never depends on reading an axis."""
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + span * pad,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK_SECONDARY,
        )


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Save a figure into results/figures/ and report where it went."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    print(f"saved -> results/figures/{name}.png")
    return path
