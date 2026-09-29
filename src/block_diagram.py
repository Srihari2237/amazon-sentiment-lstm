"""Render the system block diagram used in the project report (section 3.0).

Kept as code rather than a drawing tool so the diagram stays in sync with the
pipeline it documents and can be regenerated after any change.

The layout follows the real data flow rather than a tidy grid. The important
structure it has to show is that the pipeline produces **two** text
representations from the same review, and that they feed different families of
model: the aggressively cleaned text goes to the count-based and GloVe-based
models, while the lightly normalised text goes to DistilBERT, whose own
tokenizer expects natural text.

Usage
-----
    python -m src.block_diagram
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.viz import INK_MUTED, INK_PRIMARY, INK_SECONDARY, save_figure, use_project_style

# Palette: one hue family for the pipeline, a second for the models, green for
# the deployed artefact. Kept deliberately few so the diagram reads calmly.
SOURCE_FILL, SOURCE_EDGE = "#f1f0ec", "#c9c7bd"
STAGE_FILL, STAGE_EDGE = "#eaf2fd", "#8fbaf0"
MODEL_FILL, MODEL_EDGE = "#f6eee8", "#e0a882"
MAIN_FILL, MAIN_EDGE = "#d6e6fb", "#2a78d6"
OUT_FILL, OUT_EDGE = "#e6f4ee", "#1baf7a"
LANE_FILL = "#fafaf8"
LANE_EDGE = "#e4e3dc"


def _box(ax, x, y, w, h, title, sub=None, fill=STAGE_FILL, edge=STAGE_EDGE,
         title_size=9.0, sub_size=7.4, bold=True):
    """Rounded box with a bold title and an optional smaller subtitle."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=1.1",
            facecolor=fill, edgecolor=edge, linewidth=1.3, zorder=3,
        )
    )
    if sub:
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold" if bold else "normal",
                color=INK_PRIMARY, zorder=4)
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=sub_size, color=INK_SECONDARY, zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold" if bold else "normal",
                color=INK_PRIMARY, zorder=4)


def _down(ax, x, y_from, y_to, colour=INK_MUTED):
    ax.add_patch(FancyArrowPatch(
        (x, y_from), (x, y_to), arrowstyle="-|>", mutation_scale=11,
        linewidth=1.2, color=colour, zorder=2, shrinkA=0, shrinkB=0))


def _right(ax, x_from, x_to, y, colour=INK_MUTED):
    ax.add_patch(FancyArrowPatch(
        (x_from, y), (x_to, y), arrowstyle="-|>", mutation_scale=11,
        linewidth=1.2, color=colour, zorder=2, shrinkA=0, shrinkB=0))


def _elbow(ax, x_from, y_from, x_to, y_to, colour=INK_MUTED):
    """Vertical drop, horizontal run, then an arrow down into the target."""
    mid = (y_from + y_to) / 2
    ax.plot([x_from, x_from], [y_from, mid], color=colour, linewidth=1.2, zorder=2)
    ax.plot([x_from, x_to], [mid, mid], color=colour, linewidth=1.2, zorder=2)
    ax.add_patch(FancyArrowPatch(
        (x_to, mid), (x_to, y_to), arrowstyle="-|>", mutation_scale=11,
        linewidth=1.2, color=colour, zorder=2, shrinkA=0, shrinkB=0))


def _lane(ax, x, y, w, h, label):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
        facecolor=LANE_FILL, edgecolor=LANE_EDGE, linewidth=1.0,
        linestyle=(0, (4, 3)), zorder=1))
    ax.text(x + w / 2, y + h - 2.2, label, ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=INK_MUTED, zorder=2)


def _stage_label(ax, y, text):
    ax.text(-1.0, y, text, ha="right", va="center", fontsize=8.0,
            fontweight="bold", color=INK_MUTED, rotation=90)


def build() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12.6, 9.2))
    ax.set_xlim(-6, 106)
    ax.set_ylim(-2, 102)
    ax.axis("off")
    ax.grid(False)

    BH = 6.4          # standard box height
    CX = 50           # centre line

    # ================= 1. data preparation (single column) ================= #
    _stage_label(ax, 88, "DATA PREPARATION")

    _box(ax, 28, 93.0, 44, BH, "Amazon Reviews 2023  ·  Electronics",
         "22.6 GB JSONL, streamed from the Hugging Face Hub (never downloaded)",
         SOURCE_FILL, SOURCE_EDGE, title_size=9.6)
    _down(ax, CX, 93.0, 89.6)

    _box(ax, 28, 83.2, 44, BH, "Sample 300,000 reviews",
         "100,000-row shuffle buffer  ·  seed 42", STAGE_FILL, STAGE_EDGE)
    _down(ax, CX, 83.2, 79.8)

    _box(ax, 28, 73.4, 44, BH, "Label, clean and de-duplicate",
         "1-2 = negative  ·  3 = neutral  ·  4-5 = positive", STAGE_FILL, STAGE_EDGE)
    _down(ax, CX, 73.4, 70.0)

    _box(ax, 28, 63.6, 44, BH, "Stratified split  (seed 42)",
         "240,000 train  ·  30,000 validation  ·  30,000 test",
         STAGE_FILL, STAGE_EDGE)

    # ============ 2. two representations feeding three model lanes ========= #
    _stage_label(ax, 40, "REPRESENTATION  &  MODELS")

    # lane panels
    _lane(ax, 0.0, 21.0, 30.0, 36.0, "text_clean  →  counts")
    _lane(ax, 33.0, 21.0, 40.0, 36.0, "text_clean  →  GloVe embeddings")
    _lane(ax, 76.0, 21.0, 24.0, 36.0, "text  →  WordPiece")

    # split feeds all three lanes
    _elbow(ax, CX, 63.6, 15.0, 50.0)
    _down(ax, CX, 63.6, 50.0)
    _elbow(ax, CX, 63.6, 88.0, 50.0)

    # --- lane A: bag of words ---
    _box(ax, 2.5, 43.6, 25, BH, "TF-IDF vectoriser",
         "1-2 grams  ·  300,000 features", STAGE_FILL, STAGE_EDGE, 8.6, 7.0)
    _down(ax, 15.0, 43.6, 36.4)
    _box(ax, 2.5, 30.0, 25, BH, "Logistic Regression",
         "baseline", MODEL_FILL, MODEL_EDGE, 8.6, 7.0)

    # --- lane B: recurrent models ---
    _box(ax, 35.5, 43.6, 35, BH, "Penn-style tokenise  →  vocabulary",
         "44,811 tokens, built from training data only",
         STAGE_FILL, STAGE_EDGE, 8.6, 7.0)
    _down(ax, 53.0, 43.6, 40.0)
    _box(ax, 35.5, 33.6, 35, BH, "GloVe 100d embedding matrix",
         "73.1% of the vocabulary has a pretrained vector",
         STAGE_FILL, STAGE_EDGE, 8.6, 7.0)

    # Three recurrent models side by side, sized to sit inside the lane panel
    # (which ends at x = 73) rather than spilling into the transformer lane.
    _down(ax, 40.5, 33.6, 30.6)
    _down(ax, 53.5, 33.6, 30.6)
    _down(ax, 66.5, 33.6, 30.6)
    _box(ax, 34.75, 23.4, 11.5, 7.2, "LSTM", "unidirectional",
         MODEL_FILL, MODEL_EDGE, 8.4, 6.8)
    _box(ax, 47.75, 23.4, 11.5, 7.2, "Bi-GRU", "bidirectional",
         MODEL_FILL, MODEL_EDGE, 8.4, 6.8)
    _box(ax, 60.75, 23.4, 11.5, 7.2, "Bi-LSTM + Attn", "main model",
         MAIN_FILL, MAIN_EDGE, 8.0, 6.8)

    # --- lane C: transformer ---
    _box(ax, 78.0, 43.6, 20, BH, "WordPiece tokenizer",
         "casing and punctuation kept", STAGE_FILL, STAGE_EDGE, 8.6, 7.0)
    _down(ax, 88.0, 43.6, 30.6)
    _box(ax, 78.0, 23.4, 20, 7.2, "DistilBERT", "fine-tuned transformer",
         MODEL_FILL, MODEL_EDGE, 8.6, 7.0)

    # ===================== 3. training and evaluation ====================== #
    _stage_label(ax, 10, "EVALUATION  &  DEPLOYMENT")

    # converge the five models onto one line
    for x in (15.0, 40.5, 53.5, 66.5, 88.0):
        ax.plot([x, x], [23.4, 19.0], color=INK_MUTED, linewidth=1.2, zorder=2)
    ax.plot([15.0, 88.0], [19.0, 19.0], color=INK_MUTED, linewidth=1.2, zorder=2)
    _down(ax, CX, 19.0, 15.9)

    _box(ax, 20, 9.5, 60, BH, "Class-weighted training  ·  fp16  ·  early stopping on validation macro-F1",
         "negative 2.45  ·  neutral 4.58  ·  positive 0.42", STAGE_FILL, STAGE_EDGE,
         title_size=8.8, sub_size=7.2)

    # three outputs
    _elbow(ax, CX, 9.5, 16.0, 5.4)
    _down(ax, CX, 9.5, 5.4)
    _elbow(ax, CX, 9.5, 84.0, 5.4)

    _box(ax, 3.0, -1.0, 26, BH, "Metrics",
         "macro-F1  ·  per-class  ·  confusion", STAGE_FILL, STAGE_EDGE, 8.6, 7.0)
    _box(ax, 36.0, -1.0, 28, BH, "Post-hoc optimisation",
         "class rescaling  ·  ensemble", STAGE_FILL, STAGE_EDGE, 8.6, 7.0)
    _box(ax, 71.0, -1.0, 26, BH, "Streamlit application",
         "prediction + attention weights", OUT_FILL, OUT_EDGE, 8.6, 7.0)

    fig.suptitle(
        "Customer review sentiment analysis — system block diagram",
        x=0.5, y=0.985, fontsize=13.5, fontweight="bold",
    )
    fig.subplots_adjust(left=0.035, right=0.99, top=0.95, bottom=0.015)
    return fig


if __name__ == "__main__":
    use_project_style()
    save_figure(build(), "00_block_diagram")
