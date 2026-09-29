"""Phase 5 - build the model comparison table and chart from results/metrics.json.

Reads whatever models have been recorded so far, so it can be re-run at any point
and simply reflects the current state of the project.

Usage
-----
    python -m src.compare
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import CLASS_NAMES, load_results
from src.viz import (
    CLASS_COLORS,
    CLASS_ORDER,
    INK_MUTED,
    SEQ_BLUE,
    label_bars,
    save_figure,
    use_project_style,
)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# Presentation order: simplest model first, so the table reads as a progression.
MODEL_ORDER: list[str] = [
    "TF-IDF + LogReg",
    "LSTM",
    "Bi-GRU",
    "Bi-LSTM + Attention",
    "DistilBERT (fine-tuned)",
]


def build_table(split: str = "test") -> pd.DataFrame:
    """One row per model: headline metrics plus per-class F1."""
    results = load_results()
    rows = []
    for name in MODEL_ORDER:
        entry = results.get(name)
        if not entry or split not in entry:
            continue
        m = entry[split]
        row = {
            "model": name,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
        }
        for cls in CLASS_NAMES:
            row[f"f1_{cls}"] = m["per_class"][cls]["f1"]
        meta = entry.get("meta", {})
        row["parameters"] = meta.get("parameters")
        rows.append(row)

    if not rows:
        raise RuntimeError(f"no results recorded for split '{split}' yet")
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, split: str) -> str:
    """Render the comparison as a markdown table for the README."""
    lines = [
        f"### Model comparison ({split} split, 30,000 reviews)",
        "",
        "| Model | Accuracy | **Macro-F1** | Weighted-F1 | F1 negative | F1 neutral | F1 positive | Params |",
        "|---|---|---|---|---|---|---|---|",
    ]
    best = df["macro_f1"].max()
    for _, r in df.iterrows():
        params = "-" if pd.isna(r["parameters"]) else f"{int(r['parameters']):,}"
        name = f"**{r['model']}**" if r["macro_f1"] == best else r["model"]
        macro = f"**{r['macro_f1']:.4f}**" if r["macro_f1"] == best \
            else f"{r['macro_f1']:.4f}"
        lines.append(
            f"| {name} | {r['accuracy']:.4f} | {macro} | {r['weighted_f1']:.4f} | "
            f"{r['f1_negative']:.4f} | {r['f1_neutral']:.4f} | "
            f"{r['f1_positive']:.4f} | {params} |"
        )
    results = load_results()
    bert_meta = results.get("DistilBERT (fine-tuned)", {}).get("meta", {})
    bert_rows = bert_meta.get("train_rows")

    lines += [
        "",
        "Always predicting the majority class scores **0.7914 accuracy** but only "
        "**0.2945 macro-F1** - which is why macro-F1 is the headline metric.",
        "",
        "**Training conditions.** All models are evaluated on the same full 30,000-row "
        "validation and test splits, and all use the same class-weighted loss and "
        "early stopping on validation macro-F1.",
    ]
    full_train = 240_000
    bert_len = bert_meta.get("max_len", 128)
    if bert_rows and bert_rows < full_train:
        # Only warn about an unequal comparison when it is actually unequal.
        lines.append(
            f"DistilBERT is **not** trained on equal footing: it saw a stratified "
            f"{bert_rows:,}-row subset of the {full_train:,}-row training split at "
            f"`max_len` {bert_len}, versus 230 for the recurrent models, to fit a "
            f"6 GB GPU in reasonable time. Its score is therefore a floor on what "
            f"it could reach, not a ceiling."
        )
    elif bert_rows:
        lines.append(
            f"DistilBERT was fine-tuned on the full {bert_rows:,}-row training "
            f"split at `max_len` {bert_len} (batch 16, fp16), so it is directly "
            f"comparable with the recurrent models."
        )
    return "\n".join(lines)


def plot_macro_f1(df: pd.DataFrame) -> plt.Figure:
    """Headline chart: macro-F1 by model.

    One series, so no legend - the title names the measure. Bars are ordered by
    model complexity rather than by score, so the chart reads as a progression
    and colour never encodes rank.
    """
    fig, ax = plt.subplots(figsize=(9, 4.4))

    # Sequential ramp: darker = more complex model. Colour follows the entity.
    colors = [SEQ_BLUE[min(i + 1, len(SEQ_BLUE) - 1)] for i in range(len(df))]
    bars = ax.bar(df["model"], df["macro_f1"], color=colors, width=0.6, zorder=3)
    label_bars(ax, bars, df["macro_f1"].tolist(), fmt="{:.4f}")

    # The "always predict positive" floor, as context for every bar above it.
    ax.axhline(0.2945, color=INK_MUTED, linestyle="--", linewidth=1.2, zorder=2)
    ax.text(len(df) - 0.45, 0.3045, "always-positive baseline (0.2945)",
            ha="right", fontsize=8.5, color=INK_MUTED)

    ax.set_title("Macro-F1 by model (test split)")
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, max(0.95, df["macro_f1"].max() * 1.18))
    ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    return fig


def plot_per_class_f1(df: pd.DataFrame) -> plt.Figure:
    """Grouped bars: per-class F1 for each model.

    Three series, so a legend is present and each class keeps its fixed colour
    from the project palette - identity never depends on position.
    """
    fig, ax = plt.subplots(figsize=(10, 4.6))

    x = np.arange(len(df))
    width = 0.26
    for i, cls in enumerate(CLASS_ORDER):
        offset = (i - 1) * width
        ax.bar(x + offset, df[f"f1_{cls}"], width=width * 0.92,
               label=cls, color=CLASS_COLORS[cls], zorder=3)

    ax.set_xticks(x, df["model"], fontsize=9)
    ax.set_title("Per-class F1: every model struggles on neutral")
    ax.set_ylabel("F1")
    # Headroom so the legend never sits on top of the positive bars, which run
    # to ~0.95 in every group.
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    return fig


def main() -> int:
    use_project_style()

    df = build_table("test")
    print(df.to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "comparison.csv", index=False)
    print("saved -> results/comparison.csv")

    markdown = to_markdown(df, "test")
    (RESULTS_DIR / "comparison.md").write_text(markdown + "\n", encoding="utf-8")
    print("saved -> results/comparison.md")
    print()
    print(markdown)

    save_figure(plot_macro_f1(df), "06_model_comparison")
    save_figure(plot_per_class_f1(df), "07_per_class_f1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
