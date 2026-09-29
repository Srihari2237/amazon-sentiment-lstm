"""Shared evaluation for every model in the project.

All five models (TF-IDF baseline, LSTM, Bi-GRU, Bi-LSTM+attention, DistilBERT)
report through this module, so the Phase 5 comparison table comes from one
definition of every number rather than five near-identical reimplementations.

Results accumulate in ``results/metrics.json``, keyed by model and split, so any
phase can be re-run independently without losing earlier results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
RESULTS_DIR: Path = PROJECT_ROOT / "results"
METRICS_PATH: Path = RESULTS_DIR / "metrics.json"

CLASS_NAMES: list[str] = ["negative", "neutral", "positive"]
LABELS: list[int] = [0, 1, 2]


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict:
    """Accuracy, macro-F1, weighted-F1, per-class scores and confusion matrix.

    Macro-F1 is the headline number for this project: it averages the three
    classes equally, so the tiny neutral class counts as much as the 79%
    positive class. Accuracy is reported only to show how misleading it is.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    report = classification_report(
        y_true, y_pred, labels=LABELS, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in CLASS_NAMES
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def load_results() -> dict:
    """Load the accumulated results registry (empty dict if none yet)."""
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}


def record_result(
    model_name: str,
    split: str,
    metrics: dict,
    extra: dict | None = None,
) -> None:
    """Merge one model/split result into ``results/metrics.json``."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    entry = results.setdefault(model_name, {})
    entry[split] = metrics
    if extra:
        entry.setdefault("meta", {}).update(extra)
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"recorded {model_name} / {split} -> results/metrics.json")


def print_report(model_name: str, split: str, metrics: dict) -> None:
    """Print a readable per-class report plus the confusion matrix."""
    print()
    print("=" * 68)
    print(f"{model_name}  |  {split}")
    print("=" * 68)
    print(f"accuracy    : {metrics['accuracy']:.4f}")
    print(f"macro-F1    : {metrics['macro_f1']:.4f}   <- headline metric")
    print(f"weighted-F1 : {metrics['weighted_f1']:.4f}")
    print()
    print(f"{'class':<10}{'precision':>11}{'recall':>9}{'f1':>9}{'support':>10}")
    print("-" * 68)
    for name in CLASS_NAMES:
        m = metrics["per_class"][name]
        print(f"{name:<10}{m['precision']:>11.4f}{m['recall']:>9.4f}"
              f"{m['f1']:>9.4f}{m['support']:>10,}")
    print()
    print("confusion matrix (rows = true, cols = predicted)")
    header = " " * 12 + "".join(f"{n:>12}" for n in CLASS_NAMES)
    print(header)
    for name, row in zip(CLASS_NAMES, metrics["confusion_matrix"]):
        print(f"{name:<12}" + "".join(f"{v:>12,}" for v in row))
    print("=" * 68)


def plot_confusion_matrix(
    metrics: dict,
    title: str,
    normalise: bool = True,
):
    """Render a confusion matrix as a sequential (single-hue) heatmap.

    Magnitude is a sequential encoding, so this uses one hue light->dark rather
    than a rainbow. Every cell is annotated, so the colour never has to be
    decoded against a scale to read a value.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    from src.viz import INK_MUTED, INK_SECONDARY

    cm = np.array(metrics["confusion_matrix"], dtype=float)
    counts = cm.copy()
    if normalise:
        cm = cm / cm.sum(axis=1, keepdims=True) * 100

    # Single-hue sequential ramp (the project's blue), light -> dark.
    cmap = LinearSegmentedColormap.from_list(
        "project_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"]
    )

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=100 if normalise else cm.max())

    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(3):
        for j in range(3):
            value = cm[i, j]
            # Ink flips to white only where the fill is dark enough to need it.
            colour = "#ffffff" if value > 55 else INK_SECONDARY
            label = f"{value:.1f}%\n{counts[i, j]:,.0f}" if normalise \
                else f"{counts[i, j]:,.0f}"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=9.5, color=colour)

    cbar = fig.colorbar(im, ax=ax, shrink=0.78)
    cbar.set_label("% of true class" if normalise else "reviews", color=INK_SECONDARY)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(color=INK_MUTED, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    return fig
