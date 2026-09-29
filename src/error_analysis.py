"""Phase 6 - error analysis for the main Bi-LSTM + attention model.

Produces three things:

1. A confusion matrix for the test split.
2. The misclassified neutral reviews, dumped with their predicted probabilities
   so the failure modes can be read rather than guessed at.
3. Attention heatmaps over sample reviews, showing which words the model
   actually weighted.

Usage
-----
    python -m src.error_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.data import load_splits
from src.metrics import compute_metrics, plot_confusion_matrix
from src.predict import SentimentPredictor
from src.train import ReviewDataset, collate, evaluate
from src.viz import INK_SECONDARY, save_figure, use_project_style
from src.vocab import Vocabulary

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
RESULTS_DIR: Path = PROJECT_ROOT / "results"
MODELS_DIR: Path = PROJECT_ROOT / "models"

CLASS_NAMES: list[str] = ["negative", "neutral", "positive"]

# Single-hue sequential ramp for attention magnitude (never a rainbow).
ATTENTION_CMAP = LinearSegmentedColormap.from_list(
    "attention", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"]
)


def score_test_split(model_key: str = "bilstm_attention"):
    """Run the trained model over the test split, returning predictions+probs."""
    from torch.utils.data import DataLoader

    predictor = SentimentPredictor(model_key)
    _, _, test_df = load_splits()

    vocab = Vocabulary.load(MODELS_DIR / f"{model_key}_vocab.json")
    ds = ReviewDataset(test_df["text_clean"], test_df["label"], vocab,
                       predictor.max_len)
    loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate)

    y_true, y_pred, y_prob = evaluate(predictor.model, loader, predictor.device)
    test_df = test_df.copy()
    test_df["pred"] = y_pred
    test_df["pred_name"] = [CLASS_NAMES[p] for p in y_pred]
    for i, name in enumerate(CLASS_NAMES):
        test_df[f"p_{name}"] = y_prob[:, i]
    return predictor, test_df, compute_metrics(y_true, y_pred)


def dump_misclassified_neutrals(test_df, n: int = 20, seed: int = 42) -> list[dict]:
    """Collect misclassified neutral reviews, worst-confidence-first.

    Sampling the *most confidently wrong* cases is more informative than a random
    sample: those are where the model's reasoning breaks down hardest.
    """
    wrong = test_df[(test_df["label"] == 1) & (test_df["pred"] != 1)].copy()
    wrong["wrong_confidence"] = wrong[["p_negative", "p_positive"]].max(axis=1)
    wrong = wrong.sort_values("wrong_confidence", ascending=False).head(n)

    records = []
    for _, row in wrong.iterrows():
        records.append({
            "rating": int(row["rating"]),
            "true": "neutral",
            "predicted": row["pred_name"],
            "confidence": float(row["wrong_confidence"]),
            "p_negative": float(row["p_negative"]),
            "p_neutral": float(row["p_neutral"]),
            "p_positive": float(row["p_positive"]),
            "text": row["text"],
        })
    return records


def plot_attention_heatmaps(
    predictor: SentimentPredictor,
    reviews: list[str],
    max_tokens: int = 48,
    per_line: int = 12,
) -> plt.Figure:
    """Render tokens with background shading proportional to attention weight."""
    fig, axes = plt.subplots(len(reviews), 1,
                             figsize=(13, 2.15 * len(reviews)))
    if len(reviews) == 1:
        axes = [axes]

    for ax, review in zip(axes, reviews):
        pred = predictor.predict(review)
        tokens = pred.tokens[:max_tokens]
        weights = np.array(pred.attention[:max_tokens])
        # Normalise per review: attention is a distribution over that sequence,
        # so absolute values are not comparable across reviews of different length.
        norm = weights / weights.max() if weights.max() > 0 else weights

        ax.set_xlim(0, per_line)
        n_lines = int(np.ceil(len(tokens) / per_line))
        ax.set_ylim(n_lines, 0)
        ax.axis("off")

        for i, (tok, w) in enumerate(zip(tokens, norm)):
            row, col = divmod(i, per_line)
            ax.text(
                col + 0.5, row + 0.55, tok,
                ha="center", va="center", fontsize=9.5,
                color="#ffffff" if w > 0.62 else INK_SECONDARY,
                bbox=dict(boxstyle="round,pad=0.32", linewidth=0,
                          facecolor=ATTENTION_CMAP(float(w))),
            )

        top = ", ".join(t for t, _ in pred.top_tokens(4))
        ax.set_title(
            f"predicted {pred.label_name} ({pred.confidence:.0%})  |  "
            f"strongest: {top}",
            fontsize=10, loc="left", color=INK_SECONDARY,
        )

    fig.suptitle("Where the attention layer looks (darker = higher weight)",
                 x=0.5, y=1.0, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description="Error analysis for the main model")
    parser.add_argument("--model", default="bilstm_attention")
    parser.add_argument("--n-errors", type=int, default=20)
    args = parser.parse_args()

    use_project_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    predictor, test_df, metrics = score_test_split(args.model)
    print(f"test macro-F1 {metrics['macro_f1']:.4f}")

    save_figure(
        plot_confusion_matrix(metrics, "Bi-LSTM + Attention - test confusion matrix"),
        "08_confusion_matrix_main",
    )

    records = dump_misclassified_neutrals(test_df, args.n_errors)
    (RESULTS_DIR / "misclassified_neutrals.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    print(f"saved -> results/misclassified_neutrals.json ({len(records)} reviews)")

    # Sample reviews for the attention figure: one clear case per class plus two
    # genuinely mixed ones, which are where attention is most revealing.
    samples = [
        "the battery life is terrible and it stopped charging after two weeks",
        "works exactly as described and the sound quality is excellent",
        "the picture is sharp but the remote is cheap and the menus are slow",
        "it is okay for the price i guess nothing special either way",
        "i wanted to love this but it keeps disconnecting which ruins it",
    ]
    save_figure(plot_attention_heatmaps(predictor, samples), "09_attention_heatmaps")

    # Print the errors so they can be read and written up.
    for i, r in enumerate(records, 1):
        print(f"\n--- {i}. true neutral ({r['rating']} stars) -> "
              f"predicted {r['predicted']} at {r['confidence']:.1%} ---")
        print(r["text"][:400])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
