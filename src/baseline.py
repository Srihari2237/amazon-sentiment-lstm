"""Phase 3 - TF-IDF + Logistic Regression baseline.

Why a baseline at all: without one, "82% macro-F1" from a neural network means
nothing. This model is deliberately simple - it counts weighted words and draws
linear boundaries, with no notion of word order. Everything the recurrent models
add (sequence, context, attention) has to be justified by beating this.

The regularisation strength C is selected on the validation split only. The test
split is touched exactly once, at the end.

Usage
-----
    python -m src.baseline
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import load_splits
from src.metrics import compute_metrics, plot_confusion_matrix, print_report, record_result

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

MODEL_NAME: str = "TF-IDF + LogReg"
SEED: int = 42

# Word unigrams + bigrams: bigrams let the model see "not good" and "works well"
# as single features, which is the closest a bag-of-words model gets to order.
NGRAM_RANGE: tuple[int, int] = (1, 2)
MIN_DF: int = 3
MAX_FEATURES: int = 300_000

C_GRID: tuple[float, ...] = (0.5, 1.0, 4.0)


def build_pipeline(C: float) -> Pipeline:
    """TF-IDF features into a class-weighted multinomial logistic regression."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=NGRAM_RANGE,
                    min_df=MIN_DF,
                    max_features=MAX_FEATURES,
                    sublinear_tf=True,   # damps the effect of very repetitive text
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    # Without this the model would learn to answer "positive"
                    # almost always - 79% of the data says so.
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="TF-IDF + Logistic Regression baseline")
    parser.add_argument("--no-search", action="store_true",
                        help="skip the C search and use C=1.0")
    args = parser.parse_args()

    train_df, val_df, test_df = load_splits()
    print(f"train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,}")

    grid = (1.0,) if args.no_search else C_GRID

    best_C, best_score, best_model = None, -1.0, None
    for C in grid:
        start = time.perf_counter()
        pipe = build_pipeline(C)
        pipe.fit(train_df["text_clean"], train_df["label"])
        val_pred = pipe.predict(val_df["text_clean"])
        score = compute_metrics(val_df["label"], val_pred)["macro_f1"]
        print(f"  C={C:<5} val macro-F1 = {score:.4f}   "
              f"({time.perf_counter() - start:.0f}s)")
        if score > best_score:
            best_C, best_score, best_model = C, score, pipe

    print(f"\nselected C={best_C} on validation macro-F1 = {best_score:.4f}")

    n_features = len(best_model.named_steps["tfidf"].vocabulary_)
    print(f"vocabulary: {n_features:,} features")

    # Validation report (selection already done above).
    val_metrics = compute_metrics(val_df["label"], best_model.predict(val_df["text_clean"]))
    print_report(MODEL_NAME, "val", val_metrics)
    record_result(MODEL_NAME, "val", val_metrics,
                  extra={"C": best_C, "n_features": n_features,
                         "ngram_range": list(NGRAM_RANGE)})

    # Test split - touched once, after every decision is locked in.
    test_metrics = compute_metrics(test_df["label"],
                                   best_model.predict(test_df["text_clean"]))
    print_report(MODEL_NAME, "test", test_metrics)
    record_result(MODEL_NAME, "test", test_metrics)

    fig = plot_confusion_matrix(test_metrics, f"{MODEL_NAME} - test confusion matrix")
    from src.viz import save_figure
    save_figure(fig, "05_baseline_confusion_matrix")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "baseline_tfidf_logreg.joblib")
    print("saved model -> models/baseline_tfidf_logreg.joblib")

    return 0


if __name__ == "__main__":
    from src.viz import use_project_style

    use_project_style()
    raise SystemExit(main())
