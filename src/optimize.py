"""Post-hoc optimisation: per-class decision weights, and a model ensemble.

Motivation
----------
Every model in this project over-predicts neutral. On the test split the neutral
precision/recall gap runs from 0.22 to 0.29 - recall far ahead of precision -
which means each model is calling far more reviews neutral than really are.

That is a direct consequence of the class-weighted loss (neutral carries 11x the
weight of positive), and `argmax` then takes those inflated scores at face value.
F1 is maximised when precision and recall are balanced, so simply *scaling the
predicted probabilities per class* recovers macro-F1 without touching the model.

The scaling vector is searched on the **validation split only** and then applied
once to test - the same discipline as every other choice in this project. Nothing
here is fitted on test.

Usage
-----
    python -m src.optimize
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from src.data import load_splits
from src.metrics import compute_metrics, print_report, record_result

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
PROBS_DIR: Path = RESULTS_DIR / "probabilities"

SEED: int = 42
RECURRENT: dict[str, str] = {
    "lstm": "LSTM",
    "bigru": "Bi-GRU",
    "bilstm_attention": "Bi-LSTM + Attention",
}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# --------------------------------------------------------------------------- #
# Collecting probabilities (cached, so the search is instant on re-runs)
# --------------------------------------------------------------------------- #


def probabilities_for(model_name: str, split: str) -> np.ndarray | None:
    """Load cached probabilities for one model/split, computing them if needed."""
    PROBS_DIR.mkdir(parents=True, exist_ok=True)
    path = PROBS_DIR / f"{_slug(model_name)}_{split}.npy"
    if path.exists():
        return np.load(path)
    return None


def _save(model_name: str, split: str, probs: np.ndarray) -> None:
    PROBS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROBS_DIR / f"{_slug(model_name)}_{split}.npy", probs)


def collect_all_probabilities() -> dict[str, dict[str, np.ndarray]]:
    """Compute (and cache) validation/test probabilities for every trained model."""
    train_df, val_df, test_df = load_splits()
    frames = {"val": val_df, "test": test_df}
    out: dict[str, dict[str, np.ndarray]] = {}

    # --- TF-IDF baseline ---------------------------------------------------- #
    baseline_path = MODELS_DIR / "baseline_tfidf_logreg.joblib"
    if baseline_path.exists():
        name = "TF-IDF + LogReg"
        cached = {s: probabilities_for(name, s) for s in frames}
        if any(v is None for v in cached.values()):
            import joblib
            print(f"scoring {name}...")
            pipe = joblib.load(baseline_path)
            for split, frame in frames.items():
                probs = pipe.predict_proba(frame["text_clean"])
                _save(name, split, probs)
                cached[split] = probs
        out[name] = cached

    # --- recurrent models --------------------------------------------------- #
    import torch
    from torch.utils.data import DataLoader

    from src.train import ReviewDataset, collate, evaluate
    from src.vocab import Vocabulary

    for key, name in RECURRENT.items():
        if not (MODELS_DIR / f"{key}.pt").exists():
            continue
        cached = {s: probabilities_for(name, s) for s in frames}
        if any(v is None for v in cached.values()):
            from src.predict import SentimentPredictor
            print(f"scoring {name}...")
            predictor = SentimentPredictor(key)
            vocab = Vocabulary.load(MODELS_DIR / f"{key}_vocab.json")
            for split, frame in frames.items():
                ds = ReviewDataset(frame["text_clean"], frame["label"], vocab,
                                   predictor.max_len)
                loader = DataLoader(ds, batch_size=256, shuffle=False,
                                    collate_fn=collate)
                _, _, probs = evaluate(predictor.model, loader, predictor.device)
                _save(name, split, probs)
                cached[split] = probs
        out[name] = cached

    # --- DistilBERT ---------------------------------------------------------- #
    bert_path = MODELS_DIR / "distilbert.pt"
    if bert_path.exists():
        name = "DistilBERT (fine-tuned)"
        cached = {s: probabilities_for(name, s) for s in frames}
        if any(v is None for v in cached.values()):
            print(f"scoring {name}...")
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            from src.train_distilbert import (
                MODEL_ID,
                BertReviewDataset,
                make_collate,
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(bert_path, map_location=device)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_ID, num_labels=3
            ).to(device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            max_len = checkpoint.get("max_len", 128)

            for split, frame in frames.items():
                ds = BertReviewDataset(frame["text"], frame["label"],
                                       tokenizer, max_len)
                loader = DataLoader(ds, batch_size=128, shuffle=False,
                                    collate_fn=make_collate(tokenizer.pad_token_id))
                chunks = []
                with torch.no_grad():
                    for input_ids, attention, _ in loader:
                        input_ids = input_ids.to(device)
                        attention = attention.to(device)
                        with torch.autocast("cuda", dtype=torch.float16,
                                            enabled=device.type == "cuda"):
                            logits = model(input_ids=input_ids,
                                           attention_mask=attention).logits
                        chunks.append(
                            torch.softmax(logits.float(), dim=1).cpu().numpy()
                        )
                probs = np.concatenate(chunks)
                _save(name, split, probs)
                cached[split] = probs
        out[name] = cached

    return out


# --------------------------------------------------------------------------- #
# The search
# --------------------------------------------------------------------------- #


def search_class_weights(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_steps: int = 41,
    seed: int = SEED,
) -> tuple[np.ndarray, float]:
    """Find per-class multipliers maximising macro-F1.

    Coordinate ascent over a log-spaced grid. Only the *ratios* between the three
    multipliers matter (argmax is scale-invariant), so slot 2 (positive) is pinned
    at 1.0 and the other two are searched, then the whole vector is normalised.
    """
    from sklearn.metrics import f1_score

    grid = np.logspace(-1.0, 1.0, n_steps)   # 0.1x to 10x
    weights = np.ones(3, dtype=float)

    def macro_f1(w: np.ndarray) -> float:
        return f1_score(y_true, (probs * w).argmax(1), average="macro",
                        zero_division=0)

    best = macro_f1(weights)
    for _ in range(4):                       # a few sweeps is enough to converge
        improved = False
        for cls in (0, 1):                   # positive stays pinned at 1.0
            for value in grid:
                candidate = weights.copy()
                candidate[cls] = value
                score = macro_f1(candidate)
                if score > best + 1e-6:
                    best, weights, improved = score, candidate, True
        if not improved:
            break

    return weights / weights.sum() * 3, best


def apply_weights(probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (probs * weights).argmax(1)


def greedy_ensemble(
    val_probs: dict[str, np.ndarray],
    y_val: np.ndarray,
    n_rounds: int = 20,
) -> dict[str, float]:
    """Caruana-style greedy ensemble selection with replacement.

    Averaging *all* models is rarely optimal - a weak member drags the blend
    down. This repeatedly adds whichever model most improves validation macro-F1,
    allowing the same model to be picked again, so the final mixing weights are
    the selection counts. Models that never help simply get weight zero.

    Selection happens on validation only.
    """
    from sklearn.metrics import f1_score

    names = list(val_probs)
    counts: dict[str, int] = {n: 0 for n in names}
    running = np.zeros_like(next(iter(val_probs.values())))
    chosen = 0

    for _ in range(n_rounds):
        best_name, best_score = None, -1.0
        for name in names:
            blended = (running + val_probs[name]) / (chosen + 1)
            score = f1_score(y_val, blended.argmax(1), average="macro",
                             zero_division=0)
            if score > best_score:
                best_name, best_score = name, score
        running = running + val_probs[best_name]
        counts[best_name] += 1
        chosen += 1

    return {n: c / n_rounds for n, c in counts.items() if c}


def blend(probs: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Weighted average of per-model probability matrices."""
    total = sum(weights.values())
    out = np.zeros_like(next(iter(probs.values())))
    for name, w in weights.items():
        out += probs[name] * w
    return out / total


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-hoc threshold optimisation")
    parser.add_argument("--no-ensemble", action="store_true")
    args = parser.parse_args()

    _, val_df, test_df = load_splits()
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    all_probs = collect_all_probabilities()
    print(f"\nmodels available: {', '.join(all_probs)}\n")

    summary: list[dict] = []
    tuned_weights: dict[str, list[float]] = {}

    for name, splits in all_probs.items():
        base_test = compute_metrics(y_test, splits["test"].argmax(1))

        weights, val_score = search_class_weights(splits["val"], y_val)
        tuned_test = compute_metrics(y_test, apply_weights(splits["test"], weights))

        tuned_weights[name] = [round(float(w), 4) for w in weights]
        summary.append({
            "model": name,
            "macro_f1_argmax": base_test["macro_f1"],
            "macro_f1_tuned": tuned_test["macro_f1"],
            "delta": tuned_test["macro_f1"] - base_test["macro_f1"],
            "neutral_f1_argmax": base_test["per_class"]["neutral"]["f1"],
            "neutral_f1_tuned": tuned_test["per_class"]["neutral"]["f1"],
            "weights": tuned_weights[name],
            "val_macro_f1_tuned": val_score,
        })

        record_result(f"{name} + tuned", "test", tuned_test,
                      extra={"class_weights": tuned_weights[name],
                             "tuned_on": "validation"})

    # --- ensembles ----------------------------------------------------------- #
    if not args.no_ensemble and len(all_probs) >= 2:
        names = list(all_probs)

        # (a) plain average of everything - the naive baseline blend.
        ens_val = np.mean([all_probs[n]["val"] for n in names], axis=0)
        ens_test = np.mean([all_probs[n]["test"] for n in names], axis=0)
        base_test = compute_metrics(y_test, ens_test.argmax(1))
        weights, val_score = search_class_weights(ens_val, y_val)
        tuned_test = compute_metrics(y_test, apply_weights(ens_test, weights))
        summary.append({
            "model": f"Ensemble, mean ({len(names)})",
            "macro_f1_argmax": base_test["macro_f1"],
            "macro_f1_tuned": tuned_test["macro_f1"],
            "delta": tuned_test["macro_f1"] - base_test["macro_f1"],
            "neutral_f1_argmax": base_test["per_class"]["neutral"]["f1"],
            "neutral_f1_tuned": tuned_test["per_class"]["neutral"]["f1"],
            "weights": [round(float(w), 4) for w in weights],
            "val_macro_f1_tuned": val_score,
        })
        record_result("Ensemble (mean) + tuned", "test", tuned_test,
                      extra={"members": names,
                             "class_weights": [round(float(w), 4) for w in weights]})

        # (b) greedily weighted blend, members and weights chosen on validation.
        mix = greedy_ensemble({n: all_probs[n]["val"] for n in names}, y_val)
        print("\ngreedy ensemble mixing weights (selected on validation):")
        for n, w in sorted(mix.items(), key=lambda kv: -kv[1]):
            print(f"    {n:<28} {w:.2f}")

        g_val = blend({n: all_probs[n]["val"] for n in mix}, mix)
        g_test = blend({n: all_probs[n]["test"] for n in mix}, mix)
        base_test = compute_metrics(y_test, g_test.argmax(1))
        weights, val_score = search_class_weights(g_val, y_val)
        tuned_test = compute_metrics(y_test, apply_weights(g_test, weights))
        summary.append({
            "model": f"Ensemble, greedy ({len(mix)})",
            "macro_f1_argmax": base_test["macro_f1"],
            "macro_f1_tuned": tuned_test["macro_f1"],
            "delta": tuned_test["macro_f1"] - base_test["macro_f1"],
            "neutral_f1_argmax": base_test["per_class"]["neutral"]["f1"],
            "neutral_f1_tuned": tuned_test["per_class"]["neutral"]["f1"],
            "weights": [round(float(w), 4) for w in weights],
            "val_macro_f1_tuned": val_score,
            "mixing_weights": {k: round(v, 3) for k, v in mix.items()},
        })
        print_report("Ensemble (greedy) + tuned", "test", tuned_test)
        record_result("Ensemble (greedy) + tuned", "test", tuned_test,
                      extra={"mixing_weights": {k: round(v, 3) for k, v in mix.items()},
                             "class_weights": [round(float(w), 4) for w in weights]})

    # --- report -------------------------------------------------------------- #
    print()
    print("=" * 92)
    print("POST-HOC OPTIMISATION (weights searched on validation, applied to test)")
    print("=" * 92)
    print(f"{'model':<28}{'macro-F1':>10}{'tuned':>9}{'delta':>9}"
          f"{'neutF1':>9}{'tuned':>8}   weights")
    print("-" * 92)
    for row in summary:
        print(f"{row['model']:<28}{row['macro_f1_argmax']:>10.4f}"
              f"{row['macro_f1_tuned']:>9.4f}{row['delta']:>+9.4f}"
              f"{row['neutral_f1_argmax']:>9.4f}{row['neutral_f1_tuned']:>8.4f}"
              f"   {row['weights']}")
    print("=" * 92)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "optimization.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("saved -> results/optimization.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
