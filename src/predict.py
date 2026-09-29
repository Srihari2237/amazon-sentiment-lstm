"""Interactive inference - type a review, get the sentiment back.

Shared by the terminal REPL (``python -m src.predict``) and the Streamlit app, so
there is exactly one definition of how a user's text becomes a prediction.

The important detail: this reuses ``normalise_text`` and ``clean_for_neural``
from ``src.data``. If inference cleaned text even slightly differently from
training, the model would see a different distribution than it learned on and
quietly lose accuracy - the kind of bug that produces no error message.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.data import clean_for_neural, normalise_text
from src.models import DISPLAY_NAMES, build_model
from src.vocab import MAX_LEN, Vocabulary, tokenize

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MODELS_DIR: Path = PROJECT_ROOT / "models"

CLASS_NAMES: list[str] = ["negative", "neutral", "positive"]
# Terminal colours matching the project's diverging palette.
ANSI: dict[str, str] = {"negative": "\033[91m", "neutral": "\033[90m",
                        "positive": "\033[94m"}
RESET: str = "\033[0m"


@dataclass
class Prediction:
    """One model prediction, with everything the UIs need to explain it."""

    label: int
    label_name: str
    confidence: float
    probabilities: dict[str, float]
    tokens: list[str]
    attention: list[float]        # empty when the model has no attention layer

    def top_tokens(self, k: int = 8) -> list[tuple[str, float]]:
        """The k tokens the attention layer weighted most heavily."""
        if not self.attention:
            return []
        order = np.argsort(self.attention)[::-1][:k]
        return [(self.tokens[i], float(self.attention[i])) for i in order]


class SentimentPredictor:
    """Loads a trained checkpoint and predicts sentiment for raw review text."""

    def __init__(
        self,
        model_key: str = "bilstm_attention",
        device: str | None = None,
    ) -> None:
        checkpoint_path = MODELS_DIR / f"{model_key}.pt"
        vocab_path = MODELS_DIR / f"{model_key}_vocab.json"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No trained model at {checkpoint_path}. "
                f"Train one first:  python -m src.train --model {model_key}"
            )

        self.model_key = model_key
        self.display_name = DISPLAY_NAMES.get(model_key, model_key)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        cfg = checkpoint["config"]
        self.max_len: int = cfg.get("max_len", MAX_LEN)
        self.vocab = Vocabulary.load(vocab_path)

        extra = ({"attention_dim": cfg["attention_dim"]}
                 if model_key == "bilstm_attention" else {})
        self.model = build_model(
            model_key,
            vocab_size=checkpoint["vocab_size"],
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            embedding_matrix=None,   # weights come from the checkpoint
            **extra,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.val_macro_f1: float = checkpoint.get("val_macro_f1", float("nan"))

    @torch.no_grad()
    def predict(self, text: str) -> Prediction:
        """Predict sentiment for one raw review."""
        # Exactly the training-time cleaning - see module docstring.
        normalised = normalise_text(None, text)
        cleaned = clean_for_neural(normalised)
        if not cleaned:
            raise ValueError("no usable text after cleaning")

        # Same tokenizer as training, so the displayed tokens line up 1:1 with
        # the attention weights the model produced.
        tokens = tokenize(cleaned)[: self.max_len]
        ids = self.vocab.encode(cleaned, self.max_len)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(ids)], dtype=torch.int64)

        logits, attention = self.model(x, lengths)
        probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
        label = int(probs.argmax())

        weights: list[float] = []
        if attention is not None:
            weights = attention[0, : len(tokens)].float().cpu().numpy().tolist()

        return Prediction(
            label=label,
            label_name=CLASS_NAMES[label],
            confidence=float(probs[label]),
            probabilities={n: float(p) for n, p in zip(CLASS_NAMES, probs)},
            tokens=tokens,
            attention=weights,
        )


def _bar(value: float, width: int = 28) -> str:
    filled = int(round(value * width))
    return "#" * filled + "." * (width - filled)


def _render(prediction: Prediction) -> None:
    colour = ANSI[prediction.label_name]
    print()
    print(f"  {colour}{prediction.label_name.upper()}{RESET} "
          f"({prediction.confidence:.1%} confident)")
    print()
    for name, p in prediction.probabilities.items():
        marker = "<-" if name == prediction.label_name else "  "
        print(f"    {ANSI[name]}{name:<9}{RESET} {_bar(p)} {p:>6.1%} {marker}")

    top = prediction.top_tokens(8)
    if top:
        print()
        print("    words the model focused on:")
        print("      " + "  ".join(f"{tok} ({w:.2f})" for tok, w in top))
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive sentiment prediction for Amazon reviews",
    )
    parser.add_argument("--model", default="bilstm_attention",
                        choices=list(DISPLAY_NAMES))
    parser.add_argument("--text", help="predict one review and exit")
    args = parser.parse_args()

    predictor = SentimentPredictor(args.model)

    if args.text:
        _render(predictor.predict(args.text))
        return 0

    print("=" * 66)
    print(f"  {predictor.display_name}  |  device: {predictor.device}"
          f"  |  val macro-F1: {predictor.val_macro_f1:.4f}")
    print("=" * 66)
    print("  Type a review and press Enter. Ctrl-C or 'quit' to exit.")
    print()

    while True:
        try:
            text = input("review > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0
        if text.lower() in {"quit", "exit", "q"}:
            print("bye")
            return 0
        if not text:
            continue
        try:
            _render(predictor.predict(text))
        except ValueError as exc:
            print(f"  (skipped: {exc})\n")


if __name__ == "__main__":
    raise SystemExit(main())
