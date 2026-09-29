"""Vocabulary and GloVe embedding matrix for the recurrent models.

The vocabulary is built from the **training split only**. Building it from all
300k rows would let words that appear only in validation or test influence the
model's input space, which is a subtle form of leakage.

Words are mapped to ids; ids index into an embedding matrix whose rows are
pretrained GloVe vectors where available. Words with no GloVe vector get a small
random vector rather than zeros, so they stay distinguishable from padding and
can still be learned during fine-tuning.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MODELS_DIR: Path = PROJECT_ROOT / "models"
GLOVE_PATH: Path = PROJECT_ROOT / "data" / "glove" / "glove.6B.100d.txt"

PAD_TOKEN: str = "<pad>"
UNK_TOKEN: str = "<unk>"
PAD_ID: int = 0
UNK_ID: int = 1

MAX_VOCAB: int = 50_000
MIN_FREQ: int = 2
# EDA put the 90th percentile at 146 whitespace words and the 95th at 220.
# Tokenisation splits punctuation into its own tokens, which adds roughly 15%,
# so 230 keeps about the same coverage of full reviews.
MAX_LEN: int = 230
EMBED_DIM: int = 100

# GloVe was trained on Penn-style tokenised text: punctuation stands alone and
# contractions are split ("don't" -> "do" + "n't", "it's" -> "it" + "'s"). A
# plain str.split() leaves "great." and "don't" glued together, and neither has a
# GloVe vector - which silently sends the most sentiment-bearing words to <unk>.
# Matching GloVe's tokenisation is what makes the pretrained vectors usable.
_CONTRACTION_RE = re.compile(r"(n't|'s|'ve|'re|'m|'ll|'d)\b")
_PUNCT_RE = re.compile(r"([.,!?])")


def tokenize(text: str) -> list[str]:
    """Split cleaned text the way GloVe's own training data was split."""
    text = _CONTRACTION_RE.sub(r" \1", text)
    text = _PUNCT_RE.sub(r" \1 ", text)
    return text.split()


class Vocabulary:
    """Bidirectional token <-> id mapping with a fixed size cap."""

    def __init__(self, itos: list[str]) -> None:
        self.itos: list[str] = itos
        self.stoi: dict[str, int] = {tok: i for i, tok in enumerate(itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(
        cls,
        texts,
        max_size: int = MAX_VOCAB,
        min_freq: int = MIN_FREQ,
    ) -> "Vocabulary":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))

        # Reserve slots 0 and 1 for padding and unknown.
        keep = [
            tok for tok, n in counter.most_common(max_size - 2) if n >= min_freq
        ]
        print(f"vocabulary: {len(keep):,} kept of {len(counter):,} distinct tokens "
              f"(min_freq={min_freq}, cap={max_size:,})")
        return cls([PAD_TOKEN, UNK_TOKEN] + keep)

    def encode(self, text: str, max_len: int = MAX_LEN) -> list[int]:
        """Token string -> list of ids, truncated to ``max_len``."""
        ids = [self.stoi.get(tok, UNK_ID) for tok in tokenize(text)[:max_len]]
        return ids or [UNK_ID]   # never return an empty sequence

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.itos), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def build_embedding_matrix(
    vocab: Vocabulary,
    glove_path: Path = GLOVE_PATH,
    embed_dim: int = EMBED_DIM,
    seed: int = 42,
) -> np.ndarray:
    """Return a ``(len(vocab), embed_dim)`` matrix seeded with GloVe vectors.

    Coverage is printed because it is a genuinely diagnostic number: if only half
    the vocabulary has a pretrained vector, most of the benefit of using GloVe at
    all has been lost (usually a sign the text cleaning is too aggressive).
    """
    if not glove_path.exists():
        raise FileNotFoundError(
            f"GloVe vectors not found at {glove_path}. "
            "Run `python -m src.download_glove` first."
        )

    rng = np.random.default_rng(seed)
    # Uniform(-0.25, 0.25) is the standard init for words GloVe does not cover;
    # it matches the rough scale of real GloVe vectors.
    matrix = rng.uniform(-0.25, 0.25, size=(len(vocab), embed_dim)).astype(np.float32)
    matrix[PAD_ID] = 0.0   # padding must contribute nothing

    wanted = set(vocab.stoi)
    found = 0
    with open(glove_path, "r", encoding="utf-8") as fh:
        for line in fh:
            word, _, rest = line.partition(" ")
            if word not in wanted:
                continue
            matrix[vocab.stoi[word]] = np.fromstring(rest, sep=" ", dtype=np.float32)
            found += 1

    pct = 100 * found / len(vocab)
    print(f"GloVe coverage: {found:,}/{len(vocab):,} tokens ({pct:.1f}%) "
          f"have a pretrained vector")
    return matrix
