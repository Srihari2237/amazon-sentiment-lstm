"""Data pipeline for the Amazon Reviews 2023 (Electronics) sentiment project.

What this module does, end to end:

1.  Streams reviews from the Hugging Face Hub. The Electronics split is a single
    22.6 GB JSONL file, so it is *never* downloaded in full - we read it as a
    stream and stop once we have enough rows.
2.  Maps the 1-5 star rating onto three sentiment classes.
3.  Cleans the review text into two variants (see `clean_for_neural` below).
4.  Caches the sampled rows to Parquet so later runs cost seconds, not minutes.
5.  Produces a reproducible, stratified 80/10/10 train/val/test split and
    reports the class distribution.

Every model in this project (TF-IDF baseline, LSTM, Bi-GRU, Bi-LSTM+attention,
DistilBERT) reads the splits produced here, so they are all compared on exactly
the same data.

Usage
-----
    python -m src.data --smoke --n-rows 2000   # quick end-to-end check
    python -m src.data                         # full 300k build
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

SEED: int = 42

# Direct path to the raw JSONL on the Hub. We deliberately bypass the dataset's
# official loading script (`trust_remote_code=True`), because `datasets>=3.0`
# dropped support for script-based datasets. Reading the file through the
# built-in `json` builder works on every modern version.
HF_DATA_FILE: str = (
    "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "raw/review_categories/Electronics.jsonl"
)
HF_REPO_ID: str = "McAuley-Lab/Amazon-Reviews-2023"
HF_CONFIG: str = "raw_review_Electronics"

N_ROWS: int = 300_000
# The JSONL is stored in an arbitrary order that may correlate with product or
# user. A shuffle buffer mixes rows as they stream past so the sample is not
# simply "the top of the file".
SHUFFLE_BUFFER: int = 100_000

VAL_FRACTION: float = 0.10
TEST_FRACTION: float = 0.10

# Label scheme from CLAUDE.md: 1-2 stars = negative, 3 = neutral, 4-5 = positive.
LABEL_NAMES: dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}
NEGATIVE, NEUTRAL, POSITIVE = 0, 1, 2

# Reviews shorter than this (after cleaning) carry no usable signal.
MIN_CHARS: int = 5

# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #

_TAG_RE = re.compile(r"<[^>]+>")                      # <br />, <span>, ...
_URL_RE = re.compile(r"(?:https?://|www\.)\S+")
_WS_RE = re.compile(r"\s+")
_REPEAT_RE = re.compile(r"(.)\1{2,}")                 # "sooooo" -> "soo"
_KEEP_RE = re.compile(r"[^a-z0-9 .,!?']")             # drop everything else


def normalise_text(title: str | None, body: str | None) -> str:
    """Join the review title and body into one lightly-normalised string.

    The title is included because on Amazon it often carries the sharpest
    sentiment ("Waste of money", "Works perfectly"). Casing and punctuation are
    preserved here - this is the variant a transformer such as DistilBERT wants,
    since its tokenizer was trained on natural text.
    """
    parts = [p.strip() for p in (title or "", body or "") if p and p.strip()]
    text = ". ".join(parts)

    text = html.unescape(text)      # "&amp;" -> "&"
    text = _TAG_RE.sub(" ", text)   # strip leftover HTML markup
    text = _URL_RE.sub(" ", text)   # links are noise for sentiment
    text = _WS_RE.sub(" ", text)    # collapse newlines / runs of spaces
    return text.strip()


def clean_for_neural(text: str) -> str:
    """Aggressively normalise text for the count-based and GloVe-based models.

    TF-IDF and a GloVe vocabulary have no sub-word fallback: every unseen
    surface form is simply an out-of-vocabulary token. Lowercasing, collapsing
    character runs and dropping exotic symbols keeps the vocabulary small and
    maps more tokens onto real GloVe vectors. Basic punctuation is kept because
    "!" and "?" are genuinely predictive of sentiment.
    """
    text = text.lower()
    text = _REPEAT_RE.sub(r"\1\1", text)
    text = _KEEP_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def rating_to_label(rating: object) -> int | None:
    """Map a star rating to a sentiment class, or ``None`` if unusable.

    Ratings arrive as floats (``5.0``), so they are rounded to the nearest int
    and anything outside 1-5 is rejected rather than silently bucketed.
    """
    try:
        stars = int(round(float(rating)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

    if stars in (1, 2):
        return NEGATIVE
    if stars == 3:
        return NEUTRAL
    if stars in (4, 5):
        return POSITIVE
    return None


def _fingerprint(text: str) -> bytes:
    """Stable 8-byte hash used for de-duplication.

    ``hash()`` is not usable here: Python randomises string hashing per process,
    which would make the sample depend on the run rather than on the seed.
    """
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


def _open_stream(seed: int, shuffle_buffer: int) -> Iterable[dict]:
    """Open the Electronics reviews as a shuffled streaming dataset."""
    from datasets import load_dataset  # imported lazily: keeps --help instant

    try:
        stream = load_dataset(
            "json", data_files=HF_DATA_FILE, split="train", streaming=True
        )
    except Exception as exc:  # pragma: no cover - depends on `datasets` version
        print(
            f"[warn] direct JSONL streaming failed ({exc}); "
            f"falling back to the legacy loader script.",
            file=sys.stderr,
        )
        stream = load_dataset(
            HF_REPO_ID,
            HF_CONFIG,
            split="full",
            streaming=True,
            trust_remote_code=True,
        )

    return stream.shuffle(seed=seed, buffer_size=shuffle_buffer)


def stream_reviews(
    n_rows: int = N_ROWS,
    seed: int = SEED,
    shuffle_buffer: int = SHUFFLE_BUFFER,
) -> pd.DataFrame:
    """Stream, filter and clean reviews until ``n_rows`` usable rows are found.

    Rows are dropped when the rating is unusable, the cleaned text is too short,
    or the review is an exact duplicate of one already kept. Duplicates matter:
    Amazon contains copy-pasted and templated reviews, and if the same text
    landed in both train and test every reported score would be inflated.
    """
    # The shuffle buffer must fill before the first row is yielded, so warn the
    # user rather than leaving them staring at an idle progress bar.
    print(
        f"Streaming Electronics reviews from the Hugging Face Hub "
        f"(filling a {shuffle_buffer:,}-row shuffle buffer first)..."
    )

    stream = _open_stream(seed=seed, shuffle_buffer=shuffle_buffer)

    rows: list[dict] = []
    seen: set[bytes] = set()
    scanned = 0
    dropped = {"rating": 0, "empty": 0, "duplicate": 0}

    with tqdm(total=n_rows, unit="review", desc="collecting") as bar:
        for record in stream:
            scanned += 1

            label = rating_to_label(record.get("rating"))
            if label is None:
                dropped["rating"] += 1
                continue

            text = normalise_text(record.get("title"), record.get("text"))
            text_clean = clean_for_neural(text)
            if len(text_clean) < MIN_CHARS:
                dropped["empty"] += 1
                continue

            key = _fingerprint(text_clean)
            if key in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(key)

            rows.append(
                {
                    "text": text,                  # for DistilBERT
                    "text_clean": text_clean,      # for TF-IDF / LSTM / GRU
                    "rating": int(round(float(record["rating"]))),
                    "label": label,
                    "label_name": LABEL_NAMES[label],
                }
            )
            bar.update(1)

            if len(rows) >= n_rows:
                break

    print(
        f"Scanned {scanned:,} records -> kept {len(rows):,}. "
        f"Dropped: {dropped['rating']:,} bad rating, "
        f"{dropped['empty']:,} too short, "
        f"{dropped['duplicate']:,} duplicate."
    )

    return pd.DataFrame(rows)


def build_sample(
    n_rows: int = N_ROWS,
    seed: int = SEED,
    shuffle_buffer: int = SHUFFLE_BUFFER,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return the sampled reviews, streaming them only when necessary."""
    cache_path = RAW_DIR / f"electronics_{n_rows}.parquet"

    if use_cache and not force_refresh and cache_path.exists():
        print(f"Loading cached sample from {cache_path.relative_to(PROJECT_ROOT)}")
        return pd.read_parquet(cache_path)

    df = stream_reviews(n_rows=n_rows, seed=seed, shuffle_buffer=shuffle_buffer)

    if use_cache:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"Cached sample to {cache_path.relative_to(PROJECT_ROOT)}")

    return df


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


def stratified_split(
    df: pd.DataFrame,
    seed: int = SEED,
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test, preserving the class ratio in each part.

    Stratification matters here because neutral is a small minority: a plain
    random split could leave the validation set with a noticeably different
    neutral share than the training set, making macro-F1 hard to interpret.
    """
    holdout = val_fraction + test_fraction

    train_df, rest_df = train_test_split(
        df,
        test_size=holdout,
        stratify=df["label"],
        random_state=seed,
    )
    # Second cut divides the holdout between val and test in the right ratio.
    test_df, val_df = train_test_split(
        rest_df,
        test_size=val_fraction / holdout,
        stratify=rest_df["label"],
        random_state=seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    """Write the three splits to ``data/processed/`` as Parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
        path = PROCESSED_DIR / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        print(f"Wrote {len(frame):,} rows -> {path.relative_to(PROJECT_ROOT)}")


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the saved splits. Used by the training and evaluation modules."""
    paths = [PROCESSED_DIR / f"{name}.parquet" for name in ("train", "val", "test")]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Splits not found - run `python -m src.data` first. "
            f"Missing: {', '.join(missing)}"
        )
    train_df, val_df, test_df = (pd.read_parquet(p) for p in paths)
    return train_df, val_df, test_df


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def class_distribution(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Count and percentage per sentiment class."""
    counts = df["label"].value_counts().reindex(LABEL_NAMES.keys(), fill_value=0)
    total = int(counts.sum())
    return {
        LABEL_NAMES[label]: {
            "count": int(count),
            "percent": round(100.0 * count / total, 2) if total else 0.0,
        }
        for label, count in counts.items()
    }


def report_distribution(frames: dict[str, pd.DataFrame]) -> dict:
    """Print a class-distribution table and return it for saving."""
    report = {name: class_distribution(df) for name, df in frames.items()}

    header = f"{'class':<10}" + "".join(f"{name:>22}" for name in frames)
    print("\n" + "=" * len(header))
    print("CLASS DISTRIBUTION")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for class_name in LABEL_NAMES.values():
        cells = "".join(
            f"{report[name][class_name]['count']:>13,}"
            f" ({report[name][class_name]['percent']:>5.2f}%)"
            for name in frames
        )
        print(f"{class_name:<10}{cells}")

    print("-" * len(header))
    totals = "".join(f"{len(df):>13,}{'':>9}" for df in frames.values())
    print(f"{'TOTAL':<10}{totals}")
    print("=" * len(header) + "\n")

    return report


def save_distribution(report: dict, seed: int, n_rows: int) -> None:
    """Persist the distribution to ``results/`` (CLAUDE.md: metrics live there)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "class_distribution.json"
    payload = {
        "dataset": f"{HF_REPO_ID} / Electronics",
        "n_rows_requested": n_rows,
        "seed": seed,
        "label_scheme": {
            "negative": "1-2 stars",
            "neutral": "3 stars",
            "positive": "4-5 stars",
        },
        "distribution": report,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved class distribution -> {path.relative_to(PROJECT_ROOT)}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Amazon Electronics sentiment dataset.",
    )
    parser.add_argument(
        "--n-rows", type=int, default=N_ROWS,
        help=f"reviews to sample (default: {N_ROWS:,})",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help=f"random seed (default: {SEED})",
    )
    parser.add_argument(
        "--shuffle-buffer", type=int, default=SHUFFLE_BUFFER,
        help="streaming shuffle buffer size",
    )
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="ignore the cached sample and re-stream",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="quick check: no cache, no split files written",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # A smoke run must never touch the real cache or overwrite real splits, and
    # a 100k buffer would dwarf a 2k sample - shrink it to keep the run quick.
    shuffle_buffer = (
        min(args.shuffle_buffer, args.n_rows * 2) if args.smoke
        else args.shuffle_buffer
    )

    df = build_sample(
        n_rows=args.n_rows,
        seed=args.seed,
        shuffle_buffer=shuffle_buffer,
        use_cache=not args.smoke,
        force_refresh=args.force_refresh,
    )

    train_df, val_df, test_df = stratified_split(df, seed=args.seed)

    report = report_distribution(
        {"full sample": df, "train": train_df, "val": val_df, "test": test_df}
    )

    if args.smoke:
        print("[smoke] nothing written to disk.")
        first = df.iloc[0]
        print("\nExample cleaned review:")
        print(f"  text       : {first['text'][:160]}")
        print(f"  text_clean : {first['text_clean'][:160]}")
        print(
            f"  label      : {first['label']} "
            f"({first['label_name']}, {first['rating']} stars)"
        )
        return 0

    save_splits(train_df, val_df, test_df)
    save_distribution(report, seed=args.seed, n_rows=args.n_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
