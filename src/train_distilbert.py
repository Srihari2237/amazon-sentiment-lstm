"""Phase 5 - fine-tuned DistilBERT, the upper bound of the comparison.

This is the "how good can it get" reference point. DistilBERT arrives already
knowing English from pretraining, so it starts far ahead of a randomly
initialised LSTM; the interesting question is how much of that gap the much
smaller recurrent models close.

Deliberately written as a plain PyTorch loop rather than `transformers.Trainer`,
so that the class weighting, early stopping and evaluation are *literally the
same code path* as the recurrent models in src/train.py. A comparison table is
only trustworthy if the numbers were produced the same way.

Memory notes for a 6 GB card: max_len 128 (per CLAUDE.md), batch 32, fp16
autocast. The training subset is configurable because full fine-tuning on 240k
rows takes hours on a laptop GPU - validation and test always use the full
splits, so the comparison stays honest.

Usage
-----
    python -m src.train_distilbert --train-subset 60000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from src.data import load_splits
from src.metrics import compute_metrics, print_report, record_result

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MODELS_DIR: Path = PROJECT_ROOT / "models"

MODEL_ID: str = "distilbert-base-uncased"
DISPLAY_NAME: str = "DistilBERT (fine-tuned)"
MAX_LEN: int = 128
SEED: int = 42


class BertReviewDataset(Dataset):
    """Tokenised reviews.

    Note this uses the ``text`` column, not ``text_clean``: DistilBERT's
    WordPiece tokenizer was trained on natural text, so stripping casing and
    punctuation would throw away signal it knows how to use.
    """

    def __init__(self, texts, labels, tokenizer, max_len: int = MAX_LEN):
        self.encodings = tokenizer(
            list(texts), truncation=True, max_length=max_len, padding=False,
        )
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": int(self.labels[idx]),
        }


def make_collate(pad_token_id: int):
    """Pad dynamically to the longest sequence in the batch."""

    def collate(batch):
        longest = max(len(b["input_ids"]) for b in batch)
        input_ids = torch.full((len(batch), longest), pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(batch), longest), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            input_ids[i, :n] = torch.tensor(b["input_ids"], dtype=torch.long)
            attention[i, :n] = torch.tensor(b["attention_mask"], dtype=torch.long)
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return input_ids, attention, labels

    return collate


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    trues, preds = [], []
    for input_ids, attention, labels in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        attention = attention.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=device.type == "cuda"):
            logits = model(input_ids=input_ids, attention_mask=attention).logits
        preds.append(logits.float().argmax(1).cpu().numpy())
        trues.append(labels.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT")
    parser.add_argument("--train-subset", type=int, default=60_000,
                        help="stratified subset of the training split (0 = all)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    args = parser.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df = load_splits()

    if args.train_subset and args.train_subset < len(train_df):
        # Stratified so the subset keeps the same class ratio as the full split.
        train_df, _ = train_test_split(
            train_df, train_size=args.train_subset,
            stratify=train_df["label"], random_state=SEED,
        )
        print(f"training on a stratified {len(train_df):,}-row subset "
              f"(val/test remain full: {len(val_df):,} / {len(test_df):,})")

    print(f"loading {MODEL_ID} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=3,
    ).to(device)

    collate = make_collate(tokenizer.pad_token_id)
    loaders = {}
    for name, frame, shuffle in (("train", train_df, True),
                                 ("val", val_df, False),
                                 ("test", test_df, False)):
        ds = BertReviewDataset(frame["text"], frame["label"], tokenizer, args.max_len)
        loaders[name] = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                                   collate_fn=collate, num_workers=0,
                                   pin_memory=device.type == "cuda")

    weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]),
                                   y=train_df["label"].values)
    print("class weights: " + ", ".join(
        f"{n}={w:.2f}" for n, w in zip(("negative", "neutral", "positive"), weights)))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loaders["train"]) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=total_steps, pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / "distilbert.pt"
    best_f1, best_epoch = -1.0, -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen, start = 0.0, 0, time.perf_counter()
        for step, (input_ids, attention, labels) in enumerate(loaders["train"], 1):
            input_ids = input_ids.to(device, non_blocking=True)
            attention = attention.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(input_ids=input_ids, attention_mask=attention).logits
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()

            running += loss.item() * len(labels)
            seen += len(labels)
            if step % 200 == 0:
                rate = seen / (time.perf_counter() - start)
                print(f"  epoch {epoch} step {step}/{len(loaders['train'])} "
                      f"loss {running / seen:.4f} ({rate:.0f} rows/s)")

        y_true, y_pred = evaluate(model, loaders["val"], device)
        val = compute_metrics(y_true, y_pred)
        print(f"epoch {epoch} | train loss {running / seen:.4f} | "
              f"val macro-F1 {val['macro_f1']:.4f} | "
              f"{time.perf_counter() - start:.0f}s")

        if val["macro_f1"] > best_f1:
            best_f1, best_epoch = val["macro_f1"], epoch
            torch.save({"model_state": model.state_dict(),
                        "max_len": args.max_len,
                        "val_macro_f1": best_f1}, checkpoint_path)
            print("         new best - checkpoint saved")

    model.load_state_dict(torch.load(checkpoint_path)["model_state"])
    print(f"\nbest epoch {best_epoch} (val macro-F1 {best_f1:.4f})")

    val_metrics = compute_metrics(*evaluate(model, loaders["val"], device))
    print_report(DISPLAY_NAME, "val", val_metrics)
    record_result(DISPLAY_NAME, "val", val_metrics, extra={
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_rows": len(train_df),
        "max_len": args.max_len,
        "best_epoch": best_epoch,
    })

    test_metrics = compute_metrics(*evaluate(model, loaders["test"], device))
    print_report(DISPLAY_NAME, "test", test_metrics)
    record_result(DISPLAY_NAME, "test", test_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
