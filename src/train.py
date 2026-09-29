"""Training loop for the recurrent models (Phases 4 and 5).

One loop trains all three architectures so they share every detail except the
encoder: identical splits, class weights, optimiser, early-stopping rule and
evaluation code. That is what makes the Phase 5 comparison table meaningful.

Key choices
-----------
*Class-weighted loss* - the data is 79% positive. Without weighting, the cheapest
way to reduce the loss is to answer "positive" almost always. Weights are the
inverse class frequency, so a neutral mistake costs ~11x a positive one.

*Early stopping on validation macro-F1* - not on loss, and never on test. Macro-F1
is the metric the project is judged on, so it is the one worth stopping on.

*Mixed precision (fp16)* - the RTX 4050 has 6 GB. Autocast roughly halves
activation memory and speeds up the matmuls; GradScaler keeps the small
gradients from underflowing to zero.

Usage
-----
    python -m src.train --model bilstm_attention
    python -m src.train --model lstm --epochs 6
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from src.data import load_splits
from src.metrics import compute_metrics, print_report, record_result
from src.models import DISPLAY_NAMES, build_model, count_parameters
from src.vocab import (
    EMBED_DIM,
    MAX_LEN,
    PAD_ID,
    Vocabulary,
    build_embedding_matrix,
)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MODELS_DIR: Path = PROJECT_ROOT / "models"
SEED: int = 42


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


class ReviewDataset(Dataset):
    """Pre-encoded reviews. Encoding once up front keeps the GPU fed."""

    def __init__(self, texts, labels, vocab: Vocabulary, max_len: int = MAX_LEN):
        self.sequences = [np.array(vocab.encode(t, max_len), dtype=np.int64)
                          for t in texts]
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx]


def collate(batch):
    """Pad to the longest sequence *in this batch*, not to the global max.

    Dynamic padding matters: most reviews are ~40 words, so padding everything to
    200 would spend roughly four fifths of the compute on padding tokens.
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.int64)
    padded = torch.full((len(sequences), int(lengths.max())), PAD_ID,
                        dtype=torch.int64)
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = torch.from_numpy(seq)
    return padded, lengths, torch.tensor(labels, dtype=torch.int64)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    model: str = "bilstm_attention"
    embed_dim: int = EMBED_DIM
    hidden_dim: int = 128
    num_layers: int = 1
    attention_dim: int = 128
    dropout: float = 0.3
    max_len: int = MAX_LEN
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 8
    patience: int = 2
    grad_clip: float = 5.0
    freeze_embeddings: bool = False
    use_glove: bool = True
    seed: int = SEED
    # "ce" = plain cross-entropy; "ordinal" = distance-aware soft targets.
    loss: str = "ce"
    smoothing: float = 0.1
    # Optional suffix so a capacity experiment does not overwrite the base run.
    tag: str = ""


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #


class OrdinalCrossEntropy(nn.Module):
    """Cross-entropy with distance-aware soft targets.

    Sentiment here is *ordinal*: negative < neutral < positive. Plain
    cross-entropy does not know that - calling a negative review "positive" and
    calling it "neutral" are penalised identically, even though one is a far
    worse mistake.

    This spreads a small amount of target mass onto the other classes in inverse
    proportion to their distance from the true class, so predicting an adjacent
    class is cheaper than predicting the opposite pole. Class weights are applied
    on top, exactly as in the plain-cross-entropy path.

    For ``smoothing=0.1`` the targets are:

        true negative -> [0.900, 0.067, 0.033]
        true neutral  -> [0.050, 0.900, 0.050]
        true positive -> [0.033, 0.067, 0.900]
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        n_classes: int = 3,
        smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights)

        # targets[c] is the soft target distribution for true class c.
        targets = torch.zeros(n_classes, n_classes)
        for c in range(n_classes):
            distance = torch.abs(torch.arange(n_classes, dtype=torch.float) - c)
            share = torch.where(distance > 0, 1.0 / distance, torch.zeros(1))
            share = share / share.sum() * smoothing
            share[c] = 1.0 - smoothing
            targets[c] = share
        self.register_buffer("targets", targets)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits.float(), dim=1)
        soft = self.targets[target]                       # (B, C)
        per_sample = -(soft * log_probs).sum(dim=1)       # (B,)
        weights = self.class_weights[target]              # (B,)
        return (per_sample * weights).sum() / weights.sum()


def build_criterion(cfg: TrainConfig, class_weights: torch.Tensor) -> nn.Module:
    """Select the loss function named in the config."""
    if cfg.loss == "ordinal":
        print(f"loss: ordinal cross-entropy (smoothing={cfg.smoothing})")
        return OrdinalCrossEntropy(class_weights, smoothing=cfg.smoothing)
    print("loss: cross-entropy")
    return nn.CrossEntropyLoss(weight=class_weights)


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """Return (y_true, y_pred, y_prob) for a whole split."""
    model.eval()
    trues, preds, probs = [], [], []
    for x, lengths, y in loader:
        x, lengths = x.to(device, non_blocking=True), lengths.to(device)
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=device.type == "cuda"):
            logits, _ = model(x, lengths)
        p = torch.softmax(logits.float(), dim=1)
        preds.append(p.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())
        trues.append(y.numpy())
    return (np.concatenate(trues), np.concatenate(preds), np.concatenate(probs))


def train_model(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    display = DISPLAY_NAMES[cfg.model] + (f" ({cfg.tag})" if cfg.tag else "")
    slug = cfg.model + (f"_{cfg.tag}" if cfg.tag else "")
    print(f"\n{'=' * 68}\nTraining {display} on {device}\n{'=' * 68}")

    train_df, val_df, test_df = load_splits()

    # Vocabulary is built from training text only (see src/vocab.py).
    vocab = Vocabulary.build(train_df["text_clean"])
    embedding_matrix = (
        build_embedding_matrix(vocab, embed_dim=cfg.embed_dim)
        if cfg.use_glove else None
    )

    loaders = {}
    for name, frame, shuffle in (
        ("train", train_df, True),
        ("val", val_df, False),
        ("test", test_df, False),
    ):
        ds = ReviewDataset(frame["text_clean"], frame["label"], vocab, cfg.max_len)
        loaders[name] = DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=shuffle,
            collate_fn=collate, num_workers=0, pin_memory=device.type == "cuda",
        )

    model = build_model(
        cfg.model,
        vocab_size=len(vocab),
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        embedding_matrix=embedding_matrix,
        freeze_embeddings=cfg.freeze_embeddings,
        **({"attention_dim": cfg.attention_dim}
           if cfg.model == "bilstm_attention" else {}),
    ).to(device)
    print(f"trainable parameters: {count_parameters(model):,}")

    # Inverse-frequency class weights: a neutral mistake must cost more than a
    # positive one, or the model simply learns the prior.
    classes = np.array([0, 1, 2])
    weights = compute_class_weight("balanced", classes=classes,
                                   y=train_df["label"].values)
    print("class weights: " + ", ".join(
        f"{n}={w:.2f}" for n, w in zip(("negative", "neutral", "positive"), weights)))
    criterion = build_criterion(
        cfg, torch.tensor(weights, dtype=torch.float32, device=device)
    ).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / f"{slug}.pt"
    vocab.save(MODELS_DIR / f"{slug}_vocab.json")

    best_f1, best_epoch, epochs_without_gain = -1.0, -1, 0
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, seen, start = 0.0, 0, time.perf_counter()

        for step, (x, lengths, y) in enumerate(loaders["train"], 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            lengths = lengths.to(device)

            optimiser.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits, _ = model(x, lengths)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            # Unscale before clipping, or the clip threshold applies to scaled
            # gradients and does nothing predictable.
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimiser)
            scaler.update()

            running += loss.item() * len(y)
            seen += len(y)
            if step % 400 == 0:
                print(f"  epoch {epoch} step {step}/{len(loaders['train'])} "
                      f"loss {running / seen:.4f}")

        y_true, y_pred, _ = evaluate(model, loaders["val"], device)
        val = compute_metrics(y_true, y_pred)
        elapsed = time.perf_counter() - start
        print(f"epoch {epoch:>2} | train loss {running / seen:.4f} | "
              f"val macro-F1 {val['macro_f1']:.4f} | "
              f"val acc {val['accuracy']:.4f} | {elapsed:.0f}s")
        history.append({"epoch": epoch, "train_loss": running / seen,
                        "val_macro_f1": val["macro_f1"],
                        "val_accuracy": val["accuracy"]})

        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, epochs_without_gain = val["macro_f1"], epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(cfg),
                    "vocab_size": len(vocab),
                    "epoch": epoch,
                    "val_macro_f1": best_f1,
                },
                checkpoint_path,
            )
            print(f"         new best - checkpoint saved ({checkpoint_path.name})")
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= cfg.patience:
                print(f"early stopping: no improvement for {cfg.patience} epochs")
                break

    # Restore the best weights before touching the test split.
    model.load_state_dict(torch.load(checkpoint_path)["model_state"])
    print(f"\nbest epoch {best_epoch} (val macro-F1 {best_f1:.4f})")

    val_true, val_pred, _ = evaluate(model, loaders["val"], device)
    val_metrics = compute_metrics(val_true, val_pred)
    print_report(display, "val", val_metrics)
    record_result(display, "val", val_metrics, extra={
        "parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "history": history,
        "config": asdict(cfg),
    })

    test_true, test_pred, _ = evaluate(model, loaders["test"], device)
    test_metrics = compute_metrics(test_true, test_pred)
    print_report(display, "test", test_metrics)
    record_result(display, "test", test_metrics)

    return test_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a recurrent sentiment model")
    parser.add_argument("--model", default="bilstm_attention",
                        choices=list(DISPLAY_NAMES))
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--hidden-dim", type=int, default=TrainConfig.hidden_dim)
    parser.add_argument("--num-layers", type=int, default=TrainConfig.num_layers)
    parser.add_argument("--max-len", type=int, default=TrainConfig.max_len)
    parser.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    parser.add_argument("--no-glove", action="store_true",
                        help="random embeddings instead of pretrained GloVe")
    parser.add_argument("--loss", default="ce", choices=("ce", "ordinal"),
                        help="'ordinal' uses distance-aware soft targets, so "
                             "confusing adjacent classes costs less than "
                             "confusing the two poles")
    parser.add_argument("--smoothing", type=float, default=TrainConfig.smoothing)
    parser.add_argument("--tag", default="",
                        help="suffix for the checkpoint and result name, so an "
                             "experiment does not overwrite the baseline run")
    args = parser.parse_args()

    cfg = TrainConfig(
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        max_len=args.max_len,
        dropout=args.dropout,
        use_glove=not args.no_glove,
        loss=args.loss,
        smoothing=args.smoothing,
        tag=args.tag,
    )
    train_model(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
