"""Recurrent sentiment classifiers.

Three architectures, built to be directly comparable - same embedding layer, same
dropout, same classifier head. Only the encoder differs, so a difference in
macro-F1 is attributable to the encoder rather than to incidental choices.

1. ``LSTMClassifier``          - plain unidirectional LSTM, last hidden state.
2. ``BiGRUClassifier``         - bidirectional GRU, final states concatenated.
3. ``BiLSTMAttentionClassifier`` - the main model: bidirectional LSTM whose
   per-token outputs are pooled by a learned attention layer instead of taking
   only the final state.

Why attention is the interesting one: taking the last hidden state forces the
whole review into one vector at the final timestep, so early evidence has to
survive the entire sequence. Attention lets the model look back and weight every
token, which both helps accuracy and - because the weights are readable - gives
the "which words mattered" view the Streamlit app needs.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.vocab import PAD_ID

N_CLASSES: int = 3


class _EmbeddingBase(nn.Module):
    """Shared embedding setup so the three models differ only in their encoder."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        embedding_matrix: np.ndarray | None = None,
        freeze_embeddings: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            # Fine-tuning the vectors usually wins on a 240k-row corpus; freezing
            # is available for the low-data case.
            self.embedding.weight.requires_grad = not freeze_embeddings
        self.dropout = nn.Dropout(dropout)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embedding(x))


class LSTMClassifier(_EmbeddingBase):
    """Plain unidirectional LSTM; the final hidden state is the review vector."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        embedding_matrix: np.ndarray | None = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__(vocab_size, embed_dim, embedding_matrix,
                         freeze_embeddings, dropout)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=False,
        )
        self.fc = nn.Linear(hidden_dim, N_CLASSES)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        emb = self._embed(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.fc(self.dropout(h_n[-1])), None


class BiGRUClassifier(_EmbeddingBase):
    """Bidirectional GRU; forward and backward final states are concatenated."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        embedding_matrix: np.ndarray | None = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__(vocab_size, embed_dim, embedding_matrix,
                         freeze_embeddings, dropout)
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, N_CLASSES)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        emb = self._embed(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, h_n = self.gru(packed)
        # h_n is (layers*2, batch, hidden) - take the last layer's two directions.
        joined = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(self.dropout(joined)), None


class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention over the encoder's per-token outputs.

    score_t = v . tanh(W h_t)  ->  softmax over real tokens only.

    Padding positions are masked to -inf *before* the softmax. Masking after it
    would be wrong: the padded positions would already have taken probability
    mass from the real tokens.
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 128) -> None:
        super().__init__()
        self.project = nn.Linear(hidden_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, outputs: torch.Tensor, mask: torch.Tensor):
        # outputs (B, T, H); mask (B, T) with True on real tokens.
        scores = self.score(torch.tanh(self.project(outputs))).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)                              # (B, T)
        context = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)       # (B, H)
        return context, weights


class BiLSTMAttentionClassifier(_EmbeddingBase):
    """Main model: Bi-LSTM encoder + additive attention pooling.

    ``forward`` returns ``(logits, attention_weights)``; the weights are what the
    error analysis and the Streamlit app use to highlight influential words.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 1,
        attention_dim: int = 128,
        dropout: float = 0.3,
        embedding_matrix: np.ndarray | None = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__(vocab_size, embed_dim, embedding_matrix,
                         freeze_embeddings, dropout)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AdditiveAttention(hidden_dim * 2, attention_dim)
        self.fc = nn.Linear(hidden_dim * 2, N_CLASSES)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        emb = self._embed(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True)

        mask = torch.arange(outputs.size(1), device=x.device)[None, :] \
            < lengths[:, None].to(x.device)
        context, weights = self.attention(outputs, mask)
        return self.fc(self.dropout(context)), weights


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "lstm": LSTMClassifier,
    "bigru": BiGRUClassifier,
    "bilstm_attention": BiLSTMAttentionClassifier,
}

DISPLAY_NAMES: dict[str, str] = {
    "lstm": "LSTM",
    "bigru": "Bi-GRU",
    "bilstm_attention": "Bi-LSTM + Attention",
}


def build_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a model by its registry key."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model '{name}'; choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
