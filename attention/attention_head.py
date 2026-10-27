import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionHead(nn.Module):
    """
    A single causal self-attention head.

    Input:
        x: (batch_size, sequence_length, embed_dim)

    Output:
        out: (batch_size, sequence_length, head_dim)
    """

    def __init__(
        self,
        embed_dim: int,
        head_dim: int,
        max_seq_len: int = 256,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Learnable projections from embeddings to queries, keys, and values.
        self.query = nn.Linear(embed_dim, head_dim, bias=bias)
        self.key = nn.Linear(embed_dim, head_dim, bias=bias)
        self.value = nn.Linear(embed_dim, head_dim, bias=bias)

        self.attention_dropout = nn.Dropout(dropout)

        # True means that a token is allowed to attend to that position.
        causal_mask = torch.tril(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)
        )

        # This mask is part of the module's state, but not its saved parameters.
        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "Expected x to have shape (batch_size, sequence_length, embed_dim)"
            )

        batch_size, sequence_length, embed_dim = x.shape

        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Expected embed_dim={self.embed_dim}, got {embed_dim}"
            )

        if sequence_length > self.max_seq_len:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )

        # 1. Project the input into queries, keys, and values.
        q = self.query(x)  # (B, T, head_dim)
        k = self.key(x)    # (B, T, head_dim)
        v = self.value(x)  # (B, T, head_dim)

        # 2. Calculate query-key similarity scores.
        scores = q @ k.transpose(-2, -1)  # (B, T, T)

        # 3. Scale scores to stabilize the softmax.
        scores = scores / math.sqrt(self.head_dim)

        # 4. Prevent each token from attending to future tokens.
        mask = self.causal_mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(
            ~mask,
            torch.finfo(scores.dtype).min,
        )

        # 5. Convert scores into attention probabilities.
        weights = F.softmax(scores, dim=-1)  # (B, T, T)

        # 6. Apply dropout during training.
        weights = self.attention_dropout(weights)

        # 7. Use attention weights to combine value vectors.
        out = weights @ v  # (B, T, head_dim)

        return out