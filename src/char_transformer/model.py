"""Decoder-only (GPT-style) Transformer built from scratch in PyTorch.

The model is a stack of pre-norm Transformer blocks. Each block applies causal
multi-head self-attention followed by a position-wise feed-forward network, with
residual connections around both. Tokens are mixed with learned positional
embeddings at the input, and a final layer norm + tied-free linear head projects
back to the vocabulary for next-character prediction.

The attention is implemented by hand (Q/K/V projections, scaled dot-product, a
causal mask, softmax, weighted sum) rather than calling ``nn.MultiheadAttention``,
which is the whole point of the project. Weight initialization follows the GPT-2
scheme: normal(0, 0.02) for linears and embeddings, with residual projections
scaled by ``1/sqrt(2 * n_layer)`` to keep activation variance stable as depth
grows.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal (look-ahead) mask.

    The combined Q, K, V projection is a single linear of width ``3 * n_embd``;
    the result is split per-head into ``n_head`` slices of size ``head_dim``.
    Attention scores are masked so position ``t`` can only attend to positions
    ``<= t``, which is what makes the model autoregressive.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError(
                f"n_embd ({cfg.n_embd}) must be divisible by n_head ({cfg.n_head})"
            )
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Lower-triangular causal mask, registered as a buffer so it moves with
        # the module across devices but is not a learnable parameter.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal self-attention to ``x`` of shape ``(B, T, C)``."""
        B, T, C = x.shape

        # Project to queries, keys, values and split into heads:
        # (B, T, 3C) -> 3 x (B, n_head, T, head_dim)
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product scores: (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum of values, then re-assemble heads: (B, T, C)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear -> GELU -> Linear -> dropout.

    The hidden layer is the conventional 4x expansion of the embedding width.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """A pre-norm Transformer block: attention and feed-forward with residuals.

    Pre-norm (LayerNorm *before* each sublayer, residual added after) gives a
    clean gradient path through the residual stream and trains more stably than
    the original post-norm formulation, especially without a learning-rate
    warmup tuned to the architecture.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ffwd = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    """Decoder-only Transformer language model.

    Args:
        cfg: Model architecture config. ``cfg.vocab_size`` must be set (it comes
            from the tokenizer) before constructing the model.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.vocab_size <= 0:
            raise ValueError(
                "cfg.vocab_size must be set from the tokenizer before building the model"
            )
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        # Scale residual projection weights by 1/sqrt(2 * n_layer) per GPT-2.
        for name, param in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("net.2.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters, optionally only those that require grad."""
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run a forward pass.

        Args:
            idx: Token ids of shape ``(B, T)`` with ``T <= block_size``.
            targets: Optional next-token ids of shape ``(B, T)``. When provided,
                the mean cross-entropy loss is returned alongside the logits.

        Returns:
            ``(logits, loss)`` where ``logits`` has shape ``(B, T, vocab_size)``
            and ``loss`` is a scalar tensor or ``None`` if no targets were given.
        """
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(
                f"sequence length {T} exceeds block_size {self.cfg.block_size}"
            )

        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample ``max_new_tokens`` continuations.

        At each step the context is cropped to the last ``block_size`` tokens,
        the logits for the final position are scaled by ``temperature`` and
        optionally restricted to the ``top_k`` most likely tokens, then a token
        is drawn from the resulting categorical distribution and appended.

        Args:
            idx: Conditioning context of shape ``(B, T)``.
            max_new_tokens: Number of tokens to generate.
            temperature: Softmax temperature; lower is greedier, must be > 0.
            top_k: If set, sample only from the top-k logits at each step.

        Returns:
            The context extended with the generated tokens, shape
            ``(B, T + max_new_tokens)``.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        if was_training:
            self.train()
        return idx
