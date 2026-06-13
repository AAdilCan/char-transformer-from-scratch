"""Character-level decoder-only Transformer implemented from scratch in PyTorch."""

from .model import GPT, Block, CausalSelfAttention, FeedForward
from .trainer import Trainer, TrainHistory, estimate_loss
from .lr_schedule import cosine_lr
from .evaluation import evaluate, perplexity

__version__ = "0.1.0"

__all__ = [
    "GPT",
    "Block",
    "CausalSelfAttention",
    "FeedForward",
    "Trainer",
    "TrainHistory",
    "estimate_loss",
    "cosine_lr",
    "evaluate",
    "perplexity",
]
