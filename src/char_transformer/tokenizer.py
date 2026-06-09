"""Character-level tokenizer.

The vocabulary is the sorted set of unique characters in the training corpus.
Each character maps to a contiguous integer id, so ``vocab_size`` equals the
number of distinct characters. The mapping is tiny, so it is serialized to JSON
and stored alongside a checkpoint to guarantee that encode/decode at generation
time matches what the model was trained on.
"""

from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    """Bidirectional character <-> integer id mapping.

    Args:
        chars: The ordered vocabulary. Index in the list becomes the id.
    """

    def __init__(self, chars: list[str]) -> None:
        if len(chars) != len(set(chars)):
            raise ValueError("vocabulary contains duplicate characters")
        self.chars: list[str] = list(chars)
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.itos: dict[int, str] = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build a tokenizer from the unique characters in ``text``.

        Characters are sorted so the mapping is deterministic across runs and
        machines, which keeps a serialized vocabulary stable.
        """
        if not text:
            raise ValueError("cannot build a tokenizer from empty text")
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        """Number of distinct characters in the vocabulary."""
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        """Map a string to a list of integer ids.

        Raises:
            KeyError: If ``text`` contains a character outside the vocabulary,
                with a message identifying the offending character.
        """
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as exc:  # pragma: no cover - exercised in tests
            ch = exc.args[0]
            raise KeyError(
                f"character {ch!r} is not in the tokenizer vocabulary"
            ) from None

    def decode(self, ids: list[int]) -> str:
        """Map a sequence of integer ids back to a string."""
        try:
            return "".join(self.itos[i] for i in ids)
        except KeyError as exc:  # pragma: no cover - exercised in tests
            i = exc.args[0]
            raise KeyError(f"id {i} is outside the vocabulary [0, {self.vocab_size})") from None

    def save(self, path: str | Path) -> None:
        """Serialize the vocabulary to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"chars": self.chars}, fh, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        """Load a tokenizer previously written with :meth:`save`."""
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return cls(payload["chars"])

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        return f"CharTokenizer(vocab_size={self.vocab_size})"
