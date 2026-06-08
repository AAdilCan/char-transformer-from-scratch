"""Initial exploration of the character corpus.

Prints basic statistics that drive modeling decisions: total length, the
character vocabulary, and the most common characters. Run this after
download_data.py to sanity-check the corpus before training.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from char_transformer.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)


def explore(path: Path, top_k: int = 15) -> None:
    """Print corpus statistics: size, vocabulary, and character frequencies."""
    text = path.read_text(encoding="utf-8")
    n_chars = len(text)
    vocab = sorted(set(text))
    counts = Counter(text)

    logger.info("file: %s", path)
    logger.info("total characters: %d", n_chars)
    logger.info("unique characters (vocab size): %d", len(vocab))
    logger.info("vocabulary: %s", "".join(vocab).replace("\n", "\\n"))

    logger.info("top %d characters by frequency:", top_k)
    for ch, cnt in counts.most_common(top_k):
        display = repr(ch) if ch in {"\n", " ", "\t"} else ch
        logger.info("  %-6s %8d  (%.2f%%)", display, cnt, 100 * cnt / n_chars)

    logger.info("first 200 characters:\n%s", text[:200])


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore the character corpus")
    parser.add_argument("--path", default="data/input.txt", help="corpus text file")
    parser.add_argument("--top-k", type=int, default=15, help="how many chars to rank")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(
            f"corpus not found at {path}; run scripts/download_data.py first"
        )
    explore(path, top_k=args.top_k)


if __name__ == "__main__":
    main()
