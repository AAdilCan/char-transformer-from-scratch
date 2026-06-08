"""Download the Tiny Shakespeare corpus used for training.

The file is ~1.1 MB of plain text (the concatenated works of Shakespeare),
which is small enough to train a character-level model on CPU. It is fetched
from the canonical char-rnn mirror and cached under data/.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from char_transformer.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def download(dest: Path, force: bool = False) -> Path:
    """Download the corpus to ``dest`` unless it already exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        logger.info("corpus already present at %s (%d bytes)", dest, dest.stat().st_size)
        return dest

    logger.info("downloading corpus from %s", DATA_URL)
    with urllib.request.urlopen(DATA_URL) as resp:  # noqa: S310 (trusted URL)
        data = resp.read()
    dest.write_bytes(data)
    logger.info("saved %d bytes to %s", len(data), dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Tiny Shakespeare corpus")
    parser.add_argument("--dest", default="data/input.txt", help="output text file path")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()
    download(Path(args.dest), force=args.force)


if __name__ == "__main__":
    main()
