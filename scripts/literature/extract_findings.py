"""CLI: extract scientific findings from OpenAlex paper abstracts.

Usage:
    python scripts/literature/extract_findings.py \\
        --corpus data/corpus/normalized/openalex_neuro \\
        --config configs/literature/finding_extraction_v1.yaml \\
        --out artifacts/literature/findings_v1.jsonl \\
        [--max-papers 10000] \\
        [--resume]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Allow running from repo root without installing the package
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from neural_search.literature.finding_extractor import (
    extract_findings_from_corpus,
    load_config,
)

_KEY_NAMES = ("CLAUDE_OPUS_API_KEY", "ANTHROPIC_API_KEY")


def _load_env_local(path: Path = _REPO / ".env.local") -> None:
    """Load simple KEY=value entries from .env.local without logging secrets."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def _resolve_anthropic_api_key() -> str | None:
    """Return an Anthropic-compatible API key from supported env names."""
    _load_env_local()
    for key_name in _KEY_NAMES:
        value = os.environ.get(key_name)
        if value:
            os.environ["ANTHROPIC_API_KEY"] = value
            return value
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract neuroscience findings from abstracts.")
    parser.add_argument(
        "--corpus",
        required=True,
        type=Path,
        help="Directory containing normalized JSONL shards.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to finding_extraction YAML config.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output JSONL path for extracted findings.",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Cap on number of papers to process.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (skips already-processed papers).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    api_key = _resolve_anthropic_api_key()
    if not api_key:
        logger.warning(
            "No Anthropic key found — set ANTHROPIC_API_KEY or CLAUDE_OPUS_API_KEY."
        )
        sys.exit(0)

    config = load_config(args.config)
    if not config:
        logger.error("Config not found or empty: %s", args.config)
        sys.exit(1)

    corpus_dir: Path = args.corpus
    if not corpus_dir.exists():
        logger.warning("Corpus directory does not exist: %s — nothing to process.", corpus_dir)
        sys.exit(0)

    shards = sorted(corpus_dir.glob("*.jsonl"))
    if not shards:
        logger.warning("No JSONL shards found in %s — nothing to process.", corpus_dir)
        sys.exit(0)

    checkpoint_path: Path | None = None
    if args.resume:
        checkpoint_path = args.out.with_suffix(".checkpoint.json")

    logger.info(
        "Processing %d shards | max_papers=%s | resume=%s",
        len(shards),
        args.max_papers,
        args.resume,
    )

    total = extract_findings_from_corpus(
        shards,
        config,
        args.out,
        checkpoint_path=checkpoint_path,
        max_papers=args.max_papers,
        api_key=api_key,
    )

    logger.info("Done. Extracted %d findings -> %s", total, args.out)


if __name__ == "__main__":
    main()
