"""CLI for building field-specific embedding caches."""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_search.embeddings.base import EmbeddingProvider
from neural_search.embeddings.field_index import build_cache_from_normalized_path
from neural_search.embeddings.hashing import HashingEmbeddingProvider
from neural_search.embeddings.sentence_transformers import (
    SentenceTransformerEmbeddingProvider,
)


def _provider_from_args(args: argparse.Namespace) -> EmbeddingProvider:
    if args.provider == "hashing":
        return HashingEmbeddingProvider(dimensions=args.dimensions)
    if args.provider == "sentence-transformer":
        return SentenceTransformerEmbeddingProvider(model_name=args.model)
    raise ValueError(f"unknown provider: {args.provider}")


def _output_path(path: str) -> Path:
    output = Path(path)
    if output.suffix:
        return output
    return output / "field_embeddings.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build v0.4 field-specific embedding cache from normalized records.",
    )
    parser.add_argument("--input", required=True, help="JSON, JSONL, or directory of records")
    parser.add_argument("--out", required=True, help="Output JSONL file or directory")
    parser.add_argument(
        "--provider",
        choices=["hashing", "sentence-transformer"],
        default="hashing",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model name when provider=sentence-transformer",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=64,
        help="Hashing provider dimensions",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    provider = _provider_from_args(args)
    output = _output_path(args.out)
    records = build_cache_from_normalized_path(args.input, output, provider)
    print(
        "wrote "
        f"{len(records)} field embeddings to {output} "
        f"using {provider.provider_name}/{provider.model_name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
