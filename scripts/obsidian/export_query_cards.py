#!/usr/bin/env python3
"""Export benchmark queries as Obsidian query cards.

Usage:
    python scripts/obsidian/export_query_cards.py \
        --queries artifacts/benchmark_queries.jsonl \
        --vault obsidian_vault
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_search.eval.query_decomposition import load_query_specs
from neural_search.obsidian.io import safe_write_note
from neural_search.obsidian.templates import query_card_body, query_card_frontmatter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--vault", required=True, type=Path)
    args = parser.parse_args()

    dest = args.vault / "04_Queries"
    dest.mkdir(parents=True, exist_ok=True)
    specs = load_query_specs(args.queries)

    for spec in specs:
        note_path = dest / f"{spec.query_id}.md"
        safe_write_note(note_path, query_card_frontmatter(spec), query_card_body(spec))

    print(f"Exported {len(specs)} query cards → {dest}")


if __name__ == "__main__":
    main()
