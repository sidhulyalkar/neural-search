#!/usr/bin/env python3
"""Export corpus records as Obsidian dataset cards.

Usage:
    python scripts/obsidian/export_dataset_cards.py \
        --corpus data/corpus/normalized/combined_corpus.jsonl \
        --vault obsidian_vault
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_search.eval.evidence import dataset_evidence_from_record
from neural_search.obsidian.io import safe_write_note
from neural_search.obsidian.templates import dataset_card_body, dataset_card_frontmatter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--vault", required=True, type=Path)
    args = parser.parse_args()

    dest = args.vault / "03_Datasets"
    dest.mkdir(parents=True, exist_ok=True)

    files = sorted(args.corpus.glob("*.jsonl")) if args.corpus.is_dir() else [args.corpus]
    written = 0

    for file in files:
        with file.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                ev = dataset_evidence_from_record(record)
                safe_name = ev.record_id.replace(":", "_").replace("/", "_")
                note_path = dest / f"{safe_name}.md"
                safe_write_note(note_path, dataset_card_frontmatter(ev), dataset_card_body(ev))
                written += 1

    print(f"Exported {written} dataset cards → {dest}")


if __name__ == "__main__":
    main()
