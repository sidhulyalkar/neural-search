#!/usr/bin/env python3
"""Export high-priority audit items to Obsidian annotation notes.

Usage:
    python scripts/obsidian/export_audit_queue.py \
        --audit-queue artifacts/eval/audit_queue.jsonl \
        --vault obsidian_vault
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_search.obsidian.io import safe_write_note
from neural_search.obsidian.templates import (
    annotation_card_body,
    annotation_card_frontmatter,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-queue", required=True, type=Path)
    parser.add_argument("--vault", required=True, type=Path)
    args = parser.parse_args()

    dest = args.vault / "05_Annotations" / "Human Audits"
    dest.mkdir(parents=True, exist_ok=True)
    written = 0

    with args.audit_queue.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row["query_id"]
            rid = row["record_id"]
            annotation_id = f"{qid}__{rid.replace(':', '_')}"

            pair_ev = row.get("pair_evidence", {})
            q = pair_ev.get("query", {})
            d = pair_ev.get("dataset", {})

            fm = annotation_card_frontmatter(
                annotation_id=annotation_id,
                query_id=qid,
                record_id=rid,
                label=row.get("label"),
                confidence=row.get("confidence"),
                source=row.get("source", "bronze"),
                audit_status="pending",
            )
            body = annotation_card_body(
                query_text=q.get("query_text", ""),
                scientific_goal=q.get("scientific_goal", ""),
                hard_negatives=q.get("hard_negatives", []),
                dataset_title=d.get("title", rid),
                dataset_desc=d.get("description"),
                lf_votes=[],
                ensemble_label=row.get("label"),
                ensemble_confidence=row.get("confidence"),
            )
            note_path = dest / f"{annotation_id}.md"
            safe_write_note(note_path, fm, body)
            written += 1

    print(f"Exported {written} audit notes → {dest}")


if __name__ == "__main__":
    main()
