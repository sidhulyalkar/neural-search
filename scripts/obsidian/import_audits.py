#!/usr/bin/env python3
"""Import completed human audits from Obsidian into qrels_gold.jsonl.

Only notes with audit_status: done are imported.

Usage:
    python scripts/obsidian/import_audits.py \
        --vault obsidian_vault \
        --out artifacts/qrels_gold.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_search.obsidian.io import read_note


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    audit_dir = args.vault / "05_Annotations" / "Human Audits"
    if not audit_dir.exists():
        print(f"No audit directory found at {audit_dir}")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[tuple[str, str], dict] = {}
    if args.out.exists():
        with args.out.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                existing[(r["query_id"], r["record_id"])] = r

    imported = skipped = 0
    for note_path in sorted(audit_dir.glob("*.md")):
        fm, _ = read_note(note_path)
        if fm.get("audit_status") != "done":
            skipped += 1
            continue
        if fm.get("label") is None:
            skipped += 1
            continue

        qid = fm.get("query_id", "")
        rid = fm.get("dataset_id", "")
        if not qid or not rid:
            skipped += 1
            continue

        existing[(qid, rid)] = {
            "query_id": qid,
            "record_id": rid,
            "label": int(fm["label"]),
            "confidence": float(fm.get("confidence") or 1.0),
            "source": "gold",
            "provenance": ["human_audit"],
            "hard_negative_triggered": False,
            "disagreement": 0.0,
            "created": datetime.now(UTC).isoformat(),
        }
        imported += 1

    with args.out.open("w", encoding="utf-8") as fh:
        for record in existing.values():
            fh.write(json.dumps(record) + "\n")

    print(f"Imported {imported} gold labels (skipped {skipped}) → {args.out}")
    print(f"Total gold qrels: {len(existing)}")


if __name__ == "__main__":
    main()
