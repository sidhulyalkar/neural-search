#!/usr/bin/env python3
"""Interactive CLI for labeling query-candidate pairs with usefulness labels.

Usage:
    python scripts/annotate_usefulness.py --file data/eval/usefulness_seed_pairs.jsonl
    python scripts/annotate_usefulness.py --file data/eval/my_pairs.jsonl --start-from 10
    python scripts/annotate_usefulness.py --file data/eval/my_pairs.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VALID_LABELS = ["not_useful", "weakly_useful", "useful", "highly_useful"]
LABEL_SHORTCUTS = {"0": "not_useful", "1": "weakly_useful", "2": "useful", "3": "highly_useful"}


def _prompt_label(pair: dict) -> str | None:
    """Prompt the annotator to assign a label. Returns None to quit."""
    print("\n" + "=" * 60)
    print(f"Query [{pair['query_id']}]: {pair['query']}")
    print(f"Intent: {pair.get('intent', 'unknown')}")
    print(f"Candidate: {pair['candidate_id']}")
    print(f"Current label: {pair.get('usefulness_label', '(none)')}")
    print()
    print("Labels: 0=not_useful  1=weakly_useful  2=useful  3=highly_useful")
    print("Enter label (0-3), full name, 's' to skip, or 'q' to quit:")
    try:
        raw = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None
    if raw == "q":
        return None
    if raw == "s":
        return pair.get("usefulness_label", "not_useful")
    if raw in LABEL_SHORTCUTS:
        return LABEL_SHORTCUTS[raw]
    if raw in VALID_LABELS:
        return raw
    print(f"Invalid input '{raw}', skipping.")
    return pair.get("usefulness_label", "not_useful")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Label usefulness pairs interactively")
    parser.add_argument("--file", required=True, help="Path to JSONL pairs file")
    parser.add_argument("--start-from", type=int, default=0, help="Skip the first N pairs")
    parser.add_argument("--dry-run", action="store_true", help="Print pairs without prompting or writing")
    args = parser.parse_args(argv)

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    if args.dry_run:
        for i, pair in enumerate(pairs):
            print(f"[{i}] {pair.get('query_id')} — {pair.get('candidate_id')} — {pair.get('usefulness_label')}")
        return 0

    changed = 0
    for i, pair in enumerate(pairs):
        if i < args.start_from:
            continue
        new_label = _prompt_label(pair)
        if new_label is None:
            print("Quitting.")
            break
        if new_label != pair.get("usefulness_label"):
            pairs[i] = {**pair, "usefulness_label": new_label, "label_type": "human"}
            changed += 1

    with path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\nDone. {changed} label(s) changed. Saved to {path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
