"""Export QA-reviewed dataset cards.

Usage:
    python -m neural_search.qa.export_reviewed_cards
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_search.qa import reviewed_dataset_cards

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "data" / "reports" / "reviewed_dataset_cards.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m neural_search.qa.export_reviewed_cards",
        description="Export reviewed and trusted dataset cards as JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path. Defaults to data/reports/reviewed_dataset_cards.json.",
    )
    parser.add_argument(
        "--include-trusted-only",
        action="store_true",
        help="Export only trusted cards.",
    )
    args = parser.parse_args(argv)

    statuses = {"trusted"} if args.include_trusted_only else {"reviewed", "trusted"}
    cards = reviewed_dataset_cards(statuses=statuses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cards, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(cards)} reviewed dataset cards to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
