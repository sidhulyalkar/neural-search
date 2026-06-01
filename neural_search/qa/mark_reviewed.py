"""Mark a dataset card reviewed.

Usage:
    python -m neural_search.qa.mark_reviewed --dataset-id <ID>
"""

from __future__ import annotations

from neural_search.qa._mark import mark_main


def main(argv: list[str] | None = None) -> int:
    return mark_main("reviewed", argv)


if __name__ == "__main__":
    raise SystemExit(main())
