"""Mark a dataset card rejected.

Usage:
    python -m neural_search.qa.mark_rejected --dataset-id <ID>
"""

from __future__ import annotations

from neural_search.qa._mark import mark_main


def main(argv: list[str] | None = None) -> int:
    return mark_main("rejected", argv)


if __name__ == "__main__":
    raise SystemExit(main())
