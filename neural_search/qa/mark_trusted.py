"""Mark a dataset card trusted.

Usage:
    python -m neural_search.qa.mark_trusted --dataset-id <ID>
"""

from __future__ import annotations

from neural_search.qa._mark import mark_main


def main(argv: list[str] | None = None) -> int:
    return mark_main("trusted", argv)


if __name__ == "__main__":
    raise SystemExit(main())
