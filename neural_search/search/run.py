"""CLI for running demo search."""

from __future__ import annotations

import argparse
import json

from neural_search.search import search_datasets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neural_search.search.run")
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)
    response = search_datasets(args.query, {}, limit=args.limit)
    print(json.dumps(response.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

