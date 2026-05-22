"""Backward-compatible ontology CLI."""

from __future__ import annotations

import argparse

from neural_search.ontology.match import main as match_main
from neural_search.ontology.validate import main as validate_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neural_search.ontology")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("path")
    match = subparsers.add_parser("match")
    match.add_argument("query")
    args = parser.parse_args(argv)

    if args.command == "validate":
        return validate_main([args.path])
    if args.command == "match":
        return match_main([args.query])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

