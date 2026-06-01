"""CLI for fuzzy ontology matching."""

from __future__ import annotations

import argparse

from neural_search.ontology.matcher import match_all


def _print_group(name: str, matches) -> None:
    print(f"{name}:")
    if not matches:
        print("  none")
        return
    for match in matches:
        print(
            f"  - id: {match.id}\n"
            f"    label: {match.label}\n"
            f"    confidence: {match.confidence:.3f}\n"
            f"    evidence: {match.evidence}\n"
            f"    match_type: {match.match_type}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neural_search.ontology.match")
    parser.add_argument("query")
    args = parser.parse_args(argv)

    matches = match_all(args.query)
    _print_group("matched tasks", matches["tasks"])
    _print_group("matched behaviors", matches["behaviors"])
    _print_group("matched regions", matches["regions"])
    _print_group("matched modalities", matches["modalities"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

