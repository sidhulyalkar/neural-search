"""CLI for building a file-backed Neural Search knowledge graph."""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_search.graph.builder import build_graph_from_records, split_records
from neural_search.graph.schema import write_graph_json, write_graph_jsonl
from neural_search.normalized import load_normalized_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a v0.5 knowledge graph from normalized corpus records.",
    )
    parser.add_argument("--datasets", help="Dataset JSON/JSONL file or directory")
    parser.add_argument("--papers", help="Paper JSON/JSONL file or directory")
    parser.add_argument(
        "--records",
        help="Mixed normalized JSON/JSONL file or directory containing datasets and papers",
    )
    parser.add_argument("--out", required=True, help="Output graph JSON or JSONL path")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    datasets = []
    papers = []

    if args.records:
        datasets, papers = split_records(load_normalized_records(args.records))
    if args.datasets:
        dataset_records, _ = split_records(load_normalized_records(args.datasets))
        datasets.extend(dataset_records)
    if args.papers:
        _, paper_records = split_records(load_normalized_records(args.papers))
        papers.extend(paper_records)

    graph = build_graph_from_records(
        datasets,
        papers,
        min_confidence=args.min_confidence,
    )
    output = Path(args.out)
    if output.suffix == ".jsonl":
        write_graph_jsonl(graph, output)
    else:
        write_graph_json(graph, output)

    print(
        f"wrote graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges "
        f"to {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
