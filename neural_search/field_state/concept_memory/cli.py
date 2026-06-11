"""CLI for concept memory commands.

Usage:
    python -m neural_search.field_state.concept_memory.cli concept-build ...
    python -m neural_search.field_state.concept_memory.cli concept-search ...
    python -m neural_search.field_state.concept_memory.cli concept-basis ...
    python -m neural_search.field_state.concept_memory.cli concept-neighborhood ...
    python -m neural_search.field_state.concept_memory.cli concept-report ...
    python -m neural_search.field_state.concept_memory.cli concept-export-obsidian ...
    python -m neural_search.field_state.concept_memory.cli concept-validate ...
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import cast

from neural_search.field_state.concept_memory.basis import (
    generate_all_bases,
    read_concept_basis,
    write_concept_basis,
)
from neural_search.field_state.concept_memory.graph_builder import (
    build_concept_graph,
    build_full_concept_memory,
    get_concept_neighborhood,
    read_concept_artifacts,
)
from neural_search.field_state.concept_memory.obsidian_export import (
    export_concept_memory_to_obsidian,
)
from neural_search.field_state.concept_memory.reports import generate_all_reports
from neural_search.field_state.concept_memory.retrieval import search_concepts
from neural_search.field_state.concept_memory.validators import run_concept_validation

_DEFAULT_FIELD = "neuroscience_dataset_reuse"


def cmd_concept_build(args: argparse.Namespace) -> int:
    """Build the full concept memory from all available sources."""
    paths = build_full_concept_memory(
        field=args.field,
        vault_path=args.vault,
        corpus_path=args.corpus,
        root=args.root,
    )
    # Also write basis records
    concepts, evidence_links = read_concept_artifacts(args.root)
    bases = generate_all_bases(concepts, evidence_links)
    basis_path = write_concept_basis(bases, args.root)

    for label, path in {**paths, "concept_basis": basis_path}.items():
        print(f"  {label}: {path}")
    print(f"Summary: {len(concepts)} concepts, {len(evidence_links)} evidence links, {len(bases)} bases")
    return 0


def cmd_concept_search(args: argparse.Namespace) -> int:
    """Search concepts by query string."""
    concepts, evidence_links = read_concept_artifacts(args.root)
    graph = build_concept_graph(concepts, evidence_links)
    results = search_concepts(
        query=args.query,
        concepts=concepts,
        evidence_links=evidence_links,
        graph=graph,
        limit=args.limit,
    )
    if not results:
        print("No results found.")
        return 0
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. [{result.concept_type}] {result.canonical_name}"
            f" (score={result.score:.3f}) - {result.concept_id}"
        )
    return 0


def cmd_concept_basis(args: argparse.Namespace) -> int:
    """Print the basis record for a specific concept."""
    bases = read_concept_basis(args.root)
    for basis in bases:
        if basis.concept_id == args.concept_id:
            print(f"Concept ID:       {basis.concept_id}")
            print(f"Canonical name:   {basis.canonical_name}")
            print(f"Type:             {basis.concept_type}")
            print(f"Evidence strength:{basis.evidence_strength}")
            print(f"Summary:          {basis.summary}")
            if basis.supporting_claim_ids:
                print(f"Supporting claims:  {', '.join(basis.supporting_claim_ids)}")
            if basis.supporting_dataset_ids:
                print(f"Supporting datasets:{', '.join(basis.supporting_dataset_ids)}")
            if basis.supporting_paper_ids:
                print(f"Supporting papers:  {', '.join(basis.supporting_paper_ids)}")
            if basis.related_opportunity_ids:
                print(f"Related opportunities: {', '.join(basis.related_opportunity_ids)}")
            if basis.related_benchmark_gap_ids:
                print(f"Related benchmark gaps: {', '.join(basis.related_benchmark_gap_ids)}")
            if basis.uncertainty_notes:
                print("Uncertainty notes:")
                for note in basis.uncertainty_notes:
                    print(f"  - {note}")
            if basis.next_validation_actions:
                print("Next validation actions:")
                for action in basis.next_validation_actions:
                    print(f"  - {action}")
            return 0
    print(f"No basis found for concept_id: {args.concept_id}")
    return 1


def cmd_concept_neighborhood(args: argparse.Namespace) -> int:
    """Print the graph neighborhood of a concept."""
    concepts, evidence_links = read_concept_artifacts(args.root)
    graph = build_concept_graph(concepts, evidence_links)
    neighborhood = get_concept_neighborhood(graph, args.concept_id, depth=args.depth)

    print(f"Neighborhood of '{neighborhood['center']}' (depth={neighborhood['depth']}):")
    print(f"  Nodes ({len(neighborhood['nodes'])}):")
    for node in neighborhood["nodes"]:
        print(
            f"    [{node['concept_type']}] {node['canonical_name']} ({node['concept_id']})"
        )
    print(f"  Edges ({len(neighborhood['edges'])}):")
    for edge in neighborhood["edges"]:
        print(f"    {edge['source']} --[{edge['relation_type']}]--> {edge['target']}")
    return 0


def cmd_concept_report(args: argparse.Namespace) -> int:
    """Generate all concept memory Markdown reports."""
    concepts, evidence_links = read_concept_artifacts(args.root)
    bases = read_concept_basis(args.root)
    graph = build_concept_graph(concepts, evidence_links)
    paths = generate_all_reports(concepts, evidence_links, bases, graph=graph, root=args.root)
    print("Generated concept memory reports:")
    for label, path in paths.items():
        print(f"  {label}: {path}")
    return 0


def cmd_concept_export_obsidian(args: argparse.Namespace) -> int:
    """Export concept memory notes to an Obsidian vault."""
    concepts, evidence_links = read_concept_artifacts(args.root)
    bases = read_concept_basis(args.root)
    result = export_concept_memory_to_obsidian(
        concepts=concepts,
        evidence_links=evidence_links,
        bases=bases,
        vault_path=args.vault,
        field=args.field,
    )
    print("Exported concept memory to Obsidian:")
    print(f"  vault:         {result.vault_path}")
    print(f"  notes_created: {result.notes_created}")
    print(f"  notes_updated: {result.notes_updated}")
    print(f"  notes_skipped: {result.notes_skipped}")
    print(f"  warnings:      {len(result.warnings)}")
    return 0


def cmd_concept_validate(args: argparse.Namespace) -> int:
    """Validate the integrity of all concept memory artifacts."""
    result = run_concept_validation(root=args.root)
    status = "VALID" if result.is_valid else "INVALID"
    print(f"Concept memory validation: {status}")
    print(f"  errors:   {result.error_count}")
    print(f"  warnings: {result.warning_count}")
    for key, val in result.stats.items():
        print(f"  {key}: {val}")
    if result.errors:
        print("Errors:")
        for err in result.errors[:20]:
            print(f"  - {err}")
    return 0 if result.is_valid else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the concept memory submodule CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m neural_search.field_state.concept_memory.cli",
        description="Graph-Indexed Concept Memory CLI.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root. Defaults to the current working directory.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # concept-build
    build_p = subparsers.add_parser("concept-build", help="Build full concept memory.")
    build_p.add_argument("--field", default=_DEFAULT_FIELD, help="Field memory namespace.")
    build_p.add_argument("--vault", type=Path, default=None, help="Obsidian vault path (optional).")
    build_p.add_argument("--corpus", type=Path, default=None, help="Corpus JSONL path (optional).")
    build_p.set_defaults(func=cmd_concept_build)

    # concept-search
    search_p = subparsers.add_parser("concept-search", help="Search concepts by query.")
    search_p.add_argument("--query", required=True, help="Search query string.")
    search_p.add_argument("--field", default=_DEFAULT_FIELD, help="Field memory namespace.")
    search_p.add_argument("--limit", type=int, default=10, help="Max results to return.")
    search_p.set_defaults(func=cmd_concept_search)

    # concept-basis
    basis_p = subparsers.add_parser("concept-basis", help="Print basis record for a concept.")
    basis_p.add_argument("--concept-id", required=True, dest="concept_id", help="Concept ID.")
    basis_p.set_defaults(func=cmd_concept_basis)

    # concept-neighborhood
    nb_p = subparsers.add_parser("concept-neighborhood", help="Print concept graph neighborhood.")
    nb_p.add_argument("--concept-id", required=True, dest="concept_id", help="Concept ID.")
    nb_p.add_argument("--depth", type=int, default=2, help="Neighborhood depth (hops).")
    nb_p.set_defaults(func=cmd_concept_neighborhood)

    # concept-report
    report_p = subparsers.add_parser("concept-report", help="Generate all concept memory reports.")
    report_p.add_argument("--field", default=_DEFAULT_FIELD, help="Field memory namespace.")
    report_p.set_defaults(func=cmd_concept_report)

    # concept-export-obsidian
    export_p = subparsers.add_parser(
        "concept-export-obsidian", help="Export concept memory to Obsidian vault."
    )
    export_p.add_argument("--vault", type=Path, required=True, help="Obsidian vault path.")
    export_p.add_argument("--field", default=_DEFAULT_FIELD, help="Field memory namespace.")
    export_p.set_defaults(func=cmd_concept_export_obsidian)

    # concept-validate
    validate_p = subparsers.add_parser(
        "concept-validate", help="Validate concept memory artifact integrity."
    )
    validate_p.set_defaults(func=cmd_concept_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = cast(Callable[[argparse.Namespace], int], args.func)
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
