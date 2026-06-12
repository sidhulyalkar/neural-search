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

from neural_search.field_state.concept_memory.ablation import (
    DEFAULT_JSON_OUT as DEFAULT_ABLATION_JSON_OUT,
)
from neural_search.field_state.concept_memory.ablation import run_ablation
from neural_search.field_state.concept_memory.artifact_utils import (
    deterministic_artifacts,
    deterministic_enabled,
)
from neural_search.field_state.concept_memory.basis import (
    generate_all_bases,
    read_concept_basis,
    write_concept_basis,
)
from neural_search.field_state.concept_memory.coverage import (
    generate_coverage_report_from_artifacts,
)
from neural_search.field_state.concept_memory.evaluator import run_concept_eval
from neural_search.field_state.concept_memory.explainer import explain_from_artifacts
from neural_search.field_state.concept_memory.graph_builder import (
    build_concept_graph,
    build_full_concept_memory,
    get_concept_neighborhood,
    read_concept_artifacts,
)
from neural_search.field_state.concept_memory.manifest import (
    read_manifest,
    write_manifest,
)
from neural_search.field_state.concept_memory.obsidian_export import (
    export_concept_memory_to_obsidian,
)
from neural_search.field_state.concept_memory.reports import generate_all_reports
from neural_search.field_state.concept_memory.reranker import rerank_from_artifacts
from neural_search.field_state.concept_memory.retrieval import search_concepts
from neural_search.field_state.concept_memory.validators import run_concept_validation

_DEFAULT_FIELD = "neuroscience_dataset_reuse"


def cmd_concept_build(args: argparse.Namespace) -> int:
    """Build the full concept memory from all available sources."""
    det = deterministic_enabled(args.deterministic)
    with deterministic_artifacts(det):
        paths = build_full_concept_memory(
            field=args.field,
            vault_path=args.vault,
            corpus_path=args.corpus,
            root=args.root,
            deterministic=det,
        )
        # Also write basis records
        concepts, evidence_links = read_concept_artifacts(args.root)
        bases = generate_all_bases(concepts, evidence_links)
        basis_path = write_concept_basis(bases, args.root)
        all_paths = {**paths, "concept_basis": basis_path}
        manifest = read_manifest(args.root or Path.cwd()) or {}
        graphml_status = manifest.get("graphml_export") if isinstance(manifest, dict) else None
        manifest_path = write_manifest(
            root=args.root or Path.cwd(),
            artifact_paths=all_paths,
            deterministic=det,
            corpus_path=args.corpus,
            graphml_status=graphml_status,
        )
        all_paths["manifest"] = manifest_path

    for label, path in all_paths.items():
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
    with deterministic_artifacts(deterministic_enabled(args.deterministic)):
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


def cmd_concept_rerank(args: argparse.Namespace) -> int:
    """Rerank dataset candidates using concept memory signals."""
    results = rerank_from_artifacts(
        query=args.query,
        limit=args.limit,
        root=args.root,
        enable_concept_boost=not args.lexical_only,
        enable_evidence_boost=not args.lexical_only,
        enable_hard_negative_penalty=not args.lexical_only,
    )
    if not results:
        print("No results found.")
        return 0
    for rank, r in enumerate(results, start=1):
        print(
            f"{rank}. {r.dataset_title}"
            f"\n   id={r.dataset_id}"
            f"\n   score={r.final_score:.4f}"
            f"  (base={r.base_score:.4f}"
            f" + concept={r.concept_boost:.4f}"
            f" + evidence={r.evidence_boost:.4f}"
            f" - penalty={r.hard_negative_penalty:.4f})"
            f"\n   {r.explanation_summary}"
        )
    return 0


def cmd_concept_explain(args: argparse.Namespace) -> int:
    """Show a concept-grounded explanation for a specific dataset result."""
    explanation = explain_from_artifacts(
        query=args.query,
        dataset_id=args.dataset_id,
        root=args.root,
    )
    print(explanation.explanation_markdown)
    if args.json:
        import json
        print("\n---JSON---")
        print(json.dumps(explanation.model_dump(), indent=2))
    return 0


def cmd_concept_eval(args: argparse.Namespace) -> int:
    """Run ablation evaluation using qrels and benchmark queries."""
    out = run_concept_eval(
        queries_path=args.queries,
        qrels_path=args.qrels,
        field=args.field,
        out_path=args.out,
        root=args.root,
    )
    print(f"Concept memory evaluation report written: {out}")
    return 0


def cmd_concept_ablate_retrieval(args: argparse.Namespace) -> int:
    """Run qrels-backed Concept Memory retrieval ablation."""
    report = run_ablation(
        root=args.root,
        queries_path=args.queries,
        qrels_path=args.qrels,
        corpus_path=args.corpus,
        out_json=args.out,
        deterministic=args.deterministic,
    )
    out_json = args.out
    if args.root is not None and not out_json.is_absolute():
        out_json = args.root / out_json
    out_md = out_json.with_suffix(".md")
    print(f"Concept Memory retrieval ablation status: {report['status']}")
    print(f"  json: {out_json}")
    print(f"  markdown: {out_md}")
    for warning in report.get("warnings", []):
        print(f"  warning: {warning}")
    return 0


def cmd_concept_coverage(args: argparse.Namespace) -> int:
    """Generate concept coverage audit report."""
    out = generate_coverage_report_from_artifacts(
        field=args.field,
        out_path=args.out,
        root=args.root,
    )
    print(f"Concept coverage report written: {out}")
    return 0


def cmd_concept_validate(args: argparse.Namespace) -> int:
    """Validate the integrity of all concept memory artifacts."""
    with deterministic_artifacts(deterministic_enabled(args.deterministic)):
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
    build_p.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Use stable timestamps and manifest mode for reproducible artifacts.",
    )
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
    report_p.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Use stable report timestamps for reproducible artifacts.",
    )
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
    validate_p.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Use stable validation timestamps.",
    )
    validate_p.set_defaults(func=cmd_concept_validate)

    # concept-rerank
    rerank_p = subparsers.add_parser(
        "concept-rerank",
        help="Rerank dataset results using concept memory signals.",
    )
    rerank_p.add_argument("--query", required=True, help="Natural language search query.")
    rerank_p.add_argument("--limit", type=int, default=10, help="Max results to return.")
    rerank_p.add_argument("--field", default=_DEFAULT_FIELD, help="Field memory namespace.")
    rerank_p.add_argument(
        "--lexical-only",
        action="store_true",
        default=False,
        help="Disable concept/evidence boosts; use base lexical score only.",
    )
    rerank_p.set_defaults(func=cmd_concept_rerank)

    # concept-explain
    explain_p = subparsers.add_parser(
        "concept-explain",
        help="Show concept-grounded explanation for a dataset result.",
    )
    explain_p.add_argument("--query", required=True, help="Natural language search query.")
    explain_p.add_argument(
        "--dataset-id", required=True, dest="dataset_id", help="Dataset ID to explain."
    )
    explain_p.add_argument("--field", default=_DEFAULT_FIELD, help="Field memory namespace.")
    explain_p.add_argument(
        "--json", action="store_true", default=False, help="Also print explanation as JSON."
    )
    explain_p.set_defaults(func=cmd_concept_explain)

    # concept-eval
    eval_p = subparsers.add_parser(
        "concept-eval",
        help="Run ablation evaluation over qrels and benchmark queries.",
    )
    eval_p.add_argument(
        "--qrels", type=Path, default=None, help="Path to adjudicated_qrels.jsonl."
    )
    eval_p.add_argument(
        "--queries",
        type=Path,
        default=Path("artifacts/benchmark_queries.jsonl"),
        help="Path to benchmark_queries.jsonl.",
    )
    eval_p.add_argument("--field", default=_DEFAULT_FIELD, help="Field memory namespace.")
    eval_p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output report path (default: reports/eval/concept_memory_eval.md).",
    )
    eval_p.set_defaults(func=cmd_concept_eval)

    # concept-ablate-retrieval
    ablate_p = subparsers.add_parser(
        "concept-ablate-retrieval",
        help="Run qrels-backed retrieval ablation for Concept Memory.",
    )
    ablate_p.add_argument(
        "--qrels",
        type=Path,
        required=True,
        help="Path to qrels JSONL with query_id, record_id/dataset_id, and relevance/label.",
    )
    ablate_p.add_argument(
        "--queries",
        type=Path,
        default=Path("artifacts/benchmark_queries.jsonl"),
        help="Path to benchmark queries JSONL.",
    )
    ablate_p.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corpus/normalized/combined_corpus.jsonl"),
        help="Corpus snapshot path used for provenance hashing.",
    )
    ablate_p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_ABLATION_JSON_OUT,
        help="Output JSON path; Markdown is written next to it.",
    )
    ablate_p.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Use stable generated_at timestamps for reproducible reports.",
    )
    ablate_p.set_defaults(func=cmd_concept_ablate_retrieval)

    # concept-coverage
    coverage_p = subparsers.add_parser(
        "concept-coverage",
        help="Generate concept coverage audit report for the indexed corpus.",
    )
    coverage_p.add_argument("--field", default=_DEFAULT_FIELD, help="Field memory namespace.")
    coverage_p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output report path (default: reports/field_state/concept_memory_coverage.md).",
    )
    coverage_p.set_defaults(func=cmd_concept_coverage)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = cast(Callable[[argparse.Namespace], int], args.func)
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
