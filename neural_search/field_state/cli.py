"""CLI for lightweight field-state tracking.

Usage:
    python -m neural_search.field_state.cli init
    python -m neural_search.field_state.cli report
    python -m neural_search.field_state.cli opportunities
    python -m neural_search.field_state.cli snapshot
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import cast

from neural_search.field_state.concept_memory.cli import (
    cmd_concept_basis,
    cmd_concept_build,
    cmd_concept_export_obsidian,
    cmd_concept_neighborhood,
    cmd_concept_report,
    cmd_concept_search,
    cmd_concept_validate,
)
from neural_search.field_state.eval_memory.adjudication import adjudicate_qrels
from neural_search.field_state.eval_memory.agreement import write_qrels_agreement
from neural_search.field_state.eval_memory.claim_evidence import (
    write_claim_evidence_suggestions,
)
from neural_search.field_state.eval_memory.eval_snapshot import write_eval_snapshot
from neural_search.field_state.eval_memory.qrels_export import export_qrels_candidates
from neural_search.field_state.eval_memory.qrels_import import import_qrels_reviews
from neural_search.field_state.eval_memory.whitepaper_report import (
    write_whitepaper_validation_report,
)
from neural_search.field_state.memory.diff import compute_memory_diff, write_memory_diff
from neural_search.field_state.memory.index import write_memory_index
from neural_search.field_state.memory.review_overlay import import_review_overlays
from neural_search.field_state.obsidian.dashboard import write_dashboard
from neural_search.field_state.obsidian.task_export import (
    add_decision_note,
    export_codex_task,
)
from neural_search.field_state.obsidian.validators import (
    validate_obsidian_memory,
    write_validation_report,
)
from neural_search.field_state.obsidian.writer import export_obsidian_memory
from neural_search.field_state.reports import (
    generate_opportunities_report,
    generate_reports,
    generate_snapshot_report,
)
from neural_search.field_state.scoring import rank_opportunities
from neural_search.field_state.seeds import (
    seed_benchmark_gaps,
    seed_claims,
    seed_opportunities,
)
from neural_search.field_state.store import (
    read_opportunities,
    write_field_state,
)


def init_field_state(root: Path | None = None) -> dict[str, Path]:
    """Seed the field-state JSONL artifacts."""
    claims = seed_claims(root)
    gaps = seed_benchmark_gaps(root)
    opportunities = seed_opportunities()
    return write_field_state(claims, gaps, opportunities, root)


def cmd_init(args: argparse.Namespace) -> int:
    """Handle the init command."""
    paths = init_field_state(args.root)
    print("Initialized field-state artifacts:")
    for label, path in paths.items():
        print(f"- {label}: {path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Handle the report command."""
    paths = generate_reports(args.root)
    print("Generated field-state reports:")
    for label, path in paths.items():
        print(f"- {label}: {path}")
    return 0


def cmd_opportunities(args: argparse.Namespace) -> int:
    """Handle the opportunities command."""
    opportunities = rank_opportunities(read_opportunities(args.root))
    report_path = generate_opportunities_report(args.root)

    print("Ranked field-state opportunities:")
    if not opportunities:
        print("- none found; run `python -m neural_search.field_state.cli init` first")
    for index, opportunity in enumerate(opportunities, start=1):
        print(f"{index}. {opportunity.title} ({opportunity.total_score:.3f})")
    print(f"Report: {report_path}")
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    """Handle the snapshot command."""
    path = generate_snapshot_report(args.root)
    print(f"Generated field-state snapshot: {path}")
    return 0


def cmd_export_obsidian(args: argparse.Namespace) -> int:
    """Handle the export-obsidian command."""
    result = export_obsidian_memory(args.vault, field=args.field, root=args.root)
    print("Exported field-state Obsidian memory:")
    print(f"- vault: {result.vault_path}")
    print(f"- memory_index: {result.memory_index_path}")
    print(f"- notes_created: {result.notes_created}")
    print(f"- notes_updated: {result.notes_updated}")
    print(f"- notes_skipped: {result.notes_skipped}")
    print(f"- warnings: {len(result.warnings)}")
    return 0


def cmd_import_obsidian(args: argparse.Namespace) -> int:
    """Handle the import-obsidian command."""
    paths = import_review_overlays(args.vault, field=args.field, root=args.root)
    print("Imported Obsidian review overlays:")
    for note_type, path in paths.items():
        print(f"- {note_type}: {path}")
    return 0


def cmd_memory_validate(args: argparse.Namespace) -> int:
    """Handle the memory-validate command."""
    result = validate_obsidian_memory(args.vault, args.field)
    path = write_validation_report(args.vault, result)
    print("Validated field-state Obsidian memory:")
    print(f"- valid: {result.is_valid}")
    print(f"- errors: {result.error_count}")
    print(f"- warnings: {result.warning_count}")
    print(f"- report: {path}")
    return 0 if result.is_valid else 1


def cmd_memory_diff(args: argparse.Namespace) -> int:
    """Handle the memory-diff command."""
    diff = compute_memory_diff(args.vault, args.field)
    path = write_memory_diff(args.vault, args.field)
    print("Generated field-state memory diff:")
    print(f"- human_edited: {len(diff.human_edited)}")
    print(f"- duplicate_ids: {len(diff.duplicate_ids)}")
    print(f"- missing_from_index: {len(diff.missing_from_index)}")
    print(f"- report: {path}")
    return 0


def cmd_export_task(args: argparse.Namespace) -> int:
    """Handle the export-task command."""
    path = export_codex_task(
        opportunity_id=args.opportunity_id,
        vault_path=args.vault,
        field=args.field,
        root=args.root,
    )
    index_path = write_memory_index(args.vault, args.field)
    write_dashboard(args.vault, args.field)
    print(f"Generated Codex task note: {path}")
    print(f"Updated memory index: {index_path}")
    return 0


def cmd_decision_add(args: argparse.Namespace) -> int:
    """Handle the decision-add command."""
    path = add_decision_note(
        vault_path=args.vault,
        field=args.field,
        title=args.title,
        decision=args.decision,
        rationale=args.rationale,
        evidence=args.evidence,
        consequences=args.consequences,
        revisit_criteria=args.revisit_criteria,
    )
    index_path = write_memory_index(args.vault, args.field)
    write_dashboard(args.vault, args.field)
    print(f"Generated decision note: {path}")
    print(f"Updated memory index: {index_path}")
    return 0


def cmd_qrels_export(args: argparse.Namespace) -> int:
    """Handle qrels-export."""
    counts = export_qrels_candidates(
        pool_path=args.pool,
        queries_path=args.queries,
        corpus_path=args.corpus,
        vault_path=args.vault,
        field=args.field,
        limit=args.limit,
        root=args.root,
    )
    print("Exported qrels candidates:")
    for key, value in counts.items():
        print(f"- notes_{key}: {value}")
    return 0


def cmd_qrels_import(args: argparse.Namespace) -> int:
    """Handle qrels-import."""
    reviews = import_qrels_reviews(
        vault_path=args.vault,
        field=args.field,
        root=args.root,
    )
    print(f"Imported qrels reviews: {len(reviews)}")
    return 0


def cmd_qrels_adjudicate(args: argparse.Namespace) -> int:
    """Handle qrels-adjudicate."""
    adjudicated = adjudicate_qrels(args.root)
    needs = sum(
        1 for qrel in adjudicated if qrel.adjudication_status == "needs_adjudication"
    )
    print(f"Adjudicated qrels written: {len(adjudicated)}")
    print(f"Needs adjudication: {needs}")
    return 0


def cmd_qrels_agreement(args: argparse.Namespace) -> int:
    """Handle qrels-agreement."""
    summary = write_qrels_agreement(args.root)
    print("Generated qrels agreement:")
    print(f"- reviewed_candidates: {summary['reviewed_candidates']}")
    print(f"- disagreement_count: {summary['disagreement_count']}")
    return 0


def cmd_eval_snapshot(args: argparse.Namespace) -> int:
    """Handle eval-snapshot."""
    snapshot = write_eval_snapshot(args.root)
    status = snapshot["qrels_status"]
    print("Generated eval snapshot:")
    print(f"- candidates_exported: {status['candidates_exported']}")
    print(f"- candidates_reviewed: {status['candidates_reviewed']}")
    print(f"- candidates_adjudicated: {status['candidates_adjudicated']}")
    return 0


def cmd_claim_evidence_update(args: argparse.Namespace) -> int:
    """Handle claim-evidence-update."""
    suggestions = write_claim_evidence_suggestions(args.root)
    print(f"Generated claim evidence suggestions: {len(suggestions)}")
    return 0


def cmd_whitepaper_validation_report(args: argparse.Namespace) -> int:
    """Handle whitepaper-validation-report."""
    path = write_whitepaper_validation_report(args.root)
    print(f"Generated whitepaper validation report: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the field-state CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m neural_search.field_state.cli",
        description="Track claims, benchmark gaps, and opportunities for Neural Search.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root. Defaults to the current working directory.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Seed field-state JSONL files.")
    init_parser.set_defaults(func=cmd_init)

    report_parser = subparsers.add_parser(
        "report", help="Generate all field-state Markdown reports."
    )
    report_parser.set_defaults(func=cmd_report)

    opportunities_parser = subparsers.add_parser(
        "opportunities", help="Print ranked opportunities and write report."
    )
    opportunities_parser.set_defaults(func=cmd_opportunities)

    snapshot_parser = subparsers.add_parser(
        "snapshot", help="Generate the latest field-state snapshot."
    )
    snapshot_parser.set_defaults(func=cmd_snapshot)

    export_parser = subparsers.add_parser(
        "export-obsidian", help="Export field-state artifacts to an Obsidian vault."
    )
    export_parser.add_argument("--vault", type=Path, required=True)
    export_parser.add_argument(
        "--field", default="neuroscience_dataset_reuse", help="Field memory namespace."
    )
    export_parser.set_defaults(func=cmd_export_obsidian)

    import_parser = subparsers.add_parser(
        "import-obsidian", help="Import Obsidian human review overlays."
    )
    import_parser.add_argument("--vault", type=Path, required=True)
    import_parser.add_argument(
        "--field", default="neuroscience_dataset_reuse", help="Field memory namespace."
    )
    import_parser.set_defaults(func=cmd_import_obsidian)

    validate_parser = subparsers.add_parser(
        "memory-validate", help="Validate an Obsidian field-state vault."
    )
    validate_parser.add_argument("--vault", type=Path, required=True)
    validate_parser.add_argument(
        "--field", default="neuroscience_dataset_reuse", help="Field memory namespace."
    )
    validate_parser.set_defaults(func=cmd_memory_validate)

    diff_parser = subparsers.add_parser(
        "memory-diff", help="Diff an Obsidian field-state vault against its index."
    )
    diff_parser.add_argument("--vault", type=Path, required=True)
    diff_parser.add_argument(
        "--field", default="neuroscience_dataset_reuse", help="Field memory namespace."
    )
    diff_parser.set_defaults(func=cmd_memory_diff)

    task_parser = subparsers.add_parser(
        "export-task", help="Generate a Codex task note from an opportunity."
    )
    task_parser.add_argument("--opportunity-id", required=True)
    task_parser.add_argument("--vault", type=Path, required=True)
    task_parser.add_argument(
        "--field", default="neuroscience_dataset_reuse", help="Field memory namespace."
    )
    task_parser.set_defaults(func=cmd_export_task)

    decision_parser = subparsers.add_parser(
        "decision-add", help="Create a field-state decision note."
    )
    decision_parser.add_argument("--vault", type=Path, required=True)
    decision_parser.add_argument(
        "--field", default="neuroscience_dataset_reuse", help="Field memory namespace."
    )
    decision_parser.add_argument("--title", required=True)
    decision_parser.add_argument("--decision", default=None)
    decision_parser.add_argument("--rationale", default=None)
    decision_parser.add_argument("--evidence", default=None)
    decision_parser.add_argument("--consequences", default=None)
    decision_parser.add_argument("--revisit-criteria", default=None)
    decision_parser.set_defaults(func=cmd_decision_add)

    qrels_export_parser = subparsers.add_parser(
        "qrels-export", help="Export qrels candidates to Obsidian review notes."
    )
    qrels_export_parser.add_argument("--pool", type=Path, required=True)
    qrels_export_parser.add_argument("--queries", type=Path, required=True)
    qrels_export_parser.add_argument("--corpus", type=Path, required=True)
    qrels_export_parser.add_argument("--vault", type=Path, required=True)
    qrels_export_parser.add_argument(
        "--field", default="neuroscience_dataset_reuse", help="Field memory namespace."
    )
    qrels_export_parser.add_argument("--limit", type=int, default=None)
    qrels_export_parser.set_defaults(func=cmd_qrels_export)

    qrels_import_parser = subparsers.add_parser(
        "qrels-import", help="Import qrels review notes from Obsidian."
    )
    qrels_import_parser.add_argument("--vault", type=Path, required=True)
    qrels_import_parser.add_argument(
        "--field", default="neuroscience_dataset_reuse", help="Field memory namespace."
    )
    qrels_import_parser.set_defaults(func=cmd_qrels_import)

    qrels_adjudicate_parser = subparsers.add_parser(
        "qrels-adjudicate", help="Create adjudicated qrels from imported reviews."
    )
    qrels_adjudicate_parser.add_argument(
        "--field", default="neuroscience_dataset_reuse", help="Field memory namespace."
    )
    qrels_adjudicate_parser.set_defaults(func=cmd_qrels_adjudicate)

    qrels_agreement_parser = subparsers.add_parser(
        "qrels-agreement", help="Compute qrels review agreement metrics."
    )
    qrels_agreement_parser.add_argument(
        "--field", default="neuroscience_dataset_reuse", help="Field memory namespace."
    )
    qrels_agreement_parser.set_defaults(func=cmd_qrels_agreement)

    eval_snapshot_parser = subparsers.add_parser(
        "eval-snapshot", help="Generate field-state evaluation snapshot."
    )
    eval_snapshot_parser.add_argument(
        "--field", default="neuroscience_dataset_reuse", help="Field memory namespace."
    )
    eval_snapshot_parser.set_defaults(func=cmd_eval_snapshot)

    claim_evidence_parser = subparsers.add_parser(
        "claim-evidence-update", help="Suggest claim evidence updates from eval artifacts."
    )
    claim_evidence_parser.add_argument(
        "--field", default="neuroscience_dataset_reuse", help="Field memory namespace."
    )
    claim_evidence_parser.set_defaults(func=cmd_claim_evidence_update)

    whitepaper_parser = subparsers.add_parser(
        "whitepaper-validation-report",
        help="Generate a whitepaper-ready validation status report.",
    )
    whitepaper_parser.add_argument(
        "--field", default="neuroscience_dataset_reuse", help="Field memory namespace."
    )
    whitepaper_parser.set_defaults(func=cmd_whitepaper_validation_report)

    # ---------------------------------------------------------------------------
    # concept-* subcommands
    # ---------------------------------------------------------------------------

    concept_build_parser = subparsers.add_parser(
        "concept-build", help="Build full Graph-Indexed Concept Memory."
    )
    concept_build_parser.add_argument(
        "--field", default="neuroscience_dataset_reuse", help="Field memory namespace."
    )
    concept_build_parser.add_argument(
        "--vault", type=Path, default=None, help="Obsidian vault path (optional)."
    )
    concept_build_parser.add_argument(
        "--corpus", type=Path, default=None, help="Corpus JSONL path (optional)."
    )
    concept_build_parser.set_defaults(func=cmd_concept_build)

    concept_search_parser = subparsers.add_parser(
        "concept-search", help="Search concept memory by query string."
    )
    concept_search_parser.add_argument("--query", required=True, help="Search query string.")
    concept_search_parser.add_argument(
        "--field", default="neuroscience_dataset_reuse", help="Field memory namespace."
    )
    concept_search_parser.add_argument(
        "--limit", type=int, default=10, help="Max results to return."
    )
    concept_search_parser.set_defaults(func=cmd_concept_search)

    concept_basis_parser = subparsers.add_parser(
        "concept-basis", help="Print basis record for a concept."
    )
    concept_basis_parser.add_argument(
        "--concept-id", required=True, dest="concept_id", help="Concept ID."
    )
    concept_basis_parser.set_defaults(func=cmd_concept_basis)

    concept_nb_parser = subparsers.add_parser(
        "concept-neighborhood", help="Print the graph neighborhood of a concept."
    )
    concept_nb_parser.add_argument(
        "--concept-id", required=True, dest="concept_id", help="Concept ID."
    )
    concept_nb_parser.add_argument(
        "--depth", type=int, default=2, help="Neighborhood depth in hops."
    )
    concept_nb_parser.set_defaults(func=cmd_concept_neighborhood)

    concept_report_parser = subparsers.add_parser(
        "concept-report", help="Generate all concept memory Markdown reports."
    )
    concept_report_parser.add_argument(
        "--field", default="neuroscience_dataset_reuse", help="Field memory namespace."
    )
    concept_report_parser.set_defaults(func=cmd_concept_report)

    concept_export_parser = subparsers.add_parser(
        "concept-export-obsidian", help="Export concept memory to an Obsidian vault."
    )
    concept_export_parser.add_argument(
        "--vault", type=Path, required=True, help="Obsidian vault path."
    )
    concept_export_parser.add_argument(
        "--field", default="neuroscience_dataset_reuse", help="Field memory namespace."
    )
    concept_export_parser.set_defaults(func=cmd_concept_export_obsidian)

    concept_validate_parser = subparsers.add_parser(
        "concept-validate", help="Validate concept memory artifact integrity."
    )
    concept_validate_parser.set_defaults(func=cmd_concept_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = cast(Callable[[argparse.Namespace], int], args.func)
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
