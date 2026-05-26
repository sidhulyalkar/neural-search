"""CLI tool for relevance labeling of search results.

Usage:
    python -m neural_search.evaluation.label_relevance --query "reversal learning neuropixels" --reviewer "user123"

This tool runs a search query, displays results, and prompts for human relevance judgments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from neural_search.evaluation.relevance import (
    RELEVANCE_LEVELS,
    RelevanceJudgment,
    create_judgment,
    load_relevance_labels,
    save_relevance_labels,
)
from neural_search.search import search_datasets

DEFAULT_OUTPUT = Path("data/eval/relevance_labels.jsonl")


def format_result_for_display(result: Any, rank: int) -> str:
    """Format a search result for CLI display.

    Args:
        result: SearchResult object
        rank: Rank position (1-indexed)

    Returns:
        Formatted string for display
    """
    lines = [
        f"\n{'='*60}",
        f"Rank {rank}: {result.dataset_id}",
        f"{'='*60}",
        f"Title: {result.title}",
        f"Score: {result.score:.3f}",
    ]

    # Add score breakdown if available
    if hasattr(result, "score_breakdown") and result.score_breakdown:
        lines.append("\nScore Breakdown:")
        for component, value in sorted(result.score_breakdown.items()):
            if value > 0:
                lines.append(f"  {component}: {value:.3f}")

    # Add matched reasons
    if result.why_matched:
        lines.append("\nWhy Matched:")
        for reason in result.why_matched[:5]:
            lines.append(f"  - {reason}")

    # Add description snippet
    if hasattr(result, "description") and result.description:
        desc = result.description[:300]
        if len(result.description) > 300:
            desc += "..."
        lines.append(f"\nDescription:\n  {desc}")

    return "\n".join(lines)


def prompt_relevance_level() -> str:
    """Prompt user for relevance level.

    Returns:
        Selected relevance level
    """
    print("\nRelevance Levels:")
    print("  [1] exact           - Perfect match for query intent")
    print("  [2] highly_relevant - Good match, minor gaps")
    print("  [3] relevant        - Matches query, some limitations")
    print("  [4] partially       - Some relevance, significant gaps")
    print("  [5] not_relevant    - Wrong domain/type/species")
    print("  [6] hard_negative   - Should explicitly NOT match")
    print("  [s] skip            - Skip this result")
    print("  [q] quit            - Save and exit")

    while True:
        choice = input("\nSelect relevance [1-6, s, q]: ").strip().lower()

        if choice == "s":
            return "skip"
        if choice == "q":
            return "quit"

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(RELEVANCE_LEVELS):
                return RELEVANCE_LEVELS[idx]
        except ValueError:
            pass

        print("Invalid choice. Please enter 1-6, s, or q.")


def prompt_dimension_score(dimension: str) -> int:
    """Prompt user for dimension-specific score.

    Args:
        dimension: Name of dimension (task, modality, species, analysis)

    Returns:
        Score 0-3
    """
    print(f"\n{dimension.title()} Match Score:")
    print("  [0] Wrong/Unrelated")
    print("  [1] Related/Adjacent")
    print("  [2] Compatible/Close")
    print("  [3] Exact Match")

    while True:
        choice = input(f"  {dimension} score [0-3]: ").strip()

        try:
            score = int(choice)
            if 0 <= score <= 3:
                return score
        except ValueError:
            pass

        print("Invalid choice. Please enter 0-3.")


def prompt_confidence() -> float:
    """Prompt user for confidence level.

    Returns:
        Confidence score 0-1
    """
    print("\nConfidence in your judgment:")
    print("  [1] Low (0.5)    - Uncertain, need more context")
    print("  [2] Medium (0.75) - Reasonably confident")
    print("  [3] High (1.0)   - Very confident")

    while True:
        choice = input("  Confidence [1-3]: ").strip()

        mapping = {"1": 0.5, "2": 0.75, "3": 1.0}
        if choice in mapping:
            return mapping[choice]

        print("Invalid choice. Please enter 1-3.")


def prompt_notes() -> str:
    """Prompt user for optional notes.

    Returns:
        Notes string (may be empty)
    """
    notes = input("\nOptional notes (press Enter to skip): ").strip()
    return notes


def label_results(
    query: str,
    query_id: str,
    reviewer_id: str,
    top_k: int = 10,
    output_path: Path = DEFAULT_OUTPUT,
    verbose: bool = True,
) -> list[RelevanceJudgment]:
    """Run search and collect relevance labels interactively.

    Args:
        query: Search query string
        query_id: Identifier for the query
        reviewer_id: Identifier for the reviewer
        top_k: Number of results to label
        output_path: Path to save labels
        verbose: Whether to show detailed output

    Returns:
        List of RelevanceJudgment objects created
    """
    # Run search
    if verbose:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Query ID: {query_id}")
        print(f"Reviewer: {reviewer_id}")
        print(f"{'='*60}")
        print("\nRunning search...")

    response = search_datasets(query=query, limit=top_k)

    if not response.results:
        print("No results found for query.")
        return []

    if verbose:
        print(f"Found {len(response.results)} results.\n")
        print("For each result, you will be asked to provide relevance judgments.")
        print("Press Ctrl+C at any time to save and exit.\n")

    judgments: list[RelevanceJudgment] = []

    try:
        for rank, result in enumerate(response.results, start=1):
            print(format_result_for_display(result, rank))

            # Prompt for relevance
            relevance = prompt_relevance_level()

            if relevance == "quit":
                break
            if relevance == "skip":
                continue

            # Prompt for dimension scores
            task_match = prompt_dimension_score("task")
            modality_match = prompt_dimension_score("modality")
            species_match = prompt_dimension_score("species")
            analysis_fit = prompt_dimension_score("analysis")

            # Prompt for confidence and notes
            confidence = prompt_confidence()
            notes = prompt_notes()

            # Create judgment
            judgment = create_judgment(
                query_id=query_id,
                query_text=query,
                dataset_id=result.dataset_id,
                dataset_title=result.title,
                relevance=relevance,
                reviewer_id=reviewer_id,
                task_match=task_match,
                modality_match=modality_match,
                species_match=species_match,
                analysis_fit=analysis_fit,
                notes=notes,
                confidence=confidence,
            )

            judgments.append(judgment)
            print(f"\n✓ Judgment recorded for {result.dataset_id}: {relevance}")

    except KeyboardInterrupt:
        print("\n\nInterrupted. Saving judgments collected so far...")

    # Save judgments
    if judgments:
        save_relevance_labels(judgments, output_path, append=True)
        print(f"\nSaved {len(judgments)} judgments to {output_path}")

    return judgments


def generate_query_id(query: str) -> str:
    """Generate a query ID from query text.

    Args:
        query: Query string

    Returns:
        Query ID string
    """
    import hashlib

    # Create hash from query
    query_hash = hashlib.md5(query.lower().encode()).hexdigest()[:8]

    # Create readable prefix from first few words
    words = query.lower().split()[:3]
    prefix = "_".join(w[:8] for w in words if w.isalnum())

    return f"q_{prefix}_{query_hash}"


def main() -> None:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Label search results with human relevance judgments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Label results for a query
    python -m neural_search.evaluation.label_relevance \\
        --query "reversal learning neuropixels" \\
        --reviewer "user123"

    # Label with custom output file
    python -m neural_search.evaluation.label_relevance \\
        --query "mouse hippocampus ephys" \\
        --reviewer "expert1" \\
        --output data/eval/expert_labels.jsonl

    # Label top-5 only
    python -m neural_search.evaluation.label_relevance \\
        --query "decision making" \\
        --reviewer "user123" \\
        --top-k 5
        """,
    )

    parser.add_argument(
        "--query",
        "-q",
        required=True,
        help="Search query to label",
    )
    parser.add_argument(
        "--reviewer",
        "-r",
        required=True,
        help="Reviewer ID for attribution",
    )
    parser.add_argument(
        "--query-id",
        help="Custom query ID (auto-generated if not provided)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=10,
        help="Number of results to label (default: 10)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--show-existing",
        action="store_true",
        help="Show existing labels for this query",
    )

    args = parser.parse_args()

    # Generate query ID if not provided
    query_id = args.query_id or generate_query_id(args.query)

    # Show existing labels if requested
    if args.show_existing:
        if args.output.exists():
            existing = load_relevance_labels(args.output)
            if query_id in existing:
                label_set = existing[query_id]
                print(f"\nExisting labels for query '{args.query}':")
                for j in label_set.judgments:
                    print(f"  {j.dataset_id}: {j.relevance} (by {j.reviewer_id})")
            else:
                print(f"\nNo existing labels for query ID: {query_id}")
        else:
            print(f"\nNo existing labels file: {args.output}")
        return

    # Run labeling
    judgments = label_results(
        query=args.query,
        query_id=query_id,
        reviewer_id=args.reviewer,
        top_k=args.top_k,
        output_path=args.output,
    )

    # Summary
    if judgments:
        print("\n" + "=" * 60)
        print("Labeling Session Summary")
        print("=" * 60)
        print(f"Query: {args.query}")
        print(f"Query ID: {query_id}")
        print(f"Judgments collected: {len(judgments)}")

        relevance_counts: dict[str, int] = {}
        for j in judgments:
            relevance_counts[j.relevance] = relevance_counts.get(j.relevance, 0) + 1

        print("\nRelevance Distribution:")
        for level in RELEVANCE_LEVELS:
            count = relevance_counts.get(level, 0)
            if count > 0:
                print(f"  {level}: {count}")

        print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()
