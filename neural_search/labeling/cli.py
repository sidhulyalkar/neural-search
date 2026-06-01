"""Command-Line Interface for Relevance Labeling.

This module provides a CLI for human relevance labeling.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neural_search.labeling.session import (
    LabelingSession,
    QueryResultPair,
    RelevanceLabel,
    create_session_from_search_results,
)
from neural_search.labeling.storage import LabelStorage


def _clear_screen() -> None:
    """Clear terminal screen."""
    print("\033[2J\033[H", end="")


def _print_header(session: LabelingSession) -> None:
    """Print session header."""
    progress = session.progress_percentage()
    print("=" * 60)
    print("Neural Search - Relevance Labeling")
    print(f"Session: {session.session_id}")
    print(f"Progress: {session.completed_pairs}/{session.total_pairs} ({progress:.1f}%)")
    print("=" * 60)
    print()


def _print_pair(pair: QueryResultPair, show_score: bool = False) -> None:
    """Print a query-result pair for labeling."""
    print(f"Query: {pair.query}")
    print()
    print(f"Result #{pair.system_rank}:")
    print(f"  ID: {pair.result_id}")
    print(f"  Title: {pair.result_title}")
    print()
    if pair.result_description:
        # Wrap description
        desc = pair.result_description
        while len(desc) > 70:
            split_at = desc[:70].rfind(" ")
            if split_at == -1:
                split_at = 70
            print(f"  {desc[:split_at]}")
            desc = desc[split_at:].strip()
        if desc:
            print(f"  {desc}")
        print()

    if show_score:
        print(f"  System Score: {pair.system_score:.1f}")
        if pair.system_explanation:
            print(f"  Why Matched: {', '.join(pair.system_explanation[:3])}")
        print()


def _print_grade_options() -> None:
    """Print grading options."""
    print("Relevance Grades:")
    print("  [5] EXACT        - Perfect match for query intent")
    print("  [4] HIGHLY_REL   - Very relevant, directly useful")
    print("  [3] RELEVANT     - Relevant, would be useful")
    print("  [2] PARTIAL      - Some relevance, might be useful")
    print("  [1] NOT_RELEVANT - Not relevant to query")
    print("  [0] HARD_NEG     - Seems relevant but wrong")
    print()
    print("Commands: [s]kip, [b]ack, [q]uit, [h]elp")
    print()


def _get_grade_input() -> int | str:
    """Get grade input from user."""
    while True:
        try:
            inp = input("Grade (0-5) or command: ").strip().lower()
            if inp in ["s", "skip"]:
                return "skip"
            if inp in ["b", "back"]:
                return "back"
            if inp in ["q", "quit"]:
                return "quit"
            if inp in ["h", "help"]:
                return "help"
            grade = int(inp)
            if 0 <= grade <= 5:
                return grade
            print("Please enter a grade between 0 and 5")
        except ValueError:
            print("Invalid input. Enter a number 0-5 or a command.")


def _get_dimension_scores() -> tuple[int, int, int, int]:
    """Get dimension scores (task, modality, species, analysis)."""
    print("\nDimension Scores (0=none, 1=partial, 2=good, 3=exact):")

    def get_score(name: str) -> int:
        while True:
            try:
                inp = input(f"  {name} (0-3) [Enter=0]: ").strip()
                if not inp:
                    return 0
                score = int(inp)
                if 0 <= score <= 3:
                    return score
                print("  Please enter 0-3")
            except ValueError:
                print("  Invalid input")

    task = get_score("Task match")
    modality = get_score("Modality match")
    species = get_score("Species match")
    analysis = get_score("Analysis fit")

    return task, modality, species, analysis


def _label_pair(
    pair: QueryResultPair,
    session: LabelingSession,
) -> RelevanceLabel | None:
    """Label a single query-result pair interactively."""
    _clear_screen()
    _print_header(session)
    _print_pair(pair, show_score=session.show_system_score)
    _print_grade_options()

    result = _get_grade_input()

    if result == "quit":
        return None
    if result == "skip":
        session.skip_current()
        return None
    if result == "back":
        session.go_back()
        return None
    if result == "help":
        _print_help()
        input("\nPress Enter to continue...")
        return _label_pair(pair, session)

    # Got a grade, now get dimension scores
    task, modality, species, analysis = _get_dimension_scores()

    # Optional notes
    notes = input("\nNotes (optional): ").strip()

    return RelevanceLabel(
        query=pair.query,
        result_id=pair.result_id,
        result_title=pair.result_title,
        relevance_grade=result,
        task_match=task,
        modality_match=modality,
        species_match=species,
        analysis_fit=analysis,
        labeled_by=session.labeler_id,
        notes=notes,
        system_score=pair.system_score,
        system_rank=pair.system_rank,
    )


def _print_help() -> None:
    """Print help text."""
    print("\n" + "=" * 60)
    print("Labeling Help")
    print("=" * 60)
    print("""
RELEVANCE GRADES:
- 5 (EXACT): The dataset perfectly matches what the query is looking for.
  Example: Query asks for "mouse visual cortex recordings" and dataset
  contains exactly that.

- 4 (HIGHLY_RELEVANT): Very relevant and directly useful for the query.
  Minor differences but still excellent match.

- 3 (RELEVANT): Good match, would be useful. May not be perfect but
  researcher would likely want to see this result.

- 2 (PARTIAL): Some relevance. Could be useful in some contexts but
  not a direct match. May be missing key requirements.

- 1 (NOT_RELEVANT): Not relevant to the query intent. Wrong species,
  wrong modality, wrong task, etc.

- 0 (HARD_NEGATIVE): Superficially looks relevant but is actually
  wrong. Important for training models to distinguish.

DIMENSION SCORES (0-3 each):
- Task match: How well does the dataset's task match the query?
- Modality match: Recording modality alignment?
- Species match: Species alignment?
- Analysis fit: Would the data support the implied analysis?

COMMANDS:
- [s]kip: Skip this pair and come back later
- [b]ack: Go back to previous pair
- [q]uit: End session (progress is saved)
- [h]elp: Show this help
""")


def _print_session_summary(session: LabelingSession) -> None:
    """Print session summary."""
    summary = session.summary()
    print("\n" + "=" * 60)
    print("Session Complete!")
    print("=" * 60)
    print(f"Total pairs: {summary['total_pairs']}")
    print(f"Completed: {summary['completed']}")
    print(f"Skipped: {summary['skipped']}")
    print()
    print("Grade Distribution:")
    for grade, count in summary["grade_distribution"].items():
        if count > 0:
            print(f"  {grade}: {count}")
    print()


def run_labeling_cli(
    session: LabelingSession,
    storage: LabelStorage | None = None,
    auto_save: bool = True,
) -> LabelingSession:
    """Run the interactive labeling CLI.

    Args:
        session: LabelingSession to run
        storage: Optional LabelStorage for persistence
        auto_save: Whether to auto-save after each label

    Returns:
        Updated LabelingSession
    """
    print("\nStarting labeling session...")
    print("Type 'h' for help at any time.\n")
    input("Press Enter to begin...")

    while not session.is_complete():
        pair = session.current_pair()
        if pair is None:
            break

        label = _label_pair(pair, session)

        if label is None:
            # Skip, back, or quit
            if session.is_complete():
                break
            continue

        # Add label
        session.add_label(label)

        if storage:
            storage.add_label(label)
            if auto_save:
                storage.save()

    _print_session_summary(session)

    if storage:
        storage.save()
        print(f"Labels saved to: {storage.storage_path}")

    return session


def run_labeling_from_search(
    query: str,
    search_results: list[dict[str, Any]],
    labeler_id: str = "anonymous",
    storage_path: str | Path | None = None,
    max_pairs: int = 20,
) -> LabelingSession:
    """Convenience function to run labeling from search results.

    Args:
        query: The search query
        search_results: List of search result dictionaries
        labeler_id: ID of the person labeling
        storage_path: Path to save labels
        max_pairs: Maximum pairs to label

    Returns:
        Completed LabelingSession
    """
    session_id = f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    session = create_session_from_search_results(
        session_id=session_id,
        labeler_id=labeler_id,
        query=query,
        search_results=search_results,
        max_pairs=max_pairs,
    )

    storage = None
    if storage_path:
        storage = LabelStorage(storage_path=Path(storage_path))
        if Path(storage_path).exists():
            storage.load()

    return run_labeling_cli(session, storage)


if __name__ == "__main__":
    # Demo mode
    print("Neural Search Labeling CLI - Demo Mode")
    print()

    # Create demo session
    demo_results = [
        {
            "dataset_id": "dandi:000001",
            "title": "Mouse Visual Cortex Neuropixels",
            "description": "Neuropixels recordings from mouse visual cortex during visual stimulation",
            "score": 85.0,
            "why_matched": ["Modality: neuropixels", "Region: visual cortex", "Species: mouse"],
        },
        {
            "dataset_id": "dandi:000002",
            "title": "Rat Hippocampus Ephys",
            "description": "Extracellular recordings from rat hippocampus during navigation",
            "score": 65.0,
            "why_matched": ["Modality: ephys", "Task: navigation"],
        },
    ]

    run_labeling_from_search(
        query="Find Neuropixels recordings from mouse visual cortex",
        search_results=demo_results,
        labeler_id="demo_user",
        storage_path=Path("data/evaluation/demo_labels.json"),
    )
