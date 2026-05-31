"""Labeling Session Management.

This module provides the core session management for human relevance labeling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any

from pydantic import BaseModel, Field


class RelevanceGrade(IntEnum):
    """6-level relevance grading scale."""

    EXACT = 5          # Perfect match for the query intent
    HIGHLY_RELEVANT = 4  # Very relevant, directly useful
    RELEVANT = 3       # Relevant, would be useful
    PARTIALLY = 2      # Some relevance, might be useful in context
    NOT_RELEVANT = 1   # Not relevant to query intent
    HARD_NEGATIVE = 0  # Seems relevant but is actually wrong


class RelevanceLabel(BaseModel):
    """A single relevance label for a query-result pair."""

    # Identifiers
    query: str
    result_id: str
    result_title: str = ""

    # Grading
    relevance_grade: int = Field(ge=0, le=5)

    # Dimension scores (0-3 scale)
    task_match: int = Field(default=0, ge=0, le=3)
    modality_match: int = Field(default=0, ge=0, le=3)
    species_match: int = Field(default=0, ge=0, le=3)
    analysis_fit: int = Field(default=0, ge=0, le=3)

    # Metadata
    labeled_by: str = "anonymous"
    labeled_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    notes: str = ""

    # Context
    query_intent: str = ""
    system_score: float = 0.0
    system_rank: int = 0

    def aggregate_score(self) -> float:
        """Compute aggregate relevance score."""
        # Weighted combination of dimensions
        base_grade = self.relevance_grade / 5.0
        dimension_avg = (
            self.task_match + self.modality_match +
            self.species_match + self.analysis_fit
        ) / 12.0  # Max is 3*4 = 12
        return 0.6 * base_grade + 0.4 * dimension_avg


@dataclass
class QueryResultPair:
    """A query-result pair to be labeled."""

    query: str
    result_id: str
    result_title: str
    result_description: str
    system_score: float
    system_rank: int
    system_explanation: list[str] = field(default_factory=list)

    # Selection info
    priority: float = 0.0
    selection_reason: str = ""


@dataclass
class LabelingSession:
    """Manages a labeling session."""

    session_id: str
    labeler_id: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Session data
    pairs_to_label: list[QueryResultPair] = field(default_factory=list)
    labels: list[RelevanceLabel] = field(default_factory=list)
    current_index: int = 0

    # Progress tracking
    total_pairs: int = 0
    completed_pairs: int = 0
    skipped_pairs: int = 0

    # Session config
    show_system_score: bool = False  # Blind labeling by default
    show_system_explanation: bool = False
    randomize_order: bool = True

    def __post_init__(self) -> None:
        self.total_pairs = len(self.pairs_to_label)

    def current_pair(self) -> QueryResultPair | None:
        """Get the current pair to label."""
        if 0 <= self.current_index < len(self.pairs_to_label):
            return self.pairs_to_label[self.current_index]
        return None

    def add_label(self, label: RelevanceLabel) -> None:
        """Add a label and advance to next pair."""
        self.labels.append(label)
        self.completed_pairs += 1
        self.current_index += 1

    def skip_current(self) -> None:
        """Skip the current pair."""
        self.skipped_pairs += 1
        self.current_index += 1

    def go_back(self) -> bool:
        """Go back to previous pair."""
        if self.current_index > 0:
            self.current_index -= 1
            return True
        return False

    def is_complete(self) -> bool:
        """Check if session is complete."""
        return self.current_index >= len(self.pairs_to_label)

    def progress_percentage(self) -> float:
        """Get completion percentage."""
        if not self.pairs_to_label:
            return 100.0
        return (self.current_index / len(self.pairs_to_label)) * 100

    def summary(self) -> dict[str, Any]:
        """Get session summary."""
        return {
            "session_id": self.session_id,
            "labeler_id": self.labeler_id,
            "total_pairs": self.total_pairs,
            "completed": self.completed_pairs,
            "skipped": self.skipped_pairs,
            "remaining": len(self.pairs_to_label) - self.current_index,
            "progress": f"{self.progress_percentage():.1f}%",
            "grade_distribution": self._grade_distribution(),
        }

    def _grade_distribution(self) -> dict[str, int]:
        """Get distribution of grades assigned."""
        dist: dict[str, int] = {g.name: 0 for g in RelevanceGrade}
        for label in self.labels:
            try:
                grade_name = RelevanceGrade(label.relevance_grade).name
                dist[grade_name] += 1
            except ValueError:
                pass
        return dist


def create_session_from_search_results(
    session_id: str,
    labeler_id: str,
    query: str,
    search_results: list[dict[str, Any]],
    max_pairs: int = 20,
) -> LabelingSession:
    """Create a labeling session from search results.

    Args:
        session_id: Unique session identifier
        labeler_id: ID of the person labeling
        query: The search query
        search_results: List of search result dictionaries
        max_pairs: Maximum pairs to include

    Returns:
        LabelingSession ready for labeling
    """
    pairs = []
    for rank, result in enumerate(search_results[:max_pairs], 1):
        pairs.append(QueryResultPair(
            query=query,
            result_id=result.get("dataset_id", result.get("id", "")),
            result_title=result.get("title", ""),
            result_description=result.get("description", "")[:500],
            system_score=result.get("score", 0.0),
            system_rank=rank,
            system_explanation=result.get("why_matched", []),
        ))

    return LabelingSession(
        session_id=session_id,
        labeler_id=labeler_id,
        pairs_to_label=pairs,
    )
