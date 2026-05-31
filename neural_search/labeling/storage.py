"""Label Storage and Persistence.

This module provides storage and retrieval for relevance labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neural_search.labeling.session import RelevanceLabel


@dataclass
class LabelStorage:
    """Manages persistent storage of relevance labels."""

    storage_path: Path
    labels: list[RelevanceLabel] = field(default_factory=list)

    # Indexing
    _by_query: dict[str, list[RelevanceLabel]] = field(default_factory=dict)
    _by_result: dict[str, list[RelevanceLabel]] = field(default_factory=dict)
    _by_pair: dict[str, RelevanceLabel] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.storage_path = Path(self.storage_path)
        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        """Rebuild internal indexes."""
        self._by_query = {}
        self._by_result = {}
        self._by_pair = {}

        for label in self.labels:
            # By query
            if label.query not in self._by_query:
                self._by_query[label.query] = []
            self._by_query[label.query].append(label)

            # By result
            if label.result_id not in self._by_result:
                self._by_result[label.result_id] = []
            self._by_result[label.result_id].append(label)

            # By pair (query, result) -> most recent label
            pair_key = f"{label.query}::{label.result_id}"
            self._by_pair[pair_key] = label

    def add_label(self, label: RelevanceLabel) -> None:
        """Add a label to storage."""
        self.labels.append(label)

        # Update indexes
        if label.query not in self._by_query:
            self._by_query[label.query] = []
        self._by_query[label.query].append(label)

        if label.result_id not in self._by_result:
            self._by_result[label.result_id] = []
        self._by_result[label.result_id].append(label)

        pair_key = f"{label.query}::{label.result_id}"
        self._by_pair[pair_key] = label

    def add_labels(self, labels: list[RelevanceLabel]) -> None:
        """Add multiple labels."""
        for label in labels:
            self.add_label(label)

    def get_labels_for_query(self, query: str) -> list[RelevanceLabel]:
        """Get all labels for a query."""
        return self._by_query.get(query, [])

    def get_labels_for_result(self, result_id: str) -> list[RelevanceLabel]:
        """Get all labels for a result."""
        return self._by_result.get(result_id, [])

    def get_label_for_pair(self, query: str, result_id: str) -> RelevanceLabel | None:
        """Get the label for a specific query-result pair."""
        pair_key = f"{query}::{result_id}"
        return self._by_pair.get(pair_key)

    def has_label(self, query: str, result_id: str) -> bool:
        """Check if a pair has been labeled."""
        pair_key = f"{query}::{result_id}"
        return pair_key in self._by_pair

    def get_relevant_ids_for_query(
        self,
        query: str,
        min_grade: int = 3,
    ) -> set[str]:
        """Get IDs of results labeled as relevant for a query.

        Args:
            query: The search query
            min_grade: Minimum relevance grade to consider relevant

        Returns:
            Set of result IDs
        """
        labels = self.get_labels_for_query(query)
        return {
            label.result_id
            for label in labels
            if label.relevance_grade >= min_grade
        }

    def to_relevance_labels_dict(
        self,
        min_grade: int = 3,
    ) -> dict[str, set[str]]:
        """Convert to dict format for evaluation.

        Args:
            min_grade: Minimum grade to consider relevant

        Returns:
            Dict mapping queries to sets of relevant result IDs
        """
        result: dict[str, set[str]] = {}
        for query in self._by_query:
            relevant = self.get_relevant_ids_for_query(query, min_grade)
            if relevant:
                result[query] = relevant
        return result

    def save(self) -> None:
        """Save labels to storage path."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "saved_at": datetime.now(UTC).isoformat(),
            "total_labels": len(self.labels),
            "unique_queries": len(self._by_query),
            "unique_results": len(self._by_result),
            "labels": [label.model_dump() for label in self.labels],
        }

        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self) -> None:
        """Load labels from storage path."""
        if not self.storage_path.exists():
            return

        with open(self.storage_path) as f:
            data = json.load(f)

        self.labels = [
            RelevanceLabel(**label_data)
            for label_data in data.get("labels", [])
        ]
        self._rebuild_indexes()

    def summary(self) -> dict[str, Any]:
        """Get storage summary statistics."""
        grade_counts = {}
        for label in self.labels:
            grade = label.relevance_grade
            grade_counts[grade] = grade_counts.get(grade, 0) + 1

        return {
            "total_labels": len(self.labels),
            "unique_queries": len(self._by_query),
            "unique_results": len(self._by_result),
            "unique_pairs": len(self._by_pair),
            "grade_distribution": grade_counts,
            "storage_path": str(self.storage_path),
        }


def load_labels(path: str | Path) -> LabelStorage:
    """Load labels from a file.

    Args:
        path: Path to labels JSON file

    Returns:
        LabelStorage with loaded labels
    """
    storage = LabelStorage(storage_path=Path(path))
    storage.load()
    return storage


def save_labels(storage: LabelStorage, path: str | Path | None = None) -> None:
    """Save labels to a file.

    Args:
        storage: LabelStorage to save
        path: Optional override path
    """
    if path:
        storage.storage_path = Path(path)
    storage.save()
