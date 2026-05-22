"""Ontology matching and synonym expansion."""

import re
from dataclasses import dataclass
from typing import Optional

from .models import Ontology, Task
from .loader import get_ontology


@dataclass
class MatchResult:
    """Result of an ontology match."""

    task: Task
    match_type: str  # "exact", "synonym", "partial"
    matched_term: str
    confidence: float  # 0.0 to 1.0
    evidence: str


class OntologyMatcher:
    """
    Match text against the ontology.

    Supports:
    - Exact ID/label matching
    - Synonym expansion
    - Partial/fuzzy matching
    - Category filtering
    """

    def __init__(self, ontology: Optional[Ontology] = None):
        self.ontology = ontology or get_ontology()
        self._build_index()

    def _build_index(self) -> None:
        """Build search indices for fast matching."""
        self._id_index: dict[str, Task] = {}
        self._label_index: dict[str, Task] = {}
        self._synonym_index: dict[str, Task] = {}

        for task in self.ontology.tasks:
            self._id_index[task.id.lower()] = task
            self._label_index[task.label.lower()] = task
            for syn in task.synonyms:
                self._synonym_index[syn.lower()] = task

    def match_task(self, query: str) -> Optional[MatchResult]:
        """
        Find the best matching task for a query.

        Args:
            query: Text to match against ontology.

        Returns:
            MatchResult if found, None otherwise.
        """
        query_lower = query.lower().strip()

        # Exact ID match
        if query_lower in self._id_index:
            task = self._id_index[query_lower]
            return MatchResult(
                task=task,
                match_type="exact",
                matched_term=task.id,
                confidence=1.0,
                evidence=f"Exact match on task ID: {task.id}",
            )

        # Exact label match
        if query_lower in self._label_index:
            task = self._label_index[query_lower]
            return MatchResult(
                task=task,
                match_type="exact",
                matched_term=task.label,
                confidence=1.0,
                evidence=f"Exact match on task label: {task.label}",
            )

        # Synonym match
        if query_lower in self._synonym_index:
            task = self._synonym_index[query_lower]
            return MatchResult(
                task=task,
                match_type="synonym",
                matched_term=query,
                confidence=0.95,
                evidence=f"Synonym match: '{query}' -> {task.label}",
            )

        # Partial match
        for task in self.ontology.tasks:
            if query_lower in task.label.lower():
                return MatchResult(
                    task=task,
                    match_type="partial",
                    matched_term=task.label,
                    confidence=0.7,
                    evidence=f"Partial match in label: {task.label}",
                )
            for syn in task.synonyms:
                if query_lower in syn.lower():
                    return MatchResult(
                        task=task,
                        match_type="partial",
                        matched_term=syn,
                        confidence=0.6,
                        evidence=f"Partial match in synonym: {syn}",
                    )

        return None

    def find_all_matches(self, query: str) -> list[MatchResult]:
        """Find all matching tasks, sorted by confidence."""
        query_lower = query.lower().strip()
        results: list[MatchResult] = []
        seen_ids: set[str] = set()

        for task in self.ontology.tasks:
            if task.id in seen_ids:
                continue

            # Check for matches
            if query_lower == task.id.lower():
                results.append(
                    MatchResult(
                        task=task,
                        match_type="exact",
                        matched_term=task.id,
                        confidence=1.0,
                        evidence=f"Exact ID match",
                    )
                )
                seen_ids.add(task.id)
            elif query_lower == task.label.lower():
                results.append(
                    MatchResult(
                        task=task,
                        match_type="exact",
                        matched_term=task.label,
                        confidence=1.0,
                        evidence=f"Exact label match",
                    )
                )
                seen_ids.add(task.id)
            elif any(query_lower == syn.lower() for syn in task.synonyms):
                matched_syn = next(
                    s for s in task.synonyms if s.lower() == query_lower
                )
                results.append(
                    MatchResult(
                        task=task,
                        match_type="synonym",
                        matched_term=matched_syn,
                        confidence=0.95,
                        evidence=f"Synonym match: {matched_syn}",
                    )
                )
                seen_ids.add(task.id)
            elif query_lower in task.label.lower():
                results.append(
                    MatchResult(
                        task=task,
                        match_type="partial",
                        matched_term=task.label,
                        confidence=0.7,
                        evidence=f"Partial label match",
                    )
                )
                seen_ids.add(task.id)
            elif any(query_lower in syn.lower() for syn in task.synonyms):
                matched_syn = next(
                    s for s in task.synonyms if query_lower in s.lower()
                )
                results.append(
                    MatchResult(
                        task=task,
                        match_type="partial",
                        matched_term=matched_syn,
                        confidence=0.6,
                        evidence=f"Partial synonym match: {matched_syn}",
                    )
                )
                seen_ids.add(task.id)

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def expand_synonyms(self, task_id: str) -> list[str]:
        """Get all synonyms for a task ID."""
        task = self.ontology.get_task(task_id)
        if not task:
            return []
        return [task.label, task.id] + task.synonyms

    def get_suggested_analyses(self, task_id: str) -> list[str]:
        """Get suggested analyses for a task."""
        task = self.ontology.get_task(task_id)
        if not task:
            return []
        return task.suggested_analyses

    def get_relevant_modalities(self, task_id: str) -> list[str]:
        """Get relevant modalities for a task."""
        task = self.ontology.get_task(task_id)
        if not task:
            return []
        return task.relevant_modalities

    def get_relevant_regions(self, task_id: str) -> list[str]:
        """Get relevant brain regions for a task."""
        task = self.ontology.get_task(task_id)
        if not task:
            return []
        return task.relevant_regions

    def extract_tasks_from_text(self, text: str) -> list[MatchResult]:
        """
        Extract task mentions from free text.

        Scans text for any ontology terms.
        """
        results: list[MatchResult] = []
        seen_ids: set[str] = set()
        text_lower = text.lower()

        for task in self.ontology.tasks:
            if task.id in seen_ids:
                continue

            # Check label
            if task.label.lower() in text_lower:
                results.append(
                    MatchResult(
                        task=task,
                        match_type="exact",
                        matched_term=task.label,
                        confidence=0.9,
                        evidence=f"Found '{task.label}' in text",
                    )
                )
                seen_ids.add(task.id)
                continue

            # Check synonyms
            for syn in task.synonyms:
                if syn.lower() in text_lower:
                    results.append(
                        MatchResult(
                            task=task,
                            match_type="synonym",
                            matched_term=syn,
                            confidence=0.85,
                            evidence=f"Found synonym '{syn}' in text",
                        )
                    )
                    seen_ids.add(task.id)
                    break

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results
