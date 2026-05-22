"""Ontology loading, query expansion, deterministic matching, and CLI."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ONTOLOGY_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "ontology" / "behavioral_task_ontology.yaml"
)


@dataclass(frozen=True)
class LabelMatch:
    id: str
    label: str
    confidence: float
    evidence: str
    category: str | None = None


@dataclass(frozen=True)
class Task:
    id: str
    label: str
    category: str
    definition: str = ""
    synonyms: list[str] = field(default_factory=list)
    common_events: list[str] = field(default_factory=list)
    relevant_modalities: list[str] = field(default_factory=list)
    relevant_regions: list[str] = field(default_factory=list)
    suggested_analyses: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, item: dict[str, Any]) -> "Task":
        return cls(
            id=str(item["id"]),
            label=str(item["label"]),
            category=str(item.get("category", "")),
            definition=str(item.get("definition", "")).strip(),
            synonyms=[str(value) for value in item.get("synonyms", [])],
            common_events=[str(value) for value in item.get("common_events", [])],
            relevant_modalities=[str(value) for value in item.get("relevant_modalities", [])],
            relevant_regions=[str(value) for value in item.get("relevant_regions", [])],
            suggested_analyses=[str(value) for value in item.get("suggested_analyses", [])],
        )


@dataclass(frozen=True)
class BehaviorLabel:
    id: str
    label: str
    synonyms: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, item: dict[str, Any]) -> "BehaviorLabel":
        return cls(
            id=str(item["id"]),
            label=str(item["label"]),
            synonyms=[str(value) for value in item.get("synonyms", [])],
        )


@dataclass
class Ontology:
    tasks: list[Task]
    behavior_labels: list[BehaviorLabel]

    @property
    def task_by_id(self) -> dict[str, Task]:
        return {task.id: task for task in self.tasks}

    @property
    def behavior_by_id(self) -> dict[str, BehaviorLabel]:
        return {behavior.id: behavior for behavior in self.behavior_labels}


_ONTOLOGY: Ontology | None = None


def normalize_text(text: str) -> str:
    lowered = text.casefold()
    lowered = re.sub(r"[/_-]+", " ", lowered)
    lowered = re.sub(r"[^a-z0-9+]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _contains_phrase(normalized_text: str, phrase: str) -> bool:
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    return re.search(rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)", normalized_text) is not None


def _best_match(
    normalized_text: str,
    item_id: str,
    label: str,
    category: str | None,
    phrases: list[tuple[str, float]],
) -> LabelMatch | None:
    best: LabelMatch | None = None
    for phrase, confidence in phrases:
        if _contains_phrase(normalized_text, phrase):
            candidate = LabelMatch(
                id=item_id,
                label=label,
                confidence=confidence,
                evidence=phrase,
                category=category,
            )
            if best is None:
                best = candidate
            elif candidate.confidence > best.confidence:
                best = candidate
            elif (
                best.confidence - candidate.confidence <= 0.05
                and len(candidate.evidence) > len(best.evidence)
            ):
                best = candidate
    return best


def load_ontology(path: str | Path = DEFAULT_ONTOLOGY_PATH) -> Ontology:
    """Load ontology YAML and set it as the module-level default."""

    global _ONTOLOGY
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    ontology = Ontology(
        tasks=[Task.from_mapping(item) for item in raw.get("tasks", [])],
        behavior_labels=[
            BehaviorLabel.from_mapping(item) for item in raw.get("behavior_labels", [])
        ],
    )
    _ONTOLOGY = ontology
    return ontology


def get_ontology() -> Ontology:
    global _ONTOLOGY
    if _ONTOLOGY is None:
        _ONTOLOGY = load_ontology(DEFAULT_ONTOLOGY_PATH)
    return _ONTOLOGY


def get_all_tasks() -> list[Task]:
    return get_ontology().tasks


def get_task_by_id(task_id: str) -> Task | None:
    return get_ontology().task_by_id.get(task_id)


def match_tasks(text: str) -> list[LabelMatch]:
    """Match ontology tasks in text with evidence and confidence."""

    normalized = normalize_text(text)
    matches: list[LabelMatch] = []
    for task in get_ontology().tasks:
        phrases: list[tuple[str, float]] = [(task.label, 0.98), (task.id, 0.95)]
        phrases.extend((synonym, 0.94) for synonym in task.synonyms)
        match = _best_match(normalized, task.id, task.label, task.category, phrases)
        if match:
            matches.append(match)
    return sorted(matches, key=lambda item: item.confidence, reverse=True)


def match_behavior_labels(text: str) -> list[LabelMatch]:
    """Match cross-cutting behavior labels in text."""

    normalized = normalize_text(text)
    matches: list[LabelMatch] = []
    for behavior in get_ontology().behavior_labels:
        phrases: list[tuple[str, float]] = [(behavior.label, 0.98), (behavior.id, 0.95)]
        phrases.extend((synonym, 0.9) for synonym in behavior.synonyms)
        match = _best_match(normalized, behavior.id, behavior.label, "behavior", phrases)
        if match:
            matches.append(match)
    return sorted(matches, key=lambda item: item.confidence, reverse=True)


def expand_query_terms(query: str) -> dict[str, list[str]]:
    """Expand query terms using matched tasks and behavior labels."""

    tasks = match_tasks(query)
    behaviors = match_behavior_labels(query)
    terms = {normalize_text(query)}
    task_ids: list[str] = []
    behavior_ids: list[str] = []
    suggested_analyses: set[str] = set()

    for match in tasks:
        task = get_task_by_id(match.id)
        if not task:
            continue
        task_ids.append(task.id)
        terms.update(normalize_text(value) for value in [task.label, task.id, *task.synonyms])
        terms.update(normalize_text(value) for value in task.common_events)
        suggested_analyses.update(task.suggested_analyses)

    for match in behaviors:
        behavior = get_ontology().behavior_by_id.get(match.id)
        if not behavior:
            continue
        behavior_ids.append(behavior.id)
        terms.update(normalize_text(value) for value in [behavior.label, behavior.id, *behavior.synonyms])

    return {
        "terms": sorted(value for value in terms if value),
        "task_ids": sorted(set(task_ids)),
        "behavior_ids": sorted(set(behavior_ids)),
        "suggested_analyses": sorted(suggested_analyses),
    }


class OntologyMatcher:
    """Small compatibility wrapper for callers that prefer an object API."""

    def __init__(self, ontology: Ontology | None = None):
        self.ontology = ontology or get_ontology()

    def match_tasks(self, text: str) -> list[LabelMatch]:
        return match_tasks(text)

    def match_behavior_labels(self, text: str) -> list[LabelMatch]:
        return match_behavior_labels(text)

    def expand_query_terms(self, query: str) -> dict[str, list[str]]:
        return expand_query_terms(query)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neural_search.ontology")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("path")
    args = parser.parse_args(argv)

    if args.command == "validate":
        ontology = load_ontology(args.path)
        print(
            f"Loaded {len(ontology.tasks)} tasks and "
            f"{len(ontology.behavior_labels)} behavior labels from {args.path}"
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
