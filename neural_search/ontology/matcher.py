"""Fuzzy ontology matching for tasks, behaviors, modalities, and regions."""

from __future__ import annotations

import re
from collections.abc import Iterable
from difflib import SequenceMatcher

from neural_search.ontology.loader import get_ontology
from neural_search.ontology.models import BehaviorLabel, LabelMatch, Ontology, Task


ALIASES: dict[str, list[str]] = {
    "bci": ["bci", "brain computer interface", "brain-computer interface"],
    "calcium_imaging": ["calcium imaging", "2 photon", "two photon", "2p"],
    "extracellular_ephys": ["extracellular electrophysiology", "ephys", "spikes"],
    "fiber_photometry": ["fiber photometry", "photometry"],
    "behavior_video": ["behavior video", "video tracking"],
    "pose_tracking": ["pose tracking", "kinematics"],
    "motor_cortex": ["motor cortex", "m1", "primary motor cortex"],
    "visual_cortex": ["visual cortex", "v1", "v2", "v4"],
    "somatosensory_cortex": ["somatosensory cortex", "s1"],
    "parietal_cortex": ["parietal cortex", "ppc"],
    "OFC": ["ofc", "orbitofrontal", "orbitofrontal cortex"],
    "mPFC": ["mpfc", "medial prefrontal", "medial prefrontal cortex"],
    "ACC": ["acc", "anterior cingulate", "anterior cingulate cortex"],
}


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


def _ngrams(tokens: list[str], min_size: int, max_size: int) -> Iterable[str]:
    max_size = min(max_size, len(tokens))
    for size in range(max_size, min_size - 1, -1):
        for start in range(0, len(tokens) - size + 1):
            yield " ".join(tokens[start : start + size])


def _fuzzy_confidence(normalized_text: str, phrase: str) -> tuple[float, str] | None:
    normalized_phrase = normalize_text(phrase)
    if len(normalized_phrase) < 4:
        return None
    tokens = normalized_text.split()
    phrase_size = len(normalized_phrase.split())
    best_score = 0.0
    best_evidence = ""
    for candidate in _ngrams(tokens, max(1, phrase_size - 1), phrase_size + 1):
        score = SequenceMatcher(None, candidate, normalized_phrase).ratio()
        if score > best_score:
            best_score = score
            best_evidence = candidate
    if best_score >= 0.84:
        return round(best_score * 0.82, 3), best_evidence
    return None


def _best_phrase_match(
    text: str,
    item_id: str,
    label: str,
    category: str,
    phrases: list[tuple[str, float, str]],
) -> LabelMatch | None:
    normalized = normalize_text(text)
    best: LabelMatch | None = None
    for phrase, confidence, match_type in phrases:
        if _contains_phrase(normalized, phrase):
            candidate: LabelMatch | None = LabelMatch(
                id=item_id,
                label=label,
                confidence=confidence,
                evidence=phrase,
                category=category,
                match_type=match_type,
            )
            best = _choose_better_match(best, candidate)
    if best is not None:
        return best
    if len(normalized.split()) > 32:
        return None

    for phrase, _, _ in phrases:
        fuzzy = _fuzzy_confidence(normalized, phrase)
        if fuzzy:
            fuzzy_confidence, evidence = fuzzy
            candidate = LabelMatch(
                id=item_id,
                label=label,
                confidence=fuzzy_confidence,
                evidence=evidence,
                category=category,
                match_type="fuzzy",
            )
            best = _choose_better_match(best, candidate)
    return best


def _choose_better_match(
    best: LabelMatch | None, candidate: LabelMatch
) -> LabelMatch:
    if best is None or candidate.confidence > best.confidence:
        return candidate
    if abs(candidate.confidence - best.confidence) <= 0.05 and len(
        candidate.evidence
    ) > len(best.evidence):
        return candidate
    return best


def _aliases_for(value: str) -> list[str]:
    aliases = ALIASES.get(value, [])
    normalized_value = normalize_text(value)
    values = {value, normalized_value, normalized_value.replace(" ", "_"), *aliases}
    return [item for item in values if item]


def match_tasks(text: str, ontology: Ontology | None = None) -> list[LabelMatch]:
    ontology = ontology or get_ontology()
    matches: list[LabelMatch] = []
    for task in ontology.tasks:
        phrases = [(task.label, 0.98, "label"), (task.id, 0.95, "id")]
        phrases.extend((synonym, 0.94, "synonym") for synonym in task.synonyms)
        match = _best_phrase_match(text, task.id, task.label, task.category, phrases)
        if match:
            matches.append(match)
    return sorted(matches, key=lambda item: item.confidence, reverse=True)


def match_behavior_labels(
    text: str, ontology: Ontology | None = None
) -> list[LabelMatch]:
    ontology = ontology or get_ontology()
    matches: list[LabelMatch] = []
    for behavior in ontology.behavior_labels:
        phrases = [(behavior.label, 0.98, "label"), (behavior.id, 0.95, "id")]
        phrases.extend((synonym, 0.92, "synonym") for synonym in behavior.synonyms)
        match = _best_phrase_match(text, behavior.id, behavior.label, "behavior", phrases)
        if match:
            matches.append(match)
    return sorted(matches, key=lambda item: item.confidence, reverse=True)


def match_modalities(text: str, ontology: Ontology | None = None) -> list[LabelMatch]:
    ontology = ontology or get_ontology()
    matches: list[LabelMatch] = []
    modality_names = sorted({*ontology.modality_names, "bci"})
    for modality in modality_names:
        phrases = [(alias, 0.94 if alias != modality else 0.96, "modality") for alias in _aliases_for(modality)]
        match = _best_phrase_match(text, modality, modality, "modality", phrases)
        if match:
            matches.append(match)
    return sorted(matches, key=lambda item: item.confidence, reverse=True)


def match_brain_regions(text: str, ontology: Ontology | None = None) -> list[LabelMatch]:
    ontology = ontology or get_ontology()
    matches: list[LabelMatch] = []
    for region in ontology.region_names:
        phrases = [(alias, 0.94 if alias != region else 0.96, "region") for alias in _aliases_for(region)]
        match = _best_phrase_match(text, region, region, "brain_region", phrases)
        if match:
            matches.append(match)
    return sorted(matches, key=lambda item: item.confidence, reverse=True)


def expand_query_terms(query: str) -> dict[str, list[str]]:
    ontology = get_ontology()
    tasks = match_tasks(query, ontology)
    behaviors = match_behavior_labels(query, ontology)
    terms = {normalize_text(query)}
    suggested_analyses: set[str] = set()

    for match in tasks:
        task = ontology.task_by_id.get(match.id)
        if not task:
            continue
        terms.update(normalize_text(value) for value in [task.label, task.id, *task.synonyms])
        terms.update(normalize_text(value) for value in task.common_events)
        suggested_analyses.update(task.suggested_analyses)

    for match in behaviors:
        behavior = ontology.behavior_by_id.get(match.id)
        if not behavior:
            continue
        terms.update(normalize_text(value) for value in [behavior.label, behavior.id, *behavior.synonyms])

    return {
        "terms": sorted(value for value in terms if value),
        "task_ids": sorted({match.id for match in tasks}),
        "behavior_ids": sorted({match.id for match in behaviors}),
        "suggested_analyses": sorted(suggested_analyses),
    }


def match_all(text: str, ontology: Ontology | None = None) -> dict[str, list[LabelMatch]]:
    ontology = ontology or get_ontology()
    return {
        "tasks": match_tasks(text, ontology),
        "behaviors": match_behavior_labels(text, ontology),
        "regions": match_brain_regions(text, ontology),
        "modalities": match_modalities(text, ontology),
    }


class OntologyMatcher:
    """Object API for ontology matching."""

    def __init__(self, ontology: Ontology | None = None):
        self.ontology = ontology or get_ontology()

    def match_tasks(self, text: str) -> list[LabelMatch]:
        return match_tasks(text, self.ontology)

    def match_behavior_labels(self, text: str) -> list[LabelMatch]:
        return match_behavior_labels(text, self.ontology)

    def match_modalities(self, text: str) -> list[LabelMatch]:
        return match_modalities(text, self.ontology)

    def match_brain_regions(self, text: str) -> list[LabelMatch]:
        return match_brain_regions(text, self.ontology)

    def match_all(self, text: str) -> dict[str, list[LabelMatch]]:
        return match_all(text, self.ontology)

    def expand_query_terms(self, query: str) -> dict[str, list[str]]:
        return expand_query_terms(query)
