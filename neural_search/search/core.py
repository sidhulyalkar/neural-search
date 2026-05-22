"""Explainable search skeleton for metadata, ontology, and card signals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from neural_search.cards import generate_dataset_card_json
from neural_search.ingestion.demo_seed import build_demo_seed
from neural_search.ontology import expand_query_terms, match_behavior_labels, match_tasks, normalize_text
from neural_search.schemas import DatasetCardRead, SearchResponse, SearchResult


MODALITY_TERMS: dict[str, list[str]] = {
    "neuropixels": ["neuropixels", "neuropixel"],
    "calcium_imaging": ["calcium imaging", "two photon", "2p", "gcamp"],
    "eeg": ["eeg"],
    "ecog": ["ecog"],
    "ieeg": ["ieeg", "intracranial eeg"],
    "meg": ["meg"],
    "fmri": ["fmri"],
    "behavior_video": ["behavior video", "video"],
    "pose_tracking": ["pose tracking", "kinematics"],
}


def _get_value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _normalize_values(values: Any) -> set[str]:
    if not values:
        return set()
    if isinstance(values, str):
        return {normalize_text(values)}
    return {normalize_text(str(value)) for value in values}


def _text_for_dataset(dataset: Any, card: DatasetCardRead | Mapping[str, Any] | None = None) -> str:
    pieces = [
        _get_value(dataset, "title", ""),
        _get_value(dataset, "description", ""),
        _get_value(dataset, "source_id", ""),
        _get_value(dataset, "metadata_json", {}),
    ]
    if card:
        pieces.extend([_get_value(card, "summary", ""), _get_value(card, "why_relevant", [])])
    return normalize_text(" ".join(str(piece) for piece in pieces))


def parse_query(query: str) -> dict[str, Any]:
    """Parse search intent with ontology expansion and simple metadata hints."""

    task_matches = match_tasks(query)
    behavior_matches = match_behavior_labels(query)
    normalized = normalize_text(query)
    modalities: list[str] = []
    for modality, terms in MODALITY_TERMS.items():
        if any(normalize_text(term) in normalized for term in terms):
            modalities.append(modality)
    return {
        "query": query,
        "normalized_query": normalized,
        "tasks": [match.id for match in task_matches],
        "behaviors": [match.id for match in behavior_matches],
        "modalities": sorted(set(modalities)),
        "expanded": expand_query_terms(query),
    }


def _card_labels(card: DatasetCardRead | Mapping[str, Any], group: str) -> set[str]:
    labels = _get_value(card, "scientific_labels", {}) or {}
    values = labels.get(group, []) if isinstance(labels, Mapping) else []
    ids = set()
    for value in values:
        if isinstance(value, Mapping):
            ids.add(normalize_text(str(value.get("id", value.get("label", "")))))
    return {value for value in ids if value}


def score_dataset_against_query(
    dataset: Any,
    card: DatasetCardRead | Mapping[str, Any] | None,
    parsed_query: Mapping[str, Any],
) -> SearchResult:
    """Score one dataset against a parsed query and return explanations."""

    dataset_id = _get_value(dataset, "id", _get_value(dataset, "source_id", "unknown"))
    text = _text_for_dataset(dataset, card)
    query_terms = set(parsed_query.get("expanded", {}).get("terms", []))
    task_terms = {normalize_text(value) for value in parsed_query.get("tasks", [])}
    behavior_terms = {normalize_text(value) for value in parsed_query.get("behaviors", [])}
    modality_terms = {normalize_text(value) for value in parsed_query.get("modalities", [])}

    dataset_tasks = _normalize_values(_get_value(dataset, "tasks", [])) | (
        _card_labels(card, "tasks") if card else set()
    )
    dataset_behaviors = _normalize_values(_get_value(dataset, "behaviors", [])) | (
        _card_labels(card, "behaviors") if card else set()
    )
    dataset_modalities = _normalize_values(_get_value(dataset, "modalities", [])) | (
        _card_labels(card, "modalities") if card else set()
    )

    keyword_hits = [term for term in query_terms if term and term in text]
    semantic_similarity = min(len(keyword_hits) / max(len(query_terms), 1), 1.0)
    ontology_match = 0.0
    metadata_match = 0.0
    why: list[str] = []
    warnings: list[str] = []

    if task_terms:
        matched = sorted(task_terms & dataset_tasks)
        ontology_match += len(matched) / len(task_terms)
        why.extend(f"Task matched: {value}" for value in matched)
    if behavior_terms:
        matched = sorted(behavior_terms & dataset_behaviors)
        ontology_match += len(matched) / len(behavior_terms)
        why.extend(f"Behavior matched: {value}" for value in matched)
    ontology_match = min(ontology_match / (2 if behavior_terms and task_terms else 1), 1.0)

    if modality_terms:
        matched_modalities = sorted(modality_terms & dataset_modalities)
        metadata_match += len(matched_modalities) / len(modality_terms)
        why.extend(f"Modality matched: {value}" for value in matched_modalities)
    if not modality_terms:
        metadata_match = 0.5 if dataset_modalities else 0.0

    readiness_score = 0.0
    if card:
        readiness = _get_value(card, "analysis_readiness", None)
        readiness_score = _get_value(readiness, "score", 0) / 100
    else:
        warnings.append("No dataset card supplied; readiness contributes zero.")

    final_score = (
        0.30 * semantic_similarity
        + 0.25 * ontology_match
        + 0.20 * metadata_match
        + 0.15 * readiness_score
        + 0.10 * (1.0 if keyword_hits else 0.0)
    )
    if keyword_hits:
        why.append("Keyword evidence: " + ", ".join(sorted(keyword_hits)[:5]))
    if not why:
        warnings.append("Low explainability: no deterministic task, behavior, or modality match.")

    preview = {}
    if card:
        preview = {
            "summary": _get_value(card, "summary", ""),
            "analysis_readiness_score": int(readiness_score * 100),
            "suggested_analyses": _get_value(card, "suggested_analyses", [])[:5],
        }
    return SearchResult(
        dataset_id=dataset_id,
        score=round(final_score * 100, 2),
        why_matched=why,
        warnings=warnings,
        dataset_card_preview=preview,
    )


def _passes_filters(dataset: Any, filters: Mapping[str, Any]) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        actual = _get_value(dataset, key, None)
        if isinstance(expected, list):
            if not (_normalize_values(actual) & _normalize_values(expected)):
                return False
        elif normalize_text(str(actual)) != normalize_text(str(expected)):
            return False
    return True


def search_datasets(
    query: str,
    filters: Mapping[str, Any] | None = None,
    datasets: Sequence[Mapping[str, Any]] | None = None,
    limit: int = 10,
) -> SearchResponse:
    """Search supplied datasets or the built-in demo seed."""

    parsed = parse_query(query)
    filters = filters or {}
    records = list(datasets) if datasets is not None else build_demo_seed()
    results: list[SearchResult] = []

    for record in records:
        dataset = record.get("dataset", record)
        if not _passes_filters(dataset, filters):
            continue
        card = record.get("card")
        if card is None and "extraction" in record:
            card = generate_dataset_card_json(
                dataset, record["extraction"], record.get("papers", [])
            )
        results.append(score_dataset_against_query(dataset, card, parsed))

    results.sort(key=lambda item: item.score, reverse=True)
    return SearchResponse(query=query, parsed_query=dict(parsed), results=results[:limit])

