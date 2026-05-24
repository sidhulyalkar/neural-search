"""Explainable search skeleton for metadata, ontology, and card signals."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from neural_search.cards import generate_dataset_card_json
from neural_search.embeddings import HashingEmbeddingProvider, cosine_similarity
from neural_search.ingestion.demo_seed import build_demo_seed
from neural_search.ontology import (
    expand_query_terms,
    match_behavior_labels,
    match_brain_regions,
    match_modalities,
    match_tasks,
    normalize_text,
)
from neural_search.schemas import DatasetCardRead, SearchResponse, SearchResult
from neural_search.search.query_builder import (
    combine_query_and_structured_text,
    merge_filters,
)

RETRIEVAL_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "config" / "retrieval.yaml"
)

DEFAULT_RETRIEVAL_CONFIG: dict[str, Any] = {
    "weights": {
        "ontology": 0.30,
        "behavior": 0.22,
        "modality": 0.14,
        "metadata": 0.10,
        "semantic": 0.10,
        "readiness": 0.10,
        "paper_confidence": 0.04,
    },
    "penalties": {
        "modality_mismatch": 0.18,
        "missing_required_field": 0.04,
        "max_missing_required_fields": 5,
    },
    "thresholds": {"high_analysis_readiness": 0.80},
    "required_metadata_fields": [
        "species",
        "modalities",
        "brain_regions",
        "tasks",
        "behaviors",
        "data_standards",
        "license",
    ],
    "analysis_required_fields": {},
    "species_aliases": {},
    "analysis_intents": {},
}

MODALITY_TERMS: dict[str, list[str]] = {
    "neuropixels": ["neuropixels", "neuropixel"],
    "calcium_imaging": ["calcium imaging", "two photon", "2p", "gcamp", "optical imaging"],
    "extracellular_ephys": ["extracellular ephys", "ephys", "spikes", "electrophysiology"],
    "eeg": ["eeg", "electroencephalography"],
    "ecog": ["ecog", "electrocorticography"],
    "ieeg": ["ieeg", "intracranial eeg", "depth electrode", "seeg"],
    "meg": ["meg", "magnetoencephalography"],
    "fmri": ["fmri", "functional mri", "bold"],
    "fiber_photometry": ["fiber photometry", "photometry"],
    "behavior_video": ["behavior video", "video"],
    "pose_tracking": ["pose tracking", "kinematics", "deeplabcut", "dlc"],
}

# Generic phrases that expand to multiple modalities
GENERIC_MODALITY_PHRASES: dict[str, list[str]] = {
    "neural recordings": [
        "extracellular_ephys", "calcium_imaging", "neuropixels",
        "fiber_photometry", "ecog", "ieeg", "eeg",
    ],
    "neural activity": [
        "extracellular_ephys", "calcium_imaging", "neuropixels",
        "fiber_photometry", "ecog", "ieeg", "eeg", "fmri",
    ],
    "electrophysiology": ["extracellular_ephys", "neuropixels", "ecog", "ieeg", "eeg"],
}


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@lru_cache(maxsize=4)
def load_retrieval_config(path: str | None = None) -> dict[str, Any]:
    """Load retrieval settings from YAML, falling back to conservative defaults."""

    config_path = Path(path) if path else RETRIEVAL_CONFIG_PATH
    if not config_path.exists():
        return deepcopy(DEFAULT_RETRIEVAL_CONFIG)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        return deepcopy(DEFAULT_RETRIEVAL_CONFIG)
    return _deep_merge(DEFAULT_RETRIEVAL_CONFIG, loaded)


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


def _raw_values(values: Any) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values if value is not None]


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


def _match_aliases(
    normalized: str,
    aliases_by_id: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for concept_id, aliases in aliases_by_id.items():
        for alias in aliases:
            normalized_alias = normalize_text(str(alias))
            if not normalized_alias:
                continue
            if normalized_alias in normalized:
                matches.append(
                    {
                        "id": concept_id,
                        "label": concept_id.replace("_", " ").title(),
                        "confidence": 0.95,
                        "evidence": alias,
                        "match_type": "alias",
                    }
                )
                break
    return matches


def _contains_normalized_phrase(normalized_text: str, phrase: str) -> bool:
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    return re.search(rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)", normalized_text) is not None


def _unique_preserve(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = normalize_text(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def _match_dumps(matches: Sequence[Any]) -> list[dict[str, Any]]:
    return [match.model_dump() if hasattr(match, "model_dump") else dict(match) for match in matches]


def _parse_exclusions(query: str) -> tuple[list[str], list[str]]:
    """Parse negative modality/species exclusions from query.

    Patterns handled:
    - "NOT using X", "not X", "without X", "excluding X", "no X"
    - Returns (excluded_modalities, excluded_species)
    """
    excluded_modalities: list[str] = []
    excluded_species: list[str] = []
    normalized = normalize_text(query)

    # Patterns for exclusion detection
    exclusion_patterns = [
        r"not\s+using\s+(\w+)",
        r"not\s+(\w+)",
        r"without\s+(\w+)",
        r"excluding\s+(\w+)",
        r"no\s+(\w+)\s+(?:data|recordings?|datasets?)",
    ]

    for pattern in exclusion_patterns:
        for match in re.finditer(pattern, normalized, re.IGNORECASE):
            term = match.group(1).strip()
            # Check if it's a modality
            for mod_id, aliases in MODALITY_TERMS.items():
                if term in [normalize_text(a) for a in aliases] or term == normalize_text(mod_id):
                    if mod_id not in excluded_modalities:
                        excluded_modalities.append(mod_id)
                    break

    return excluded_modalities, excluded_species


def parse_query(
    query: str,
    retrieval_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse search intent with ontology expansion and simple metadata hints."""

    config = dict(retrieval_config or load_retrieval_config())
    task_matches = match_tasks(query)
    behavior_matches = match_behavior_labels(query)
    brain_region_matches = match_brain_regions(query)
    normalized = normalize_text(query)

    # Parse exclusions first
    excluded_modalities, excluded_species = _parse_exclusions(query)

    modalities: list[str] = [match.id for match in match_modalities(query)]

    # Check specific modality terms
    for modality, terms in MODALITY_TERMS.items():
        if any(_contains_normalized_phrase(normalized, term) for term in terms):
            modalities.append(modality)

    # Expand generic phrases to multiple modalities
    for phrase, expanded_mods in GENERIC_MODALITY_PHRASES.items():
        if normalize_text(phrase) in normalized:
            modalities.extend(expanded_mods)

    # Remove excluded modalities from positive matches (must be done after all expansions)
    modalities = [m for m in modalities if m not in excluded_modalities]

    expanded = expand_query_terms(query)
    analysis_intents = _match_aliases(
        normalized,
        config.get("analysis_intents", {}),
    )
    analysis_ids = [match["id"] for match in analysis_intents]
    query_analysis_terms = {
        normalize_text(value)
        for value in [
            *analysis_ids,
            *expanded.get("suggested_analyses", []),
        ]
    }
    for suggested in expanded.get("suggested_analyses", []):
        normalized_suggested = normalize_text(suggested)
        if normalized_suggested in normalized and suggested not in analysis_ids:
            analysis_intents.append(
                {
                    "id": suggested,
                    "label": suggested.replace("_", " ").title(),
                    "confidence": 0.90,
                    "evidence": suggested,
                    "match_type": "ontology_suggestion",
                }
            )
            analysis_ids.append(suggested)

    species_intents = _match_aliases(
        normalized,
        config.get("species_aliases", {}),
    )

    return {
        "query": query,
        "normalized_query": normalized,
        "tasks": [match.id for match in task_matches],
        "behaviors": [match.id for match in behavior_matches],
        "modalities": sorted(set(modalities)),
        "excluded_modalities": excluded_modalities,
        "excluded_species": excluded_species,
        "species": sorted({match["id"] for match in species_intents}),
        "brain_regions": [match.id for match in brain_region_matches],
        "analysis": sorted(set(analysis_ids)),
        "task_intent": _match_dumps(task_matches),
        "behavior_intent": _match_dumps(behavior_matches),
        "modality_intent": [
            *[
                {
                    "id": match.id,
                    "label": match.label,
                    "confidence": match.confidence,
                    "evidence": match.evidence,
                    "match_type": match.match_type,
                }
                for match in match_modalities(query)
            ],
            *[
                {
                    "id": modality,
                    "label": modality.replace("_", " ").title(),
                    "confidence": 0.82,
                    "evidence": "generic neural modality phrase",
                    "match_type": "expanded",
                }
                for phrase, expanded_mods in GENERIC_MODALITY_PHRASES.items()
                if normalize_text(phrase) in normalized
                for modality in expanded_mods
            ],
        ],
        "species_intent": species_intents,
        "brain_region_intent": _match_dumps(brain_region_matches),
        "analysis_intent": analysis_intents,
        "inferred_concepts": sorted(
            query_analysis_terms
            | {normalize_text(match.category or "") for match in task_matches}
        ),
        "expanded": expanded,
    }


def _card_labels(card: DatasetCardRead | Mapping[str, Any], group: str) -> set[str]:
    labels = _get_value(card, "scientific_labels", {}) or {}
    values = labels.get(group, []) if isinstance(labels, Mapping) else []
    ids = set()
    for value in values:
        if isinstance(value, Mapping):
            ids.add(normalize_text(str(value.get("id", value.get("label", "")))))
    return {value for value in ids if value}


def _linked_paper_score(dataset: Any, card: DatasetCardRead | Mapping[str, Any] | None) -> float:
    linked_ids = _raw_values(_get_value(dataset, "linked_paper_ids", []))
    metadata = _get_value(dataset, "metadata_json", {}) or {}
    if isinstance(metadata, Mapping):
        linked_ids.extend(_raw_values(metadata.get("linked_paper_ids", [])))
    count = len({value for value in linked_ids if value})
    if card:
        provenance = _get_value(card, "provenance", {}) or {}
        if isinstance(provenance, Mapping):
            count = max(count, int(provenance.get("linked_paper_count", 0) or 0))
    return min(count / 3, 1.0)


def _missing_required_fields(
    dataset: Any,
    parsed_query: Mapping[str, Any],
    retrieval_config: Mapping[str, Any],
) -> list[str]:
    required = set(retrieval_config.get("required_metadata_fields", []))
    analysis_required = retrieval_config.get("analysis_required_fields", {})
    if isinstance(analysis_required, Mapping):
        for analysis_id in parsed_query.get("analysis", []):
            required.update(analysis_required.get(analysis_id, []))

    missing: list[str] = []
    for field in sorted(required):
        value = _get_value(dataset, field, None)
        if value in (None, "", [], {}, False):
            missing.append(field)
    return missing


def _evidence_snippets(text: str, terms: Sequence[str], limit: int = 3) -> list[str]:
    if not text:
        return []
    normalized_text = normalize_text(text)
    snippets: list[str] = []
    for term in _unique_preserve(list(terms)):
        index = normalized_text.find(term)
        if index == -1:
            continue
        start = max(index - 48, 0)
        end = min(index + len(term) + 72, len(normalized_text))
        snippet = normalized_text[start:end].strip()
        if snippet and snippet not in snippets:
            snippets.append(snippet)
        if len(snippets) >= limit:
            break
    return snippets


@lru_cache(maxsize=1)
def _default_embedding_provider() -> HashingEmbeddingProvider:
    return HashingEmbeddingProvider()


def _embedding_semantic_score(query: str, text: str) -> float:
    provider = _default_embedding_provider()
    query_vector, text_vector = provider.embed_texts([query, text])
    return cosine_similarity(query_vector, text_vector)


def _field_score(query_values: set[str], dataset_values: set[str]) -> tuple[float, list[str]]:
    if not query_values:
        return (0.5 if dataset_values else 0.0), []
    matched = sorted(query_values & dataset_values)
    return len(matched) / len(query_values), matched


def _reusable_reason(
    dataset: Any,
    readiness_score: float,
    matched_labels: Sequence[str],
    paper_score: float,
) -> str:
    strengths: list[str] = []
    standards = ", ".join(_raw_values(_get_value(dataset, "data_standards", []))[:2])
    if standards:
        strengths.append(f"standardized {standards} metadata")
    if _get_value(dataset, "has_trials", False):
        strengths.append("trial structure")
    if _get_value(dataset, "has_behavior", False):
        strengths.append("behavioral annotations")
    if readiness_score >= 0.8:
        strengths.append("high analysis readiness")
    if paper_score > 0:
        strengths.append("linked paper provenance")
    if matched_labels:
        strengths.append("direct scientific label matches")
    if not strengths:
        return "Reusable evidence is limited until core metadata and provenance are filled in."
    return "Scientifically reusable because it has " + ", ".join(strengths[:4]) + "."


def score_dataset_against_query(
    dataset: Any,
    card: DatasetCardRead | Mapping[str, Any] | None,
    parsed_query: Mapping[str, Any],
    retrieval_config: Mapping[str, Any] | None = None,
) -> SearchResult:
    """Score one dataset against a parsed query and return explanations."""

    config = dict(retrieval_config or load_retrieval_config())
    weights = config.get("weights", {})
    penalties = config.get("penalties", {})
    dataset_id = _get_value(dataset, "id", _get_value(dataset, "source_id", "unknown"))
    text = _text_for_dataset(dataset, card)
    query_terms = set(parsed_query.get("expanded", {}).get("terms", []))
    task_terms = {normalize_text(value) for value in parsed_query.get("tasks", [])}
    behavior_terms = {normalize_text(value) for value in parsed_query.get("behaviors", [])}
    modality_terms = {normalize_text(value) for value in parsed_query.get("modalities", [])}
    species_terms = {normalize_text(value) for value in parsed_query.get("species", [])}
    region_terms = {normalize_text(value) for value in parsed_query.get("brain_regions", [])}
    analysis_terms = {normalize_text(value) for value in parsed_query.get("analysis", [])}

    dataset_tasks = _normalize_values(_get_value(dataset, "tasks", [])) | (
        _card_labels(card, "tasks") if card else set()
    )
    dataset_behaviors = _normalize_values(_get_value(dataset, "behaviors", [])) | (
        _card_labels(card, "behaviors") if card else set()
    )
    dataset_modalities = _normalize_values(_get_value(dataset, "modalities", [])) | (
        _card_labels(card, "modalities") if card else set()
    )
    dataset_species = _normalize_values(_get_value(dataset, "species", [])) | (
        _card_labels(card, "species") if card else set()
    )
    dataset_regions = _normalize_values(_get_value(dataset, "brain_regions", [])) | (
        _card_labels(card, "brain_regions") if card else set()
    )
    dataset_analyses = _normalize_values(_get_value(card, "suggested_analyses", [])) if card else set()

    keyword_hits = [term for term in query_terms if term and term in text]
    keyword_similarity = min(len(keyword_hits) / max(len(query_terms), 1), 1.0)
    embedding_similarity = _embedding_semantic_score(
        str(parsed_query.get("query", "")),
        text,
    )
    semantic_similarity = max(keyword_similarity, embedding_similarity)
    task_score, matched_tasks = _field_score(task_terms, dataset_tasks)
    behavior_score, matched_behaviors = _field_score(behavior_terms, dataset_behaviors)
    modality_score, matched_modalities = _field_score(modality_terms, dataset_modalities)
    species_score, matched_species = _field_score(species_terms, dataset_species)
    region_score, matched_regions = _field_score(region_terms, dataset_regions)
    analysis_score, matched_analyses = _field_score(analysis_terms, dataset_analyses)

    ontology_match = task_score if task_terms else 0.0
    behavior_match = behavior_score if behavior_terms else 0.0
    metadata_scores = []
    if species_terms:
        metadata_scores.append(species_score)
    if region_terms:
        metadata_scores.append(region_score)
    if analysis_terms:
        metadata_scores.append(analysis_score)
    metadata_match = sum(metadata_scores) / len(metadata_scores) if metadata_scores else 0.0

    why: list[str] = []
    warnings: list[str] = []
    missing_metadata_warnings: list[str] = []
    matched_labels = [
        *matched_tasks,
        *matched_behaviors,
        *matched_modalities,
        *matched_species,
        *matched_regions,
        *matched_analyses,
    ]

    why.extend(f"Task matched: {value}" for value in matched_tasks)
    why.extend(f"Behavior matched: {value}" for value in matched_behaviors)
    why.extend(f"Modality matched: {value}" for value in matched_modalities)
    why.extend(f"Species matched: {value}" for value in matched_species)
    why.extend(f"Brain region matched: {value}" for value in matched_regions)
    why.extend(f"Analysis matched: {value}" for value in matched_analyses)

    modality_penalty = 0.0
    if modality_terms and dataset_modalities and not matched_modalities:
        modality_penalty = float(penalties.get("modality_mismatch", 0.0))
        warnings.append(
            "Modality mismatch: query requested "
            + ", ".join(sorted(modality_terms))
            + " but dataset lists "
            + ", ".join(sorted(dataset_modalities))
            + "."
        )

    # Apply exclusion penalty for hard-negative modalities
    exclusion_penalty = 0.0
    excluded_modalities = {normalize_text(m) for m in parsed_query.get("excluded_modalities", [])}
    if excluded_modalities:
        violated_exclusions = excluded_modalities & dataset_modalities
        if violated_exclusions:
            # Strong penalty for violating exclusion constraints
            exclusion_penalty = float(penalties.get("exclusion_violation", 0.5))
            warnings.append(
                f"Exclusion violation: dataset contains {', '.join(sorted(violated_exclusions))} "
                "which was explicitly excluded from the query."
            )
    else:
        violated_exclusions = set()

    readiness_score = 0.0
    if card:
        readiness = _get_value(card, "analysis_readiness", None)
        readiness_score = _get_value(readiness, "score", 0) / 100
    else:
        warnings.append("No dataset card supplied; readiness contributes zero.")

    missing_fields = sorted(
        set(_get_value(card, "missing_fields", []) if card else [])
        | set(_missing_required_fields(dataset, parsed_query, config))
    )
    max_missing = max(int(penalties.get("max_missing_required_fields", 5)), 1)
    missing_penalty = min(
        len(missing_fields),
        max_missing,
    ) * float(penalties.get("missing_required_field", 0.0))
    for field in missing_fields:
        missing_metadata_warnings.append(f"Missing metadata field: {field}")

    paper_score = _linked_paper_score(dataset, card)
    readiness_threshold = float(
        config.get("thresholds", {}).get("high_analysis_readiness", 0.80)
    )

    final_score = (
        float(weights.get("ontology", 0.0)) * ontology_match
        + float(weights.get("behavior", 0.0)) * behavior_match
        + float(weights.get("modality", 0.0)) * modality_score
        + float(weights.get("metadata", 0.0)) * metadata_match
        + float(weights.get("semantic", 0.0)) * semantic_similarity
        + float(weights.get("readiness", 0.0)) * readiness_score
        + float(weights.get("paper_confidence", 0.0)) * paper_score
        - modality_penalty
        - missing_penalty
        - exclusion_penalty
    )
    final_score = max(0.0, min(final_score, 1.0))
    ontology_score = (
        task_score
        + behavior_score
        + modality_score
        + species_score
        + region_score
    ) / 5
    provenance_score = paper_score
    usability_score = readiness_score
    analysis_fit_score = analysis_score if analysis_terms else 0.0
    negative_constraint_score = max(0.0, 1.0 - exclusion_penalty)
    score_breakdown = {
        "lexical_score": round(keyword_similarity, 3),
        "ontology_score": round(ontology_score, 3),
        "semantic_score": round(semantic_similarity, 3),
        "provenance_score": round(provenance_score, 3),
        "usability_score": round(usability_score, 3),
        "analysis_fit_score": round(analysis_fit_score, 3),
        "negative_constraint_score": round(negative_constraint_score, 3),
        "final_score": round(final_score, 3),
        # Backward-compatible aliases for v0.2 UI/tests.
        "ontology": round(ontology_match, 3),
        "behavior": round(behavior_match, 3),
        "modality": round(modality_score, 3),
        "metadata": round(metadata_match, 3),
        "semantic": round(semantic_similarity, 3),
        "keyword_semantic": round(keyword_similarity, 3),
        "embedding_semantic": round(embedding_similarity, 3),
        "readiness": round(readiness_score, 3),
        "paper_confidence": round(paper_score, 3),
        "penalties": round(modality_penalty + missing_penalty + exclusion_penalty, 3),
    }
    if keyword_hits:
        why.append("Keyword evidence: " + ", ".join(sorted(keyword_hits)[:6]))
    if readiness_score >= readiness_threshold:
        why.append(f"High analysis readiness: {int(readiness_score * 100)}/100")
    if paper_score > 0:
        why.append("Linked papers increase confidence in provenance.")
    if not why:
        warnings.append("Low explainability: no deterministic task, behavior, or modality match.")

    warnings.extend(missing_metadata_warnings)

    preview = {}
    if card:
        preview = {
            "summary": _get_value(card, "summary", ""),
            "analysis_readiness_score": int(readiness_score * 100),
            "suggested_analyses": _get_value(card, "suggested_analyses", [])[:5],
            "score_breakdown": score_breakdown,
        }
    evidence_text = " ".join(
        str(part)
        for part in [
            _get_value(dataset, "title", ""),
            _get_value(dataset, "description", ""),
            _get_value(card, "summary", "") if card else "",
            _get_value(card, "why_relevant", []) if card else [],
        ]
    )
    return SearchResult(
        dataset_id=dataset_id,
        score=round(final_score * 100, 2),
        why_matched=why,
        warnings=warnings,
        matched_terms=sorted(set(matched_labels) | set(keyword_hits[:8])),
        inferred_concepts=list(parsed_query.get("inferred_concepts", [])),
        evidence_snippets=_evidence_snippets(evidence_text, [*matched_labels, *keyword_hits]),
        missing_metadata_warnings=missing_metadata_warnings,
        missing_requirements=missing_fields,
        negative_constraint_matches=sorted(violated_exclusions),
        evidence=[
            {"type": "label_match", "value": value}
            for value in sorted(set(matched_labels))
        ],
        explanation=_reusable_reason(dataset, readiness_score, matched_labels, paper_score),
        reusable_reason=_reusable_reason(dataset, readiness_score, matched_labels, paper_score),
        dataset_card_preview=preview,
        score_breakdown=score_breakdown,
    )


def _passes_filters(
    dataset: Any,
    filters: Mapping[str, Any],
    card: DatasetCardRead | Mapping[str, Any] | None = None,
) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        if key == "min_analysis_readiness_score":
            readiness = _get_value(card, "analysis_readiness", None)
            score = int(_get_value(readiness, "score", 0))
            if score < int(expected):
                return False
            continue
        if key == "reviewed_trusted_only":
            if expected and _get_value(dataset, "qa_status", "auto_generated") not in {
                "reviewed",
                "trusted",
            }:
                return False
            continue
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
    structured_query: Mapping[str, Any] | None = None,
    datasets: Sequence[Mapping[str, Any]] | None = None,
    limit: int = 10,
    retrieval_config: Mapping[str, Any] | None = None,
) -> SearchResponse:
    """Search supplied datasets or the built-in demo seed."""

    config = dict(retrieval_config or load_retrieval_config())
    combined_query = combine_query_and_structured_text(query, structured_query)
    parsed = parse_query(combined_query, config)
    filters = merge_filters(filters, structured_query)
    records = list(datasets) if datasets is not None else build_demo_seed()
    results: list[SearchResult] = []

    for record in records:
        dataset = record.get("dataset", record)
        card = record.get("card")
        if card is None and "extraction" in record:
            card = generate_dataset_card_json(
                dataset, record["extraction"], record.get("papers", [])
            )
        if not _passes_filters(dataset, filters, card):
            continue
        results.append(score_dataset_against_query(dataset, card, parsed, config))

    results.sort(key=lambda item: item.score, reverse=True)
    return SearchResponse(query=combined_query, parsed_query=dict(parsed), results=results[:limit])


def hybrid_search_with_latent(
    query: str,
    filters: Mapping[str, Any] | None = None,
    structured_query: Mapping[str, Any] | None = None,
    datasets: Sequence[Mapping[str, Any]] | None = None,
    limit: int = 10,
    latent_weight: float = 0.2,
    query_dataset_id: str | None = None,
    retrieval_config: Mapping[str, Any] | None = None,
) -> SearchResponse:
    """Search with combined ontology and latent neural similarity scoring.

    This experimental function adds latent feature similarity to the
    standard ontology-based search. When a query_dataset_id is provided,
    it finds datasets with similar neural/behavioral characteristics.

    Args:
        query: Text search query
        filters: Optional metadata filters
        structured_query: Structured query parameters
        datasets: Optional dataset records to search
        limit: Maximum results to return
        latent_weight: Weight for latent similarity (0-1, default 0.2)
        query_dataset_id: Dataset ID to use as similarity anchor
        retrieval_config: Custom retrieval configuration

    Returns:
        SearchResponse with combined scoring
    """
    from neural_search.latent import extract_session_features
    from neural_search.latent.search import search_by_neural_similarity

    # First, run standard ontology search
    base_response = search_datasets(
        query=query,
        filters=filters,
        structured_query=structured_query,
        datasets=datasets,
        limit=limit * 2,  # Get more for reranking
        retrieval_config=retrieval_config,
    )

    if latent_weight <= 0 or not query_dataset_id:
        # No latent scoring requested
        base_response.results = base_response.results[:limit]
        return base_response

    # Build latent index for all datasets
    records = list(datasets) if datasets is not None else build_demo_seed()
    sessions = []
    query_session = None

    for record in records:
        dataset = record.get("dataset", record)
        session = extract_session_features(dataset)
        sessions.append(session)
        if dataset.get("source_id") == query_dataset_id or dataset.get("id") == query_dataset_id:
            query_session = session

    if query_session is None:
        # Query dataset not found, return base results
        base_response.results = base_response.results[:limit]
        return base_response

    # Compute latent similarity
    latent_results = search_by_neural_similarity(
        query_session, sessions, top_k=limit * 2
    )
    latent_scores = {lr.dataset_id: lr.similarity_score for lr in latent_results}

    # Combine scores
    ontology_weight = 1.0 - latent_weight
    for result in base_response.results:
        latent_score = latent_scores.get(result.dataset_id, 0.0)
        # Recompute combined score
        base_score = result.score / 100.0  # Scores are 0-100
        combined = ontology_weight * base_score + latent_weight * latent_score
        result.score = round(combined * 100, 1)
        if latent_score > 0:
            result.why_matched.append(f"Latent similarity: {latent_score:.2f}")

    # Re-sort by combined score
    base_response.results.sort(key=lambda r: r.score, reverse=True)
    base_response.results = base_response.results[:limit]

    return base_response
