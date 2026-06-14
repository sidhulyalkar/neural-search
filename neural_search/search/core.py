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
from neural_search.field_state.retrieval_bridge import (
    compute_memory_graph_evidence,
    compute_memory_graph_score,
    load_memory_graph_store,
)
from neural_search.graph.search_features import (
    compute_graph_features_for_result,
    graph_context_score,
    load_graph_if_exists,
)
from neural_search.graph.transitive import (
    expand_query_with_graph,
)
from neural_search.ingestion.demo_seed import build_combined_corpus
from neural_search.ontology import (
    expand_query_terms,
    match_affordances,
    match_behavior_labels,
    match_brain_regions,
    match_modalities,
    match_recording_scales,
    match_tasks,
    normalize_text,
)
from neural_search.retrieval.dataset_context_bridge import dataset_context_from_record
from neural_search.retrieval.query_intent import (
    classify_query_intent as classify_usefulness_intent,
)
from neural_search.retrieval.usefulness_scorer import DatasetContext, score_usefulness
from neural_search.schemas import DatasetCardRead, SearchResponse, SearchResult
from neural_search.search.constraints import (
    negative_constraint_violations,
    parse_hard_negative_constraints,
)
from neural_search.search.explanation import (
    ExplanationContext,
    MatchGroup,
    generate_explanation,
)
from neural_search.search.field_semantic import (
    field_semantic_score_for_result,
    load_field_semantic_index,
)
from neural_search.search.intent import (
    blend_weights,
    classify_query_intent,
)
from neural_search.search.query_builder import (
    combine_query_and_structured_text,
    merge_filters,
)
from neural_search.species import (
    species_exclusions_for_only_query,
    species_query_matches,
    species_terms_for_values,
)


# Lazy import for awareness to avoid circular imports
def _get_awareness_modules():
    from neural_search.awareness.scoring import score_dataset_awareness
    from neural_search.awareness.taxonomy import infer_query_awareness
    return infer_query_awareness, score_dataset_awareness


# Lazy import for planner to avoid circular imports
def _get_planner():
    from neural_search.intelligence.planner import plan_search_intelligence
    return plan_search_intelligence


RETRIEVAL_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "config" / "retrieval.yaml"
)

DEFAULT_RETRIEVAL_CONFIG: dict[str, Any] = {
    "weights": {
        "ontology": 0.28,
        "behavior": 0.20,
        "modality": 0.12,
        "affordance": 0.10,
        "metadata": 0.10,
        "semantic": 0.08,
        "field_semantic": 0.06,
        "graph": 0.04,
        "readiness": 0.08,
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
    "graph": {
        "enabled": False,
        "path": "data/graph/neural_search_graph.demo_v05.json",
        "weights": {},
    },
    "field_embeddings": {
        "enabled": False,
        "path": "data/embeddings/demo_v05.field_embeddings.jsonl",
        "field_weights": {},
    },
    "awareness": {
        "enabled": False,
        "weight": 0.0,
        "rerank": False,
    },
    "planner": {
        "enabled": False,
        "blend_factor": 0.5,  # How much to weight planner weights vs base (0=base, 1=planner)
    },
    "memory_graph": {
        "enabled": False,
        "nodes_path": "artifacts/field_state/memory_graph_nodes.jsonl",
        "edges_path": "artifacts/field_state/memory_graph_edges.jsonl",
        "weight": 0.06,  # conservative initial weight
    },
    "coverage_gap": {
        "enabled": False,
        "db_path": "data/coverage/ledger.duckdb",
    },
    "diversity": {
        "enabled": True,
        "max_per_source": 3,
    },
    "specter2": {
        "enabled": False,
        "embeddings_path": "data/embeddings/specter2_corpus.jsonl",
        "weight": 0.20,
        "model": "allenai/specter2_base",
    },
    "llm_expansion": {
        "enabled": False,
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 256,
    },
    "hard_negative_filters": {"enabled": True},
}

MODALITY_TERMS: dict[str, list[str]] = {
    # Electrophysiology
    "neuropixels": ["neuropixels", "neuropixel"],
    "calcium_imaging": ["calcium imaging", "two photon", "2p", "gcamp", "optical imaging"],
    "extracellular_ephys": ["extracellular ephys", "ephys", "spikes", "electrophysiology"],
    "eeg": ["eeg", "electroencephalography"],
    "ecog": ["ecog", "electrocorticography"],
    "ieeg": ["ieeg", "intracranial eeg", "depth electrode", "seeg"],
    "meg": ["meg", "magnetoencephalography"],
    "fmri": ["fmri", "functional mri", "bold"],
    "fiber_photometry": ["fiber photometry", "photometry"],
    "patch_clamp": ["patch clamp", "whole cell recording", "intracellular"],
    # Behavioral
    "behavior_video": ["behavior video", "video"],
    "pose_tracking": ["pose tracking", "kinematics", "deeplabcut", "dlc"],
    # Transcriptomics
    "single_cell_rnaseq": ["single cell rna", "scrna-seq", "scrnaseq", "single cell rna-seq",
                          "single cell transcriptomics", "10x genomics"],
    "single_nucleus_rnaseq": ["single nucleus rna", "snrna-seq", "snrnaseq",
                              "single nuclei rna", "nuclear rna-seq"],
    "bulk_rnaseq": ["bulk rna-seq", "rnaseq", "rna sequencing", "transcriptome"],
    "spatial_transcriptomics": ["spatial transcriptomics", "spatial rna", "spatial gene expression"],
    "merfish": ["merfish"],
    "visium": ["visium"],
    # Epigenomics
    "single_cell_atacseq": ["single cell atac", "scatac-seq", "scatacseq", "chromatin accessibility"],
    "single_nucleus_atacseq": ["single nucleus atac", "snatac-seq", "snatacsec"],
    "multiome": ["multiome", "multi-omic", "multiomic", "rna atac"],
    "chip_seq": ["chip-seq", "chipseq", "histone modification"],
    "methylation": ["methylation", "bisulfite", "methylome"],
    # Multi-modal genomics
    "patch_seq": ["patch-seq", "patchseq", "ephys transcriptomics"],
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
    # Genomic expansions
    "transcriptomics": ["single_cell_rnaseq", "single_nucleus_rnaseq", "bulk_rnaseq",
                        "spatial_transcriptomics"],
    "single cell": ["single_cell_rnaseq", "single_cell_atacseq"],
    "single nucleus": ["single_nucleus_rnaseq", "single_nucleus_atacseq"],
    "cell type atlas": ["single_cell_rnaseq", "single_nucleus_rnaseq", "spatial_transcriptomics"],
    "epigenomics": ["single_cell_atacseq", "single_nucleus_atacseq", "chip_seq", "methylation"],
    "chromatin": ["single_cell_atacseq", "single_nucleus_atacseq", "chip_seq"],
}


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _retrieval_config_with_defaults(
    retrieval_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if retrieval_config is None:
        return load_retrieval_config()
    return _deep_merge(DEFAULT_RETRIEVAL_CONFIG, retrieval_config)


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
        _get_value(dataset, "recording_scales", []),
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

    config = _retrieval_config_with_defaults(retrieval_config)
    task_matches = match_tasks(query)
    behavior_matches = match_behavior_labels(query)
    brain_region_matches = match_brain_regions(query)
    recording_scale_matches = match_recording_scales(query)
    affordance_matches = match_affordances(query)
    normalized = normalize_text(query)

    # Parse exclusions first
    excluded_modalities, excluded_species = _parse_exclusions(query)
    negative_constraints = parse_hard_negative_constraints(query, config)
    only_species_exclusions = species_exclusions_for_only_query(query)
    if only_species_exclusions:
        negative_constraints["hard_excluded_species"] = sorted(
            {
                *negative_constraints["hard_excluded_species"],
                *only_species_exclusions,
            }
        )
    excluded_modalities = sorted(
        set(excluded_modalities) | set(negative_constraints["hard_excluded_modalities"])
    )
    excluded_species = sorted(
        set(excluded_species) | set(negative_constraints["hard_excluded_species"])
    )

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

    # Add affordance matches to analysis intents
    for match in affordance_matches:
        if match.id not in analysis_ids:
            analysis_intents.append(
                {
                    "id": match.id,
                    "label": match.label,
                    "confidence": match.confidence,
                    "evidence": match.evidence,
                    "match_type": "affordance",
                }
            )
            analysis_ids.append(match.id)

    query_analysis_terms = {
        normalize_text(value)
        for value in [
            *analysis_ids,
            *expanded.get("suggested_analyses", []),
            *expanded.get("affordance_ids", []),
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
    existing_species_ids = {match["id"] for match in species_intents}
    for match in species_query_matches(query):
        if match["id"] not in existing_species_ids:
            species_intents.append(match)
            existing_species_ids.add(match["id"])

    return {
        "query": query,
        "normalized_query": normalized,
        "tasks": [match.id for match in task_matches],
        "behaviors": [match.id for match in behavior_matches],
        "modalities": sorted(set(modalities)),
        "excluded_modalities": excluded_modalities,
        "excluded_species": excluded_species,
        "excluded_tasks": negative_constraints["hard_excluded_tasks"],
        "excluded_sources": negative_constraints["hard_excluded_sources"],
        "excluded_regions": negative_constraints["hard_excluded_regions"],
        "excluded_dataset_types": negative_constraints["hard_excluded_dataset_types"],
        "excluded_analysis_affordances": negative_constraints[
            "hard_excluded_analysis_affordances"
        ],
        "excluded_recording_devices": negative_constraints[
            "hard_excluded_recording_devices"
        ],
        "negative_constraints": negative_constraints,
        "species": sorted({match["id"] for match in species_intents}),
        "species_constraints": {
            "matches": species_intents,
            "only_query_exclusions": only_species_exclusions,
        },
        "brain_regions": [match.id for match in brain_region_matches],
        "recording_scales": [match.id for match in recording_scale_matches],
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
        "recording_scale_intent": _match_dumps(recording_scale_matches),
        "affordance_intent": _match_dumps(affordance_matches),
        "analysis_intent": analysis_intents,
        "inferred_concepts": sorted(
            query_analysis_terms
            | {normalize_text(match.category or "") for match in task_matches}
        ),
        "expanded": expanded,
        "affordances": [match.id for match in affordance_matches],
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
    card: DatasetCardRead | Mapping[str, Any] | None,
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
        if value in (None, "", [], {}, False) and card is not None:
            label_group = "behaviors" if field == "behaviors" else field
            if _card_labels(card, label_group):
                value = "card_label"
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

    config = _retrieval_config_with_defaults(retrieval_config)
    weights = config.get("weights", {})
    penalties = config.get("penalties", {})
    dataset_id = _get_value(dataset, "id", _get_value(dataset, "source_id", "unknown"))
    dataset_source = _get_value(dataset, "source", "")
    text = _text_for_dataset(dataset, card)
    query_terms = set(parsed_query.get("expanded", {}).get("terms", []))
    task_terms = {normalize_text(value) for value in parsed_query.get("tasks", [])}
    behavior_terms = {normalize_text(value) for value in parsed_query.get("behaviors", [])}
    modality_terms = {normalize_text(value) for value in parsed_query.get("modalities", [])}
    recording_scale_terms = {
        normalize_text(value) for value in parsed_query.get("recording_scales", [])
    }
    species_terms = {
        normalize_text(value).replace(" ", "_")
        for value in parsed_query.get("species", [])
    }
    region_terms = {normalize_text(value) for value in parsed_query.get("brain_regions", [])}
    analysis_terms = {normalize_text(value) for value in parsed_query.get("analysis", [])}
    affordance_terms = {normalize_text(value) for value in parsed_query.get("affordances", [])}

    dataset_tasks = _normalize_values(_get_value(dataset, "tasks", [])) | (
        _card_labels(card, "tasks") if card else set()
    )
    dataset_behaviors = _normalize_values(_get_value(dataset, "behaviors", [])) | (
        _card_labels(card, "behaviors") if card else set()
    )
    dataset_modalities = _normalize_values(_get_value(dataset, "modalities", [])) | (
        _card_labels(card, "modalities") if card else set()
    )
    dataset_recording_scales = _normalize_values(
        _get_value(dataset, "recording_scales", [])
    ) | (_card_labels(card, "recording_scales") if card else set())
    raw_dataset_species = [
        *_raw_values(_get_value(dataset, "species", [])),
        *list(_card_labels(card, "species") if card else set()),
    ]
    dataset_species = (
        _normalize_values(_get_value(dataset, "species", []))
        | (_card_labels(card, "species") if card else set())
        | species_terms_for_values(raw_dataset_species)
    )
    dataset_regions = _normalize_values(_get_value(dataset, "brain_regions", [])) | (
        _card_labels(card, "brain_regions") if card else set()
    )
    dataset_analyses = _normalize_values(_get_value(card, "suggested_analyses", [])) if card else set()
    # Get affordances from dataset and card
    dataset_affordances = _normalize_values(_get_value(dataset, "analysis_affordances", []))
    if card:
        card_affordances = _get_value(card, "suggested_analyses", [])
        dataset_affordances |= _normalize_values(card_affordances)
        # Also check scientific labels for affordances
        labels = _get_value(card, "scientific_labels", {}) or {}
        if isinstance(labels, Mapping):
            dataset_affordances |= _normalize_values(labels.get("analysis_affordances", []))

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
    recording_scale_score, matched_recording_scales = _field_score(
        recording_scale_terms,
        dataset_recording_scales,
    )
    species_score, matched_species = _field_score(species_terms, dataset_species)
    region_score, matched_regions = _field_score(region_terms, dataset_regions)
    analysis_score, matched_analyses = _field_score(analysis_terms, dataset_analyses)
    affordance_score, matched_affordances = _field_score(affordance_terms, dataset_affordances)

    ontology_match = task_score if task_terms else 0.0
    behavior_match = behavior_score if behavior_terms else 0.0
    metadata_scores = []
    if species_terms:
        metadata_scores.append(species_score)
    if region_terms:
        metadata_scores.append(region_score)
    if analysis_terms:
        metadata_scores.append(analysis_score)
    if recording_scale_terms:
        metadata_scores.append(recording_scale_score)
    metadata_match = sum(metadata_scores) / len(metadata_scores) if metadata_scores else 0.0

    why: list[str] = []
    warnings: list[str] = []
    missing_metadata_warnings: list[str] = []
    matched_labels = [
        *matched_tasks,
        *matched_behaviors,
        *matched_modalities,
        *matched_recording_scales,
        *matched_species,
        *matched_regions,
        *matched_analyses,
        *matched_affordances,
    ]

    why.extend(f"Task matched: {value}" for value in matched_tasks)
    why.extend(f"Behavior matched: {value}" for value in matched_behaviors)
    why.extend(f"Modality matched: {value}" for value in matched_modalities)
    why.extend(f"Recording scale matched: {value}" for value in matched_recording_scales)
    why.extend(f"Species matched: {value}" for value in matched_species)
    why.extend(f"Brain region matched: {value}" for value in matched_regions)
    why.extend(f"Analysis matched: {value}" for value in matched_analyses)
    why.extend(f"Affordance matched: {value}" for value in matched_affordances)

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

    # Preserve direct scoring behavior: explicit hard negatives are still visible
    # when this lower-level scorer is called outside the main filtered search path.
    exclusion_penalty = 0.0
    excluded_modalities = {normalize_text(m) for m in parsed_query.get("excluded_modalities", [])}
    violated_exclusions = set(
        negative_constraint_violations(
            dataset,
            card if isinstance(card, Mapping) else None,
            parsed_query.get("negative_constraints", {}),
        )
    )
    if excluded_modalities:
        violated_exclusions |= excluded_modalities & dataset_modalities
    if violated_exclusions:
        exclusion_penalty = float(penalties.get("exclusion_violation", 0.5))
        warnings.append(
            f"Exclusion violation: dataset contains {', '.join(sorted(violated_exclusions))} "
            "which was explicitly excluded from the query."
        )

    readiness_score = 0.0
    if card:
        readiness = _get_value(card, "analysis_readiness", None)
        readiness_score = _get_value(readiness, "score", 0) / 100
    else:
        warnings.append("No dataset card supplied; readiness contributes zero.")

    missing_fields = sorted(
        set(_get_value(card, "missing_fields", []) if card else [])
        | set(_missing_required_fields(dataset, card, parsed_query, config))
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

    # Affordance match contributes when query specifies affordances
    affordance_match = affordance_score if affordance_terms else 0.0

    # Coverage rarity boost — upweights results covering underrepresented dimensions
    coverage_gap_boost = 0.0
    gap_cfg = config.get("coverage_gap", {})
    if gap_cfg.get("enabled") and (region_terms or modality_terms or species_terms):
        try:
            from neural_search.coverage.gap_boost import get_global_booster
            booster = get_global_booster()
            coverage_gap_boost = booster.score(
                region_ids=dataset_regions & region_terms if region_terms else None,
                modality_ids=dataset_modalities & modality_terms if modality_terms else None,
                species_ids=dataset_species & species_terms if species_terms else None,
            )
        except Exception:
            pass

    final_score = (
        float(weights.get("ontology", 0.0)) * ontology_match
        + float(weights.get("behavior", 0.0)) * behavior_match
        + float(weights.get("modality", 0.0)) * modality_score
        + float(weights.get("affordance", 0.0)) * affordance_match
        + float(weights.get("metadata", 0.0)) * metadata_match
        + float(weights.get("semantic", 0.0)) * semantic_similarity
        + float(weights.get("readiness", 0.0)) * readiness_score
        + float(weights.get("paper_confidence", 0.0)) * paper_score
        + coverage_gap_boost
        - modality_penalty
        - missing_penalty
        - exclusion_penalty
    )
    final_score = max(0.0, min(final_score, 1.0))
    ontology_score = (
        task_score
        + behavior_score
        + modality_score
        + recording_scale_score
        + species_score
        + region_score
    ) / 6
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
        "affordance_score": round(affordance_match, 3),
        "recording_scale_score": round(
            recording_scale_score if recording_scale_terms else 0.0,
            3,
        ),
        "negative_constraint_score": round(negative_constraint_score, 3),
        "coverage_gap_boost": round(coverage_gap_boost, 4),
        "final_score": round(final_score, 3),
        # Backward-compatible aliases for v0.2 UI/tests.
        "ontology": round(ontology_match, 3),
        "behavior": round(behavior_match, 3),
        "modality": round(modality_score, 3),
        "affordance": round(affordance_match, 3),
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
    if matched_affordances:
        why.append(f"Supports analysis: {', '.join(matched_affordances[:3])}")
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
            "matched_affordances": matched_affordances[:5],
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
        source=dataset_source,
        score=round(final_score * 100, 2),
        why_matched=why,
        warnings=warnings,
        matched_terms=sorted(set(matched_labels) | set(keyword_hits[:8])),
        inferred_concepts=list(parsed_query.get("inferred_concepts", [])),
        evidence_snippets=_evidence_snippets(evidence_text, [*matched_labels, *keyword_hits]),
        missing_metadata_warnings=missing_metadata_warnings,
        missing_metadata=missing_fields,
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


def _augment_result_with_optional_scores(
    result: SearchResult,
    query: str,
    parsed_query: Mapping[str, Any],
    retrieval_config: Mapping[str, Any],
    *,
    graph: Any,
    graph_config: Mapping[str, Any],
    field_index: Any,
    field_config: Mapping[str, Any],
    dataset: Any = None,
    awareness_config: Mapping[str, Any] | None = None,
    query_awareness: Any = None,
    memory_graph_store: Any = None,
    memory_graph_config: Mapping[str, Any] | None = None,
) -> None:
    weights = retrieval_config.get("weights", {})
    base_final = float(result.score_breakdown.get("final_score", result.score / 100.0))
    extra_score = 0.0

    if field_config.get("enabled"):
        field_score = field_semantic_score_for_result(
            field_index,
            str(result.dataset_id),
            query,
            dict(field_config),
        )
        result.score_breakdown["field_semantic_score"] = round(field_score.score, 3)
        if field_score.matched_fields:
            result.why_matched.append(
                "Field semantic matches: " + ", ".join(field_score.matched_fields)
            )
        extra_score += float(weights.get("field_semantic", 0.0)) * field_score.score

    if graph_config.get("enabled"):
        graph_score = graph_context_score(
            graph,
            str(result.dataset_id),
            dict(parsed_query),
            weights=dict(graph_config.get("weights", {})),
            use_edge_confidence=bool(graph_config.get("use_edge_confidence", False)),
        )
        result.score_breakdown["graph_score"] = round(graph_score, 3)
        if graph_score > 0:
            graph_features = compute_graph_features_for_result(
                graph,
                str(result.dataset_id),
                dict(parsed_query),
            )
            requirement_matches = graph_features.get("requirement_matches", {})
            requirement_labels = [
                str(item.get("requirement"))
                for values in requirement_matches.values()
                for item in values
                if item.get("requirement")
            ]
            result.why_matched.append(f"Graph context score: {graph_score:.2f}")
            if requirement_labels:
                result.why_matched.append(
                    "Graph requirements matched: "
                    + ", ".join(sorted(dict.fromkeys(requirement_labels))[:5])
                )
            species_context = graph_features.get("species_context", {})
            taxon_matches = graph_features.get("matched_query_context", {}).get(
                "taxon_groups",
                [],
            )
            if taxon_matches:
                result.why_matched.append(
                    "Graph species context matched: "
                    + ", ".join(sorted(dict.fromkeys(taxon_matches))[:5])
                )
            result.dataset_card_preview["graph_context"] = {
                "linked_papers": graph_features.get("linked_papers", [])[:5],
                "analysis_affordances": graph_features.get("analysis_affordances", [])[:5],
                "matched_query_context": graph_features.get("matched_query_context", {}),
                "requirement_matches": requirement_matches,
                "species": graph_features.get("species", [])[:5],
                "species_context": species_context,
            }
            result.graph_context = result.dataset_card_preview["graph_context"]
            result.linked_papers = graph_features.get("linked_papers", [])[:5]
        extra_score += float(weights.get("graph", 0.0)) * graph_score

    # Memory-graph scoring
    if memory_graph_config and memory_graph_config.get("enabled") and memory_graph_store is not None:
        try:
            mg_evidence = compute_memory_graph_evidence(
                memory_graph_store, result, parsed_query
            )
            result.memory_graph_evidence = mg_evidence
            mg_score = compute_memory_graph_score(memory_graph_store, result, parsed_query)
            result.score_breakdown["memory_graph_score"] = round(mg_score, 3)
            extra_score += float(memory_graph_config.get("weight", 0.06)) * mg_score
            # Build human-readable why_matched entry from evidence
            _mg_parts: list[str] = []
            if mg_evidence.get("modality_matches"):
                _mg_parts.append("modality: " + ", ".join(mg_evidence["modality_matches"][:2]))
            if mg_evidence.get("recording_scale_matches"):
                _mg_parts.append(
                    "scale: " + ", ".join(mg_evidence["recording_scale_matches"][:2])
                )
            if mg_evidence.get("species_matches"):
                _mg_parts.append("species: " + ", ".join(mg_evidence["species_matches"][:2]))
            if mg_evidence.get("region_matches"):
                _mg_parts.append("region: " + ", ".join(mg_evidence["region_matches"][:2]))
            if mg_evidence.get("affordance_matches"):
                _mg_parts.append("supports: " + ", ".join(mg_evidence["affordance_matches"][:2]))
            if mg_evidence.get("has_raw_signal"):
                _mg_parts.append("raw signal confirmed")
            if mg_evidence.get("contraindicated"):
                _mg_parts.append("⚠ contraindicated: " + ", ".join(mg_evidence["contraindicated"][:1]))
            if _mg_parts:
                result.why_matched.append("Graph: " + " · ".join(_mg_parts))
        except Exception:
            # Memory graph scoring is optional; never fail retrieval
            result.score_breakdown["memory_graph_score"] = 0.0

    # Awareness scoring
    if awareness_config and awareness_config.get("enabled") and dataset is not None:
        try:
            _, score_dataset_awareness = _get_awareness_modules()
            if query_awareness is not None:
                awareness_result = score_dataset_awareness(dataset, query_awareness)
                awareness_score = awareness_result.score
                result.score_breakdown["awareness_score"] = round(awareness_score, 3)
                result.dataset_card_preview["data_form_awareness"] = awareness_result.model_dump()

                if awareness_result.matched_data_forms:
                    result.why_matched.append(
                        "Data forms matched: " + ", ".join(awareness_result.matched_data_forms)
                    )
                if awareness_result.matched_analysis_families:
                    result.why_matched.append(
                        "Analysis families matched: "
                        + ", ".join(awareness_result.matched_analysis_families)
                    )
                if awareness_result.cross_modal_opportunities:
                    result.why_matched.append(
                        "Cross-modal opportunities: "
                        + ", ".join(awareness_result.cross_modal_opportunities[:4])
                    )
                for warning in awareness_result.warnings:
                    result.warnings.append(f"Awareness: {warning}")

                awareness_weight = float(awareness_config.get("weight", 0.0))
                extra_score += awareness_weight * awareness_score
        except Exception:
            # Awareness scoring is optional; don't fail retrieval if it errors
            result.score_breakdown["awareness_score"] = 0.0

    if extra_score:
        final_score = max(0.0, min(base_final + extra_score, 1.0))
        result.score_breakdown["final_score"] = round(final_score, 3)
        result.score = round(final_score * 100, 2)

    # Usefulness scoring — attach to result.usefulness_score
    try:
        query_ctx = parsed_query.get("_query_usefulness_ctx")
        usefulness_intent = parsed_query.get("_usefulness_intent")
        if query_ctx is not None:
            # Use raw corpus record (has tasks/modalities/species) not card_preview (summary only)
            raw_record = dataset if isinstance(dataset, Mapping) else {}
            cand_ctx = dataset_context_from_record(raw_record)
            if not cand_ctx.dataset_id:
                cand_ctx = DatasetContext(dataset_id=str(result.dataset_id))
            u_score = score_usefulness(query_ctx, cand_ctx, usefulness_intent, graph=graph)
            result.usefulness_score = {
                "total_score": round(u_score.total_score, 4),
                "intent": u_score.intent.value,
                "dimension_scores": {k: round(v, 4) for k, v in u_score.dimension_scores.items()},
                "warnings": u_score.warnings,
            }
    except Exception:
        pass  # Usefulness scoring is always optional — never fail search


def _generate_rich_explanation(
    result: SearchResult,
    query: str,
    dataset: Any,
    parsed_query: Mapping[str, Any],
) -> None:
    """Generate rich explanation for a search result.

    Updates the result's explanation field with a more detailed explanation
    generated from the match context.
    """
    try:
        # Build match groups from the result
        match_groups: list[MatchGroup] = []

        # Tasks
        task_terms = [normalize_text(t) for t in parsed_query.get("tasks", [])]
        matched_tasks = [t for t in result.matched_terms if t in task_terms]
        if task_terms or matched_tasks:
            match_groups.append(MatchGroup(
                category="task",
                query_terms=task_terms,
                matched_terms=matched_tasks,
                match_quality="full" if matched_tasks else "none",
            ))

        # Modalities
        modality_terms = [normalize_text(m) for m in parsed_query.get("modalities", [])]
        matched_modalities = [m for m in result.matched_terms if m in modality_terms]
        if modality_terms or matched_modalities:
            match_groups.append(MatchGroup(
                category="modality",
                query_terms=modality_terms,
                matched_terms=matched_modalities,
                match_quality="full" if matched_modalities else "none",
            ))

        # Recording scales
        recording_scale_terms = [
            normalize_text(s) for s in parsed_query.get("recording_scales", [])
        ]
        matched_recording_scales = [
            s for s in result.matched_terms if s in recording_scale_terms
        ]
        if recording_scale_terms or matched_recording_scales:
            match_groups.append(MatchGroup(
                category="recording_scale",
                query_terms=recording_scale_terms,
                matched_terms=matched_recording_scales,
                match_quality="full" if matched_recording_scales else "none",
            ))

        # Species
        species_terms = [normalize_text(s) for s in parsed_query.get("species", [])]
        matched_species = [s for s in result.matched_terms if s in species_terms]
        if species_terms or matched_species:
            match_groups.append(MatchGroup(
                category="species",
                query_terms=species_terms,
                matched_terms=matched_species,
                match_quality="full" if matched_species else "none",
            ))

        # Brain regions
        region_terms = [normalize_text(r) for r in parsed_query.get("brain_regions", [])]
        matched_regions = [r for r in result.matched_terms if r in region_terms]
        if region_terms or matched_regions:
            match_groups.append(MatchGroup(
                category="brain_region",
                query_terms=region_terms,
                matched_terms=matched_regions,
                match_quality="full" if matched_regions else "none",
            ))

        # Build explanation context
        context = ExplanationContext(
            query_text=query,
            dataset_title=_get_value(dataset, "title", str(result.dataset_id)),
            dataset_id=str(result.dataset_id),
            match_groups=match_groups,
            score=result.score,
            score_breakdown=dict(result.score_breakdown),
            warnings=list(result.warnings),
            missing_metadata=list(result.missing_metadata),
            graph_context=result.graph_context,
            linked_papers=list(result.linked_papers),
        )

        # Generate explanation
        explanation_result = generate_explanation(context)

        # Update result with detailed explanation
        result.explanation = explanation_result.detailed
        result.dataset_card_preview["explanation"] = {
            "brief": explanation_result.brief,
            "detailed": explanation_result.detailed,
            "technical": explanation_result.technical,
            "quality_grade": explanation_result.quality_grade,
            "match_summary": explanation_result.match_summary,
        }
    except Exception:
        # Explanation generation is optional; don't fail search if it errors
        pass


def search_datasets(
    query: str,
    filters: Mapping[str, Any] | None = None,
    structured_query: Mapping[str, Any] | None = None,
    datasets: Sequence[Mapping[str, Any]] | None = None,
    limit: int = 10,
    retrieval_config: Mapping[str, Any] | None = None,
) -> SearchResponse:
    """Search supplied datasets or the built-in demo seed."""

    config = _retrieval_config_with_defaults(retrieval_config)
    combined_query = combine_query_and_structured_text(query, structured_query)
    parsed = parse_query(combined_query, config)

    # LLM query expansion fallback — fires only when rule-based parsing finds nothing
    llm_cfg = config.get("llm_expansion", {})
    if (
        llm_cfg.get("enabled")
        and not parsed.get("brain_regions")
        and not parsed.get("tasks")
    ):
        try:
            from neural_search.search.llm_expansion import expand_query_with_llm
            llm_terms = expand_query_with_llm(
                combined_query,
                model=str(llm_cfg.get("model", "claude-haiku-4-5-20251001")),
                max_tokens=int(llm_cfg.get("max_tokens", 256)),
            )
            if llm_terms:
                for dim in ("brain_regions", "tasks", "modalities", "species"):
                    if llm_terms.get(dim):
                        existing = list(parsed.get(dim, []))
                        merged = list(dict.fromkeys(existing + llm_terms[dim]))
                        parsed[dim] = merged
                parsed["llm_expansion_applied"] = True
                parsed["llm_expanded_terms"] = llm_terms
        except Exception:
            pass

    # Classify query intent and adjust weights
    intent = classify_query_intent(combined_query, parsed)
    if intent.weight_overrides:
        base_weights = config.get("weights", {})
        blended_weights = blend_weights(
            base_weights,
            intent.weight_overrides,
            intent.confidence,
            confidence_threshold=0.70,
        )
        config = _deep_merge(config, {"weights": blended_weights})
    parsed["query_intent"] = {
        "primary": intent.primary_intent.value,
        "confidence": round(intent.confidence, 3),
        "secondary": [i.value for i in intent.secondary_intents],
    }

    # Build query DatasetContext for usefulness scoring (once per search call)
    _query_usefulness_ctx = DatasetContext(
        dataset_id="__query__",
        modalities=list(parsed.get("modalities", [])),
        tasks=list(parsed.get("tasks", [])),
        species=list(parsed.get("species", [])),
        brain_regions=list(parsed.get("brain_regions", [])),
        affordances=list(parsed.get("affordances", [])),
    )
    _usefulness_intent_cls = classify_usefulness_intent(combined_query)
    parsed["_query_usefulness_ctx"] = _query_usefulness_ctx
    parsed["_usefulness_intent"] = _usefulness_intent_cls.intent

    # Search intelligence planner integration
    planner_config = (
        config.get("planner", {})
        if isinstance(config.get("planner", {}), Mapping)
        else {}
    )
    if planner_config.get("enabled"):
        try:
            plan_search_intelligence = _get_planner()
            plan = plan_search_intelligence(combined_query)
            parsed["search_plan"] = plan.model_dump()

            # Blend planner weights with current weights
            blend_factor = float(planner_config.get("blend_factor", 0.5))
            if blend_factor > 0 and plan.retrieval_weights:
                base_weights = dict(config.get("weights", {}))
                planner_weights = dict(plan.retrieval_weights)
                blended = {}
                all_keys = set(base_weights.keys()) | set(planner_weights.keys())
                for key in all_keys:
                    base_val = base_weights.get(key, 0.0)
                    plan_val = planner_weights.get(key, 0.0)
                    blended[key] = (1.0 - blend_factor) * base_val + blend_factor * plan_val
                config = _deep_merge(config, {"weights": blended})

            # Add planner warnings to parsed query
            if plan.warnings:
                parsed["planner_warnings"] = list(plan.warnings)
        except Exception:
            # Planner is optional; don't fail retrieval if it errors
            pass

    filters = merge_filters(filters, structured_query)
    records = list(datasets) if datasets is not None else build_combined_corpus()
    results: list[SearchResult] = []
    filtered_constraints: list[dict[str, Any]] = []
    hard_filters_enabled = bool(config.get("hard_negative_filters", {}).get("enabled", True))
    graph_config = config.get("graph", {}) if isinstance(config.get("graph", {}), Mapping) else {}
    graph = (
        load_graph_if_exists(graph_config.get("path"))
        if graph_config.get("enabled")
        else None
    )

    # Expand query with graph relationships
    if graph is not None:
        max_hops = int(graph_config.get("max_transitive_hops", 2))
        parsed = expand_query_with_graph(graph, parsed, max_hops=max_hops)

    field_config = (
        config.get("field_embeddings", {})
        if isinstance(config.get("field_embeddings", {}), Mapping)
        else {}
    )
    field_index = (
        load_field_semantic_index(field_config.get("path"))
        if field_config.get("enabled")
        else None
    )

    # Awareness scoring setup
    awareness_config = (
        config.get("awareness", {})
        if isinstance(config.get("awareness", {}), Mapping)
        else {}
    )
    query_awareness = None
    if awareness_config.get("enabled"):
        try:
            infer_query_awareness, _ = _get_awareness_modules()
            query_awareness = infer_query_awareness(combined_query)
            parsed["query_awareness"] = query_awareness.model_dump()
        except Exception:
            # Awareness is optional; don't fail retrieval if it errors
            pass

    # Memory graph scoring setup
    memory_graph_config = (
        config.get("memory_graph", {})
        if isinstance(config.get("memory_graph", {}), Mapping)
        else {}
    )
    memory_graph_store = (
        load_memory_graph_store(
            str(memory_graph_config.get("nodes_path", "")),
            str(memory_graph_config.get("edges_path", "")),
        )
        if memory_graph_config.get("enabled")
        else None
    )

    for record in records:
        dataset = record.get("dataset", record)
        card = record.get("card")
        if card is None and "extraction" in record:
            card = generate_dataset_card_json(
                dataset, record["extraction"], record.get("papers", [])
            )
        if not _passes_filters(dataset, filters, card):
            continue
        violations = negative_constraint_violations(
            dataset,
            card if isinstance(card, Mapping) else None,
            parsed.get("negative_constraints", {}),
        )
        if hard_filters_enabled and violations:
            filtered_constraints.append(
                {
                    "dataset_id": _get_value(dataset, "id", _get_value(dataset, "source_id", "unknown")),
                    "violations": violations,
                }
            )
            continue
        result = score_dataset_against_query(dataset, card, parsed, config)
        _augment_result_with_optional_scores(
            result,
            combined_query,
            parsed,
            config,
            graph=graph,
            graph_config=graph_config,
            field_index=field_index,
            field_config=field_config,
            dataset=dataset,
            awareness_config=awareness_config,
            query_awareness=query_awareness,
            memory_graph_store=memory_graph_store,
            memory_graph_config=memory_graph_config,
        )
        # Generate rich explanation after all scores are computed
        _generate_rich_explanation(result, combined_query, dataset, parsed)
        results.append(result)

    results.sort(key=lambda item: item.score, reverse=True)

    # SPECTER2 hybrid fusion — additive cosine signal from scientific paper embeddings
    specter2_cfg = config.get("specter2", {})
    if specter2_cfg.get("enabled") and results:
        try:
            from neural_search.search.specter2_fusion import augment_with_specter2
            augment_with_specter2(results, combined_query, specter2_cfg)
            results.sort(key=lambda item: item.score, reverse=True)
        except Exception:
            pass

    # Source diversity — prevent a single large corpus from monopolising the top-K
    diversity_cfg = config.get("diversity", {})
    if diversity_cfg.get("enabled"):
        from neural_search.search.diversity import apply_source_diversity
        results = apply_source_diversity(
            results,
            max_per_source=int(diversity_cfg.get("max_per_source", 3)),
            limit=limit,
        )
    else:
        results = results[:limit]

    # Strip private/internal keys (DatasetContext objects, enum instances) before serialising
    parsed_response = {k: v for k, v in parsed.items() if not k.startswith("_")}
    if filtered_constraints:
        parsed_response["filtered_negative_constraints"] = filtered_constraints
    return SearchResponse(
        query=combined_query,
        parsed_query=parsed_response,
        results=results,
        filtered_constraints=filtered_constraints,
    )


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
    records = list(datasets) if datasets is not None else build_combined_corpus()
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
