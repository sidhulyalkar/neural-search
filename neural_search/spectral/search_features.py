"""Search-facing glue for the aperiodic spectral phenotype affordance.

Query-side trigger terms live in the ontology
(``data/ontology/behavioral_task_ontology.yaml`` ->
``analysis_affordances`` -> ``aperiodic_spectral_parameterization`` ->
``query_synonyms``) so they flow through the existing
``neural_search.ontology.matcher.match_affordances`` pipeline and the
existing ``affordance`` weight channel in
``neural_search.search.core.score_dataset_against_query`` — no parallel
scoring path is introduced. This module exposes the canonical trigger-term
list (for tests/docs) and a small human-readable explain helper.
"""

from __future__ import annotations

from neural_search.ontology import match_affordances
from neural_search.schemas import NormalizedDatasetRecord
from neural_search.spectral.eligibility import detect_aperiodic_eligibility

APERIODIC_AFFORDANCE_ID = "aperiodic_spectral_parameterization"

# Mirrors the ontology's query_synonyms for this affordance. Kept here too so
# tests and documentation have a single, importable source of truth.
SPECTRAL_TRIGGER_TERMS: tuple[str, ...] = (
    "aperiodic",
    "1/f",
    "one over f",
    "spectral slope",
    "power law",
    "broadband",
    "E/I balance",
    "excitation inhibition",
    "intrinsic timescale",
    "neural timescale",
    "offset",
    "knee",
    "FOOOF",
    "specparam",
    "IRASA",
    "periodic vs aperiodic",
)


def query_matches_spectral_affordance(query: str) -> bool:
    """True if ``query`` triggers the aperiodic spectral affordance match."""

    return any(match.id == APERIODIC_AFFORDANCE_ID for match in match_affordances(query))


def explain_spectral_search_match(query: str, record: NormalizedDatasetRecord) -> dict:
    """Explain whether/why ``record`` should be boosted for ``query`` on the
    aperiodic spectral phenotype affordance."""

    eligibility = detect_aperiodic_eligibility(record)
    matched = next(
        (match for match in match_affordances(query) if match.id == APERIODIC_AFFORDANCE_ID),
        None,
    )
    boost_applicable = matched is not None and eligibility.support_level in ("high", "medium")

    explanation: list[str] = []
    if matched is not None:
        explanation.append(
            f"Query matched aperiodic/1-over-f spectral terms (match confidence {matched.confidence:.2f})."
        )
    if eligibility.support_level in ("high", "medium"):
        explanation.append(
            f"Dataset eligibility for aperiodic spectral parameterization: "
            f"{eligibility.support_level} (confidence {eligibility.confidence:.2f})."
        )
    elif eligibility.support_level == "unsupported":
        explanation.append(
            "Dataset is not eligible for aperiodic spectral parameterization given current metadata."
        )

    return {
        "matched_query_affordance": matched.id if matched is not None else None,
        "match_confidence": matched.confidence if matched is not None else 0.0,
        "eligibility_support_level": eligibility.support_level,
        "eligibility_confidence": eligibility.confidence,
        "boost_applicable": boost_applicable,
        "explanation": explanation,
    }
