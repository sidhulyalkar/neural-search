"""Query-aware retrieval planning for broad neuroscience search."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from neural_search.awareness.taxonomy import (
    DATA_FORMS,
    QueryAwareness,
    infer_query_awareness,
)


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


@dataclass(frozen=True)
class SearchIntelligencePlan:
    """Actionable query plan derived from neuroscience awareness signals."""

    query: str
    mode: str
    intent: str
    confidence: float
    query_awareness: QueryAwareness
    retrieval_weights: dict[str, float]
    required_data_forms: tuple[str, ...] = ()
    excluded_data_forms: tuple[str, ...] = ()
    required_signals: tuple[str, ...] = ()
    complementary_data_forms: tuple[str, ...] = ()
    quality_checks: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    recommended_benchmark_tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "mode": self.mode,
            "intent": self.intent,
            "confidence": self.confidence,
            "query_awareness": self.query_awareness.model_dump(),
            "retrieval_weights": dict(self.retrieval_weights),
            "required_data_forms": list(self.required_data_forms),
            "excluded_data_forms": list(self.excluded_data_forms),
            "required_signals": list(self.required_signals),
            "complementary_data_forms": list(self.complementary_data_forms),
            "quality_checks": list(self.quality_checks),
            "warnings": list(self.warnings),
            "recommended_benchmark_tags": list(self.recommended_benchmark_tags),
            "metadata": dict(self.metadata),
        }


def _detect_intent(query: str, awareness: QueryAwareness) -> tuple[str, float]:
    normalized = _norm(query)
    if re.search(r"\b(dandi|openneuro|dataset|ds\d{3,}|[0-9]{6})\b", normalized):
        return "dataset_lookup", 0.84
    if awareness.excluded_data_forms or any(term in normalized for term in ("not", "without")):
        return "hard_negative", 0.82
    if any(
        term in normalized
        for term in ("similar", "related", "like this", "same lab", "linked paper")
    ):
        return "graph_similarity", 0.78
    if awareness.analysis_families:
        return "analysis_affordance", 0.76
    if len(awareness.requested_data_forms) >= 2:
        return "cross_modal", 0.74
    if awareness.requested_data_forms:
        return "data_form_search", 0.70
    if normalized.startswith(("what ", "which ", "find ", "show ", "explore ")):
        return "exploratory", 0.62
    return "general_neuroscience", 0.55


def _mode_for_intent(intent: str, awareness: QueryAwareness) -> str:
    if intent == "hard_negative":
        return "constraint_filter_first"
    if len(awareness.requested_data_forms) >= 2:
        return "cross_modal_fit"
    if intent == "analysis_affordance":
        return "analysis_readiness"
    if intent == "dataset_lookup":
        return "exact_then_context"
    if intent == "exploratory":
        return "recall_first"
    return "balanced"


def _weights_for_intent(intent: str) -> dict[str, float]:
    profiles: dict[str, dict[str, float]] = {
        "dataset_lookup": {
            "metadata": 0.32,
            "semantic": 0.18,
            "graph": 0.12,
            "awareness": 0.08,
            "negative_constraint": 0.10,
        },
        "hard_negative": {
            "negative_constraint": 0.30,
            "awareness": 0.18,
            "ontology": 0.18,
            "modality": 0.14,
            "semantic": 0.10,
            "metadata": 0.10,
        },
        "analysis_affordance": {
            "affordance": 0.26,
            "awareness": 0.18,
            "readiness": 0.16,
            "ontology": 0.14,
            "field_semantic": 0.12,
            "semantic": 0.10,
        },
        "cross_modal": {
            "awareness": 0.24,
            "ontology": 0.18,
            "field_semantic": 0.14,
            "graph": 0.14,
            "semantic": 0.12,
            "metadata": 0.10,
        },
        "graph_similarity": {
            "graph": 0.26,
            "field_semantic": 0.18,
            "semantic": 0.16,
            "awareness": 0.14,
            "ontology": 0.12,
            "metadata": 0.08,
        },
        "exploratory": {
            "semantic": 0.24,
            "awareness": 0.18,
            "ontology": 0.16,
            "field_semantic": 0.14,
            "metadata": 0.12,
            "readiness": 0.08,
        },
    }
    return profiles.get(
        intent,
        {
            "awareness": 0.18,
            "ontology": 0.18,
            "semantic": 0.16,
            "field_semantic": 0.12,
            "metadata": 0.12,
            "readiness": 0.08,
            "graph": 0.08,
        },
    )


def _complementary_forms(awareness: QueryAwareness) -> tuple[str, ...]:
    forms: list[str] = []
    for form_id in awareness.requested_data_forms:
        forms.extend(DATA_FORMS[form_id].complementary_forms)
    return tuple(sorted(_unique(forms)))


def _quality_checks(awareness: QueryAwareness) -> tuple[str, ...]:
    checks = ["preserve_hard_negative_filtering"]
    if awareness.required_signals:
        checks.append("verify_required_signals")
    if awareness.analysis_families:
        checks.append("verify_analysis_affordances")
    if len(awareness.requested_data_forms) >= 2:
        checks.append("verify_cross_modal_alignment")
    if awareness.species_terms:
        checks.append("verify_species_metadata")
    return tuple(checks)


def _coverage_warnings(
    awareness: QueryAwareness,
    corpus_profile: Mapping[str, Any] | None,
) -> list[str]:
    warnings: list[str] = []
    if not awareness.requested_data_forms:
        warnings.append("No specific neuroscience data form detected; use recall-oriented ranking.")
    if not corpus_profile:
        return warnings

    underrepresented = set(corpus_profile.get("underrepresented_data_forms", []))
    counts = corpus_profile.get("data_form_counts", {})
    for form_id in awareness.requested_data_forms:
        if form_id in underrepresented or int(counts.get(form_id, 0) or 0) == 0:
            warnings.append(f"Corpus coverage is thin for requested data form: {form_id}")
    return warnings


def plan_search_intelligence(
    query: str,
    *,
    corpus_profile: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SearchIntelligencePlan:
    """Create a deterministic query plan for neuroscience-aware retrieval."""

    awareness = infer_query_awareness(query)
    intent, confidence = _detect_intent(query, awareness)
    tags = [intent, *awareness.requested_data_forms, *awareness.analysis_families]
    return SearchIntelligencePlan(
        query=query,
        mode=_mode_for_intent(intent, awareness),
        intent=intent,
        confidence=round(confidence, 3),
        query_awareness=awareness,
        retrieval_weights=_weights_for_intent(intent),
        required_data_forms=awareness.requested_data_forms,
        excluded_data_forms=awareness.excluded_data_forms,
        required_signals=awareness.required_signals,
        complementary_data_forms=_complementary_forms(awareness),
        quality_checks=_quality_checks(awareness),
        warnings=tuple(_coverage_warnings(awareness, corpus_profile)),
        recommended_benchmark_tags=tuple(sorted(_unique(tags))),
        metadata=dict(metadata or {}),
    )


def _load_corpus_profile(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if "data_form_counts" in payload:
        return payload
    return {
        "data_form_counts": payload.get("data_form_counts", {}),
        "underrepresented_data_forms": [
            gap.get("data_form")
            for gap in payload.get("gaps", [])
            if gap.get("priority") in {"critical", "high"}
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a neuroscience-aware retrieval plan for one query."
    )
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--corpus-profile",
        help="Optional JSON coverage profile from search-intelligence-report.",
    )
    parser.add_argument("--out", help="Optional JSON output path.")
    args = parser.parse_args(argv)

    plan = plan_search_intelligence(
        args.query,
        corpus_profile=_load_corpus_profile(args.corpus_profile),
    )
    payload = json.dumps(plan.model_dump(), indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
