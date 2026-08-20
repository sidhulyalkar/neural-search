"""Conservative, evidence-aware reanalysis planning.

The planner separates *feasibility* from *novelty*. A method can be a strong fit
for the data while already having precedent on the target dataset, in which case
it becomes a replication/extension candidate rather than a claim of novelty.
All output remains advisory and explicitly requires scientific review.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from neural_search.awareness.taxonomy import DATA_FORMS, detect_data_forms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHOD_REGISTRY_PATH = PROJECT_ROOT / "data" / "methods" / "method_registry.yaml"
METHOD_TAXONOMY_PATH = PROJECT_ROOT / "data" / "methods" / "methods_taxonomy.yaml"


@dataclass(frozen=True)
class ReanalysisEvidence:
    kind: str
    source_id: str
    summary: str
    confidence: float | None = None
    evidence_tier: str = "heuristic_candidate"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReanalysisCandidate:
    method_id: str
    method_label: str
    analysis_family: str
    data_form: str
    priority_score: float
    feasibility_score: float
    feasibility_status: str
    novelty_status: str
    rationale: str
    required_signals: list[str]
    present_required_signals: list[str]
    missing_required_signals: list[str]
    assumptions: list[str]
    limitations: list[str]
    computes: list[str]
    precedent_datasets: list[dict[str, Any]]
    evidence: list[ReanalysisEvidence]
    requires_human_review: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReanalysisPlan:
    dataset_id: str
    dataset_title: str
    generated_at: str
    matched_data_forms: list[str]
    candidates: list[ReanalysisCandidate]
    uncovered_analysis_families: list[str]
    warnings: list[str]
    evidence_policy: str = (
        "Candidate methods are hypotheses for scientific review. A missing paper/method "
        "link is not proof that an analysis has never been performed."
    )
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@lru_cache(maxsize=1)
def _registry_links() -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(METHOD_REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    return {
        str(link["analysis_family"]): dict(link)
        for link in list(raw.get("links") or [])
        if link.get("analysis_family")
    }


@lru_cache(maxsize=1)
def _method_taxonomy() -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(METHOD_TAXONOMY_PATH.read_text(encoding="utf-8")) or {}
    methods: dict[str, dict[str, Any]] = {}
    for category in list(raw.get("categories") or []):
        for method in list(category.get("methods") or []):
            method_id = str(method.get("id") or "").strip()
            if method_id:
                methods[method_id] = dict(method)
    return methods


def _string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        candidate = value.get("label") or value.get("id") or value.get("name")
        return [str(candidate)] if candidate else []
    if isinstance(value, list | tuple | set):
        result: list[str] = []
        for item in value:
            result.extend(_string_values(item))
        return result
    return [str(value)]


def _dataset_id(record: Mapping[str, Any]) -> str:
    dataset = record.get("dataset") if isinstance(record.get("dataset"), Mapping) else record
    source = str(dataset.get("source") or record.get("source") or "unknown")
    source_id = str(
        dataset.get("source_id")
        or dataset.get("id")
        or record.get("source_id")
        or record.get("dataset_id")
        or "unknown"
    )
    if ":" in source_id:
        return source_id
    return f"{source}:{source_id}"


def _dataset_title(record: Mapping[str, Any]) -> str:
    dataset = record.get("dataset") if isinstance(record.get("dataset"), Mapping) else record
    return str(dataset.get("title") or record.get("title") or _dataset_id(record))


def _record_text(record: Mapping[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, default=str).replace("_", " ").lower()


def _detected_data_forms(record: Mapping[str, Any]) -> list[str]:
    dataset = record.get("dataset") if isinstance(record.get("dataset"), Mapping) else record
    terms: list[str] = []
    for field_name in (
        "modalities",
        "tasks",
        "species",
        "brain_regions",
        "data_standards",
        "standards",
        "description",
        "title",
    ):
        terms.extend(_string_values(dataset.get(field_name)))
        if dataset is not record:
            terms.extend(_string_values(record.get(field_name)))
    return detect_data_forms(terms)


def _signal_presence(record_text: str, required_signals: tuple[str, ...]) -> tuple[list[str], list[str]]:
    present: list[str] = []
    missing: list[str] = []
    aliases = {
        "events": ("event", "events", "trial", "trials", "timestamps"),
        "trials": ("trial", "trials", "event", "events"),
        "channels": ("channel", "channels", "electrode", "electrodes"),
        "sampling_rate": ("sampling rate", "sampling frequency", "sample rate", "fs"),
        "participants": ("participant", "participants", "subject", "subjects", "patient"),
        "sessions": ("session", "sessions", "visit", "visits"),
        "images": ("image", "images", "nifti", "bold", "fmri", "mri"),
        "units": ("unit", "units", "spike", "spikes"),
        "spike_times": ("spike time", "spike times", "spikes", "units"),
        "fluorescence": ("fluorescence", "dff", "df/f", "calcium", "gcamp"),
        "roi_masks": ("roi", "rois", "mask", "masks"),
        "electrodes": ("electrode", "electrodes", "channel", "channels"),
    }
    for signal in required_signals:
        candidates = aliases.get(signal, (signal.replace("_", " "),))
        if any(candidate.lower() in record_text for candidate in candidates):
            present.append(signal)
        else:
            missing.append(signal)
    return present, missing


def _flatten_assumptions(method: Mapping[str, Any]) -> list[str]:
    assumptions = method.get("assumptions") or {}
    if isinstance(assumptions, Mapping):
        return [f"{key}: {value}" for key, value in assumptions.items()][:8]
    return _string_values(assumptions)[:8]


def _flatten_limitations(method: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    values.extend(_string_values(method.get("limitations")))
    values.extend(_string_values(method.get("pitfalls")))
    values.extend(_string_values(method.get("critical_limitation")))
    values.extend(_string_values(method.get("critical_misunderstanding")))
    return values[:10]


def _candidate_key(method_id: str, analysis_family: str) -> tuple[str, str]:
    return method_id, analysis_family


def build_reanalysis_plan(
    record: Mapping[str, Any],
    *,
    existing_method_evidence: Mapping[str, Mapping[str, Any]] | None = None,
    precedent_by_method: Mapping[str, list[dict[str, Any]]] | None = None,
    literature_by_method: Mapping[str, list[dict[str, Any]]] | None = None,
    limit: int = 12,
) -> ReanalysisPlan:
    """Build a ranked analysis plan from metadata, method requirements, and evidence."""

    existing_method_evidence = existing_method_evidence or {}
    precedent_by_method = precedent_by_method or {}
    literature_by_method = literature_by_method or {}
    forms = _detected_data_forms(record)
    record_text = _record_text(record)
    registry = _registry_links()
    methods = _method_taxonomy()
    warnings: list[str] = []
    uncovered: set[str] = set()
    candidates_by_key: dict[tuple[str, str], ReanalysisCandidate] = {}

    if not forms:
        warnings.append(
            "No broad data form could be inferred from available metadata; reanalysis coverage is incomplete."
        )

    for form_id in forms:
        form = DATA_FORMS[form_id]
        present, missing = _signal_presence(record_text, form.required_signals)
        signal_coverage = (
            len(present) / len(form.required_signals) if form.required_signals else 1.0
        )
        for analysis_family in form.analysis_families:
            link = registry.get(analysis_family)
            if not link:
                uncovered.add(analysis_family)
                continue
            registry_confidence = float(link.get("confidence", 0.5))
            for method_id in list(link.get("taxonomy_method_ids") or []):
                method_id = str(method_id)
                method = methods.get(method_id, {})
                method_label = str(method.get("label") or method_id.replace("_", " ").title())
                target_evidence = dict(existing_method_evidence.get(method_id) or {})
                precedents = list(precedent_by_method.get(method_id) or [])[:5]
                literature = list(literature_by_method.get(method_id) or [])[:3]

                feasibility = min(1.0, registry_confidence * (0.55 + 0.45 * signal_coverage))
                if not missing:
                    feasibility_status = "supported_by_metadata"
                elif signal_coverage >= 0.5:
                    feasibility_status = "conditional_missing_signals"
                else:
                    feasibility_status = "blocked_by_missing_signals"

                novelty_status = (
                    "existing_use_evidence"
                    if target_evidence
                    else "possible_new_use_unverified"
                )
                novelty_component = 0.25 if target_evidence else 1.0
                evidence_component = min(
                    1.0,
                    0.35
                    + (0.25 if precedents else 0.0)
                    + (0.2 if literature else 0.0)
                    + (0.2 if target_evidence else 0.0),
                )
                priority = min(
                    1.0,
                    0.55 * feasibility
                    + 0.30 * novelty_component
                    + 0.15 * evidence_component,
                )

                evidence: list[ReanalysisEvidence] = [
                    ReanalysisEvidence(
                        kind="method_registry",
                        source_id=f"analysis_family:{analysis_family}",
                        summary=str(link.get("rationale") or "Method registry match."),
                        confidence=registry_confidence,
                        evidence_tier="heuristic_candidate",
                        metadata={"data_form": form_id},
                    )
                ]
                if target_evidence:
                    evidence.append(
                        ReanalysisEvidence(
                            kind="existing_target_paper_method",
                            source_id=str(target_evidence.get("paper_openalex_id") or "paper"),
                            summary=(
                                f"Existing paper-method evidence already links this dataset to {method_label}; "
                                "treat as replication/extension rather than a novel-use claim."
                            ),
                            confidence=float(target_evidence.get("method_confidence") or 0.5),
                            evidence_tier="paper_linked_method_evidence",
                            metadata=target_evidence,
                        )
                    )
                for precedent in precedents[:3]:
                    evidence.append(
                        ReanalysisEvidence(
                            kind="precedent_dataset",
                            source_id=str(precedent.get("dataset_id") or "dataset"),
                            summary=str(
                                precedent.get("summary")
                                or f"A related dataset has evidence for {method_label}."
                            ),
                            confidence=(
                                float(precedent["confidence"])
                                if precedent.get("confidence") is not None
                                else None
                            ),
                            evidence_tier="evidence_backed_bridge",
                            metadata=precedent,
                        )
                    )
                for finding in literature:
                    evidence.append(
                        ReanalysisEvidence(
                            kind="literature_finding",
                            source_id=str(finding.get("finding_id") or finding.get("paper_id") or "finding"),
                            summary=str(finding.get("finding_text") or "Related literature finding."),
                            confidence=(
                                float(finding["relevance_score"])
                                if finding.get("relevance_score") is not None
                                else None
                            ),
                            evidence_tier="extracted_literature_finding",
                            metadata={
                                "paper_id": finding.get("paper_id"),
                                "paper_title": finding.get("paper_title"),
                            },
                        )
                    )

                candidate = ReanalysisCandidate(
                    method_id=method_id,
                    method_label=method_label,
                    analysis_family=analysis_family,
                    data_form=form_id,
                    priority_score=round(priority, 4),
                    feasibility_score=round(feasibility, 4),
                    feasibility_status=feasibility_status,
                    novelty_status=novelty_status,
                    rationale=(
                        f"{method_label} is linked to the {analysis_family} analysis family for "
                        f"{form.label.lower()} data. Metadata supports {len(present)}/"
                        f"{len(form.required_signals)} expected signal classes."
                    ),
                    required_signals=list(form.required_signals),
                    present_required_signals=present,
                    missing_required_signals=missing,
                    assumptions=_flatten_assumptions(method),
                    limitations=_flatten_limitations(method),
                    computes=_string_values(method.get("computes"))[:8],
                    precedent_datasets=precedents,
                    evidence=evidence,
                )
                key = _candidate_key(method_id, analysis_family)
                previous = candidates_by_key.get(key)
                if previous is None or candidate.priority_score > previous.priority_score:
                    candidates_by_key[key] = candidate

    candidates = sorted(
        candidates_by_key.values(),
        key=lambda item: (
            item.feasibility_status == "blocked_by_missing_signals",
            -item.priority_score,
            item.method_label,
        ),
    )[:limit]

    if uncovered:
        warnings.append(
            "Some detected analysis families have no defensible method-registry mapping yet: "
            + ", ".join(sorted(uncovered))
        )
    if any(candidate.novelty_status == "possible_new_use_unverified" for candidate in candidates):
        warnings.append(
            "'possible_new_use_unverified' means no current paper-method evidence was found; it is not proof of novelty."
        )

    return ReanalysisPlan(
        dataset_id=_dataset_id(record),
        dataset_title=_dataset_title(record),
        generated_at=datetime.now(UTC).isoformat(),
        matched_data_forms=forms,
        candidates=candidates,
        uncovered_analysis_families=sorted(uncovered),
        warnings=warnings,
    )
