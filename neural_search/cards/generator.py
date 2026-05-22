"""Dataset-card generation with provenance-first summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from neural_search.ontology import get_task_by_id
from neural_search.readiness import compute_analysis_readiness
from neural_search.schemas import AnalysisReadiness, DatasetCardRead, ExtractionResult


def _get_value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _label_list(values: Sequence[Any]) -> list[str]:
    labels: list[str] = []
    for value in values:
        if isinstance(value, Mapping):
            labels.append(str(value.get("label", value.get("id", ""))))
        else:
            labels.append(str(getattr(value, "label", getattr(value, "id", ""))))
    return [label for label in labels if label]


def _ids(values: Sequence[Any]) -> list[str]:
    ids: list[str] = []
    for value in values:
        if isinstance(value, Mapping):
            ids.append(str(value.get("id", "")))
        else:
            ids.append(str(getattr(value, "id", "")))
    return [value for value in ids if value]


def _summary(dataset: Any, extraction: ExtractionResult) -> str:
    title = _get_value(dataset, "title", "Untitled dataset")
    labels = _label_list(extraction.tasks)[:2]
    modalities = _label_list(extraction.modalities)[:2]
    parts = [str(title)]
    if labels:
        parts.append("matched to " + ", ".join(labels))
    if modalities:
        parts.append("with " + ", ".join(modalities))
    return "; ".join(parts) + "."


def _why_matched(extraction: ExtractionResult) -> list[str]:
    reasons: list[str] = []
    for label in [*extraction.tasks, *extraction.behaviors, *extraction.modalities]:
        reasons.append(
            f"{label.label} matched from evidence '{label.evidence}' "
            f"(confidence {label.confidence:.2f})."
        )
    return reasons


def _suggested_analyses(extraction: ExtractionResult) -> list[str]:
    suggestions: set[str] = set()
    for task_id in _ids(extraction.tasks):
        task = get_task_by_id(task_id)
        if task:
            suggestions.update(task.suggested_analyses)
    return sorted(suggestions)


def generate_dataset_card_json(
    dataset: Any,
    extraction: ExtractionResult | Mapping[str, Any],
    linked_papers: Sequence[Any] | None = None,
) -> DatasetCardRead:
    """Generate a dataset-card JSON object without inventing unsupported claims."""

    extraction_obj = (
        extraction
        if isinstance(extraction, ExtractionResult)
        else ExtractionResult.model_validate(extraction)
    )
    readiness = compute_analysis_readiness(dataset, extraction_obj, linked_papers or [])
    dataset_id = _get_value(dataset, "id", _get_value(dataset, "source_id", "unknown"))
    card = DatasetCardRead(
        dataset_id=dataset_id,
        summary=_summary(dataset, extraction_obj),
        why_relevant=_why_matched(extraction_obj),
        scientific_labels={
            "tasks": [item.model_dump() for item in extraction_obj.tasks],
            "behaviors": [item.model_dump() for item in extraction_obj.behaviors],
            "modalities": [item.model_dump() for item in extraction_obj.modalities],
            "brain_regions": [item.model_dump() for item in extraction_obj.brain_regions],
            "species": [item.model_dump() for item in extraction_obj.species],
            "data_standards": [item.model_dump() for item in extraction_obj.data_standards],
        },
        analysis_readiness=readiness,
        missing_fields=extraction_obj.missing_fields,
        suggested_analyses=_suggested_analyses(extraction_obj),
        provenance={
            "dataset_source": _get_value(dataset, "source", None),
            "dataset_source_id": _get_value(dataset, "source_id", None),
            "linked_paper_count": len(linked_papers or []),
            "claim_policy": "Only deterministic labels with evidence are included.",
        },
    )
    card.card_markdown = generate_dataset_card_markdown(card)
    return card


def _section(title: str, values: Sequence[str]) -> str:
    if not values:
        return f"## {title}\n\nNone detected.\n"
    lines = "\n".join(f"- {value}" for value in values)
    return f"## {title}\n\n{lines}\n"


def generate_dataset_card_markdown(card: DatasetCardRead | Mapping[str, Any]) -> str:
    """Render a dataset-card object as Markdown."""

    card_obj = card if isinstance(card, DatasetCardRead) else DatasetCardRead.model_validate(card)
    readiness: AnalysisReadiness = card_obj.analysis_readiness
    label_groups = card_obj.scientific_labels
    scientific_labels: list[str] = []
    for group_name in ["tasks", "behaviors", "modalities", "brain_regions", "species", "data_standards"]:
        labels = []
        for item in label_groups.get(group_name, []):
            label_id = item.get("id", "")
            label = item.get("label", label_id)
            labels.append(f"{label} ({label_id})" if label_id else label)
        if labels:
            scientific_labels.append(f"{group_name}: {', '.join(labels)}")

    parts = [
        f"# Dataset Card: {card_obj.dataset_id}",
        "",
        "## Summary",
        "",
        card_obj.summary,
        "",
        _section("Why Matched", card_obj.why_relevant),
        _section("Scientific Labels", scientific_labels),
        "## Analysis Readiness",
        "",
        f"Score: {readiness.score}/100",
        "",
        _section("Strengths", readiness.strengths),
        _section("Limitations", readiness.limitations),
        _section("Suggested Analyses", card_obj.suggested_analyses),
        _section("Missing Metadata", card_obj.missing_fields),
        "## Provenance",
        "",
    ]
    parts.extend(f"- {key}: {value}" for key, value in card_obj.provenance.items())
    return "\n".join(parts).strip() + "\n"
