"""Dataset-card generation with provenance-first summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from neural_search.notebooks.templates import available_templates_for_dataset
from neural_search.ontology import get_task_by_id
from neural_search.readiness import compute_analysis_readiness
from neural_search.recipes import match_recipes_for_tasks
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


def _dataset_list(dataset: Any, name: str) -> list[str]:
    values = _get_value(dataset, name, []) or []
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values if value is not None]


def _metadata(dataset: Any) -> Mapping[str, Any]:
    value = _get_value(dataset, "metadata_json", {}) or {}
    return value if isinstance(value, Mapping) else {}


def _label_ids_or_dataset(
    extraction_values: Sequence[Any],
    dataset: Any,
    dataset_field: str,
) -> list[str]:
    ids = _ids(extraction_values)
    return ids or _dataset_list(dataset, dataset_field)


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


def _first_analysis(suggested: Sequence[str]) -> str | None:
    if not suggested:
        return None
    preferred = [
        "event_aligned_activity",
        "post_reversal_adaptation",
        "perseveration_analysis",
        "choice_decoding",
        "reaction_time_analysis",
        "trial_outcome_decoding",
        "hit_false_alarm_comparison",
        "endpoint_error_analysis",
    ]
    for item in preferred:
        if item in suggested:
            return item
    return suggested[0]


def _advanced_analysis(suggested: Sequence[str]) -> str | None:
    if not suggested:
        return None
    preferred = [
        "latent_state_modeling",
        "q_learning_modeling",
        "neural_trajectory_analysis",
        "cursor_velocity_decoding",
        "stimulus_choice_separation",
    ]
    for item in preferred:
        if item in suggested:
            return item
    return suggested[-1] if len(suggested) > 1 else None


def _paper_record(paper: Any, dataset: Any) -> dict[str, Any]:
    title = str(_get_value(paper, "title", "Untitled linked paper"))
    abstract = str(_get_value(paper, "abstract", "") or "")
    concepts = _dataset_list(paper, "concepts")
    dataset_labels = {
        *(_dataset_list(dataset, "tasks")),
        *(_dataset_list(dataset, "behaviors")),
        *(_dataset_list(dataset, "modalities")),
        *(_dataset_list(dataset, "brain_regions")),
    }
    evidence_text = f"{title} {abstract} {' '.join(concepts)}".casefold()
    overlap = [
        label
        for label in dataset_labels
        if label.casefold() in evidence_text
        or label.replace("_", " ").casefold() in evidence_text
    ]
    confidence = min(0.65 + 0.08 * len(overlap), 0.95) if paper else 0.0
    authors = _get_value(paper, "authors_json", []) or []
    author_names = [
        str(author.get("name", ""))
        for author in authors
        if isinstance(author, Mapping) and author.get("name")
    ]
    return {
        "id": _get_value(paper, "id", title),
        "title": title,
        "authors": author_names,
        "year": _get_value(paper, "publication_year", None),
        "doi": _get_value(paper, "doi", None),
        "openalex_id": _get_value(paper, "openalex_id", None),
        "url": _get_value(paper, "url", None),
        "concepts": concepts,
        "confidence": round(confidence, 2),
        "link_evidence": overlap[:5],
    }


def _asset_records(assets: Sequence[Any] | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for asset in assets or []:
        records.append(
            {
                "id": _get_value(asset, "id", _get_value(asset, "path", "asset")),
                "path": _get_value(asset, "path", ""),
                "asset_type": _get_value(asset, "asset_type", None),
                "file_format": _get_value(asset, "file_format", None),
                "size_bytes": _get_value(asset, "size_bytes", None),
                "modality": _get_value(asset, "modality", None),
            }
        )
    return records


def _trial_event_structure(dataset: Any, extraction: ExtractionResult) -> list[str]:
    metadata = _metadata(dataset)
    values = []
    for key in ["trial_columns", "events", "event_columns", "columns"]:
        item = metadata.get(key)
        if isinstance(item, list):
            values.extend(str(value) for value in item)
    values.extend(_ids(extraction.behaviors))
    if _get_value(dataset, "has_trials", False):
        values.append("trials")
    return sorted({value for value in values if value})


def generate_dataset_card_json(
    dataset: Any,
    extraction: ExtractionResult | Mapping[str, Any],
    linked_papers: Sequence[Any] | None = None,
    assets: Sequence[Any] | None = None,
) -> DatasetCardRead:
    """Generate a dataset-card JSON object without inventing unsupported claims."""

    extraction_obj = (
        extraction
        if isinstance(extraction, ExtractionResult)
        else ExtractionResult.model_validate(extraction)
    )
    readiness = compute_analysis_readiness(dataset, extraction_obj, linked_papers or [])
    dataset_id = _get_value(dataset, "id", _get_value(dataset, "source_id", "unknown"))
    suggested_analyses = _suggested_analyses(extraction_obj)
    tasks = _label_ids_or_dataset(extraction_obj.tasks, dataset, "tasks")
    behaviors = _label_ids_or_dataset(extraction_obj.behaviors, dataset, "behaviors")
    modalities = _label_ids_or_dataset(extraction_obj.modalities, dataset, "modalities")
    brain_regions = _label_ids_or_dataset(extraction_obj.brain_regions, dataset, "brain_regions")
    species = _label_ids_or_dataset(extraction_obj.species, dataset, "species")
    data_standards = _label_ids_or_dataset(extraction_obj.data_standards, dataset, "data_standards")
    data_standard = data_standards[0] if data_standards else None
    paper_records = [_paper_record(paper, dataset) for paper in linked_papers or []]
    asset_records = _asset_records(assets)
    metadata = _metadata(dataset)
    trial_events = _trial_event_structure(dataset, extraction_obj)
    reward_outcomes = [
        value
        for value in behaviors + trial_events
        if value in {"reward", "omission", "correct", "error", "feedback", "hit", "miss"}
    ]
    stimuli = [
        value
        for value in trial_events
        if "stimulus" in value or "cue" in value or "target" in value or "offer" in value
    ]
    first_analysis = _first_analysis(suggested_analyses)
    advanced_analysis = _advanced_analysis(suggested_analyses)
    matched_recipes = match_recipes_for_tasks(tasks)
    notebook_templates = available_templates_for_dataset(dataset)
    starter_recipes = [
        {
            "id": recipe["id"],
            "title": recipe.get("title", recipe["id"]),
            "summary": recipe.get("summary", ""),
            "level": recipe.get("level", "starter"),
            "analyses": recipe.get("analyses", []),
            "required_fields": recipe.get("required_fields", []),
            "matched_tasks": recipe.get("matched_tasks", []),
            "match_score": recipe.get("match_score", 0),
        }
        for recipe in matched_recipes
    ]
    scientific_use_case = (
        f"Use for {tasks[0].replace('_', ' ')} analyses involving "
        f"{', '.join(behaviors[:3]).replace('_', ' ')}."
        if tasks and behaviors
        else "Use for exploratory neural and behavioral dataset reuse."
    )
    why_matters = (
        "It combines task labels, behavioral events, neural metadata, and provenance "
        "into an analysis-ready starting point."
    )
    reuse_steps = [
        "Open the source dataset link and inspect license/access constraints.",
        "Generate the starter notebook from this card.",
        f"Begin with {first_analysis.replace('_', ' ') if first_analysis else 'basic event and metadata inspection'}.",
    ]
    limitations = readiness.limitations
    if extraction_obj.missing_fields:
        limitations = [*limitations, "Missing metadata should be reviewed before publication-grade reuse."]

    card = DatasetCardRead(
        dataset_id=dataset_id,
        title=_get_value(dataset, "title", str(dataset_id)),
        source=_get_value(dataset, "source", None),
        source_id=_get_value(dataset, "source_id", None),
        url=_get_value(dataset, "url", None),
        doi=_get_value(dataset, "doi", None),
        license=_get_value(dataset, "license", None),
        data_standard=data_standard,
        species=species,
        modalities=modalities,
        brain_regions=brain_regions,
        tasks=tasks,
        behaviors=behaviors,
        assets=asset_records,
        related_papers=paper_records,
        summary=_summary(dataset, extraction_obj),
        summary_details={
            "one_sentence": _summary(dataset, extraction_obj),
            "scientific_use_case": scientific_use_case,
            "why_this_dataset_matters": why_matters,
        },
        experimental_structure={
            "task_labels": tasks,
            "behavior_labels": behaviors,
            "trial_event_structure": trial_events,
            "stimuli": stimuli,
            "rewards_outcomes": sorted(set(reward_outcomes)),
            "species": species,
            "subjects": metadata.get("subjects", metadata.get("subject_count")),
            "sessions": metadata.get("sessions", metadata.get("session_count")),
        },
        neural_data={
            "modalities": modalities,
            "brain_regions": brain_regions,
            "file_formats": sorted(
                {
                    str(asset.get("file_format")).upper()
                    for asset in asset_records
                    if asset.get("file_format")
                }
                | {standard.upper() for standard in data_standards}
            ),
            "available_assets": asset_records,
            "has_raw_data": bool(_get_value(dataset, "has_raw_data", False)),
            "has_processed_data": bool(_get_value(dataset, "has_processed_data", False)),
        },
        analysis_plan={
            "readiness_score": readiness.score,
            "strengths": readiness.strengths,
            "limitations": limitations,
            "missing_metadata": extraction_obj.missing_fields,
            "suggested_first_analysis": first_analysis,
            "suggested_advanced_analysis": advanced_analysis,
            "starter_recipes": starter_recipes,
        },
        linked_literature={
            "candidate_papers": paper_records,
            "link_confidence_summary": (
                "Candidate links are based on fixture/source metadata and overlapping extracted concepts."
                if paper_records
                else "No linked literature was supplied."
            ),
        },
        reuse_instructions={
            "source_link": _get_value(dataset, "url", None),
            "how_to_load": (
                "Use PyNWB for NWB assets."
                if data_standard == "NWB"
                else "Use the dataset source tooling and inspect tabular/event files."
            ),
            "notebook_generation_status": "available",
            "notebook_templates": notebook_templates,
            "known_caveats": limitations,
            "recommended_first_steps": reuse_steps,
        },
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
        readiness={
            "score": readiness.score,
            "strengths": readiness.strengths,
            "limitations": limitations,
            "missing_metadata": extraction_obj.missing_fields,
            "suggested_analyses": suggested_analyses,
        },
        missing_fields=extraction_obj.missing_fields,
        missing_metadata=extraction_obj.missing_fields,
        suggested_analyses=suggested_analyses,
        provenance={
            "dataset_source": _get_value(dataset, "source", None),
            "dataset_source_id": _get_value(dataset, "source_id", None),
            "source_metadata": {
                "license": _get_value(dataset, "license", None),
                "data_standards": data_standards,
                "has_behavior": _get_value(dataset, "has_behavior", False),
                "has_trials": _get_value(dataset, "has_trials", False),
            },
            "extraction_method": "deterministic ontology and metadata extraction",
            "confidence_scores": {
                group: [
                    {
                        "id": item.id,
                        "confidence": item.confidence,
                        "evidence": item.evidence,
                    }
                    for item in values
                ]
                for group, values in {
                    "tasks": extraction_obj.tasks,
                    "behaviors": extraction_obj.behaviors,
                    "modalities": extraction_obj.modalities,
                    "brain_regions": extraction_obj.brain_regions,
                    "species": extraction_obj.species,
                    "data_standards": extraction_obj.data_standards,
                }.items()
            },
            "review_status": "machine_generated_needs_scientific_review",
            "linked_paper_count": len(linked_papers or []),
            "claim_policy": "Only deterministic labels with evidence are included.",
        },
        generated_at=datetime.now(timezone.utc),
    )
    card.card_markdown = generate_dataset_card_markdown(card)
    card.markdown = card.card_markdown
    return card


def _section(title: str, values: Sequence[str]) -> str:
    if not values:
        return f"## {title}\n\nNone detected.\n"
    lines = "\n".join(f"- {value}" for value in values)
    return f"## {title}\n\n{lines}\n"


def _fmt(value: Any) -> str:
    if value in (None, "", [], {}):
        return "Not available"
    if isinstance(value, list):
        return ", ".join(str(item).replace("_", " ") for item in value) if value else "Not available"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value).replace("_", " ")


def _kv_section(title: str, values: Mapping[str, Any]) -> list[str]:
    lines = [f"## {title}", ""]
    if not values:
        return [*lines, "Not available.", ""]
    for key, value in values.items():
        if isinstance(value, list) and value and isinstance(value[0], Mapping):
            lines.append(f"- **{key.replace('_', ' ').title()}**:")
            for item in value:
                label = item.get("title") or item.get("path") or item.get("id") or str(item)
                extra = []
                if item.get("doi"):
                    extra.append(f"DOI: {item['doi']}")
                if item.get("openalex_id"):
                    extra.append(f"OpenAlex: {item['openalex_id']}")
                if item.get("confidence") is not None:
                    extra.append(f"confidence {item['confidence']}")
                lines.append(f"  - {label}" + (f" ({'; '.join(extra)})" if extra else ""))
        else:
            lines.append(f"- **{key.replace('_', ' ').title()}**: {_fmt(value)}")
    lines.append("")
    return lines


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
        f"# Scientific Reuse Card: {card_obj.title or card_obj.dataset_id}",
        "",
        f"Dataset ID: `{card_obj.dataset_id}`",
        "",
        "## Summary",
        "",
        card_obj.summary_details.get("one_sentence", card_obj.summary),
        "",
        f"- **Scientific use case**: {_fmt(card_obj.summary_details.get('scientific_use_case'))}",
        f"- **Why this dataset matters**: {_fmt(card_obj.summary_details.get('why_this_dataset_matters'))}",
        "",
        *_kv_section("Experimental Structure", card_obj.experimental_structure),
        *_kv_section("Neural Data", card_obj.neural_data),
        "## Analysis Readiness",
        "",
        f"Score: {readiness.score}/100",
        "",
        _section("Strengths", card_obj.analysis_plan.get("strengths", readiness.strengths)),
        _section("Limitations", card_obj.analysis_plan.get("limitations", readiness.limitations)),
        f"- **Suggested first analysis**: {_fmt(card_obj.analysis_plan.get('suggested_first_analysis'))}",
        f"- **Suggested advanced analysis**: {_fmt(card_obj.analysis_plan.get('suggested_advanced_analysis'))}",
        "",
        _section("Missing Metadata", card_obj.missing_fields),
        *_kv_section("Linked Literature", card_obj.linked_literature),
        *_kv_section("Reuse Instructions", card_obj.reuse_instructions),
        _section("Why Matched", card_obj.why_relevant),
        _section("Scientific Labels", scientific_labels),
        "## Provenance",
        "",
    ]
    for key, value in card_obj.provenance.items():
        if key == "confidence_scores" and isinstance(value, Mapping):
            parts.append("- **Confidence scores**:")
            for group, scores in value.items():
                formatted = ", ".join(
                    f"{item.get('id')}={item.get('confidence')}"
                    for item in scores
                    if isinstance(item, Mapping)
                )
                parts.append(f"  - {group}: {formatted or 'none'}")
        elif key == "source_metadata" and isinstance(value, Mapping):
            parts.append("- **Source metadata**:")
            for source_key, source_value in value.items():
                parts.append(f"  - {source_key}: {_fmt(source_value)}")
        else:
            parts.append(f"- **{key.replace('_', ' ').title()}**: {_fmt(value)}")
    return "\n".join(str(part) for part in parts).strip() + "\n"
