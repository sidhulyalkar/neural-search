"""Dataset scoring against broad neuroscience data-form awareness."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from neural_search.awareness.taxonomy import (
    DATA_FORMS,
    QueryAwareness,
    detect_data_forms,
)
from neural_search.schemas import NormalizedDatasetRecord


def _norm_signal(value: str) -> str:
    return value.casefold().replace("_", " ").replace("-", " ")


@dataclass(frozen=True)
class DatasetAwareness:
    """Broad data-form summary inferred for one dataset."""

    dataset_id: str
    data_forms: tuple[str, ...] = ()
    families: tuple[str, ...] = ()
    scales: tuple[str, ...] = ()
    analysis_families: tuple[str, ...] = ()
    species: tuple[str, ...] = ()
    missing_requirements: tuple[str, ...] = ()


@dataclass(frozen=True)
class AwarenessScore:
    """Explanation-rich awareness fit for a dataset/query pair."""

    dataset_id: str
    score: float
    data_form_score: float
    analysis_score: float
    scale_score: float
    species_score: float
    matched_data_forms: tuple[str, ...] = ()
    matched_analysis_families: tuple[str, ...] = ()
    cross_modal_opportunities: tuple[str, ...] = ()
    missing_requirements: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    awareness: DatasetAwareness | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "score": self.score,
            "data_form_score": self.data_form_score,
            "analysis_score": self.analysis_score,
            "scale_score": self.scale_score,
            "species_score": self.species_score,
            "matched_data_forms": list(self.matched_data_forms),
            "matched_analysis_families": list(self.matched_analysis_families),
            "cross_modal_opportunities": list(self.cross_modal_opportunities),
            "missing_requirements": list(self.missing_requirements),
            "warnings": list(self.warnings),
            "awareness": (
                {
                    "dataset_id": self.awareness.dataset_id,
                    "data_forms": list(self.awareness.data_forms),
                    "families": list(self.awareness.families),
                    "scales": list(self.awareness.scales),
                    "analysis_families": list(self.awareness.analysis_families),
                    "species": list(self.awareness.species),
                    "missing_requirements": list(self.awareness.missing_requirements),
                }
                if self.awareness
                else None
            ),
        }


def _label_values(values: Any) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        return [values]
    output: list[str] = []
    for value in values:
        if hasattr(value, "label"):
            output.append(str(value.label))
        elif isinstance(value, Mapping):
            output.append(str(value.get("label") or value.get("id") or value))
        else:
            output.append(str(value))
    return [item for item in output if item]


def _get(dataset: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(dataset, Mapping):
        return dataset.get(field_name, default)
    return getattr(dataset, field_name, default)


def _dataset_id(dataset: Any) -> str:
    return str(
        _get(
            dataset,
            "dataset_id",
            _get(dataset, "id", _get(dataset, "source_id", "unknown")),
        )
    )


def _metadata_text(dataset: Any) -> str:
    if isinstance(dataset, NormalizedDatasetRecord):
        pieces = [
            dataset.title,
            dataset.description or "",
            *_label_values(dataset.modalities),
            *_label_values(dataset.data_standards),
            *_label_values(dataset.behavioral_events),
            *_label_values(dataset.tasks),
            *_label_values(dataset.analysis_goals),
            *_label_values(dataset.file_formats),
        ]
    else:
        pieces = [
            _get(dataset, "title", ""),
            _get(dataset, "description", ""),
            *_label_values(_get(dataset, "modalities", [])),
            *_label_values(_get(dataset, "data_standards", [])),
            *_label_values(_get(dataset, "behaviors", [])),
            *_label_values(_get(dataset, "tasks", [])),
            *_label_values(_get(dataset, "file_formats", [])),
            str(_get(dataset, "metadata_json", "")),
        ]
    return " ".join(str(piece) for piece in pieces if piece)


def infer_dataset_awareness(dataset: Any) -> DatasetAwareness:
    """Infer broad data-form coverage for a normalized or legacy dataset."""

    text = _metadata_text(dataset)
    forms = detect_data_forms(text)
    families = sorted({DATA_FORMS[form].family for form in forms})
    scales = sorted({DATA_FORMS[form].scale for form in forms})
    analyses = sorted(
        {
            analysis
            for form in forms
            for analysis in DATA_FORMS[form].analysis_families
        }
    )
    species = sorted(
        {
            value.casefold().replace(" ", "_")
            for value in _label_values(_get(dataset, "species", []))
        }
    )
    if isinstance(dataset, NormalizedDatasetRecord):
        species = sorted(
            {
                value.casefold().replace(" ", "_")
                for value in _label_values(dataset.species)
            }
        )

    required = {
        signal
        for form in forms
        for signal in DATA_FORMS[form].required_signals
    }
    lower_text = _norm_signal(text)
    missing = sorted(signal for signal in required if _norm_signal(signal) not in lower_text)
    return DatasetAwareness(
        dataset_id=_dataset_id(dataset),
        data_forms=tuple(forms),
        families=tuple(families),
        scales=tuple(scales),
        analysis_families=tuple(analyses),
        species=tuple(species),
        missing_requirements=tuple(missing),
    )


def _overlap_score(requested: tuple[str, ...], actual: tuple[str, ...]) -> tuple[float, tuple[str, ...]]:
    if not requested:
        return (0.5 if actual else 0.0), ()
    matched = tuple(sorted(set(requested) & set(actual)))
    return (len(matched) / len(set(requested)), matched)


def _cross_modal_opportunities(
    query_awareness: QueryAwareness,
    dataset_awareness: DatasetAwareness,
) -> tuple[str, ...]:
    opportunities: list[str] = []
    dataset_forms = set(dataset_awareness.data_forms)
    for requested in query_awareness.requested_data_forms:
        complementary = set(DATA_FORMS[requested].complementary_forms)
        overlaps = sorted(complementary & dataset_forms)
        for overlap in overlaps:
            opportunities.append(f"{requested}+{overlap}")
    return tuple(sorted(dict.fromkeys(opportunities)))


def score_dataset_awareness(
    dataset: Any,
    query_awareness: QueryAwareness,
) -> AwarenessScore:
    """Score how well a dataset fits broad neuroscience query needs."""

    awareness = infer_dataset_awareness(dataset)
    data_form_score, matched_forms = _overlap_score(
        query_awareness.requested_data_forms,
        awareness.data_forms,
    )
    analysis_score, matched_analyses = _overlap_score(
        query_awareness.analysis_families,
        awareness.analysis_families,
    )
    scale_score, _ = _overlap_score(query_awareness.scale_terms, awareness.scales)
    species_score, _ = _overlap_score(query_awareness.species_terms, awareness.species)

    excluded = sorted(set(query_awareness.excluded_data_forms) & set(awareness.data_forms))
    required_missing = tuple(
        sorted(set(query_awareness.required_signals) & set(awareness.missing_requirements))
    )
    warnings: list[str] = []
    if excluded:
        warnings.append("Dataset contains excluded data forms: " + ", ".join(excluded))
    if required_missing:
        warnings.append("Missing query-required signals: " + ", ".join(required_missing))

    raw_score = (
        0.45 * data_form_score
        + 0.25 * analysis_score
        + 0.15 * scale_score
        + 0.15 * species_score
    )
    if excluded:
        raw_score *= 0.2
    if required_missing:
        raw_score *= max(0.4, 1.0 - 0.1 * len(required_missing))

    return AwarenessScore(
        dataset_id=awareness.dataset_id,
        score=round(max(0.0, min(raw_score, 1.0)), 4),
        data_form_score=round(data_form_score, 4),
        analysis_score=round(analysis_score, 4),
        scale_score=round(scale_score, 4),
        species_score=round(species_score, 4),
        matched_data_forms=matched_forms,
        matched_analysis_families=matched_analyses,
        cross_modal_opportunities=_cross_modal_opportunities(query_awareness, awareness),
        missing_requirements=required_missing,
        warnings=tuple(warnings),
        awareness=awareness,
    )
