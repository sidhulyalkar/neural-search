"""Analysis-readiness scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from neural_search.schemas import AnalysisReadiness, ExtractionResult


def _get_value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _has_any(values: Any) -> bool:
    return bool(values)


def _ids(values: Sequence[Any]) -> set[str]:
    ids: set[str] = set()
    for value in values:
        if isinstance(value, Mapping):
            ids.add(str(value.get("id", value.get("label", ""))))
        else:
            ids.add(str(getattr(value, "id", getattr(value, "label", ""))))
    return {value for value in ids if value}


def compute_analysis_readiness(
    dataset: Any, extraction: ExtractionResult | Mapping[str, Any], linked_papers: Sequence[Any] | None
) -> AnalysisReadiness:
    """Compute a transparent 0-100 analysis-readiness score."""

    extraction_obj = (
        extraction
        if isinstance(extraction, ExtractionResult)
        else ExtractionResult.model_validate(extraction)
    )
    score = 0
    strengths: list[str] = []
    limitations: list[str] = []

    standards = _ids(extraction_obj.data_standards) | set(_get_value(dataset, "data_standards", []) or [])
    if {"NWB", "BIDS"} & standards:
        score += 20
        strengths.append("Uses NWB or BIDS metadata standards.")
    else:
        limitations.append("No NWB or BIDS standard detected.")

    if _has_any(extraction_obj.behaviors) or _get_value(dataset, "has_behavior", False):
        score += 15
        strengths.append("Behavioral variables are present or inferable.")
    else:
        limitations.append("Behavioral variables were not detected.")

    metadata = _get_value(dataset, "metadata_json", {}) or {}
    trial_terms = " ".join(str(value) for value in [metadata, _get_value(dataset, "description", "")])
    if _get_value(dataset, "has_trials", False) or any(
        term in trial_terms.casefold() for term in ["trial", "event", "events.tsv"]
    ):
        score += 15
        strengths.append("Trial or event structure is available.")
    else:
        limitations.append("Trial or event structure is unclear.")

    if _has_any(extraction_obj.modalities) or _get_value(dataset, "modalities", []):
        score += 10
        strengths.append("Neural or behavioral modality is identified.")
    else:
        limitations.append("Recording modality is missing.")

    if _has_any(extraction_obj.tasks) or _get_value(dataset, "tasks", []):
        score += 10
        strengths.append("Behavioral task label is identified.")
    else:
        limitations.append("Behavioral task label is missing.")

    if _has_any(extraction_obj.brain_regions) or _get_value(dataset, "brain_regions", []):
        score += 10
        strengths.append("Brain region metadata is present.")
    else:
        limitations.append("Brain region metadata is missing.")

    if linked_papers:
        score += 10
        strengths.append("Linked paper or abstract is available.")
    else:
        limitations.append("No linked paper was provided.")

    if _get_value(dataset, "license", None) or metadata.get("license"):
        score += 5
        strengths.append("License information is present.")
    else:
        limitations.append("License information is missing.")

    if _get_value(dataset, "has_processed_data", False) or metadata.get("has_processed_data"):
        score += 5
        strengths.append("Processed data are indicated.")
    else:
        limitations.append("Processed data availability is unclear.")

    return AnalysisReadiness(score=min(score, 100), strengths=strengths, limitations=limitations)

