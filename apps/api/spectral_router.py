"""Read-only API endpoints for aperiodic spectral phenotype reanalysis.

Mirrors the self-contained-router convention used by ``claims_router.py`` /
``graph_router.py`` — this module loads the demo/combined corpus directly
from ``neural_search.ingestion.demo_seed`` instead of importing from
``apps.api.main``, avoiding a circular import.

These endpoints only report metadata-derived eligibility and any
already-computed spectral features for a dataset; they never run a spectral
analysis on demand (that is the job of
``scripts/reanalysis/run_aperiodic_one.py``).
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException

from neural_search.normalized import make_dataset_id, make_evidence_label_id
from neural_search.schemas import EvidenceLabel, NormalizedDatasetRecord, UsabilityFlags
from neural_search.spectral.eligibility import detect_aperiodic_eligibility
from neural_search.spectral.features import summarize_for_card

router = APIRouter()

_DEMO_MODE = os.getenv("NEURAL_SEARCH_DEMO_MODE", "").lower() in ("1", "true", "yes")

_demo_data: list[dict[str, Any]] | None = None


def _ensure_demo_data() -> list[dict[str, Any]]:
    global _demo_data
    if _demo_data is None:
        from neural_search.ingestion.demo_seed import (
            build_combined_corpus,
            build_demo_seed,
        )

        _demo_data = build_demo_seed() if _DEMO_MODE else build_combined_corpus()
    return _demo_data


def _find_dataset_record(dataset_id: str) -> dict[str, Any] | None:
    for record in _ensure_demo_data():
        ds = record["dataset"]
        if ds.get("id") == dataset_id or ds.get("source_id") == dataset_id:
            return record
    return None


def _evidence_labels(label_type: str, values: list[Any]) -> list[EvidenceLabel]:
    labels: list[EvidenceLabel] = []
    for value in values or []:
        value_str = str(value)
        if not value_str:
            continue
        labels.append(
            EvidenceLabel(
                id=make_evidence_label_id(label_type, value_str),
                label=value_str.replace("_", " "),
                label_type=label_type,
                confidence=0.9,
                source_field=label_type,
                source_value=value_str,
                extractor_name="apps.api.spectral_router",
            )
        )
    return labels


def _record_from_dataset_dict(ds: dict[str, Any], extraction: Any = None) -> NormalizedDatasetRecord:
    """Adapt the demo corpus's flat dataset dict into the
    ``NormalizedDatasetRecord`` shape that ``detect_aperiodic_eligibility``
    expects. The demo corpus does not carry full provenance-rich labels, so
    this is a best-effort, conservative adapter."""

    source = str(ds.get("source") or "unknown")
    source_id = str(ds.get("source_id") or ds.get("id") or "unknown")
    missing_fields = list(getattr(extraction, "missing_fields", None) or ds.get("missing_fields") or [])

    return NormalizedDatasetRecord(
        dataset_id=str(ds.get("id") or make_dataset_id(source, source_id)),
        source=source,
        source_id=source_id,
        title=str(ds.get("title") or source_id),
        description=ds.get("description"),
        url=ds.get("url"),
        modalities=_evidence_labels("modality", ds.get("modalities", [])),
        data_standards=_evidence_labels("data_standard", ds.get("data_standards", [])),
        usability_flags=UsabilityFlags(
            has_raw_data=ds.get("has_raw_data"),
            has_processed_data=ds.get("has_processed_data"),
            has_neural_data=bool(ds.get("modalities")) or None,
            has_behavior=ds.get("has_behavior"),
            has_trials=ds.get("has_trials"),
        ),
        missing_fields=missing_fields,
    )


@router.get("/api/datasets/{dataset_id}/aperiodic/eligibility")
async def get_aperiodic_eligibility(dataset_id: str) -> dict[str, Any]:
    """Metadata-only eligibility for aperiodic spectral parameterization."""

    record = _find_dataset_record(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    normalized = _record_from_dataset_dict(record["dataset"], record.get("extraction"))
    eligibility = detect_aperiodic_eligibility(normalized)
    return {"dataset_id": dataset_id, "eligibility": eligibility.model_dump(mode="json")}


@router.get("/api/datasets/{dataset_id}/spectral/features")
async def get_spectral_features(dataset_id: str) -> dict[str, Any]:
    """Already-computed spectral feature bundles for a dataset, if any.

    No features are computed on demand here — only previously persisted
    bundles (e.g. produced by ``scripts/reanalysis/run_aperiodic_one.py``
    and attached via ``record.get("spectral_feature_bundle")``) are
    reported.
    """

    record = _find_dataset_record(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    bundle = record.get("spectral_feature_bundle")
    return {
        "dataset_id": dataset_id,
        "has_computed_features": bundle is not None,
        "feature_bundle": bundle.model_dump(mode="json") if bundle is not None else None,
    }


@router.get("/api/datasets/{dataset_id}/spectral/qc")
async def get_spectral_qc(dataset_id: str) -> dict[str, Any]:
    """Eligibility-plus-QC summary suitable for dataset-card display."""

    record = _find_dataset_record(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    normalized = _record_from_dataset_dict(record["dataset"], record.get("extraction"))
    eligibility = detect_aperiodic_eligibility(normalized)
    bundle = record.get("spectral_feature_bundle")
    return {"dataset_id": dataset_id, "summary": summarize_for_card(eligibility, bundle)}
