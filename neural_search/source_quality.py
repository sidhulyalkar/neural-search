"""Source quality profiles kept separate from scientific relevance scoring."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

TrustLevel = Literal["high", "medium", "low", "unknown"]


@dataclass(frozen=True)
class SourceQualityProfile:
    """A deterministic source-level quality profile."""

    source: str
    display_name: str
    trust_level: TrustLevel
    reliability_score: float
    supported_standards: tuple[str, ...] = ()
    access_mode: str = "unknown"
    fixture_backed: bool = False
    notes: str = ""


@dataclass(frozen=True)
class SourceQualityAssessment:
    """Quality assessment for one record's source provenance."""

    record_id: str
    source: str
    source_id: str | None
    trust_level: TrustLevel
    quality_score: float
    matched_standards: tuple[str, ...] = ()
    missing_expected_standards: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    fixture_backed: bool = False


DEFAULT_SOURCE_QUALITY_PROFILES: dict[str, SourceQualityProfile] = {
    "dandi": SourceQualityProfile(
        source="dandi",
        display_name="DANDI Archive",
        trust_level="high",
        reliability_score=0.92,
        supported_standards=("NWB",),
        access_mode="api",
        fixture_backed=True,
        notes="Primary archive for NWB-backed neurophysiology and behavior.",
    ),
    "openneuro": SourceQualityProfile(
        source="openneuro",
        display_name="OpenNeuro",
        trust_level="high",
        reliability_score=0.9,
        supported_standards=("BIDS",),
        access_mode="api",
        fixture_backed=True,
        notes="Primary archive for BIDS human neuroimaging and electrophysiology.",
    ),
    "openalex": SourceQualityProfile(
        source="openalex",
        display_name="OpenAlex",
        trust_level="medium",
        reliability_score=0.78,
        access_mode="api",
        fixture_backed=True,
        notes="Bibliographic source; dataset links require evidence review.",
    ),
    "modeldb": SourceQualityProfile(
        source="modeldb",
        display_name="ModelDB",
        trust_level="medium",
        reliability_score=0.8,
        access_mode="fixture_or_scrape",
        fixture_backed=True,
        notes="Computational model registry with heterogeneous metadata.",
    ),
    "cellxgene": SourceQualityProfile(
        source="cellxgene",
        display_name="cellxgene",
        trust_level="medium",
        reliability_score=0.82,
        supported_standards=("AnnData", "h5ad"),
        access_mode="api",
        fixture_backed=True,
        notes="Single-cell atlas source; neuroscience relevance depends on collection.",
    ),
    "microns": SourceQualityProfile(
        source="microns",
        display_name="MICrONS",
        trust_level="high",
        reliability_score=0.88,
        access_mode="public_portal",
        fixture_backed=True,
        notes="Connectomics source with strong provenance but specialized access.",
    ),
    "allen_brain": SourceQualityProfile(
        source="allen_brain",
        display_name="Allen Brain Atlas",
        trust_level="high",
        reliability_score=0.9,
        supported_standards=("AnnData", "NWB"),
        access_mode="api",
        fixture_backed=False,
        notes="High-value atlas source across cell types, imaging, and connectivity.",
    ),
    "nemo_archive": SourceQualityProfile(
        source="nemo_archive",
        display_name="NeMO Archive",
        trust_level="high",
        reliability_score=0.86,
        supported_standards=("AnnData", "h5ad", "FASTQ", "BAM"),
        access_mode="api",
        fixture_backed=False,
        notes="BRAIN Initiative multi-omic archive.",
    ),
    "demo": SourceQualityProfile(
        source="demo",
        display_name="Demo fixture",
        trust_level="low",
        reliability_score=0.55,
        access_mode="fixture",
        fixture_backed=True,
        notes="Synthetic/local demo source; useful for tests, not scientific claims.",
    ),
    "manual": SourceQualityProfile(
        source="manual",
        display_name="Manual curated source",
        trust_level="medium",
        reliability_score=0.72,
        access_mode="curated",
        fixture_backed=True,
        notes="Manual records require reviewer provenance before promotion.",
    ),
}


def _record_dict(record: Any) -> dict[str, Any]:
    if isinstance(record, BaseModel):
        return record.model_dump(mode="json", exclude_none=True)
    if isinstance(record, Mapping):
        payload = dict(record)
        nested = payload.get("dataset", payload)
        return dict(nested) if isinstance(nested, Mapping) else payload
    raise TypeError(f"record must be a mapping or Pydantic model, got {type(record)!r}")


def _source_key(value: Any) -> str:
    return str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")


def _record_id(payload: Mapping[str, Any]) -> str:
    for key in ("dataset_id", "paper_id", "id", "source_id"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return "unknown"


def _values(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key, ())
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    values: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            values.append(str(item.get("label", item.get("id", ""))))
        else:
            values.append(str(item))
    return tuple(item for item in values if item)


def assess_source_quality(
    record: Any,
    *,
    profiles: Mapping[str, SourceQualityProfile] | None = None,
) -> SourceQualityAssessment:
    """Assess source quality without changing scientific relevance scores."""

    payload = _record_dict(record)
    source = _source_key(payload.get("source"))
    source_id = str(payload.get("source_id") or "").strip() or None
    active_profiles = profiles or DEFAULT_SOURCE_QUALITY_PROFILES
    profile = active_profiles.get(source)
    warnings: list[str] = []
    if profile is None:
        profile = SourceQualityProfile(
            source=source or "unknown",
            display_name=source or "Unknown source",
            trust_level="unknown",
            reliability_score=0.4,
            notes="No source quality profile is registered.",
        )
        warnings.append("source profile is not registered")

    standards = {_source_key(value) for value in _values(payload, "data_standards")}
    expected = {_source_key(value) for value in profile.supported_standards}
    matched = sorted(standards & expected)
    missing = sorted(expected - standards)

    score = profile.reliability_score
    if expected and matched:
        score += 0.05
    if expected and missing:
        score -= 0.08
        warnings.append("record lacks an expected source data standard")
    if not source_id:
        score -= 0.08
        warnings.append("record lacks a source_id")
    if not payload.get("url") and source not in {"demo", "manual"}:
        score -= 0.04
        warnings.append("record lacks a source URL")
    if profile.trust_level == "low":
        warnings.append("source is suitable for fixtures but not promotion")

    return SourceQualityAssessment(
        record_id=_record_id(payload),
        source=profile.source,
        source_id=source_id,
        trust_level=profile.trust_level,
        quality_score=round(max(0.0, min(score, 1.0)), 3),
        matched_standards=tuple(matched),
        missing_expected_standards=tuple(missing),
        warnings=tuple(sorted(dict.fromkeys(warnings))),
        fixture_backed=profile.fixture_backed,
    )


def summarize_source_quality(
    records: Iterable[Any],
    *,
    profiles: Mapping[str, SourceQualityProfile] | None = None,
) -> dict[str, Any]:
    """Summarize source quality for readiness and promotion gates."""

    assessments = [
        assess_source_quality(record, profiles=profiles)
        for record in records
    ]
    if not assessments:
        return {
            "record_count": 0,
            "mean_quality_score": 0.0,
            "trust_level_counts": {},
            "source_counts": {},
            "warning_count": 0,
            "warnings": [],
        }

    trust_counts = Counter(item.trust_level for item in assessments)
    source_counts = Counter(item.source for item in assessments)
    warnings = [
        {
            "record_id": item.record_id,
            "source": item.source,
            "warnings": list(item.warnings),
        }
        for item in assessments
        if item.warnings
    ]
    return {
        "record_count": len(assessments),
        "mean_quality_score": round(
            sum(item.quality_score for item in assessments) / len(assessments),
            3,
        ),
        "trust_level_counts": dict(sorted(trust_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "fixture_backed_count": sum(1 for item in assessments if item.fixture_backed),
        "warning_count": len(warnings),
        "warnings": warnings,
    }
