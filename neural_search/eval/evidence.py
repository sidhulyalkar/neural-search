"""Core evidence data models for weak-supervision labeling.

QuerySpec       — structured decomposition of a benchmark query
DatasetEvidence — normalized representation of a corpus record
PairEvidence    — a (query, dataset) pair ready for labeling functions
LFVote          — a single labeling function's output
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


_COMPLETENESS_FIELDS = [
    "title", "description", "species", "modalities", "brain_regions",
    "tasks", "license", "url", "has_raw_data",
]


def compute_metadata_completeness(record: dict) -> float:
    """Return 0.0–1.0 fraction of key fields that are populated."""
    present = sum(
        1 for f in _COMPLETENESS_FIELDS
        if record.get(f) not in (None, [], "", {}, False)
    )
    return present / len(_COMPLETENESS_FIELDS)


@dataclass
class QuerySpec:
    query_id: str
    query_text: str
    intent: str
    scientific_goal: str
    required_modalities: list[str] = field(default_factory=list)
    preferred_modalities: list[str] = field(default_factory=list)
    required_species: list[str] = field(default_factory=list)
    preferred_species: list[str] = field(default_factory=list)
    brain_regions: list[str] = field(default_factory=list)
    task_constraints: list[str] = field(default_factory=list)
    data_level_requirements: list[str] = field(default_factory=list)
    hard_negatives: list[str] = field(default_factory=list)
    analysis_affordances: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DatasetEvidence:
    record_id: str
    source: str
    title: str
    description: str | None
    species: list[str] = field(default_factory=list)
    modalities: list[str] = field(default_factory=list)
    data_levels: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    license: str | None = None
    doi: str | None = None
    url: str | None = None
    raw_data_available: bool = False
    metadata_completeness: float = 0.0
    has_behavior: bool = False
    has_trials: bool = False
    data_standards: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PairEvidence:
    query_id: str
    record_id: str
    query: QuerySpec
    dataset: DatasetEvidence
    pooled_from: list[str] = field(default_factory=list)
    min_rank: int = 1000
    priority: str = "normal"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LFVote:
    lf_name: str
    label: int
    confidence: float
    rationale: str
    abstain: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _infer_data_levels(record: dict) -> list[str]:
    levels: list[str] = []
    if record.get("has_raw_data"):
        levels.append("raw")
    if record.get("has_processed_data"):
        levels.append("processed")
    return levels


def _flatten_str_list(values: list) -> list[str]:
    """Normalize a list that may contain strings or dicts with an 'id'/'label' key."""
    result = []
    for item in values:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            canonical = str(item.get("id") or item.get("label") or "")
            if canonical:
                result.append(canonical)
    return result


def dataset_evidence_from_record(record: dict) -> DatasetEvidence:
    """Build a DatasetEvidence from a normalized corpus record dict."""
    source = record.get("source", "")
    source_id = record.get("source_id", "")
    record_id = f"{source}:{source_id}"

    doi: str | None = None
    metadata = record.get("metadata_json") or {}
    if isinstance(metadata, dict):
        doi = metadata.get("doi") or metadata.get("identifier")

    return DatasetEvidence(
        record_id=record_id,
        source=source,
        title=record.get("title") or "",
        description=record.get("description"),
        species=_flatten_str_list(list(record.get("species") or [])),
        modalities=_flatten_str_list(list(record.get("modalities") or [])),
        data_levels=_infer_data_levels(record),
        tasks=_flatten_str_list(list(record.get("tasks") or [])),
        regions=_flatten_str_list(list(record.get("brain_regions") or [])),
        license=record.get("license"),
        doi=doi,
        url=record.get("url"),
        raw_data_available=bool(record.get("has_raw_data")),
        metadata_completeness=compute_metadata_completeness(record),
        has_behavior=bool(record.get("has_behavior")),
        has_trials=bool(record.get("has_trials")),
        data_standards=_flatten_str_list(list(record.get("data_standards") or [])),
    )
