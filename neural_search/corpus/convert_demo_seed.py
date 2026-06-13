"""Convert demo seed YAML into canonical normalized corpus JSONL files."""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from neural_search.analysis_affordances import detect_analysis_affordances
from neural_search.normalized import (
    NormalizedRecord,
    make_dataset_id,
    make_evidence_label_id,
    make_paper_id,
    write_jsonl,
)
from neural_search.schemas import (
    EvidenceLabel,
    NormalizedDatasetRecord,
    NormalizedPaperRecord,
    UsabilityFlags,
)
from neural_search.scientific_labels import enrich_record_with_scientific_labels

DEFAULT_CREATED_AT = "2026-05-24T00:00:00+00:00"
DEFAULT_EXTRACTOR_VERSION = "v0.6.0"

FIELD_LABEL_TYPES = {
    "species": "species",
    "modalities": "modality",
    "brain_regions": "brain_region",
    "tasks": "task",
    "behaviors": "behavioral_event",
    "data_standards": "data_standard",
}

BEHAVIOR_ONLY_MODALITIES = {
    "behavior_video",
    "behavior_tracking",
    "pose_tracking",
    "deeplabcut",
    "sleap",
    "facemap",
    "pupil_tracking",
    "whisker_tracking",
    "running_wheel",
    "position_tracking",
}

NEURAL_MODALITIES = {
    "calcium_imaging",
    "two_photon",
    "extracellular_ephys",
    "electrophysiology",
    "neuropixels",
    "spikes",
    "lfp",
    "eeg",
    "ecog",
    "ieeg",
    "seeg",
    "meg",
    "fmri",
    "functional_mri",
    "bold",
    "fiber_photometry",
    "tetrode",
    "utah_array",
}

ANALYSIS_HINT_RE = re.compile(
    r"supports analyses including (?P<items>[^.]+)",
    flags=re.IGNORECASE,
)


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping at {path}")
    return payload


def _clean_values(values: Any) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        return [values.strip()] if values.strip() else []
    return [str(value).strip() for value in values if str(value).strip()]


def _label(
    *,
    label_type: str,
    label: str,
    source_field: str,
    source_value: str | None = None,
    confidence: float = 0.99,
) -> EvidenceLabel:
    return EvidenceLabel(
        id=make_evidence_label_id(label_type, label),
        label=label,
        label_type=label_type,
        confidence=confidence,
        evidence_text=source_value or label,
        source_field=source_field,
        source_value=source_value or label,
        extractor_name="neural_search.corpus.convert_demo_seed",
        extractor_version=DEFAULT_EXTRACTOR_VERSION,
    )


def _labels_for_field(fixture: Mapping[str, Any], field_name: str) -> list[EvidenceLabel]:
    label_type = FIELD_LABEL_TYPES[field_name]
    return [
        _label(
            label_type=label_type,
            label=value,
            source_field=field_name,
            source_value=value,
        )
        for value in _clean_values(fixture.get(field_name, []))
    ]


def _merge_labels(*groups: Iterable[EvidenceLabel]) -> list[EvidenceLabel]:
    by_id: dict[str, EvidenceLabel] = {}
    for group in groups:
        for label in group:
            if label.id not in by_id or label.confidence > by_id[label.id].confidence:
                by_id[label.id] = label
    return sorted(by_id.values(), key=lambda item: (item.label_type, item.id))


def _paper_analysis_hints(papers: Iterable[Mapping[str, Any]]) -> list[str]:
    hints: list[str] = []
    for paper in papers:
        abstract = str(paper.get("abstract") or "")
        match = ANALYSIS_HINT_RE.search(abstract)
        if not match:
            continue
        for item in re.split(r",|\band\b", match.group("items")):
            cleaned = item.strip().strip(".")
            if cleaned:
                hints.append(cleaned)
    return list(dict.fromkeys(hints))


def _file_format_labels(fixture: Mapping[str, Any]) -> list[EvidenceLabel]:
    formats: list[str] = []
    standards = {value.casefold() for value in _clean_values(fixture.get("data_standards", []))}
    if "nwb" in standards or "dandi" in standards:
        formats.append("nwb")
    if "bids" in standards or "openneuro" in standards:
        formats.extend(["tsv", "json"])
    for asset in fixture.get("assets", []) or []:
        if isinstance(asset, Mapping) and asset.get("file_format"):
            formats.append(str(asset["file_format"]))
    return [
        _label(label_type="file_format", label=value, source_field="data_standards")
        for value in dict.fromkeys(formats)
    ]


def _usability_flags(fixture: Mapping[str, Any]) -> UsabilityFlags:
    modalities = {value.casefold() for value in _clean_values(fixture.get("modalities", []))}
    standards = {value.casefold() for value in _clean_values(fixture.get("data_standards", []))}
    has_neural = bool(modalities & NEURAL_MODALITIES)
    has_behavior = bool(fixture.get("has_behavior")) or bool(modalities & BEHAVIOR_ONLY_MODALITIES)
    return UsabilityFlags(
        has_trials=bool(fixture.get("has_trials")),
        has_behavior=has_behavior,
        has_neural_data=has_neural,
        has_continuous_behavior=bool(
            modalities & {"pose_tracking", "position_tracking", "facemap", "pupil_tracking"}
        ),
        has_event_timestamps=bool(fixture.get("has_trials")) or bool(fixture.get("behaviors")),
        has_raw_data=bool(fixture.get("has_raw_data", True)),
        has_processed_data=bool(fixture.get("has_processed_data", False)),
        has_standard_format=bool(standards & {"nwb", "bids", "dandi", "openneuro"}),
    )


def _missing_fields(fixture: Mapping[str, Any], record: NormalizedDatasetRecord) -> list[str]:
    missing: list[str] = []
    for field_name in FIELD_LABEL_TYPES:
        if not _clean_values(fixture.get(field_name, [])):
            missing.append(field_name)
    if not fixture.get("description"):
        missing.append("description")
    if not fixture.get("license"):
        missing.append("license")
    if not record.usability_flags.has_trials:
        missing.append("trials")
    return sorted(dict.fromkeys(missing))


def convert_dataset_fixture(
    fixture: Mapping[str, Any],
    *,
    paper_lookup: Mapping[str, Mapping[str, Any]] | None = None,
    created_at: str = DEFAULT_CREATED_AT,
    raw_payload_path: str = "data/seed/demo_datasets.yaml",
) -> NormalizedDatasetRecord:
    """Convert one demo dataset fixture into a normalized dataset record."""

    source = str(fixture.get("source") or "demo")
    source_id = str(fixture["source_id"])
    linked_paper_ids = [
        make_paper_id("demo", paper_id)
        for paper_id in _clean_values(fixture.get("linked_paper_ids", []))
    ]
    linked_papers = [
        paper_lookup[paper_id]
        for paper_id in _clean_values(fixture.get("linked_paper_ids", []))
        if paper_lookup and paper_id in paper_lookup
    ]
    analysis_goals = [
        _label(
            label_type="analysis_goal",
            label=value,
            source_field="linked_paper_abstracts",
            confidence=0.82,
        )
        for value in _paper_analysis_hints(linked_papers)
    ]
    record = NormalizedDatasetRecord(
        dataset_id=make_dataset_id(source, source_id),
        source=source,
        source_id=source_id,
        title=str(fixture["title"]),
        description=fixture.get("description"),
        url=fixture.get("url"),
        raw_payload_path=raw_payload_path,
        species=_labels_for_field(fixture, "species"),
        modalities=_labels_for_field(fixture, "modalities"),
        brain_regions=_labels_for_field(fixture, "brain_regions"),
        tasks=_labels_for_field(fixture, "tasks"),
        behavioral_events=_labels_for_field(fixture, "behaviors"),
        analysis_goals=analysis_goals,
        data_standards=_labels_for_field(fixture, "data_standards"),
        file_formats=_file_format_labels(fixture),
        linked_papers=linked_paper_ids,
        usability_flags=_usability_flags(fixture),
        created_at=created_at,
        extractor_version=DEFAULT_EXTRACTOR_VERSION,
    )
    enriched = enrich_record_with_scientific_labels(record)
    assert isinstance(enriched, NormalizedDatasetRecord)
    record = enriched.model_copy(
        update={
            "species": _merge_labels(record.species, enriched.species),
            "modalities": _merge_labels(record.modalities, enriched.modalities),
            "brain_regions": _merge_labels(record.brain_regions, enriched.brain_regions),
            "tasks": _merge_labels(record.tasks, enriched.tasks),
            "behavioral_events": _merge_labels(
                record.behavioral_events,
                enriched.behavioral_events,
            ),
            "analysis_goals": _merge_labels(record.analysis_goals, enriched.analysis_goals),
            "data_standards": _merge_labels(record.data_standards, enriched.data_standards),
            "file_formats": _merge_labels(record.file_formats, enriched.file_formats),
        },
        deep=True,
    )
    record = record.model_copy(update={"missing_fields": _missing_fields(fixture, record)})
    return record.model_copy(
        update={"analysis_affordances": detect_analysis_affordances(record)},
        deep=True,
    )


def convert_paper_fixture(
    fixture: Mapping[str, Any],
    *,
    linked_dataset_ids: Iterable[str] = (),
    created_at: str = DEFAULT_CREATED_AT,
) -> NormalizedPaperRecord:
    """Convert one demo paper fixture into a normalized paper record."""

    source_id = str(fixture["id"])
    concepts = " ".join(_clean_values(fixture.get("concepts", [])))
    abstract = " ".join(
        part for part in [str(fixture.get("abstract") or ""), f"Concepts: {concepts}"] if part
    ).strip()
    authors = [
        str(author.get("name", "")).strip()
        for author in fixture.get("authors_json", [])
        if isinstance(author, Mapping) and str(author.get("name", "")).strip()
    ]
    record = NormalizedPaperRecord(
        paper_id=make_paper_id("demo", source_id),
        source="demo",
        source_id=source_id,
        title=str(fixture["title"]),
        abstract=abstract or None,
        doi=fixture.get("doi"),
        url=fixture.get("url"),
        year=fixture.get("publication_year"),
        authors=authors,
        linked_datasets=list(dict.fromkeys(linked_dataset_ids)),
        raw_payload_path="data/seed/demo_papers.yaml",
        created_at=created_at,
        extractor_version=DEFAULT_EXTRACTOR_VERSION,
    )
    enriched = enrich_record_with_scientific_labels(record)
    assert isinstance(enriched, NormalizedPaperRecord)
    return enriched


def convert_demo_seed(
    datasets_path: str | Path,
    papers_path: str | Path,
    out_dir: str | Path,
    *,
    prefix: str = "demo_v05",
    created_at: str = DEFAULT_CREATED_AT,
) -> dict[str, Path]:
    """Convert demo seed YAML files and write deterministic normalized JSONL."""

    datasets_payload = _load_yaml_mapping(datasets_path)
    papers_payload = _load_yaml_mapping(papers_path)
    dataset_fixtures = datasets_payload.get("datasets", [])
    paper_fixtures = papers_payload.get("papers", [])
    if not isinstance(dataset_fixtures, list) or not isinstance(paper_fixtures, list):
        raise ValueError("demo seed YAML must contain datasets and papers lists")

    paper_lookup = {
        str(paper["id"]): paper
        for paper in paper_fixtures
        if isinstance(paper, Mapping) and paper.get("id")
    }
    datasets = [
        convert_dataset_fixture(fixture, paper_lookup=paper_lookup, created_at=created_at)
        for fixture in dataset_fixtures
    ]
    linked_dataset_ids_by_paper: dict[str, list[str]] = {}
    for dataset in datasets:
        for paper_id in dataset.linked_papers:
            linked_dataset_ids_by_paper.setdefault(paper_id, []).append(dataset.dataset_id)

    papers = [
        convert_paper_fixture(
            fixture,
            linked_dataset_ids=linked_dataset_ids_by_paper.get(
                make_paper_id("demo", str(fixture["id"])),
                [],
            ),
            created_at=created_at,
        )
        for fixture in paper_fixtures
    ]

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = write_jsonl(datasets, output / f"{prefix}.datasets.jsonl")
    paper_path = write_jsonl(papers, output / f"{prefix}.papers.jsonl")
    mixed_records: list[NormalizedRecord] = [*datasets, *papers]
    records_path = write_jsonl(mixed_records, output / f"{prefix}.records.jsonl")
    return {"datasets": dataset_path, "papers": paper_path, "records": records_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert demo seed YAML into normalized corpus JSONL.",
    )
    parser.add_argument("--datasets", required=True, help="data/seed/demo_datasets.yaml")
    parser.add_argument("--papers", required=True, help="data/seed/demo_papers.yaml")
    parser.add_argument("--out-dir", required=True, help="Output normalized corpus directory")
    parser.add_argument("--prefix", default="demo_v05")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = convert_demo_seed(
        args.datasets,
        args.papers,
        args.out_dir,
        prefix=args.prefix,
    )
    print(
        "wrote normalized demo corpus: "
        + ", ".join(f"{name}={path}" for name, path in paths.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
