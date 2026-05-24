"""Manifest-driven real-corpus fixture ingestion."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from neural_search.analysis_affordances import detect_analysis_affordances
from neural_search.corpus.manifest import (
    CorpusManifest,
    CorpusManifestEntry,
    load_manifest,
)
from neural_search.file_inspection import FileInspectionClaim, inspect_dataset_files
from neural_search.normalized import (
    make_dataset_id,
    make_evidence_label_id,
    make_paper_id,
    write_jsonl,
)
from neural_search.schemas import (
    AnalysisAffordance,
    EvidenceLabel,
    NormalizedDatasetRecord,
    NormalizedPaperRecord,
    UsabilityFlags,
)
from neural_search.scientific_labels import enrich_record_with_scientific_labels

DEFAULT_CREATED_AT = "2026-05-24T00:00:00+00:00"
EXTRACTOR_VERSION = "v0.7.0"

DATASET_LABEL_FIELDS = {
    "species": "species",
    "modalities": "modality",
    "brain_regions": "brain_region",
    "tasks": "task",
    "behaviors": "behavioral_event",
    "behavioral_events": "behavioral_event",
    "data_standards": "data_standard",
    "file_formats": "file_format",
}

CLAIM_FIELD_TO_RECORD_FIELD = {
    "species": "species",
    "modalities": "modalities",
    "brain_regions": "brain_regions",
    "tasks": "tasks",
    "behavioral_events": "behavioral_events",
    "data_standards": "data_standards",
    "file_formats": "file_formats",
}


def _clean_values(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(item).strip() for item in value if str(item).strip()]


def _label(
    *,
    label_type: str,
    label: str,
    source_field: str,
    source_value: str | None = None,
    confidence: float = 0.95,
    extractor_name: str = "neural_search.corpus.ingest_manifest",
) -> EvidenceLabel:
    return EvidenceLabel(
        id=make_evidence_label_id(label_type, label, extractor_name),
        label=label,
        label_type=label_type,
        confidence=confidence,
        evidence_text=source_value or label,
        source_field=source_field,
        source_value=source_value or label,
        extractor_name=extractor_name,
        extractor_version=EXTRACTOR_VERSION,
    )


def _labels_from_metadata(metadata: Mapping[str, Any], field_name: str) -> list[EvidenceLabel]:
    label_type = DATASET_LABEL_FIELDS[field_name]
    return [
        _label(label_type=label_type, label=value, source_field=f"manifest:{field_name}")
        for value in _clean_values(metadata.get(field_name, []))
    ]


def _claim_label(claim: FileInspectionClaim, label_type: str) -> EvidenceLabel:
    return _label(
        label_type=label_type,
        label=claim.value,
        source_field=f"file_inspection:{claim.field}",
        source_value=f"{claim.claim_id}: {claim.evidence}",
        confidence=claim.confidence,
        extractor_name=claim.extractor,
    )


def _merge_labels(*groups: Iterable[EvidenceLabel]) -> list[EvidenceLabel]:
    labels: dict[str, EvidenceLabel] = {}
    for group in groups:
        for label in group:
            existing = labels.get(label.id)
            if existing is None or label.confidence > existing.confidence:
                labels[label.id] = label
    return sorted(labels.values(), key=lambda item: (item.label_type, item.label))


def _usability_flags(
    metadata: Mapping[str, Any],
    claims: Iterable[FileInspectionClaim],
) -> UsabilityFlags:
    claim_values: dict[str, set[str]] = defaultdict(set)
    for claim in claims:
        claim_values[claim.field].add(claim.value.casefold())

    modalities = {value.casefold() for value in _clean_values(metadata.get("modalities", []))}
    standards = {value.casefold() for value in _clean_values(metadata.get("data_standards", []))}
    neural_modalities = {
        "neuropixels",
        "extracellular_ephys",
        "eeg",
        "ecog",
        "ieeg",
        "calcium_imaging",
        "fiber_photometry",
    }
    return UsabilityFlags(
        has_trials=bool(metadata.get("has_trials")) or "true" in claim_values["has_trials"],
        has_behavior=bool(metadata.get("has_behavior"))
        or bool(_clean_values(metadata.get("behaviors", [])))
        or "true" in claim_values["has_event_timestamps"],
        has_neural_data=bool(modalities & neural_modalities)
        or bool(claim_values["units"])
        or "eeg" in claim_values["modalities"],
        has_continuous_behavior=bool(metadata.get("has_continuous_behavior")),
        has_event_timestamps=bool(metadata.get("has_event_timestamps"))
        or "true" in claim_values["has_event_timestamps"],
        has_raw_data=bool(metadata.get("has_raw_data", True)),
        has_processed_data=bool(metadata.get("has_processed_data"))
        or "present" in claim_values["derivatives"],
        has_standard_format=bool(standards & {"nwb", "bids", "dandi", "openneuro"})
        or "nwb" in claim_values["data_standards"]
        or "bids" in claim_values["data_standards"],
    )


def _missing_fields(metadata: Mapping[str, Any], record: NormalizedDatasetRecord) -> list[str]:
    missing: list[str] = []
    for field in ["species", "modalities", "brain_regions", "tasks", "data_standards"]:
        if not _clean_values(metadata.get(field, [])) and not getattr(record, field):
            missing.append(field)
    if not metadata.get("description"):
        missing.append("description")
    if not record.usability_flags.has_trials:
        missing.append("trials")
    return sorted(dict.fromkeys(missing))


def _claim_affordances(claims: Iterable[FileInspectionClaim]) -> list[AnalysisAffordance]:
    affordances: dict[str, AnalysisAffordance] = {}
    for claim in claims:
        if claim.claim_type != "analysis_affordance":
            continue
        existing = affordances.get(claim.value)
        if existing and existing.confidence >= claim.confidence:
            continue
        affordances[claim.value] = AnalysisAffordance(
            analysis_id=claim.value,
            support_level="medium" if claim.confidence < 0.85 else "high",
            confidence=claim.confidence,
            helpful_fields_present=[claim.field],
            evidence=[f"{claim.claim_id}: {claim.evidence}"],
            detector_name=claim.extractor,
            detector_version=EXTRACTOR_VERSION,
        )
    return sorted(affordances.values(), key=lambda item: item.analysis_id)


def normalize_dataset_entry(
    entry: CorpusManifestEntry,
    claims: Iterable[FileInspectionClaim],
) -> NormalizedDatasetRecord:
    metadata = entry.metadata
    dataset_id = make_dataset_id(entry.source, entry.source_id)
    claim_list = list(claims)
    claim_labels: dict[str, list[EvidenceLabel]] = defaultdict(list)
    for claim in claim_list:
        target_field = CLAIM_FIELD_TO_RECORD_FIELD.get(claim.field)
        if not target_field:
            continue
        label_type = DATASET_LABEL_FIELDS[target_field]
        claim_labels[target_field].append(_claim_label(claim, label_type))

    record = NormalizedDatasetRecord(
        dataset_id=dataset_id,
        source=entry.source,
        source_id=entry.source_id,
        title=str(metadata.get("title") or f"{entry.source} {entry.source_id}"),
        description=metadata.get("description"),
        url=metadata.get("url"),
        raw_payload_path=metadata.get("raw_payload_path"),
        species=_merge_labels(_labels_from_metadata(metadata, "species"), claim_labels["species"]),
        modalities=_merge_labels(
            _labels_from_metadata(metadata, "modalities"),
            claim_labels["modalities"],
        ),
        brain_regions=_merge_labels(
            _labels_from_metadata(metadata, "brain_regions"),
            claim_labels["brain_regions"],
        ),
        tasks=_merge_labels(_labels_from_metadata(metadata, "tasks"), claim_labels["tasks"]),
        behavioral_events=_merge_labels(
            _labels_from_metadata(metadata, "behaviors"),
            claim_labels["behavioral_events"],
        ),
        data_standards=_merge_labels(
            _labels_from_metadata(metadata, "data_standards"),
            claim_labels["data_standards"],
        ),
        file_formats=_merge_labels(
            _labels_from_metadata(metadata, "file_formats"),
            claim_labels["file_formats"],
        ),
        linked_papers=[
            make_paper_id("openalex", value)
            if not str(value).startswith("paper:")
            else str(value)
            for value in _clean_values(metadata.get("linked_paper_ids", []))
        ],
        usability_flags=_usability_flags(metadata, claim_list),
        analysis_affordances=_claim_affordances(claim_list),
        created_at=DEFAULT_CREATED_AT,
        extractor_version=EXTRACTOR_VERSION,
    )
    enriched = enrich_record_with_scientific_labels(record)
    assert isinstance(enriched, NormalizedDatasetRecord)
    record = record.model_copy(
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
            "analysis_affordances": [
                *record.analysis_affordances,
                *detect_analysis_affordances(record),
            ],
        },
        deep=True,
    )
    return record.model_copy(update={"missing_fields": _missing_fields(metadata, record)})


def normalize_paper_entry(entry: CorpusManifestEntry) -> NormalizedPaperRecord:
    metadata = entry.metadata
    record = NormalizedPaperRecord(
        paper_id=make_paper_id(entry.source, entry.source_id),
        source=entry.source,
        source_id=entry.source_id,
        title=str(metadata.get("title") or f"{entry.source} {entry.source_id}"),
        abstract=metadata.get("abstract"),
        doi=metadata.get("doi"),
        url=metadata.get("url"),
        year=metadata.get("year"),
        authors=_clean_values(metadata.get("authors", [])),
        linked_datasets=[
            value if str(value).startswith("dataset:") else make_dataset_id("dandi", str(value))
            for value in _clean_values(metadata.get("linked_dataset_ids", []))
        ],
        raw_payload_path=metadata.get("raw_payload_path"),
        created_at=DEFAULT_CREATED_AT,
        extractor_version=EXTRACTOR_VERSION,
    )
    enriched = enrich_record_with_scientific_labels(record)
    assert isinstance(enriched, NormalizedPaperRecord)
    return enriched


def _write_raw_payload(entry: CorpusManifestEntry, raw_dir: Path) -> Path:
    output = raw_dir / entry.source / f"{entry.source_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": entry.source,
        "source_id": entry.source_id,
        "record_type": entry.record_type,
        "metadata": entry.metadata,
        "fetch": entry.fetch,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def ingest_manifest(
    manifest: CorpusManifest,
    *,
    out_dir: str | Path,
    claims_out: str | Path | None = None,
    raw_dir: str | Path | None = None,
    write_raw: bool = False,
    dry_run: bool = False,
    prefix: str | None = None,
) -> dict[str, Any]:
    """Normalize manifest entries, optionally inspecting local fixture files."""

    output_dir = Path(out_dir)
    corpus_tag = prefix or manifest.corpus_tag
    counts = Counter()
    warnings: list[str] = []
    if dry_run:
        for entry in manifest.entries:
            counts[f"would_{entry.record_type}"] += 1
            counts[f"source:{entry.source}"] += 1
        return {
            "corpus_tag": corpus_tag,
            "dry_run": True,
            "counts": dict(counts),
            "warnings": warnings,
        }

    claims_by_dataset: dict[str, list[FileInspectionClaim]] = defaultdict(list)
    for entry in manifest.entries:
        if entry.record_type != "dataset":
            continue
        dataset_id = make_dataset_id(entry.source, entry.source_id)
        entry_claims = inspect_dataset_files(entry.inspection_paths, dataset_id)
        claims_by_dataset[dataset_id].extend(entry_claims)
        warnings.extend(
            f"{dataset_id}: {claim.evidence}"
            for claim in entry_claims
            if claim.claim_type == "warning"
        )

    datasets: list[NormalizedDatasetRecord] = []
    papers: list[NormalizedPaperRecord] = []
    for entry in manifest.entries:
        counts[f"source:{entry.source}"] += 1
        if write_raw and raw_dir:
            _write_raw_payload(entry, Path(raw_dir))
        if entry.status == "skipped":
            counts["skipped"] += 1
            continue
        try:
            if entry.record_type == "dataset":
                dataset_id = make_dataset_id(entry.source, entry.source_id)
                datasets.append(normalize_dataset_entry(entry, claims_by_dataset[dataset_id]))
                counts["normalized_datasets"] += 1
            else:
                papers.append(normalize_paper_entry(entry))
                counts["normalized_papers"] += 1
        except Exception as exc:
            counts["failed"] += 1
            warnings.append(f"{entry.source}:{entry.source_id}: {exc}")

    all_claims = [claim for claims in claims_by_dataset.values() for claim in claims]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "datasets": str(write_jsonl(datasets, output_dir / f"{corpus_tag}.datasets.jsonl")),
        "papers": str(write_jsonl(papers, output_dir / f"{corpus_tag}.papers.jsonl")),
        "records": str(
            write_jsonl(
                [*datasets, *papers],
                output_dir / f"{corpus_tag}.records.jsonl",
            )
        ),
    }
    if claims_out:
        claims_path = Path(claims_out)
        claims_path.parent.mkdir(parents=True, exist_ok=True)
        with claims_path.open("w", encoding="utf-8") as handle:
            for claim in all_claims:
                handle.write(claim.model_dump_json())
                handle.write("\n")
        paths["claims"] = str(claims_path)
    counts["claims"] = len(all_claims)
    return {
        "corpus_tag": corpus_tag,
        "dry_run": False,
        "counts": dict(counts),
        "warnings": warnings,
        "paths": paths,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a real corpus from a manifest.")
    parser.add_argument("--manifest", required=True, help="Manifest YAML path")
    parser.add_argument("--out", required=True, help="Output normalized corpus directory")
    parser.add_argument("--claims-out", help="Output claim JSONL path")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw payload output root")
    parser.add_argument("--write-raw", action="store_true", help="Persist raw payload JSON")
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize only")
    parser.add_argument("--prefix", help="Override output file prefix")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    summary = ingest_manifest(
        manifest,
        out_dir=args.out,
        claims_out=args.claims_out,
        raw_dir=args.raw_dir,
        write_raw=args.write_raw,
        dry_run=args.dry_run,
        prefix=args.prefix,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not summary["counts"].get("failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
