"""Build regional coverage maps from normalized corpus records."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from neural_search.schemas import EvidenceLabel, NormalizedDatasetRecord

DEFAULT_INPUTS = (
    Path("data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl"),
    Path("data/corpus/normalized/coverage_depth/coverage_depth.records.jsonl"),
)
DEFAULT_TARGETS = Path("data/config/regional_map_targets.yaml")
DEFAULT_OUT_DIR = Path("data/reports/regional_map")
RecordLike = dict[str, Any] | NormalizedDatasetRecord


def _slug(value: str) -> str:
    cleaned = value.strip().lower().replace("-", "_").replace(" ", "_")
    return "_".join(part for part in re.split(r"_+", cleaned) if part)


def _as_mapping(record: RecordLike) -> dict[str, Any]:
    if isinstance(record, NormalizedDatasetRecord):
        return record.model_dump(mode="json")
    return dict(record)


def _record_id(record: RecordLike) -> str:
    return str(_as_mapping(record)["dataset_id"])


def _label_text(label: Any) -> str:
    if isinstance(label, EvidenceLabel):
        return label.label
    if isinstance(label, dict):
        return str(label.get("label") or label.get("id") or "")
    return str(label)


def _label_extra_text(label: Any) -> list[str]:
    if isinstance(label, EvidenceLabel):
        return [label.label, label.evidence_text or "", label.source_value or ""]
    if isinstance(label, dict):
        return [
            str(label.get("label") or ""),
            str(label.get("id") or ""),
            str(label.get("evidence_text") or ""),
            str(label.get("source_value") or ""),
        ]
    return [str(label)]


def _label_values(labels: Any) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for label in labels or []:
        value = _slug(_label_text(label))
        if value and value not in seen:
            seen.add(value)
            values.append(value)
    return values


def _record_text(record: RecordLike) -> str:
    payload = _as_mapping(record)
    parts = [
        payload.get("title") or "",
        payload.get("description") or "",
        payload.get("source") or "",
        payload.get("source_id") or "",
        payload.get("raw_payload_path") or "",
    ]
    for field in (
        "species",
        "modalities",
        "brain_regions",
        "tasks",
        "behavioral_events",
        "analysis_goals",
        "data_standards",
        "file_formats",
    ):
        for label in payload.get(field) or []:
            parts.extend(_label_extra_text(label))
    return " ".join(str(part) for part in parts if part)


def _compile_alias_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias.strip().lower())
    escaped = escaped.replace(r"\ ", r"[\s_-]+")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")


def load_region_targets(path: str | Path = DEFAULT_TARGETS) -> list[dict[str, Any]]:
    """Load atlas-style region targets used for candidate coverage mapping."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    targets = payload.get("regional_targets", [])
    if not isinstance(targets, list):
        raise ValueError("regional map targets must define regional_targets list")
    normalized: list[dict[str, Any]] = []
    for target in targets:
        if not isinstance(target, dict):
            continue
        region_id = _slug(str(target.get("id", "")))
        if not region_id:
            continue
        aliases = [_slug(region_id).replace("_", " ")]
        aliases.extend(str(alias).strip() for alias in target.get("aliases", []) if str(alias).strip())
        normalized.append(
            {
                "id": region_id,
                "label": str(target.get("label") or region_id),
                "system": str(target.get("system") or "unassigned"),
                "aliases": sorted(set(aliases), key=str.lower),
            },
        )
    return normalized


def _load_loose_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    if input_path.is_dir():
        records: list[dict[str, Any]] = []
        for child in sorted([*input_path.glob("*.jsonl"), *input_path.glob("*.json")]):
            records.extend(_load_loose_records(child))
        return records
    if input_path.suffix == ".jsonl":
        records = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return records

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def load_dataset_records(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Load dataset records from one or more normalized JSON/JSONL paths."""

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for record in _load_loose_records(path):
            dataset_id = record.get("dataset_id")
            if dataset_id and dataset_id not in seen:
                seen.add(str(dataset_id))
                records.append(record)
    return records


def find_candidate_regions(
    record: RecordLike,
    targets: list[dict[str, Any]],
) -> list[str]:
    """Find atlas target mentions in visible normalized record text."""

    text = _record_text(record).lower()
    hits: list[str] = []
    for target in targets:
        aliases = target.get("aliases", [])
        if any(_compile_alias_pattern(alias).search(text) for alias in aliases):
            hits.append(str(target["id"]))
    return sorted(set(hits))


def build_regional_map(
    records: list[RecordLike],
    targets: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a machine-readable region coverage map and review queue."""

    target_by_id = {str(target["id"]): target for target in targets}
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    by_species: dict[str, Counter[str]] = defaultdict(Counter)
    by_modality: dict[str, Counter[str]] = defaultdict(Counter)
    region_records: dict[str, set[str]] = defaultdict(set)
    region_sources: dict[str, set[str]] = defaultdict(set)
    region_species: dict[str, set[str]] = defaultdict(set)
    region_modalities: dict[str, set[str]] = defaultdict(set)
    candidate_records: dict[str, set[str]] = defaultdict(set)
    missing_region_records: list[dict[str, Any]] = []

    for record in records:
        payload = _as_mapping(record)
        record_id = _record_id(record)
        source = _slug(str(payload.get("source") or "unknown"))
        species = _label_values(payload.get("species"))
        modalities = _label_values(payload.get("modalities"))
        regions = _label_values(payload.get("brain_regions"))
        candidates = find_candidate_regions(record, targets)

        for region in regions:
            region_records[region].add(record_id)
            region_sources[region].add(source)
            for value in species:
                region_species[region].add(value)
            for value in modalities:
                region_modalities[region].add(value)
            by_source[source][region] += 1
            for value in species:
                by_species[value][region] += 1
            for value in modalities:
                by_modality[value][region] += 1

        for candidate in candidates:
            if candidate not in regions:
                candidate_records[candidate].add(record_id)

        if not regions:
            missing_region_records.append(
                {
                    "record_id": record_id,
                    "source": source,
                    "source_id": str(payload.get("source_id") or ""),
                    "title": str(payload.get("title") or ""),
                    "candidate_regions": candidates,
                    "species": species,
                    "modalities": modalities,
                },
            )

    target_rows = []
    for target in targets:
        region_id = str(target["id"])
        verified = len(region_records.get(region_id, set()))
        candidates = len(candidate_records.get(region_id, set()))
        if verified:
            status = "verified"
        elif candidates:
            status = "candidate_only"
        else:
            status = "uncovered"
        target_rows.append(
            {
                "region_id": region_id,
                "label": target["label"],
                "system": target["system"],
                "verified_record_count": verified,
                "candidate_record_count": candidates,
                "status": status,
            },
        )

    region_rows = [
        {
            "region_id": region,
            "label": target_by_id.get(region, {}).get("label", region),
            "system": target_by_id.get(region, {}).get("system", "unassigned"),
            "record_count": len(record_ids),
            "candidate_record_count": len(candidate_records.get(region, set())),
            "sources": sorted(region_sources[region]),
            "species": sorted(region_species[region]),
            "modalities": sorted(region_modalities[region]),
            "example_records": sorted(record_ids)[:8],
        }
        for region, record_ids in region_records.items()
    ]
    region_rows.sort(key=lambda item: (-int(item["record_count"]), str(item["region_id"])))

    total_records = len(records)
    with_regions = sum(1 for record in records if _as_mapping(record).get("brain_regions"))
    with_candidates = sum(1 for item in missing_region_records if item["candidate_regions"])
    verified_targets = sum(1 for item in target_rows if item["status"] == "verified")
    candidate_only_targets = sum(1 for item in target_rows if item["status"] == "candidate_only")
    system_totals: dict[str, dict[str, int]] = {}
    for row in target_rows:
        system = str(row["system"])
        totals = system_totals.setdefault(
            system,
            {
                "target_regions": 0,
                "verified_regions": 0,
                "candidate_only_regions": 0,
                "uncovered_regions": 0,
                "verified_records": 0,
                "candidate_records": 0,
            },
        )
        totals["target_regions"] += 1
        totals[f"{row['status']}_regions"] += 1
        totals["verified_records"] += int(row["verified_record_count"])
        totals["candidate_records"] += int(row["candidate_record_count"])

    return {
        "summary": {
            "dataset_records": total_records,
            "records_with_verified_regions": with_regions,
            "records_without_verified_regions": total_records - with_regions,
            "region_coverage": round(with_regions / total_records, 4) if total_records else 0.0,
            "regionless_records_with_candidate_mentions": with_candidates,
            "atlas_target_regions": len(target_rows),
            "atlas_targets_verified": verified_targets,
            "atlas_targets_candidate_only": candidate_only_targets,
            "atlas_targets_uncovered": len(target_rows) - verified_targets - candidate_only_targets,
        },
        "regions": region_rows,
        "atlas_targets": sorted(
            target_rows,
            key=lambda item: (str(item["status"]), str(item["system"]), str(item["region_id"])),
        ),
        "missing_region_records": missing_region_records,
        "by_source": _counter_matrix(by_source),
        "by_species": _counter_matrix(by_species),
        "by_modality": _counter_matrix(by_modality),
        "system_totals": dict(sorted(system_totals.items())),
        "frontend_regions": [
            {
                "region_id": row["region_id"],
                "label": row["label"],
                "system": row["system"],
                "verified_record_count": row["record_count"],
                "candidate_record_count": row["candidate_record_count"],
                "species": row["species"],
                "modalities": row["modalities"],
                "sources": row["sources"],
                "example_records": row["example_records"][:3],
            }
            for row in region_rows
        ],
        "candidate_region_mentions": {
            region: sorted(record_ids)
            for region, record_ids in sorted(candidate_records.items())
            if record_ids
        },
    }


def _counter_matrix(matrix: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {
        key: dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))
        for key, counter in sorted(matrix.items())
    }


def render_regional_map_report(regional_map: dict[str, Any]) -> str:
    """Render a human-readable regional coverage report."""

    summary = regional_map["summary"]
    lines = [
        "# Regional Coverage Map",
        "",
        f"- Dataset records: {summary['dataset_records']}",
        f"- Records with verified regions: {summary['records_with_verified_regions']}",
        f"- Region coverage: {summary['region_coverage']:.1%}",
        f"- Records without verified regions: {summary['records_without_verified_regions']}",
        "- Regionless records with candidate mentions: "
        f"{summary['regionless_records_with_candidate_mentions']}",
        f"- Atlas target regions: {summary['atlas_target_regions']}",
        f"- Atlas targets verified: {summary['atlas_targets_verified']}",
        f"- Atlas targets candidate-only: {summary['atlas_targets_candidate_only']}",
        f"- Atlas targets uncovered: {summary['atlas_targets_uncovered']}",
        "",
        "## Top Verified Regions",
        "",
        "| Region | Records | Sources | Species | Modalities |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for row in regional_map["regions"][:25]:
        lines.append(
            "| {region} | {count} | {sources} | {species} | {modalities} |".format(
                region=row["region_id"],
                count=row["record_count"],
                sources=", ".join(row["sources"]) or "-",
                species=", ".join(row["species"][:6]) or "-",
                modalities=", ".join(row["modalities"][:6]) or "-",
            ),
        )

    lines.extend(
        [
            "",
            "## System Coverage",
            "",
            "| System | Targets | Verified | Candidate-only | Uncovered | Verified Records | Candidate Records |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ],
    )
    for system, totals in regional_map.get("system_totals", {}).items():
        lines.append(
            "| {system} | {targets} | {verified} | {candidate_only} | {uncovered} | {verified_records} | {candidate_records} |".format(
                system=system,
                targets=totals["target_regions"],
                verified=totals["verified_regions"],
                candidate_only=totals["candidate_only_regions"],
                uncovered=totals["uncovered_regions"],
                verified_records=totals["verified_records"],
                candidate_records=totals["candidate_records"],
            ),
        )

    lines.extend(
        [
            "",
            "## Candidate-Only Atlas Targets",
            "",
            "| Region | System | Candidate Records |",
            "| --- | --- | ---: |",
        ],
    )
    for row in regional_map["atlas_targets"]:
        if row["status"] != "candidate_only":
            continue
        lines.append(
            f"| {row['region_id']} | {row['system']} | {row['candidate_record_count']} |",
        )

    lines.extend(
        [
            "",
            "## Regionless Review Queue",
            "",
            "| Record | Source | Candidate Regions | Title |",
            "| --- | --- | --- | --- |",
        ],
    )
    for row in regional_map["missing_region_records"][:50]:
        candidates = ", ".join(row["candidate_regions"]) or "-"
        title = str(row["title"]).replace("|", "\\|")
        lines.append(f"| {row['record_id']} | {row['source']} | {candidates} | {title} |")
    return "\n".join(lines).rstrip() + "\n"


def write_regional_map_artifacts(
    input_paths: list[str | Path] | None = None,
    target_path: str | Path = DEFAULT_TARGETS,
    out_dir: str | Path = DEFAULT_OUT_DIR,
) -> dict[str, Path]:
    """Write regional map JSON, Markdown report, and review queue JSON."""

    paths = list(input_paths) if input_paths is not None else list(DEFAULT_INPUTS)
    records = load_dataset_records(paths)
    targets = load_region_targets(target_path)
    regional_map = build_regional_map(records, targets)

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "regional_map.json"
    report_path = output / "regional_map.md"
    queue_path = output / "regionless_review_queue.json"
    json_path.write_text(json.dumps(regional_map, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(render_regional_map_report(regional_map), encoding="utf-8")
    queue_path.write_text(
        json.dumps(regional_map["missing_region_records"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"json": json_path, "report": report_path, "review_queue": queue_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build regional corpus coverage map.")
    parser.add_argument("--input", action="append", dest="inputs")
    parser.add_argument("--targets", default=str(DEFAULT_TARGETS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args(argv)

    paths = [Path(item) for item in args.inputs] if args.inputs else list(DEFAULT_INPUTS)
    outputs = write_regional_map_artifacts(paths, args.targets, args.out_dir)
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
