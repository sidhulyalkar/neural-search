"""Coverage reports for real-corpus fixture artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from neural_search.corpus.manifest import load_manifest
from neural_search.file_inspection.claims import FileInspectionClaim
from neural_search.normalized import load_normalized_records
from neural_search.schemas import NormalizedDatasetRecord


def _load_claims(path: str | Path) -> list[FileInspectionClaim]:
    claims_path = Path(path)
    if not claims_path.exists():
        return []
    claims: list[FileInspectionClaim] = []
    with claims_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                claims.append(FileInspectionClaim.model_validate_json(line))
    return claims


def _label_count(records: list[NormalizedDatasetRecord], field: str) -> int:
    return sum(len(getattr(record, field)) for record in records)


def generate_real_corpus_report(
    *,
    manifest_path: str | Path,
    records_path: str | Path,
    claims_path: str | Path,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    records = load_normalized_records(records_path)
    datasets = [record for record in records if isinstance(record, NormalizedDatasetRecord)]
    claims = _load_claims(claims_path)
    missing_counter: Counter[str] = Counter()
    for dataset in datasets:
        missing_counter.update(dataset.missing_fields)

    return {
        "corpus_tag": manifest.corpus_tag,
        "manifest_entries": len(manifest.entries),
        "source_counts": manifest.source_counts(),
        "record_type_counts": manifest.record_type_counts(),
        "normalized_dataset_count": len(datasets),
        "claim_count": len(claims),
        "claim_type_counts": dict(Counter(claim.claim_type for claim in claims)),
        "claim_field_counts": dict(Counter(claim.field for claim in claims)),
        "dataset_label_counts": {
            "species": _label_count(datasets, "species"),
            "modalities": _label_count(datasets, "modalities"),
            "brain_regions": _label_count(datasets, "brain_regions"),
            "tasks": _label_count(datasets, "tasks"),
            "behavioral_events": _label_count(datasets, "behavioral_events"),
            "data_standards": _label_count(datasets, "data_standards"),
            "analysis_affordances": sum(len(item.analysis_affordances) for item in datasets),
        },
        "missing_metadata": dict(missing_counter),
        "datasets": [
            {
                "dataset_id": dataset.dataset_id,
                "title": dataset.title,
                "source": dataset.source,
                "missing_fields": dataset.missing_fields,
                "claim_count": len([claim for claim in claims if claim.dataset_id == dataset.dataset_id]),
            }
            for dataset in datasets
        ],
    }


def generate_real_corpus_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Real Corpus v0.7 Report",
        "",
        f"- Corpus tag: `{report['corpus_tag']}`",
        f"- Manifest entries: {report['manifest_entries']}",
        f"- Normalized datasets: {report['normalized_dataset_count']}",
        f"- File-inspection claims: {report['claim_count']}",
        "",
        "## Source Coverage",
        "",
    ]
    for source, count in sorted(report["source_counts"].items()):
        lines.append(f"- `{source}`: {count}")
    lines.extend(["", "## Claim Coverage", ""])
    for claim_type, count in sorted(report["claim_type_counts"].items()):
        lines.append(f"- `{claim_type}`: {count}")
    lines.extend(["", "## Missing Metadata", ""])
    if report["missing_metadata"]:
        for field, count in sorted(report["missing_metadata"].items()):
            lines.append(f"- `{field}`: {count} datasets")
    else:
        lines.append("- No missing metadata fields recorded.")
    lines.extend(["", "## Dataset Claims", ""])
    for dataset in report["datasets"]:
        lines.append(
            f"- `{dataset['dataset_id']}`: {dataset['claim_count']} claims, "
            f"missing={dataset['missing_fields']}"
        )
    return "\n".join(lines) + "\n"


def write_real_corpus_reports(
    *,
    manifest_path: str | Path,
    records_path: str | Path,
    claims_path: str | Path,
    out_dir: str | Path,
) -> dict[str, str]:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = generate_real_corpus_report(
        manifest_path=manifest_path,
        records_path=records_path,
        claims_path=claims_path,
    )
    json_path = output / "real_corpus_report.json"
    md_path = output / "real_corpus_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(generate_real_corpus_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate real-corpus artifact reports.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--claims", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = write_real_corpus_reports(
        manifest_path=args.manifest,
        records_path=args.records,
        claims_path=args.claims,
        out_dir=args.out,
    )
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
