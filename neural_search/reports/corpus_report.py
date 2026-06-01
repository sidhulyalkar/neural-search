"""Corpus QA report generation for normalized v0.3 records."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from neural_search.normalized import NormalizedRecord, load_normalized_records
from neural_search.schemas import (
    EvidenceLabel,
    NormalizedDatasetRecord,
    NormalizedPaperRecord,
)

LABEL_FIELDS = [
    "species",
    "modalities",
    "brain_regions",
    "tasks",
    "behavioral_events",
    "analysis_goals",
    "data_standards",
    "file_formats",
]


def _labels(record: NormalizedRecord) -> list[EvidenceLabel]:
    if isinstance(record, NormalizedPaperRecord):
        return record.extracted_labels
    labels: list[EvidenceLabel] = []
    for field in LABEL_FIELDS:
        labels.extend(getattr(record, field, []))
    return labels


def _dataset_records(records: list[NormalizedRecord]) -> list[NormalizedDatasetRecord]:
    return [record for record in records if isinstance(record, NormalizedDatasetRecord)]


def _paper_records(records: list[NormalizedRecord]) -> list[NormalizedPaperRecord]:
    return [record for record in records if isinstance(record, NormalizedPaperRecord)]


def summarize_corpus(records: list[NormalizedRecord]) -> dict[str, Any]:
    """Build deterministic counts and QA findings for normalized records."""

    source_counts = Counter(record.source for record in records)
    label_counts: dict[str, Counter[str]] = {
        field: Counter() for field in [*LABEL_FIELDS, "paper_labels", "analysis_affordances"]
    }
    missing_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    low_confidence: list[dict[str, Any]] = []

    for record in records:
        if isinstance(record, NormalizedDatasetRecord):
            for field in LABEL_FIELDS:
                labels = getattr(record, field)
                label_counts[field].update(label.id for label in labels)
            for field in record.missing_fields:
                missing_by_source[record.source][field] += 1
            label_counts["analysis_affordances"].update(
                item.analysis_id for item in record.analysis_affordances
            )
        else:
            label_counts["paper_labels"].update(label.id for label in record.extracted_labels)

        for label in _labels(record):
            if label.confidence < 0.7:
                low_confidence.append(
                    {
                        "record_id": getattr(record, "dataset_id", getattr(record, "paper_id", "")),
                        "source": record.source,
                        "label_id": label.id,
                        "label_type": label.label_type,
                        "confidence": label.confidence,
                        "evidence_text": label.evidence_text,
                    }
                )

    dataset_paper_links = []
    paper_ids = {paper.paper_id for paper in _paper_records(records)} | {
        paper.source_id for paper in _paper_records(records)
    }
    for dataset in _dataset_records(records):
        for linked in dataset.linked_papers:
            dataset_paper_links.append(
                {
                    "dataset_id": dataset.dataset_id,
                    "linked_paper": linked,
                    "paper_present": linked in paper_ids,
                    "evidence": "declared linked_papers field",
                }
            )

    return {
        "total_records": len(records),
        "dataset_records": len(_dataset_records(records)),
        "paper_records": len(_paper_records(records)),
        "source_counts": dict(sorted(source_counts.items())),
        "label_counts": {
            field: dict(counter.most_common())
            for field, counter in sorted(label_counts.items())
            if counter
        },
        "missing_by_source": {
            source: dict(counter.most_common())
            for source, counter in sorted(missing_by_source.items())
        },
        "low_confidence_labels": sorted(
            low_confidence,
            key=lambda item: (item["confidence"], item["source"], item["record_id"]),
        ),
        "dataset_paper_links": dataset_paper_links,
    }


def _table(counter: dict[str, int]) -> list[str]:
    if not counter:
        return ["No records."]
    lines = ["| Value | Count |", "|-------|------:|"]
    lines.extend(f"| {key} | {value} |" for key, value in counter.items())
    return lines


def write_corpus_reports(summary: dict[str, Any], out_dir: str | Path) -> dict[str, str]:
    """Write all corpus QA reports requested by v0.3."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    coverage = [
        "# Corpus Coverage Report",
        "",
        f"Total records: {summary['total_records']}",
        f"Dataset records: {summary['dataset_records']}",
        f"Paper records: {summary['paper_records']}",
        "",
        "## Labels",
        "",
    ]
    for field, counts in summary["label_counts"].items():
        coverage.extend([f"### {field}", "", *_table(counts), ""])
    path = out / "corpus_coverage_report.md"
    path.write_text("\n".join(coverage), encoding="utf-8")
    paths["coverage"] = str(path)

    missing = ["# Missing Metadata Report", ""]
    for source, counts in summary["missing_by_source"].items():
        missing.extend([f"## {source}", "", *_table(counts), ""])
    if len(missing) == 2:
        missing.append("No missing metadata recorded.")
    path = out / "missing_metadata_report.md"
    path.write_text("\n".join(missing), encoding="utf-8")
    paths["missing"] = str(path)

    confidence = [
        "# Label Confidence Report",
        "",
        f"Low-confidence labels: {len(summary['low_confidence_labels'])}",
        "",
    ]
    for item in summary["low_confidence_labels"][:50]:
        confidence.append(
            "- {record_id} `{label_id}` ({label_type}) confidence={confidence:.2f}".format(
                **item
            )
        )
    path = out / "label_confidence_report.md"
    path.write_text("\n".join(confidence), encoding="utf-8")
    paths["confidence"] = str(path)

    source_distribution = [
        "# Source Distribution Report",
        "",
        *_table(summary["source_counts"]),
        "",
    ]
    path = out / "source_distribution_report.md"
    path.write_text("\n".join(source_distribution), encoding="utf-8")
    paths["sources"] = str(path)

    links = ["# Dataset Paper Linking Report", ""]
    for item in summary["dataset_paper_links"]:
        status = "present" if item["paper_present"] else "missing"
        links.append(
            f"- {item['dataset_id']} -> {item['linked_paper']} ({status}); {item['evidence']}"
        )
    if len(links) == 2:
        links.append("No dataset-paper links recorded.")
    path = out / "dataset_paper_linking_report.md"
    path.write_text("\n".join(links), encoding="utf-8")
    paths["links"] = str(path)

    path = out / "low_confidence_labels.json"
    path.write_text(
        json.dumps(summary["low_confidence_labels"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["low_confidence"] = str(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neural_search.reports.corpus_report")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    records = load_normalized_records(args.input)
    paths = write_corpus_reports(summarize_corpus(records), args.out)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
