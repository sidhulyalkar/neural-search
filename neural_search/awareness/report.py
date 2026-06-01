"""Corpus reports for broad neuroscience data-form awareness."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from neural_search.awareness.scoring import infer_dataset_awareness
from neural_search.normalized import load_normalized_records
from neural_search.schemas import NormalizedDatasetRecord


def build_awareness_report(records_path: str | Path) -> dict[str, Any]:
    """Summarize data-form and analysis-family coverage for normalized records."""

    records = load_normalized_records(records_path)
    datasets_by_id = {
        record.dataset_id: record
        for record in records
        if isinstance(record, NormalizedDatasetRecord)
    }
    datasets = [datasets_by_id[key] for key in sorted(datasets_by_id)]
    awareness = [infer_dataset_awareness(record) for record in datasets]
    form_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    scale_counts: Counter[str] = Counter()
    analysis_counts: Counter[str] = Counter()
    species_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    for item in awareness:
        form_counts.update(item.data_forms)
        family_counts.update(item.families)
        scale_counts.update(item.scales)
        analysis_counts.update(item.analysis_families)
        species_counts.update(item.species)
        missing_counts.update(item.missing_requirements)

    underrepresented_forms = [
        form
        for form in [
            "intracellular_ephys",
            "mri",
            "eeg_meg",
            "connectomics",
            "molecular",
            "computational_model",
        ]
        if form_counts[form] == 0
    ]
    return {
        "record_path": str(records_path),
        "dataset_count": len(datasets),
        "data_form_counts": dict(form_counts),
        "family_counts": dict(family_counts),
        "scale_counts": dict(scale_counts),
        "analysis_family_counts": dict(analysis_counts),
        "species_counts": dict(species_counts),
        "missing_requirement_counts": dict(missing_counts),
        "underrepresented_data_forms": underrepresented_forms,
        "datasets": [
            {
                "dataset_id": item.dataset_id,
                "data_forms": list(item.data_forms),
                "families": list(item.families),
                "scales": list(item.scales),
                "analysis_families": list(item.analysis_families),
                "missing_requirements": list(item.missing_requirements),
            }
            for item in awareness
        ],
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Neuroscience Awareness Report",
        "",
        f"- Records: `{report['record_path']}`",
        f"- Datasets: {report['dataset_count']}",
        "",
        "## Data Forms",
        "",
    ]
    if report["data_form_counts"]:
        for form, count in sorted(report["data_form_counts"].items()):
            lines.append(f"- `{form}`: {count}")
    else:
        lines.append("- No data forms detected.")
    lines.extend(["", "## Analysis Families", ""])
    for family, count in sorted(report["analysis_family_counts"].items()):
        lines.append(f"- `{family}`: {count}")
    lines.extend(["", "## Underrepresented Data Forms", ""])
    if report["underrepresented_data_forms"]:
        for form in report["underrepresented_data_forms"]:
            lines.append(f"- `{form}`")
    else:
        lines.append("- None among tracked priority forms.")
    lines.extend(["", "## Dataset Awareness", ""])
    for dataset in report["datasets"][:50]:
        lines.append(
            f"- `{dataset['dataset_id']}`: forms={dataset['data_forms']}, "
            f"analysis={dataset['analysis_families']}"
        )
    return "\n".join(lines) + "\n"


def write_awareness_report(records_path: str | Path, out_dir: str | Path) -> dict[str, str]:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = build_awareness_report(records_path)
    json_path = output / "neuroscience_awareness_report.json"
    md_path = output / "neuroscience_awareness_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate neuroscience awareness reports.")
    parser.add_argument("--records", required=True, help="Normalized record JSONL or directory")
    parser.add_argument("--out", required=True, help="Output report directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = write_awareness_report(args.records, args.out)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
