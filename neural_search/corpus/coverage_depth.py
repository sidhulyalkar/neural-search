"""Build reviewed coverage-depth corpus pack and reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from neural_search.corpus.convert_demo_seed import convert_dataset_fixture
from neural_search.normalized import write_jsonl
from neural_search.schemas import NormalizedDatasetRecord

DEFAULT_SEED_PATH = Path("data/seed/coverage_depth_datasets.yaml")
DEFAULT_OUT_DIR = Path("data/corpus/normalized/coverage_depth")
DEFAULT_PREFIX = "coverage_depth"


def _load_seed(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"coverage seed root must be a mapping: {path}")
    datasets = payload.get("datasets", [])
    targets = payload.get("source_targets", [])
    if not isinstance(datasets, list):
        raise ValueError("coverage seed must contain a datasets list")
    if not isinstance(targets, list):
        raise ValueError("coverage seed source_targets must be a list")
    return {"datasets": datasets, "source_targets": targets}


def build_coverage_depth_records(
    seed_path: str | Path = DEFAULT_SEED_PATH,
) -> tuple[list[NormalizedDatasetRecord], list[dict[str, Any]]]:
    """Convert reviewed coverage-depth seed YAML into normalized records."""

    payload = _load_seed(seed_path)
    records = [
        convert_dataset_fixture(
            fixture,
            paper_lookup={},
            raw_payload_path=str(seed_path),
        )
        for fixture in payload["datasets"]
        if isinstance(fixture, Mapping)
    ]
    return records, list(payload["source_targets"])


def _label_counter(records: list[NormalizedDatasetRecord], field: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        labels = getattr(record, field)
        for label in labels:
            counts[label.label] += 1
    return counts


def _tag_counter(records: list[NormalizedDatasetRecord]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        tags = []
        if record.description:
            # The reviewed YAML is the source of truth for tags; retain a small
            # report-friendly fallback by source family when tags are absent.
            tags.append(record.source)
        for tag in tags:
            counts[tag] += 1
    return counts


def render_coverage_depth_report(
    records: list[NormalizedDatasetRecord],
    source_targets: list[dict[str, Any]],
) -> str:
    """Render a compact markdown report for the reviewed coverage pack."""

    lines = [
        "# Coverage Depth Corpus Pack",
        "",
        f"- Reviewed dataset records: {len(records)}",
        f"- Open source-target gaps: {len(source_targets)}",
        "",
        "## Species",
        "",
    ]
    for label, count in sorted(_label_counter(records, "species").items()):
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## Modalities", ""])
    for label, count in sorted(_label_counter(records, "modalities").items()):
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## Brain Regions", ""])
    for label, count in sorted(_label_counter(records, "brain_regions").items()):
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## Analysis Affordances", ""])
    affordances: Counter[str] = Counter()
    for record in records:
        for affordance in record.analysis_affordances:
            if affordance.support_level in {"high", "medium"}:
                affordances[affordance.analysis_id] += 1
    for affordance, count in sorted(affordances.items()):
        lines.append(f"- {affordance}: {count}")
    lines.extend(["", "## Source Targets", ""])
    if not source_targets:
        lines.append("- No open source targets.")
    for target in source_targets:
        lines.append(f"### {target.get('id', 'unknown')}")
        lines.append(f"- Priority: {target.get('priority', 'unknown')}")
        lines.append(f"- Rationale: {target.get('rationale', '')}")
        candidate_sources = target.get("candidate_sources") or []
        if candidate_sources:
            lines.append("- Candidate sources:")
            for source in candidate_sources:
                lines.append(f"  - {source}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_coverage_depth_pack(
    seed_path: str | Path = DEFAULT_SEED_PATH,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    *,
    prefix: str = DEFAULT_PREFIX,
) -> dict[str, Path]:
    """Write normalized JSONL records, source targets, and coverage report."""

    records, source_targets = build_coverage_depth_records(seed_path)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = write_jsonl(records, output / f"{prefix}.records.jsonl")
    targets_path = output / f"{prefix}.source_targets.json"
    targets_path.write_text(
        json.dumps(source_targets, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path = output / f"{prefix}_report.md"
    report_path.write_text(
        render_coverage_depth_report(records, source_targets),
        encoding="utf-8",
    )
    return {"records": records_path, "source_targets": targets_path, "report": report_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build coverage-depth corpus pack.")
    parser.add_argument("--seed", default=str(DEFAULT_SEED_PATH))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args(argv)
    paths = write_coverage_depth_pack(args.seed, args.out_dir, prefix=args.prefix)
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
