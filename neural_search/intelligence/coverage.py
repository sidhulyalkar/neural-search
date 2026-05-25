"""Coverage-driven planning for corpus and benchmark expansion."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural_search.awareness.scoring import infer_dataset_awareness
from neural_search.awareness.taxonomy import DATA_FORMS, infer_query_awareness
from neural_search.evaluation.run_benchmark import load_benchmark_queries
from neural_search.normalized import load_normalized_records
from neural_search.schemas import NormalizedDatasetRecord


@dataclass(frozen=True)
class CoverageGap:
    """One data-form coverage gap across corpus records and benchmark queries."""

    data_form: str
    label: str
    corpus_count: int
    benchmark_query_count: int
    target_corpus_count: int
    target_benchmark_query_count: int
    priority: str
    recommended_queries: tuple[str, ...]
    recommended_sources: tuple[str, ...]

    def model_dump(self) -> dict[str, Any]:
        return {
            "data_form": self.data_form,
            "label": self.label,
            "corpus_count": self.corpus_count,
            "benchmark_query_count": self.benchmark_query_count,
            "target_corpus_count": self.target_corpus_count,
            "target_benchmark_query_count": self.target_benchmark_query_count,
            "priority": self.priority,
            "recommended_queries": list(self.recommended_queries),
            "recommended_sources": list(self.recommended_sources),
        }


@dataclass(frozen=True)
class SearchCoveragePlan:
    """Corpus and benchmark coverage summary for search intelligence work."""

    dataset_count: int
    benchmark_query_count: int
    data_form_counts: dict[str, int]
    benchmark_data_form_counts: dict[str, int]
    gaps: tuple[CoverageGap, ...]

    def model_dump(self) -> dict[str, Any]:
        return {
            "dataset_count": self.dataset_count,
            "benchmark_query_count": self.benchmark_query_count,
            "data_form_counts": dict(self.data_form_counts),
            "benchmark_data_form_counts": dict(self.benchmark_data_form_counts),
            "gaps": [gap.model_dump() for gap in self.gaps],
        }


SOURCE_HINTS: dict[str, tuple[str, ...]] = {
    "eeg_meg": ("OpenNeuro", "clinical EDF/BIDS repositories"),
    "mri": ("OpenNeuro", "Human Connectome Project"),
    "intracranial_human_ephys": ("OpenNeuro iEEG", "DANDI"),
    "extracellular_ephys": ("DANDI", "IBL", "Allen Brain Observatory"),
    "intracellular_ephys": ("DANDI", "NeuroElectro-style curated tables"),
    "optical_imaging": ("DANDI", "Allen Brain Observatory"),
    "fiber_photometry": ("DANDI",),
    "behavior": ("DANDI", "OpenNeuro", "manual landmark corpus"),
    "clinical": ("OpenNeuro", "clinical BIDS/EDF repositories"),
    "connectomics": ("MICrONS", "DANDI", "NeuroMorpho"),
    "molecular": ("cellxgene", "Allen Brain Cell Atlas"),
    "computational_model": ("ModelDB", "Open Source Brain"),
}


def _example_queries(form_id: str) -> tuple[str, ...]:
    form = DATA_FORMS[form_id]
    analysis = form.analysis_families[0] if form.analysis_families else "reuse"
    alias = form.aliases[0] if form.aliases else form.label
    return (
        f"{alias} datasets for {analysis.replace('_', ' ')}",
        f"{form.label} with behavior and reusable metadata",
        f"{form.label} excluding unrelated modalities",
    )


def _priority(corpus_gap: int, benchmark_gap: int) -> str:
    score = corpus_gap * 2 + benchmark_gap
    if score >= 6:
        return "critical"
    if score >= 3:
        return "high"
    if score > 0:
        return "medium"
    return "covered"


def _dataset_records(path: str | Path) -> list[NormalizedDatasetRecord]:
    return [
        record
        for record in load_normalized_records(path)
        if isinstance(record, NormalizedDatasetRecord)
    ]


def _benchmark_counts(path: str | Path | None) -> tuple[int, Counter[str]]:
    counts: Counter[str] = Counter()
    if path is None or not Path(path).exists():
        return 0, counts
    queries = load_benchmark_queries(Path(path))
    for query in queries:
        awareness = infer_query_awareness(query.query)
        counts.update(awareness.requested_data_forms)
    return len(queries), counts


def build_search_coverage_plan(
    records_path: str | Path,
    benchmark_path: str | Path | None = None,
    *,
    target_corpus_count: int = 3,
    target_benchmark_query_count: int = 2,
) -> SearchCoveragePlan:
    """Build a coverage plan from normalized records and optional benchmark queries."""

    datasets = _dataset_records(records_path)
    data_form_counts: Counter[str] = Counter()
    for dataset in datasets:
        data_form_counts.update(infer_dataset_awareness(dataset).data_forms)

    benchmark_query_count, benchmark_counts = _benchmark_counts(benchmark_path)
    gaps: list[CoverageGap] = []
    for form_id, form in DATA_FORMS.items():
        corpus_count = int(data_form_counts.get(form_id, 0))
        query_count = int(benchmark_counts.get(form_id, 0))
        corpus_gap = max(target_corpus_count - corpus_count, 0)
        benchmark_gap = max(target_benchmark_query_count - query_count, 0)
        priority = _priority(corpus_gap, benchmark_gap)
        if priority == "covered":
            continue
        gaps.append(
            CoverageGap(
                data_form=form_id,
                label=form.label,
                corpus_count=corpus_count,
                benchmark_query_count=query_count,
                target_corpus_count=target_corpus_count,
                target_benchmark_query_count=target_benchmark_query_count,
                priority=priority,
                recommended_queries=_example_queries(form_id),
                recommended_sources=SOURCE_HINTS.get(form_id, ("manual curation",)),
            )
        )

    priority_order = {"critical": 0, "high": 1, "medium": 2}
    gaps.sort(
        key=lambda gap: (
            priority_order.get(gap.priority, 9),
            gap.corpus_count,
            gap.benchmark_query_count,
            gap.data_form,
        )
    )
    return SearchCoveragePlan(
        dataset_count=len(datasets),
        benchmark_query_count=benchmark_query_count,
        data_form_counts=dict(sorted(data_form_counts.items())),
        benchmark_data_form_counts=dict(sorted(benchmark_counts.items())),
        gaps=tuple(gaps),
    )


def _markdown(plan: SearchCoveragePlan) -> str:
    lines = [
        "# Search Intelligence Coverage Plan",
        "",
        f"- Datasets inspected: {plan.dataset_count}",
        f"- Benchmark queries inspected: {plan.benchmark_query_count}",
        f"- Open gaps: {len(plan.gaps)}",
        "",
        "## Highest Priority Gaps",
        "",
        "| Priority | Data Form | Corpus | Benchmarks | Recommended Sources |",
        "|---|---|---:|---:|---|",
    ]
    for gap in plan.gaps:
        lines.append(
            "| "
            + " | ".join(
                [
                    gap.priority,
                    gap.label,
                    f"{gap.corpus_count}/{gap.target_corpus_count}",
                    f"{gap.benchmark_query_count}/{gap.target_benchmark_query_count}",
                    ", ".join(gap.recommended_sources),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Query Seeds", ""])
    for gap in plan.gaps:
        lines.append(f"### {gap.label}")
        for query in gap.recommended_queries:
            lines.append(f"- {query}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_search_coverage_plan(
    plan: SearchCoveragePlan,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write JSON and Markdown coverage reports."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "search_coverage_plan.json"
    md_path = out / "search_coverage_plan.md"
    json_path.write_text(
        json.dumps(plan.model_dump(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_path.write_text(_markdown(plan), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate coverage-driven search intelligence reports."
    )
    parser.add_argument("--records", required=True, help="Normalized records path or directory.")
    parser.add_argument("--benchmark", help="Optional benchmark query YAML path.")
    parser.add_argument("--out", required=True, help="Output report directory.")
    parser.add_argument("--target-corpus-count", type=int, default=3)
    parser.add_argument("--target-benchmark-query-count", type=int, default=2)
    args = parser.parse_args(argv)

    plan = build_search_coverage_plan(
        args.records,
        args.benchmark,
        target_corpus_count=args.target_corpus_count,
        target_benchmark_query_count=args.target_benchmark_query_count,
    )
    paths = write_search_coverage_plan(plan, args.out)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
