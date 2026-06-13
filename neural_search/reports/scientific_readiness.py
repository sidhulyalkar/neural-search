"""Scientific readiness audit for corpus, graph, evaluation, and agent workflows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neural_search.graph.schema import KnowledgeGraph, read_graph_json
from neural_search.normalized import NormalizedRecord, load_normalized_records
from neural_search.schemas import NormalizedDatasetRecord, NormalizedPaperRecord
from neural_search.source_quality import summarize_source_quality
from neural_search.species import get_species_profile

DEFAULT_CORPUS_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "corpus"
    / "normalized"
    / "real_v07.records.jsonl"
)
DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "corpus"
    / "normalized"
    / "real_v07.datasets.jsonl"
)
DEFAULT_PAPER_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "corpus"
    / "normalized"
    / "real_v07.papers.jsonl"
)
DEFAULT_GRAPH_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "graph"
    / "neural_search_graph.real_v07.json"
)

LABEL_FIELDS = (
    "species",
    "modalities",
    "brain_regions",
    "tasks",
    "behavioral_events",
    "data_standards",
    "file_formats",
)


def _load_default_records() -> list[NormalizedRecord]:
    if DEFAULT_CORPUS_PATH.exists():
        return load_normalized_records(DEFAULT_CORPUS_PATH)
    records: list[NormalizedRecord] = []
    for path in (DEFAULT_DATASET_PATH, DEFAULT_PAPER_PATH):
        records.extend(load_normalized_records(path))
    return records


def _labels(record: NormalizedDatasetRecord, field: str) -> list[str]:
    return [label.label for label in getattr(record, field, [])]


def _counter(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(value for value in values if value).items()))


def _dataset_records(records: Iterable[NormalizedRecord]) -> list[NormalizedDatasetRecord]:
    return [record for record in records if isinstance(record, NormalizedDatasetRecord)]


def _paper_records(records: Iterable[NormalizedRecord]) -> list[NormalizedPaperRecord]:
    return [record for record in records if isinstance(record, NormalizedPaperRecord)]


def _corpus_section(records: list[NormalizedRecord]) -> dict[str, Any]:
    datasets = _dataset_records(records)
    papers = _paper_records(records)
    label_counts = {
        field: _counter(label for dataset in datasets for label in _labels(dataset, field))
        for field in LABEL_FIELDS
    }
    canonical_species = Counter()
    taxon_groups = Counter()
    for dataset in datasets:
        for value in _labels(dataset, "species"):
            profile = get_species_profile(value)
            if profile is None:
                canonical_species[value] += 1
                continue
            canonical_species[profile.species_id] += 1
            taxon_groups.update(profile.taxon_groups)
    linked_dataset_count = sum(1 for dataset in datasets if dataset.linked_papers)
    paper_link_count = sum(len(dataset.linked_papers) for dataset in datasets)
    return {
        "total_records": len(records),
        "dataset_records": len(datasets),
        "paper_records": len(papers),
        "source_counts": _counter(record.source for record in records),
        "label_counts": label_counts,
        "canonical_species_counts": dict(sorted(canonical_species.items())),
        "taxon_group_counts": dict(sorted(taxon_groups.items())),
        "dataset_paper_link_density": round(
            paper_link_count / max(len(datasets), 1),
            3,
        ),
        "datasets_with_linked_papers": linked_dataset_count,
        "datasets_with_analysis_affordances": sum(
            1 for dataset in datasets if dataset.analysis_affordances
        ),
        "datasets_with_standard": sum(1 for dataset in datasets if dataset.data_standards),
    }


def _graph_section(graph: KnowledgeGraph | None) -> dict[str, Any]:
    if graph is None:
        return {
            "available": False,
            "node_count": 0,
            "edge_count": 0,
            "requirement_edge_count": 0,
            "species_context_edge_count": 0,
            "dataset_nodes": 0,
            "paper_nodes": 0,
        }
    requirement_edges = [
        edge for edge in graph.edges.values() if edge.edge_type.startswith("analysis_requires_")
    ]
    species_edges = [
        edge
        for edge in graph.edges.values()
        if edge.edge_type
        in {
            "species_in_taxon_group",
            "species_has_model_role",
            "species_has_animal_type",
        }
    ]
    return {
        "available": True,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "requirement_edge_count": len(requirement_edges),
        "species_context_edge_count": len(species_edges),
        "dataset_nodes": sum(1 for node in graph.nodes.values() if node.node_type == "dataset"),
        "paper_nodes": sum(1 for node in graph.nodes.values() if node.node_type == "paper"),
        "species_nodes": sum(1 for node in graph.nodes.values() if node.node_type == "species"),
        "taxon_group_nodes": sum(
            1 for node in graph.nodes.values() if node.node_type == "taxon_group"
        ),
    }


def _load_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    target = Path(path)
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def _evaluation_section(
    benchmark_reports: Iterable[str | Path] = (),
    calibration_report: str | Path | None = None,
) -> dict[str, Any]:
    reports = [_load_json(path) for path in benchmark_reports]
    reports = [report for report in reports if report is not None]
    calibration = _load_json(calibration_report)
    return {
        "benchmark_report_count": len(reports),
        "benchmark_suites": [str(report.get("suite", "unknown")) for report in reports],
        "mean_precision_at_5": [
            report.get("mean_precision_at_5")
            for report in reports
            if "mean_precision_at_5" in report
        ],
        "mean_label_recall_at_10": [
            report.get("mean_label_recall_at_10")
            for report in reports
            if "mean_label_recall_at_10" in report
        ],
        "calibration_available": calibration is not None,
        "calibration_summary": calibration.get("summary", calibration) if calibration else {},
    }


def _agent_section() -> dict[str, Any]:
    return {
        "dataset_discovery_workflow": True,
        "benchmark_audit_workflow": True,
        "readiness_report": True,
        "gap_analysis_endpoint": False,
        "experiment_plan_endpoint": False,
        "cohort_builder_endpoint": False,
        "paper_dataset_linking_endpoint": False,
    }


def _warnings(corpus: dict[str, Any], graph: dict[str, Any], evaluation: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if corpus["dataset_records"] < 50:
        warnings.append("Corpus is still small for broad neuroscience ranking; expand real datasets.")
    if len(corpus["canonical_species_counts"]) < 6:
        warnings.append("Species coverage is sparse; add more model organisms and human/NHP records.")
    if corpus["dataset_paper_link_density"] < 0.5:
        warnings.append("Dataset-paper linking is weak; prioritize provenance links.")
    if corpus["datasets_with_standard"] < corpus["dataset_records"]:
        warnings.append("Some datasets lack explicit data standards.")
    if not graph["available"]:
        warnings.append("Graph artifact is unavailable; graph explanations and association paths are limited.")
    elif graph["requirement_edge_count"] == 0:
        warnings.append("Graph lacks analysis requirement edges.")
    if graph.get("species_context_edge_count", 0) == 0:
        warnings.append("Graph lacks species/taxon context edges.")
    if evaluation["benchmark_report_count"] == 0:
        warnings.append("No benchmark result reports supplied to readiness audit.")
    if not evaluation["calibration_available"]:
        warnings.append("No calibration report supplied; score reliability is not audited here.")
    return warnings


def _source_quality_warnings(source_quality: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    trust_counts = source_quality.get("trust_level_counts", {})
    if trust_counts.get("unknown", 0):
        warnings.append("Some records use sources without registered quality profiles.")
    if trust_counts.get("low", 0):
        warnings.append("Some records come from low-trust fixture/demo sources.")
    if float(source_quality.get("mean_quality_score", 0.0)) < 0.7:
        warnings.append("Mean source quality score is below promotion-ready threshold.")
    return warnings


def build_scientific_readiness_report(
    records: list[NormalizedRecord] | None = None,
    *,
    corpus_path: str | Path | None = None,
    graph_path: str | Path | None = DEFAULT_GRAPH_PATH,
    benchmark_reports: Iterable[str | Path] = (),
    calibration_report: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic v0.8 scientific readiness report."""

    loaded_records = (
        records
        if records is not None
        else load_normalized_records(corpus_path) if corpus_path else _load_default_records()
    )
    graph = None
    if graph_path and Path(graph_path).exists():
        graph = read_graph_json(graph_path)
    corpus = _corpus_section(loaded_records)
    graph_summary = _graph_section(graph)
    evaluation = _evaluation_section(benchmark_reports, calibration_report)
    source_quality = summarize_source_quality(loaded_records)
    report = {
        "report": "scientific_readiness",
        "version": "v0.8.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus": corpus,
        "graph": graph_summary,
        "evaluation": evaluation,
        "source_quality": source_quality,
        "agent_readiness": _agent_section(),
        "warnings": [
            *_warnings(corpus, graph_summary, evaluation),
            *_source_quality_warnings(source_quality),
        ],
    }
    return report


def render_scientific_readiness_markdown(report: dict[str, Any]) -> str:
    """Render the readiness report as concise Markdown."""

    corpus = report["corpus"]
    graph = report["graph"]
    evaluation = report["evaluation"]
    source_quality = report.get("source_quality", {})
    lines = [
        "# Scientific Readiness Report",
        "",
        f"Version: {report['version']}",
        f"Generated at: {report['generated_at']}",
        "",
        "## Corpus",
        "",
        f"- Records: {corpus['total_records']}",
        f"- Datasets: {corpus['dataset_records']}",
        f"- Papers: {corpus['paper_records']}",
        f"- Sources: {', '.join(corpus['source_counts']) or 'none'}",
        f"- Canonical species: {', '.join(corpus['canonical_species_counts']) or 'none'}",
        f"- Paper-link density: {corpus['dataset_paper_link_density']}",
        "",
        "## Graph",
        "",
        f"- Available: {graph['available']}",
        f"- Nodes: {graph['node_count']}",
        f"- Edges: {graph['edge_count']}",
        f"- Requirement edges: {graph['requirement_edge_count']}",
        f"- Species context edges: {graph['species_context_edge_count']}",
        "",
        "## Evaluation",
        "",
        f"- Benchmark reports: {evaluation['benchmark_report_count']}",
        f"- Calibration available: {evaluation['calibration_available']}",
        "",
        "## Source Quality",
        "",
        f"- Mean quality score: {source_quality.get('mean_quality_score', 0.0)}",
        f"- Trust levels: {source_quality.get('trust_level_counts', {})}",
        f"- Records with warnings: {source_quality.get('warning_count', 0)}",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in report["warnings"])
    if not report["warnings"]:
        lines.append("- No readiness warnings.")
    return "\n".join(lines) + "\n"


def write_scientific_readiness_reports(
    report: dict[str, Any],
    out_dir: str | Path,
) -> dict[str, str]:
    """Write JSON and Markdown scientific readiness reports."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "scientific_readiness_report.json"
    md_path = output / "scientific_readiness_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_scientific_readiness_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m neural_search.reports.scientific_readiness"
    )
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--benchmark-report", action="append", default=[])
    parser.add_argument("--calibration-report", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    report = build_scientific_readiness_report(
        corpus_path=args.corpus,
        graph_path=args.graph,
        benchmark_reports=args.benchmark_report,
        calibration_report=args.calibration_report,
    )
    print(json.dumps(write_scientific_readiness_reports(report, args.out), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
