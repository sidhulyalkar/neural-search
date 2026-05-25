"""Corpus and knowledge-base expansion planning."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural_search.awareness.taxonomy import DATA_FORMS
from neural_search.graph.schema import read_graph_json
from neural_search.intelligence.coverage import (
    SOURCE_HINTS,
    CoverageGap,
    build_search_coverage_plan,
)
from neural_search.normalized import load_normalized_records
from neural_search.schemas import NormalizedDatasetRecord, NormalizedPaperRecord

REQUIRED_GRAPH_NODE_TYPES = (
    "dataset",
    "paper",
    "modality",
    "task",
    "species",
    "analysis_affordance",
    "data_standard",
)

REQUIRED_GRAPH_EDGE_TYPES = (
    "dataset_has_modality",
    "dataset_has_task",
    "dataset_has_species",
    "dataset_supports_analysis",
    "dataset_uses_standard",
    "paper_mentions_dataset",
    "paper_uses_dataset",
)


@dataclass(frozen=True)
class ExpansionTask:
    """One actionable corpus or knowledge-base expansion task."""

    task_id: str
    title: str
    priority: str
    track: str
    rationale: str
    data_form: str | None = None
    target_sources: tuple[str, ...] = ()
    target_concepts: tuple[str, ...] = ()
    target_graph_edges: tuple[str, ...] = ()
    target_benchmark_queries: int = 0
    acceptance_checks: tuple[str, ...] = ()

    def model_dump(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "priority": self.priority,
            "track": self.track,
            "rationale": self.rationale,
            "data_form": self.data_form,
            "target_sources": list(self.target_sources),
            "target_concepts": list(self.target_concepts),
            "target_graph_edges": list(self.target_graph_edges),
            "target_benchmark_queries": self.target_benchmark_queries,
            "acceptance_checks": list(self.acceptance_checks),
        }


@dataclass(frozen=True)
class CorpusKnowledgeExpansionPlan:
    """Regeneratable plan for corpus and graph knowledge-base expansion."""

    dataset_count: int
    paper_count: int
    source_counts: dict[str, int]
    graph_node_type_counts: dict[str, int]
    graph_edge_type_counts: dict[str, int]
    data_form_counts: dict[str, int]
    benchmark_data_form_counts: dict[str, int]
    tasks: tuple[ExpansionTask, ...]

    def model_dump(self) -> dict[str, Any]:
        return {
            "dataset_count": self.dataset_count,
            "paper_count": self.paper_count,
            "source_counts": dict(self.source_counts),
            "graph_node_type_counts": dict(self.graph_node_type_counts),
            "graph_edge_type_counts": dict(self.graph_edge_type_counts),
            "data_form_counts": dict(self.data_form_counts),
            "benchmark_data_form_counts": dict(self.benchmark_data_form_counts),
            "tasks": [task.model_dump() for task in self.tasks],
        }


def _safe_id(value: str) -> str:
    return "_".join(value.casefold().replace("-", "_").split())


def _records_summary(records_path: str | Path) -> tuple[int, int, dict[str, int]]:
    records = load_normalized_records(records_path)
    source_counts: Counter[str] = Counter()
    dataset_count = 0
    paper_count = 0
    for record in records:
        if isinstance(record, NormalizedDatasetRecord):
            dataset_count += 1
            source_counts[record.source] += 1
        elif isinstance(record, NormalizedPaperRecord):
            paper_count += 1
            source_counts[record.source] += 1
    return dataset_count, paper_count, dict(sorted(source_counts.items()))


def _graph_summary(graph_path: str | Path | None) -> tuple[dict[str, int], dict[str, int]]:
    if graph_path is None:
        return {}, {}
    path = Path(graph_path)
    if not path.exists():
        return {}, {}
    graph = read_graph_json(path)
    node_counts = Counter(node.node_type for node in graph.nodes.values())
    edge_counts = Counter(edge.edge_type for edge in graph.edges.values())
    return dict(sorted(node_counts.items())), dict(sorted(edge_counts.items()))


def _concept_targets(gap: CoverageGap) -> tuple[str, ...]:
    data_form = DATA_FORMS[gap.data_form]
    concepts = (
        *data_form.modalities,
        *data_form.standards,
        *data_form.analysis_families,
        *data_form.required_signals,
        *data_form.complementary_forms,
    )
    seen: set[str] = set()
    ordered: list[str] = []
    for concept in concepts:
        if concept not in seen:
            seen.add(concept)
            ordered.append(concept)
    return tuple(ordered[:12])


def _gap_task(gap: CoverageGap) -> ExpansionTask:
    missing_queries = max(
        gap.target_benchmark_query_count - gap.benchmark_query_count,
        0,
    )
    return ExpansionTask(
        task_id=f"task24_expand_{_safe_id(gap.data_form)}",
        title=f"Expand {gap.label} corpus and benchmark coverage",
        priority=gap.priority,
        track="corpus_and_benchmark",
        data_form=gap.data_form,
        rationale=(
            f"{gap.label} has {gap.corpus_count}/{gap.target_corpus_count} "
            f"target corpus records and {gap.benchmark_query_count}/"
            f"{gap.target_benchmark_query_count} target benchmark queries."
        ),
        target_sources=SOURCE_HINTS.get(gap.data_form, ("manual curation",)),
        target_concepts=_concept_targets(gap),
        target_graph_edges=(
            "dataset_has_modality",
            "dataset_supports_analysis",
            "dataset_uses_standard",
        ),
        target_benchmark_queries=missing_queries,
        acceptance_checks=(
            "normalized records include source, source_id, title, labels, and provenance",
            "graph artifact contains modality, analysis, and standard edges for new records",
            "benchmark seeds include reviewed expected IDs or explicit review_required notes",
        ),
    )


def _missing_graph_tasks(
    node_counts: dict[str, int],
    edge_counts: dict[str, int],
) -> list[ExpansionTask]:
    tasks: list[ExpansionTask] = []
    missing_nodes = tuple(
        node_type for node_type in REQUIRED_GRAPH_NODE_TYPES if node_counts.get(node_type, 0) == 0
    )
    if missing_nodes:
        tasks.append(
            ExpansionTask(
                task_id="task25_fill_graph_node_types",
                title="Fill missing knowledge graph node types",
                priority="high",
                track="knowledge_graph",
                rationale=(
                    "The graph is missing required scientific node categories "
                    "needed for cross-source reasoning."
                ),
                target_concepts=missing_nodes,
                acceptance_checks=(
                    "graph reports list nonzero counts for required node types where data exists",
                    "placeholder nodes are clearly marked and resolved when normalized records arrive",
                ),
            )
        )

    missing_edges = tuple(
        edge_type for edge_type in REQUIRED_GRAPH_EDGE_TYPES if edge_counts.get(edge_type, 0) == 0
    )
    if missing_edges:
        tasks.append(
            ExpansionTask(
                task_id="task25_fill_graph_edge_types",
                title="Fill missing knowledge graph relationship types",
                priority="high",
                track="knowledge_graph",
                rationale=(
                    "Search intelligence needs explicit dataset-paper-concept "
                    "relationships rather than only lexical labels."
                ),
                target_graph_edges=missing_edges,
                acceptance_checks=(
                    "graph artifact validates with all edge endpoints resolved",
                    "graph reports expose edge coverage by relationship type",
                ),
            )
        )
    return tasks


def _source_balance_tasks(source_counts: dict[str, int]) -> list[ExpansionTask]:
    tasks: list[ExpansionTask] = []
    required_sources = {
        "dandi": "NWB animal physiology and imaging",
        "openneuro": "BIDS human neuroimaging and electrophysiology",
        "openalex": "paper and citation context",
        "modeldb": "computational model coverage",
        "cellxgene": "single-cell molecular coverage",
        "microns": "connectomics coverage",
    }
    missing = tuple(source for source in required_sources if source_counts.get(source, 0) == 0)
    if missing:
        tasks.append(
            ExpansionTask(
                task_id="task26_add_source_families",
                title="Add missing source families to corpus intake",
                priority="critical",
                track="source_intake",
                rationale=(
                    "A general neuroscience search corpus needs public-source "
                    "coverage across physiology, imaging, literature, models, "
                    "molecular data, and connectomics."
                ),
                target_sources=missing,
                target_concepts=tuple(required_sources[source] for source in missing),
                acceptance_checks=(
                    "each source family has fixture-backed normalized records",
                    "network-backed ingestion remains optional outside CI",
                    "source counts appear in expansion and corpus reports",
                ),
            )
        )
    return tasks


def build_corpus_knowledge_expansion_plan(
    records_path: str | Path,
    benchmark_path: str | Path | None = None,
    *,
    graph_path: str | Path | None = None,
    target_corpus_count: int = 5,
    target_benchmark_query_count: int = 3,
) -> CorpusKnowledgeExpansionPlan:
    """Build corpus and knowledge-base expansion tasks from local artifacts."""

    coverage = build_search_coverage_plan(
        records_path,
        benchmark_path,
        target_corpus_count=target_corpus_count,
        target_benchmark_query_count=target_benchmark_query_count,
    )
    dataset_count, paper_count, source_counts = _records_summary(records_path)
    node_counts, edge_counts = _graph_summary(graph_path)
    tasks = [_gap_task(gap) for gap in coverage.gaps]
    tasks.extend(_missing_graph_tasks(node_counts, edge_counts))
    tasks.extend(_source_balance_tasks(source_counts))

    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    tasks.sort(
        key=lambda task: (
            priority_order.get(task.priority, 9),
            task.track,
            task.task_id,
        )
    )
    return CorpusKnowledgeExpansionPlan(
        dataset_count=dataset_count,
        paper_count=paper_count,
        source_counts=source_counts,
        graph_node_type_counts=node_counts,
        graph_edge_type_counts=edge_counts,
        data_form_counts=coverage.data_form_counts,
        benchmark_data_form_counts=coverage.benchmark_data_form_counts,
        tasks=tuple(tasks),
    )


def _markdown(plan: CorpusKnowledgeExpansionPlan) -> str:
    lines = [
        "# Corpus and Knowledge Base Expansion Plan",
        "",
        f"- Dataset records: {plan.dataset_count}",
        f"- Paper records: {plan.paper_count}",
        f"- Expansion tasks: {len(plan.tasks)}",
        "",
        "## Source Counts",
        "",
    ]
    if plan.source_counts:
        for source, count in plan.source_counts.items():
            lines.append(f"- {source}: {count}")
    else:
        lines.append("- No normalized source records found.")

    lines.extend(
        [
            "",
            "## Graph Coverage",
            "",
            "### Node Types",
            "",
        ]
    )
    if plan.graph_node_type_counts:
        for node_type, count in plan.graph_node_type_counts.items():
            lines.append(f"- {node_type}: {count}")
    else:
        lines.append("- No graph artifact supplied.")

    lines.extend(["", "### Edge Types", ""])
    if plan.graph_edge_type_counts:
        for edge_type, count in plan.graph_edge_type_counts.items():
            lines.append(f"- {edge_type}: {count}")
    else:
        lines.append("- No graph artifact supplied.")

    lines.extend(
        [
            "",
            "## Expansion Tasks",
            "",
            "| Priority | Task | Track | Targets |",
            "|---|---|---|---|",
        ]
    )
    for task in plan.tasks:
        targets = ", ".join(
            [*task.target_sources[:4], *task.target_concepts[:4], *task.target_graph_edges[:4]]
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    task.priority,
                    task.title,
                    task.track,
                    targets or task.data_form or "general",
                ]
            )
            + " |"
        )

    lines.extend(["", "## Acceptance Checks", ""])
    for task in plan.tasks:
        lines.append(f"### {task.task_id}")
        lines.append(task.rationale)
        for check in task.acceptance_checks:
            lines.append(f"- {check}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_corpus_knowledge_expansion_plan(
    plan: CorpusKnowledgeExpansionPlan,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write JSON and Markdown corpus/knowledge expansion reports."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "corpus_knowledge_expansion_plan.json"
    md_path = out / "corpus_knowledge_expansion_plan.md"
    json_path.write_text(
        json.dumps(plan.model_dump(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_path.write_text(_markdown(plan), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate corpus and knowledge-base expansion tasks."
    )
    parser.add_argument("--records", required=True, help="Normalized records path or directory.")
    parser.add_argument("--benchmark", help="Optional benchmark query YAML path.")
    parser.add_argument("--graph", help="Optional knowledge graph JSON artifact.")
    parser.add_argument("--out", required=True, help="Output report directory.")
    parser.add_argument("--target-corpus-count", type=int, default=5)
    parser.add_argument("--target-benchmark-query-count", type=int, default=3)
    args = parser.parse_args(argv)

    plan = build_corpus_knowledge_expansion_plan(
        args.records,
        args.benchmark,
        graph_path=args.graph,
        target_corpus_count=args.target_corpus_count,
        target_benchmark_query_count=args.target_benchmark_query_count,
    )
    paths = write_corpus_knowledge_expansion_plan(plan, args.out)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
