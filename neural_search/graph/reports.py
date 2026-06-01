"""Markdown reports for Neural Search knowledge graphs."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from neural_search.graph.query import find_nodes_by_type, find_papers_for_dataset
from neural_search.graph.schema import KnowledgeGraph, read_graph_json


def _heading(title: str) -> str:
    return f"# {title}\n\n"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No records._\n"
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(output) + "\n"


def _node_counts(graph: KnowledgeGraph) -> Counter[str]:
    return Counter(node.node_type for node in graph.nodes.values())


def _edge_counts(graph: KnowledgeGraph) -> Counter[str]:
    return Counter(edge.edge_type for edge in graph.edges.values())


def graph_summary_report(graph: KnowledgeGraph) -> str:
    """Return node/edge counts and basic connectivity stats as Markdown."""

    node_counts = _node_counts(graph)
    edge_counts = _edge_counts(graph)
    degrees: Counter[str] = Counter()
    for edge in graph.edges.values():
        degrees[edge.source_node_id] += 1
        degrees[edge.target_node_id] += 1
    isolated = len([node_id for node_id in graph.nodes if degrees[node_id] == 0])
    top_connected = degrees.most_common(10)

    return (
        _heading("Graph Summary Report")
        + f"- Nodes: {len(graph.nodes)}\n"
        + f"- Edges: {len(graph.edges)}\n"
        + f"- Isolated nodes: {isolated}\n\n"
        + "## Node Counts\n\n"
        + _table("Node type Count".split(), [[key, str(value)] for key, value in node_counts.most_common()])
        + "\n## Edge Counts\n\n"
        + _table("Edge type Count".split(), [[key, str(value)] for key, value in edge_counts.most_common()])
        + "\n## Top Connected Nodes\n\n"
        + _table(
            ["Node ID", "Label", "Degree"],
            [
                [
                    node_id,
                    graph.nodes[node_id].label if node_id in graph.nodes else "",
                    str(degree),
                ]
                for node_id, degree in top_connected
            ],
        )
    )


def graph_scientific_coverage_report(graph: KnowledgeGraph) -> str:
    """Return task/modality/region/species coverage as Markdown."""

    sections = [_heading("Graph Scientific Coverage Report")]
    for node_type in [
        "task",
        "modality",
        "brain_region",
        "species",
        "behavioral_event",
        "analysis_affordance",
        "required_signal",
        "data_standard",
    ]:
        rows = [
            [node.node_id, node.label, str(len(node.source_ids)), f"{node.confidence:.2f}"]
            for node in find_nodes_by_type(graph, node_type)
        ]
        rows.sort(key=lambda row: row[1])
        sections.append(f"## {node_type.replace('_', ' ').title()}\n\n")
        sections.append(_table(["Node ID", "Label", "Sources", "Confidence"], rows))
        sections.append("\n")
    return "".join(sections)


def graph_requirement_report(graph: KnowledgeGraph) -> str:
    """Return analysis requirement edges as Markdown."""

    requirement_edges = [
        edge
        for edge in graph.edges.values()
        if edge.edge_type.startswith("analysis_requires_")
    ]
    rows = []
    for edge in requirement_edges:
        source = graph.nodes[edge.source_node_id]
        target = graph.nodes[edge.target_node_id]
        rows.append(
            [
                edge.edge_type,
                source.label,
                target.node_type,
                target.label,
                str(edge.properties.get("data_form", "")),
                f"{edge.confidence:.2f}",
            ]
        )
    rows.sort(key=lambda row: (row[0], row[4], row[1], row[3]))
    return (
        _heading("Graph Requirement Report")
        + f"- Analysis requirement edges: {len(requirement_edges)}\n\n"
        + _table(
            ["Edge Type", "Analysis", "Target Type", "Requirement", "Data Form", "Confidence"],
            rows,
        )
    )


def graph_linking_report(graph: KnowledgeGraph) -> str:
    """Return dataset-paper link analysis as Markdown."""

    rows: list[list[str]] = []
    linked_dataset_count = 0
    for dataset in find_nodes_by_type(graph, "dataset"):
        papers = find_papers_for_dataset(graph, dataset.node_id)
        if papers:
            linked_dataset_count += 1
        rows.append(
            [
                dataset.node_id,
                dataset.label,
                str(len(papers)),
                ", ".join(paper.label for paper in papers[:5]),
            ]
        )
    rows.sort(key=lambda row: (-int(row[2]), row[1]))
    return (
        _heading("Graph Linking Report")
        + f"- Datasets with linked papers: {linked_dataset_count}\n"
        + f"- Total datasets: {len(find_nodes_by_type(graph, 'dataset'))}\n\n"
        + _table(["Dataset ID", "Dataset", "Linked Papers", "Paper Labels"], rows)
    )


def graph_gap_report(graph: KnowledgeGraph) -> str:
    """Return missing-data and sparse-linkage gaps as Markdown."""

    missing_by_field: Counter[str] = Counter()
    datasets_without_papers: list[list[str]] = []
    datasets_without_analysis: list[list[str]] = []
    for dataset in find_nodes_by_type(graph, "dataset"):
        for field in dataset.properties.get("missing_fields", []):
            missing_by_field[str(field)] += 1
        if not find_papers_for_dataset(graph, dataset.node_id):
            datasets_without_papers.append([dataset.node_id, dataset.label])
        has_analysis = any(
            edge.source_node_id == dataset.node_id
            and edge.edge_type == "dataset_supports_analysis"
            for edge in graph.edges.values()
        )
        if not has_analysis:
            datasets_without_analysis.append([dataset.node_id, dataset.label])

    return (
        _heading("Graph Gap Report")
        + "## Missing Metadata Fields\n\n"
        + _table(
            ["Field", "Dataset Count"],
            [[field, str(count)] for field, count in missing_by_field.most_common()],
        )
        + "\n## Datasets Without Linked Papers\n\n"
        + _table(["Dataset ID", "Dataset"], datasets_without_papers)
        + "\n## Datasets Without Analysis Affordances\n\n"
        + _table(["Dataset ID", "Dataset"], datasets_without_analysis)
    )


def generate_graph_reports(graph: KnowledgeGraph) -> dict[str, str]:
    """Generate all graph reports keyed by output filename."""

    return {
        "graph_summary_report.md": graph_summary_report(graph),
        "graph_scientific_coverage_report.md": graph_scientific_coverage_report(graph),
        "graph_requirement_report.md": graph_requirement_report(graph),
        "graph_linking_report.md": graph_linking_report(graph),
        "graph_gap_report.md": graph_gap_report(graph),
    }


def write_graph_reports(graph: KnowledgeGraph, out_dir: str | Path) -> dict[str, Path]:
    """Write all graph reports to a directory."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for filename, content in generate_graph_reports(graph).items():
        path = output / filename
        path.write_text(content, encoding="utf-8")
        paths[filename] = path
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate knowledge graph reports.")
    parser.add_argument("--graph", required=True, help="Input graph JSON file")
    parser.add_argument("--out", required=True, help="Output report directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    graph = read_graph_json(args.graph)
    paths = write_graph_reports(graph, args.out)
    print(f"wrote {len(paths)} graph reports to {Path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
