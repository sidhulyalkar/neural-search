"""Generate reports/methodology_coverage_report.md.

Reports how much of the analysis_family vocabulary (populated into the real
corpus graph by build_taxonomy_requirement_subgraph) is now bridged to named
techniques from methods_taxonomy.yaml via the methodology registry overlay,
plus the existing analysis_requires_* requirement-edge report.
"""

from __future__ import annotations

import logging
from pathlib import Path

from neural_search.graph.builder import build_graph_from_records
from neural_search.graph.method_registry_builder import load_method_registry
from neural_search.graph.reports import graph_requirement_report
from neural_search.kg.schemas.method_registry import (
    known_analysis_families,
    known_taxonomy_method_ids,
)

REPORT_PATH = Path(__file__).parent.parent / "reports" / "methodology_coverage_report.md"


def _coverage_section() -> str:
    registry = load_method_registry()
    all_families = known_analysis_families()
    all_methods = known_taxonomy_method_ids()

    linked_families = {link.analysis_family for link in registry.links}
    linked_methods: set[str] = set()
    for link in registry.links:
        linked_methods.update(link.taxonomy_method_ids)

    unlinked_families = sorted(all_families - linked_families)
    unlinked_methods = sorted(all_methods - linked_methods)
    needs_review = sorted(
        link.analysis_family for link in registry.links if link.requires_human_review
    )

    lines = [
        "# Methodology Coverage Report",
        "",
        f"- Analysis families linked: {len(linked_families)}/{len(all_families)}",
        f"- Taxonomy methods cross-linked to >=1 analysis family: "
        f"{len(linked_methods)}/{len(all_methods)}",
        f"- Links flagged requires_human_review: {len(needs_review)}/{len(registry.links)}",
        "",
        "## Unlinked analysis families (open gap)",
        "",
    ]
    if unlinked_families:
        lines.extend(f"- {family}" for family in unlinked_families)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Taxonomy methods not yet cross-linked to any analysis family",
            "",
        ]
    )
    if unlinked_methods:
        lines.extend(f"- {method}" for method in unlinked_methods)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Links flagged for human review",
            "",
        ]
    )
    if needs_review:
        lines.extend(f"- {family}" for family in needs_review)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    graph = build_graph_from_records()
    method_edges = [
        e for e in graph.edges.values() if e.edge_type == "method_supports_analysis"
    ]
    method_nodes = [n for n in graph.nodes.values() if n.node_type == "method"]

    sections = [
        _coverage_section(),
        "## Live graph counts",
        "",
        f"- `method` nodes in merged graph: {len(method_nodes)}",
        f"- `method_supports_analysis` edges in merged graph: {len(method_edges)}",
        "",
        graph_requirement_report(graph),
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
