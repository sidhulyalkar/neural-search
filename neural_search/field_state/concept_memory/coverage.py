"""Concept coverage audit for the Neural Search dataset corpus.

Reports how many datasets have task, modality, species, brain-region, method,
and analysis-affordance concept links — helping identify metadata enrichment targets.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx  # type: ignore[import-untyped]

from neural_search.field_state.concept_memory.graph_builder import (
    build_concept_graph,
    read_concept_artifacts,
)
from neural_search.field_state.concept_memory.normalize import normalize_concept_name
from neural_search.field_state.concept_memory.schema import ConceptNode, EvidenceLink
from neural_search.field_state.store import resolve_path

_COVERAGE_REPORT_PATH = Path("reports/field_state/concept_memory_coverage.md")

# Concept types we track for coverage
_TRACKED_TYPES: list[str] = [
    "task",
    "modality",
    "species",
    "brain_region",
    "method",
    "analysis_affordance",
    "experimental_protocol",
]


# ---------------------------------------------------------------------------
# Internal data classes
# ---------------------------------------------------------------------------


@dataclass
class DatasetCoverage:
    dataset_id: str
    dataset_title: str
    source_prefix: str
    covered_types: set[str] = field(default_factory=set)
    connected_concept_ids: list[str] = field(default_factory=list)

    @property
    def coverage_count(self) -> int:
        return len(self.covered_types)

    @property
    def is_well_covered(self) -> bool:
        core = {"task", "modality", "species"}
        return bool(self.covered_types & core)


@dataclass
class SourceCoverage:
    source_prefix: str
    total: int = 0
    covered_by_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _source_prefix(dataset_id: str) -> str:
    """Extract source system prefix from a dataset ID."""
    if "_" in dataset_id:
        return dataset_id.split("_")[0]
    return dataset_id[:8] if len(dataset_id) > 8 else dataset_id


def _build_coverage_map(
    concepts: list[ConceptNode],
    evidence_links: list[EvidenceLink],
    graph: nx.DiGraph,
) -> list[DatasetCoverage]:
    """Build per-dataset coverage records."""
    # Index: concept_id → ConceptNode for fast type lookup
    concept_index: dict[str, ConceptNode] = {c.concept_id: c for c in concepts}

    # Build adjacency: dataset_concept_id → [(connected_concept_id, relation_type)]
    dataset_ids = {c.concept_id for c in concepts if c.concept_type == "dataset"}
    adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for link in evidence_links:
        src = link.source_concept_id
        tgt = link.target_concept_id
        if src in dataset_ids and tgt is not None:
            adjacency[src].append((tgt, link.relation_type))
        elif tgt is not None and tgt in dataset_ids:
            adjacency[tgt].append((src, link.relation_type))

    coverages: list[DatasetCoverage] = []
    for c in concepts:
        if c.concept_type != "dataset":
            continue

        dataset_id = c.source_ids[0] if c.source_ids else c.concept_id
        covered_types: set[str] = set()
        connected_ids: list[str] = []

        for connected_id, _ in adjacency.get(c.concept_id, []):
            connected_ids.append(connected_id)
            if connected_id in concept_index:
                ctype = concept_index[connected_id].concept_type
                if ctype in _TRACKED_TYPES:
                    covered_types.add(ctype)

        coverages.append(
            DatasetCoverage(
                dataset_id=dataset_id,
                dataset_title=c.canonical_name,
                source_prefix=_source_prefix(dataset_id),
                covered_types=covered_types,
                connected_concept_ids=connected_ids,
            )
        )

    return coverages


def _alias_normalization_sample(
    concepts: list[ConceptNode],
    max_examples: int = 20,
) -> list[tuple[str, str]]:
    """Return sample (original_alias, normalized_canonical) pairs."""
    samples: list[tuple[str, str]] = []
    for c in concepts:
        if c.concept_type not in _TRACKED_TYPES:
            continue
        for alias in c.aliases:
            normalized = normalize_concept_name(alias)
            if normalized != alias.lower() and normalized == c.canonical_name.lower():
                samples.append((alias, c.canonical_name))
                if len(samples) >= max_examples:
                    return samples
    return samples


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _render_coverage_markdown(
    coverages: list[DatasetCoverage],
    alias_samples: list[tuple[str, str]],
    generated_at: str,
    field: str,
) -> str:
    total = len(coverages)
    if total == 0:
        return (
            "# Concept Coverage Audit\n\n"
            "_No dataset concepts found. Run `concept-build` to index the corpus._\n"
        )

    # Per-type coverage counts
    type_counts: dict[str, int] = dict.fromkeys(_TRACKED_TYPES, 0)
    for dc in coverages:
        for t in dc.covered_types:
            if t in type_counts:
                type_counts[t] += 1

    # Per-source coverage
    source_stats: dict[str, SourceCoverage] = {}
    for dc in coverages:
        sp = dc.source_prefix
        if sp not in source_stats:
            source_stats[sp] = SourceCoverage(source_prefix=sp)
        source_stats[sp].total += 1
        for t in dc.covered_types:
            source_stats[sp].covered_by_type[t] += 1

    # Well-covered vs bare
    well_covered = sum(1 for dc in coverages if dc.is_well_covered)
    bare = total - well_covered

    # Underrepresented types
    underrepresented = [
        (t, type_counts[t])
        for t in _TRACKED_TYPES
        if type_counts[t] / max(total, 1) < 0.20
    ]
    underrepresented.sort(key=lambda x: x[1])

    # Poor-coverage sources (< 10% well-covered)
    poor_sources = [
        sp
        for sp, sc in sorted(source_stats.items())
        if sc.total >= 5
        and sum(1 for dc in coverages if dc.source_prefix == sp and dc.is_well_covered)
        / sc.total
        < 0.10
    ]

    lines: list[str] = [
        "# Concept Coverage Audit",
        "",
        f"**Field:** {field}  ",
        f"**Generated:** {generated_at}  ",
        f"**Total dataset concepts:** {total}  ",
        f"**Well-covered (≥1 core type):** {well_covered} ({100 * well_covered // max(total, 1)}%)  ",
        f"**No core type coverage:** {bare} ({100 * bare // max(total, 1)}%)",
        "",
        "## Coverage by Concept Type",
        "",
        "| Concept Type | Datasets Covered | Coverage % |",
        "|--------------|-----------------|------------|",
    ]
    for t in _TRACKED_TYPES:
        cnt = type_counts[t]
        pct = 100 * cnt // max(total, 1)
        lines.append(f"| {t} | {cnt} | {pct}% |")
    lines.append("")

    lines += [
        "## Coverage by Source Repository",
        "",
        "| Source | Total | task | modality | species | brain_region | method | affordance |",
        "|--------|-------|------|----------|---------|--------------|--------|------------|",
    ]
    for sp, sc in sorted(source_stats.items(), key=lambda x: -x[1].total):
        row = [f"| {sp}", f"{sc.total}"]
        for t in ["task", "modality", "species", "brain_region", "method", "analysis_affordance"]:
            cnt = sc.covered_by_type.get(t, 0)
            pct = 100 * cnt // max(sc.total, 1)
            row.append(f"{cnt} ({pct}%)")
        lines.append(" | ".join(row) + " |")
    lines.append("")

    if underrepresented:
        lines += [
            "## Underrepresented Concept Types",
            "",
            "Types with <20% dataset coverage — these are prime enrichment targets:",
            "",
        ]
        for t, cnt in underrepresented:
            pct = 100 * cnt // max(total, 1)
            lines.append(f"- **{t}**: {cnt} datasets ({pct}%)")
        lines.append("")

    if poor_sources:
        lines += [
            "## Sources with Poor Core Coverage",
            "",
            "Sources where <10% of datasets have any task/modality/species concept:",
            "",
        ]
        for sp in poor_sources:
            sc = source_stats[sp]
            lines.append(f"- **{sp}** ({sc.total} datasets)")
        lines.append("")

    if alias_samples:
        lines += [
            "## Alias Normalization Examples",
            "",
            "Sample of alias → canonical concept name mappings applied during indexing:",
            "",
            "| Alias | Canonical Name |",
            "|-------|---------------|",
        ]
        for alias, canonical in alias_samples:
            lines.append(f"| {alias} | {canonical} |")
        lines.append("")

    lines += [
        "## Methodology",
        "",
        "- Coverage is determined by evidence links connecting dataset concepts to typed concept nodes.",
        "- 'Well-covered' means at least one task, modality, or species concept is linked.",
        "- Alias normalization is applied at concept-build time via `normalize_concept_name()`.",
        "- These counts reflect the indexed concept memory and may not match raw corpus metadata.",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_coverage_report(
    concepts: list[ConceptNode],
    evidence_links: list[EvidenceLink],
    field: str = "neuroscience_dataset_reuse",
    out_path: Path | None = None,
    root: Path | None = None,
) -> Path:
    """Compute concept coverage and write a Markdown report. Returns report path."""
    graph = build_concept_graph(concepts, evidence_links)
    coverages = _build_coverage_map(concepts, evidence_links, graph)
    alias_samples = _alias_normalization_sample(concepts)

    report_md = _render_coverage_markdown(
        coverages=coverages,
        alias_samples=alias_samples,
        generated_at=datetime.now(UTC).isoformat(),
        field=field,
    )

    out = resolve_path(out_path or _COVERAGE_REPORT_PATH, root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_md, encoding="utf-8")
    return out


def generate_coverage_report_from_artifacts(
    field: str = "neuroscience_dataset_reuse",
    out_path: Path | None = None,
    root: Path | None = None,
) -> Path:
    """Load artifacts from disk, compute coverage, write report."""
    concepts, evidence_links = read_concept_artifacts(root)
    return generate_coverage_report(
        concepts=concepts,
        evidence_links=evidence_links,
        field=field,
        out_path=out_path,
        root=root,
    )
