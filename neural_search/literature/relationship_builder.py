"""Cross-finding relationship builder.

Materialises three types of relationships from a normalized findings JSONL:

  FindingEdge types:
    supports      — same region + task, same direction, different papers (≥2 papers)
    contradicts   — same region + task, opposite direction, different papers
    co_occurs_in  — two regions mentioned together in the same finding

  RegionEdge types:
    region_co_occurs_with — region A and region B co-mentioned in at least N findings

These edges are written as JSONL so they can be directly ingested by the KG builder.

Usage:
    from neural_search.literature.relationship_builder import (
        build_cross_finding_edges,
        build_region_cooccurrence_edges,
    )
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Pairs of directions that are considered contradictory
_CONTRADICTIONS: set[frozenset[str]] = {
    frozenset({"increase", "decrease"}),
    frozenset({"increase", "no_change"}),
    frozenset({"decrease", "no_change"}),
}


# ---------------------------------------------------------------------------
# Edge data classes
# ---------------------------------------------------------------------------


@dataclass
class FindingEdge:
    edge_type: str           # supports | contradicts
    finding_id_a: str
    finding_id_b: str
    paper_id_a: str
    paper_id_b: str
    shared_regions: list[str]
    shared_tasks: list[str]
    direction_a: str
    direction_b: str
    n_supporting_papers: int  # how many papers agree (for supports edges)
    confidence: float


@dataclass
class RegionCooccurrenceEdge:
    edge_type: str            # always "region_co_occurs_with"
    region_a: str
    region_b: str
    n_findings: int           # number of findings with both regions
    finding_ids: list[str]    # up to 20 examples
    confidence: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_findings(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _region_key(regions: list[str]) -> frozenset[str]:
    return frozenset(r.strip().lower() for r in regions if r.strip())


def _task_key(tasks: list[str]) -> frozenset[str]:
    return frozenset(t.strip().lower() for t in tasks if t.strip())


# ---------------------------------------------------------------------------
# Cross-finding relationship builder
# ---------------------------------------------------------------------------


def build_cross_finding_edges(
    findings_path: Path,
    *,
    min_shared_regions: int = 1,
    min_shared_tasks: int = 0,
    max_edges: int = 100_000,
) -> list[FindingEdge]:
    """Build supports/contradicts edges across findings from different papers.

    Groups findings by (region_set, task_set) signature and then compares
    directions within each group.  Only cross-paper comparisons are made.

    Args:
        findings_path: Path to normalized JSONL.
        min_shared_regions: Minimum overlapping regions for a match (default 1).
        min_shared_tasks: Minimum overlapping tasks (default 0 — allows region-only match).
        max_edges: Safety cap on total edges produced.
    """
    # Index findings by (region, task) signature
    # We use individual (region, task) pairs to allow partial overlap matching
    region_task_index: dict[tuple[str, str], list[dict]] = defaultdict(list)

    findings = list(_iter_findings(findings_path))
    logger.info("build_cross_finding_edges: loaded %d findings", len(findings))

    for f in findings:
        regions = [r.strip().lower() for r in f.get("regions", []) if r.strip()]
        tasks = [t.strip().lower() for t in f.get("tasks", []) if t.strip()]
        if not regions:
            continue
        for region in regions:
            task_group = tuple(sorted(tasks)) if tasks else ("_any",)
            region_task_index[(region, task_group)].append(f)

    edges: list[FindingEdge] = []
    seen_pairs: set[frozenset[str]] = set()

    for (_region, _task_group), group in region_task_index.items():
        if len(group) < 2:
            continue

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a["paper_id"] == b["paper_id"]:
                    continue  # same paper

                pair_key = frozenset({a["finding_id"], b["finding_id"]})
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                dir_a = a.get("result_direction", "other")
                dir_b = b.get("result_direction", "other")
                shared_regions = list(
                    _region_key(a.get("regions", [])) & _region_key(b.get("regions", []))
                )
                shared_tasks = list(
                    _task_key(a.get("tasks", [])) & _task_key(b.get("tasks", []))
                )

                if len(shared_regions) < min_shared_regions:
                    continue
                if min_shared_tasks > 0 and len(shared_tasks) < min_shared_tasks:
                    continue

                if dir_a == dir_b and dir_a not in ("other", "correlation"):
                    edge_type = "supports"
                    conf = min(0.5 + 0.1 * len(shared_regions) + 0.1 * len(shared_tasks), 0.95)
                elif frozenset({dir_a, dir_b}) in _CONTRADICTIONS:
                    edge_type = "contradicts"
                    conf = min(0.5 + 0.1 * len(shared_regions) + 0.1 * len(shared_tasks), 0.90)
                else:
                    continue

                edges.append(
                    FindingEdge(
                        edge_type=edge_type,
                        finding_id_a=a["finding_id"],
                        finding_id_b=b["finding_id"],
                        paper_id_a=a["paper_id"],
                        paper_id_b=b["paper_id"],
                        shared_regions=shared_regions,
                        shared_tasks=shared_tasks,
                        direction_a=dir_a,
                        direction_b=dir_b,
                        n_supporting_papers=1,
                        confidence=conf,
                    )
                )

                if len(edges) >= max_edges:
                    logger.warning(
                        "build_cross_finding_edges: reached max_edges=%d, stopping early",
                        max_edges,
                    )
                    return edges

    logger.info(
        "build_cross_finding_edges: produced %d edges (%d supports, %d contradicts)",
        len(edges),
        sum(1 for e in edges if e.edge_type == "supports"),
        sum(1 for e in edges if e.edge_type == "contradicts"),
    )
    return edges


# ---------------------------------------------------------------------------
# Region co-occurrence edge builder
# ---------------------------------------------------------------------------


def build_region_cooccurrence_edges(
    findings_path: Path,
    *,
    min_cooccurrences: int = 2,
    max_example_ids: int = 20,
) -> list[RegionCooccurrenceEdge]:
    """Build region_co_occurs_with edges from multi-region findings.

    Two regions are co-occurring when they appear together in the same finding,
    indicating they are mentioned in the same experimental context (e.g., the
    paper measured both regions simultaneously or compared them directly).

    Args:
        findings_path: Path to normalized JSONL.
        min_cooccurrences: Minimum number of findings sharing the pair (default 2).
        max_example_ids: Max finding_ids to store per edge (default 20).
    """
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    pair_examples: dict[tuple[str, str], list[str]] = defaultdict(list)

    for f in _iter_findings(findings_path):
        regions = list({r.strip().lower() for r in f.get("regions", []) if r.strip()})
        if len(regions) < 2:
            continue

        finding_id = f.get("finding_id", "")
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                pair = tuple(sorted([regions[i], regions[j]]))
                pair_counts[pair] += 1
                if len(pair_examples[pair]) < max_example_ids:
                    pair_examples[pair].append(finding_id)

    edges: list[RegionCooccurrenceEdge] = []
    for pair, count in pair_counts.items():
        if count < min_cooccurrences:
            continue
        conf = min(0.5 + 0.02 * count, 0.95)
        edges.append(
            RegionCooccurrenceEdge(
                edge_type="region_co_occurs_with",
                region_a=pair[0],
                region_b=pair[1],
                n_findings=count,
                finding_ids=pair_examples[pair],
                confidence=conf,
            )
        )

    edges.sort(key=lambda e: -e.n_findings)
    logger.info(
        "build_region_cooccurrence_edges: produced %d region-pair edges (min=%d co-occurrences)",
        len(edges),
        min_cooccurrences,
    )
    return edges


# ---------------------------------------------------------------------------
# Consensus summary builder
# ---------------------------------------------------------------------------


@dataclass
class ConsensusRecord:
    region: str
    direction: str
    task: str | None
    n_findings: int
    n_papers: int
    finding_ids: list[str]
    consensus_strength: float  # 0–1; proportion of findings agreeing on direction


def build_consensus_summaries(
    findings_path: Path,
    *,
    min_papers: int = 2,
) -> list[ConsensusRecord]:
    """Compute consensus strength for (region, direction, task) triples.

    A consensus record captures how many papers agree that 'X increases/decreases
    in region R during task T'.  High consensus → established fact.  Mixed
    directions → contested finding.

    Args:
        findings_path: Path to normalized JSONL.
        min_papers: Minimum distinct papers required to form a consensus record.
    """
    # key: (region, task_or_none) → {direction: [(paper_id, finding_id)]}
    index: dict[
        tuple[str, str | None],
        dict[str, list[tuple[str, str]]],
    ] = defaultdict(lambda: defaultdict(list))

    for f in _iter_findings(findings_path):
        regions = [r.strip().lower() for r in f.get("regions", []) if r.strip()]
        direction = f.get("result_direction", "other")
        tasks = [t.strip().lower() for t in f.get("tasks", []) if t.strip()]
        paper_id = f.get("paper_id", "")
        finding_id = f.get("finding_id", "")

        if direction in ("other",):
            continue

        for region in regions:
            task = tasks[0] if tasks else None
            index[(region, task)][direction].append((paper_id, finding_id))

    records: list[ConsensusRecord] = []
    for (region, task), direction_groups in index.items():
        total_findings = sum(len(v) for v in direction_groups.values())
        for direction, entries in direction_groups.items():
            n_papers = len({p for p, _ in entries})
            if n_papers < min_papers:
                continue
            strength = len(entries) / total_findings
            records.append(
                ConsensusRecord(
                    region=region,
                    direction=direction,
                    task=task,
                    n_findings=len(entries),
                    n_papers=n_papers,
                    finding_ids=[fid for _, fid in entries[:20]],
                    consensus_strength=round(strength, 3),
                )
            )

    records.sort(key=lambda r: (-r.n_papers, -r.consensus_strength))
    logger.info(
        "build_consensus_summaries: produced %d consensus records (%d with strength≥0.8)",
        len(records),
        sum(1 for r in records if r.consensus_strength >= 0.8),
    )
    return records


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def write_edges_jsonl(edges: list[Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for edge in edges:
            fh.write(json.dumps(asdict(edge)) + "\n")
    logger.info("Wrote %d edges to %s", len(edges), out_path)
