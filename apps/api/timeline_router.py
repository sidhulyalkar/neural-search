"""Topic taxonomy and temporal timeline API endpoints."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Query

from neural_search.graph.temporal import (
    build_topic_timeline,
    get_intellectual_ancestors,
)

REPO_ROOT = Path(__file__).parent.parent.parent
TAXONOMY_PATH = REPO_ROOT / "data" / "ontology" / "topic_taxonomy.yaml"
TIMELINES_PATH = REPO_ROOT / "artifacts" / "topics" / "timelines.json"
CITATIONS_PATH = REPO_ROOT / "artifacts" / "citations" / "citation_edges.jsonl"
FINDINGS_PATH = REPO_ROOT / "artifacts" / "literature" / "findings_v1.jsonl"

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/topics", tags=["topics"])

# ── Module-level caches ────────────────────────────────────────────────────────

_taxonomy: dict[str, Any] | None = None
_timelines: dict[str, Any] | None = None
_citation_edges: list[dict[str, Any]] | None = None


def _get_taxonomy() -> dict[str, Any]:
    global _taxonomy
    if _taxonomy is None:
        if not TAXONOMY_PATH.exists():
            _taxonomy = {"topics": []}
        else:
            with TAXONOMY_PATH.open(encoding="utf-8") as f:
                _taxonomy = yaml.safe_load(f) or {"topics": []}
    return _taxonomy


def _get_timelines() -> dict[str, Any]:
    global _timelines
    if _timelines is None:
        if TIMELINES_PATH.exists():
            _timelines = json.loads(TIMELINES_PATH.read_text(encoding="utf-8"))
        else:
            _timelines = {}
    return _timelines


def _get_citation_edges() -> list[dict[str, Any]]:
    global _citation_edges
    if _citation_edges is None:
        if not CITATIONS_PATH.exists():
            _citation_edges = []
        else:
            edges = []
            with CITATIONS_PATH.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        edges.append(json.loads(line))
            _citation_edges = edges
    return _citation_edges


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _get_corpus_papers() -> list[dict[str, Any]]:
    """Load papers from the corpus (JSONL shards under data/corpus/normalized)."""
    import glob
    papers = []
    corpus_dir = REPO_ROOT / "data" / "corpus" / "normalized"
    for json_path in glob.glob(str(corpus_dir / "**" / "*.json"), recursive=True):
        try:
            with open(json_path, encoding="utf-8") as f:
                record = json.load(f)
            if isinstance(record, list):
                papers.extend(record)
            elif isinstance(record, dict):
                papers.append(record)
        except Exception:
            pass
    return papers


@router.get("/")
def list_topics() -> list[dict[str, Any]]:
    """List all topics from the taxonomy with basic stats."""
    taxonomy = _get_taxonomy()
    timelines = _get_timelines()
    result = []
    for topic in taxonomy.get("topics", []):
        tid = topic.get("id", "")
        tl = timelines.get(tid, {})
        result.append({
            "id": tid,
            "label": topic.get("label", ""),
            "description": topic.get("description", ""),
            "color": topic.get("color"),
            "companion_topics": topic.get("companion_topics", []),
            "total_papers": tl.get("total_papers", 0),
            "total_findings": tl.get("total_findings", 0),
            "total_datasets": tl.get("total_datasets", 0),
            "year_range": tl.get("year_range"),
        })
    return result


@router.get("/{topic_id}")
def get_topic(topic_id: str) -> dict[str, Any]:
    """Get full topic details including companion topics."""
    taxonomy = _get_taxonomy()
    topic = next((t for t in taxonomy.get("topics", []) if t.get("id") == topic_id), None)
    if topic is None:
        raise HTTPException(status_code=404, detail=f"Topic {topic_id!r} not found")
    return topic


@router.get("/{topic_id}/timeline")
def get_topic_timeline(
    topic_id: str,
    min_year: int = Query(1990),
    max_year: int = Query(2025),
) -> dict[str, Any]:
    """
    Return the timeline for a topic, filtered by year range.

    Falls back to building it on-the-fly if artifacts/topics/timelines.json
    doesn't exist yet.
    """
    taxonomy = _get_taxonomy()
    topic = next((t for t in taxonomy.get("topics", []) if t.get("id") == topic_id), None)
    if topic is None:
        raise HTTPException(status_code=404, detail=f"Topic {topic_id!r} not found")

    prebuilt = _get_timelines().get(topic_id)
    if prebuilt:
        # Filter entries by year range
        entries = [
            e for e in prebuilt.get("entries", [])
            if min_year <= e.get("year", 0) <= max_year
        ]
        return {**prebuilt, "entries": entries, "min_year": min_year, "max_year": max_year}

    # Build on-the-fly
    logger.info("Building timeline on-the-fly for topic %s", topic_id)
    papers = _get_corpus_papers()
    findings = _load_jsonl(FINDINGS_PATH)
    # Use empty dataset list since we don't have corpus datasets readily available
    from dataclasses import asdict
    tl = build_topic_timeline(topic_id, taxonomy, papers, findings, datasets=[])
    result = {**asdict(tl), "year_range": list(tl.year_range)}
    entries = [
        e for e in result.get("entries", [])
        if min_year <= e.get("year", 0) <= max_year
    ]
    return {**result, "entries": entries, "min_year": min_year, "max_year": max_year}


@router.get("/{topic_id}/ancestors")
def get_intellectual_ancestors_endpoint(topic_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """Return foundational papers for a topic (most cited within topic)."""
    taxonomy = _get_taxonomy()
    topic = next((t for t in taxonomy.get("topics", []) if t.get("id") == topic_id), None)
    if topic is None:
        raise HTTPException(status_code=404, detail=f"Topic {topic_id!r} not found")

    papers = _get_corpus_papers()
    citation_edges = _get_citation_edges()

    topic_tasks: list[str] = topic.get("tasks", [])
    topic_regions: list[str] = topic.get("regions", [])

    # Filter to topic papers
    def _matches(paper: dict) -> bool:
        tags = list(paper.get("tasks") or []) + list(paper.get("brain_regions") or [])
        tags_lower = {t.lower() for t in tags}
        for task in topic_tasks:
            if task.lower() in tags_lower:
                return True
        for region in topic_regions:
            if region.lower() in tags_lower:
                return True
        return False

    topic_papers = [p for p in papers if _matches(p)]
    ancestors = get_intellectual_ancestors(topic_id, topic_papers, citation_edges)
    return ancestors[:limit]


@router.get("/{topic_id}/subgraph")
def get_topic_subgraph(topic_id: str) -> dict[str, Any]:
    """
    Return a graph-compatible {nodes, links} dict for the topic.

    Nodes: papers, regions, methods. Links: paper->region, paper->method, cites.
    Powers the KnowledgeExplorerPage topic view.
    """
    taxonomy = _get_taxonomy()
    topic = next((t for t in taxonomy.get("topics", []) if t.get("id") == topic_id), None)
    if topic is None:
        raise HTTPException(status_code=404, detail=f"Topic {topic_id!r} not found")

    topic_tasks: list[str] = topic.get("tasks", [])
    topic_regions: list[str] = topic.get("regions", [])

    papers = _get_corpus_papers()
    citation_edges = _get_citation_edges()

    def _matches(paper: dict) -> bool:
        tags = list(paper.get("tasks") or []) + list(paper.get("brain_regions") or [])
        tags_lower = {t.lower() for t in tags}
        for task in topic_tasks:
            if task.lower() in tags_lower:
                return True
        for region in topic_regions:
            if region.lower() in tags_lower:
                return True
        return False

    topic_papers = [p for p in papers if _matches(p)][:100]
    paper_ids: set[str] = set()
    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []

    for paper in topic_papers:
        pid = str(paper.get("openalex_id") or paper.get("id", ""))
        if not pid or pid in paper_ids:
            continue
        paper_ids.add(pid)
        nodes.append({
            "id": pid,
            "type": "paper",
            "label": (paper.get("title") or "")[:80],
            "year": paper.get("publication_year"),
            "size": max(1, min(20, (paper.get("cited_by_count") or 0) // 10)),
        })
        for region in paper.get("brain_regions") or []:
            rid = f"region:{region}"
            links.append({"source": pid, "target": rid, "type": "mentions_region"})
        for mod in paper.get("modalities") or []:
            mid = f"method:{mod}"
            links.append({"source": pid, "target": mid, "type": "uses_method"})

    # Add region and method nodes referenced by links
    extra_ids: set[str] = {lnk["target"] for lnk in links}
    for nid in extra_ids:
        if nid.startswith("region:"):
            nodes.append({"id": nid, "type": "brain_region", "label": nid[7:], "size": 8})
        elif nid.startswith("method:"):
            nodes.append({"id": nid, "type": "modality", "label": nid[7:], "size": 6})

    # Add inner-corpus citation links among topic papers
    for edge in citation_edges:
        src = edge.get("citing_paper_id", "")
        tgt = edge.get("cited_paper_id", "")
        if src in paper_ids and tgt in paper_ids:
            links.append({"source": src, "target": tgt, "type": "cites"})

    return {
        "topic_id": topic_id,
        "topic_label": topic.get("label", topic_id),
        "nodes": nodes,
        "links": links,
        "meta": {"node_count": len(nodes), "edge_count": len(links)},
    }
