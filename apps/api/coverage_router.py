"""Coverage statistics and gap analysis API endpoints.

Route ordering matters — static paths (/topics/summary) are defined before
parameterized paths (/topics/{topic_id}) so FastAPI matches them correctly.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException

REPO_ROOT = Path(__file__).parent.parent.parent
TAXONOMY_PATH = REPO_ROOT / "data/ontology/topic_taxonomy.yaml"
COVERAGE_PATH = REPO_ROOT / "artifacts/topics/coverage.json"
GAPS_PATH = REPO_ROOT / "artifacts/topics/gaps.json"
SUBGRAPHS_PATH = REPO_ROOT / "artifacts/topics/subgraphs.json"

router = APIRouter(prefix="/api/coverage", tags=["coverage"])

# Module-level lazy caches
_taxonomy: dict[str, Any] | None = None
_coverage: dict[str, Any] | None = None
_gaps: list[dict[str, Any]] | None = None
_subgraphs: dict[str, Any] | None = None


def _get_taxonomy() -> dict[str, Any]:
    global _taxonomy
    if _taxonomy is None:
        if not TAXONOMY_PATH.exists():
            _taxonomy = {"topics": []}
        else:
            with TAXONOMY_PATH.open(encoding="utf-8") as f:
                _taxonomy = yaml.safe_load(f) or {"topics": []}
    return _taxonomy


def _get_coverage() -> dict[str, Any]:
    global _coverage
    if _coverage is None:
        if not COVERAGE_PATH.exists():
            _coverage = {}
        else:
            _coverage = json.loads(COVERAGE_PATH.read_text(encoding="utf-8"))
    return _coverage


def _get_gaps() -> list[dict[str, Any]]:
    global _gaps
    if _gaps is None:
        if not GAPS_PATH.exists():
            _gaps = []
        else:
            _gaps = json.loads(GAPS_PATH.read_text(encoding="utf-8"))
    return _gaps


def _get_subgraphs() -> dict[str, Any]:
    global _subgraphs
    if _subgraphs is None:
        if not SUBGRAPHS_PATH.exists():
            _subgraphs = {}
        else:
            _subgraphs = json.loads(SUBGRAPHS_PATH.read_text(encoding="utf-8"))
    return _subgraphs


def _load_live_findings() -> list[dict]:
    findings_path = REPO_ROOT / "artifacts/literature/findings_v1.jsonl"
    findings: list[dict] = []
    if findings_path.exists():
        with findings_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        findings.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return findings


def _load_live_corpus() -> tuple[list[dict], list[dict]]:
    """Return (papers, datasets) from the demo seed corpus."""
    try:
        from neural_search.ingestion.demo_seed import build_combined_corpus
        corpus = build_combined_corpus()
    except Exception:
        corpus = []
    datasets = [r["dataset"] for r in corpus if "dataset" in r]
    papers: list[dict] = []
    for r in corpus:
        papers.extend(r.get("papers") or [])
    return papers, datasets


def _build_live_coverage(topic_id: str) -> dict[str, Any] | None:
    """Compute coverage on-the-fly if the pre-built artifact isn't available."""
    from neural_search.graph.coverage_stats import compute_topic_coverage

    taxonomy = _get_taxonomy()
    topics_by_id = {t["id"]: t for t in (taxonomy.get("topics") or [])}
    topic = topics_by_id.get(topic_id)
    if not topic:
        return None

    papers, datasets = _load_live_corpus()
    findings = _load_live_findings()

    cov = compute_topic_coverage(topic_id, topic, papers, findings, datasets)
    cov_dict = asdict(cov)
    if cov_dict.get("year_range"):
        cov_dict["year_range"] = list(cov_dict["year_range"])
    return cov_dict


# ── Static routes (must come before parameterized routes) ──────────────────────

@router.get("/topics")
def list_topic_coverage() -> list[dict[str, Any]]:
    """All topics with coverage scores, sorted by coverage_score desc."""
    taxonomy = _get_taxonomy()
    coverage = _get_coverage()

    results = []
    for topic in taxonomy.get("topics") or []:
        tid = topic["id"]
        cov = coverage.get(tid, {})
        results.append({
            "topic_id": tid,
            "topic_label": topic.get("label", tid),
            "color": topic.get("color", "#6366f1"),
            "coverage_score": cov.get("coverage_score", 0.0),
            "total_papers": cov.get("total_papers", 0),
            "total_datasets": cov.get("total_datasets", 0),
            "total_findings": cov.get("total_findings", 0),
            "n_data_gaps": len(cov.get("data_gaps") or []),
        })

    results.sort(key=lambda x: -x["coverage_score"])
    return results


@router.get("/topics/summary")
def get_coverage_summary() -> dict[str, Any]:
    """Global topic coverage summary across all topics."""
    taxonomy = _get_taxonomy()
    coverage = _get_coverage()
    gaps = _get_gaps()

    topics = taxonomy.get("topics") or []
    scores: list[float] = []
    topic_summaries: list[dict[str, Any]] = []
    total_papers = 0
    total_datasets = 0

    for topic in topics:
        tid = topic["id"]
        cov = coverage.get(tid, {})
        score = cov.get("coverage_score", 0.0)
        scores.append(score)
        total_papers += cov.get("total_papers", 0)
        total_datasets += cov.get("total_datasets", 0)
        topic_summaries.append({
            "id": tid,
            "label": topic.get("label", tid),
            "score": score,
        })

    sorted_summaries = sorted(topic_summaries, key=lambda x: -x["score"])
    avg_score = round(sum(scores) / len(scores), 3) if scores else 0.0
    critical_gaps = sum(1 for g in gaps if g.get("severity") == "critical")

    return {
        "total_topics": len(topics),
        "total_papers": total_papers,
        "total_datasets": total_datasets,
        "avg_coverage_score": avg_score,
        "most_covered_topics": sorted_summaries[:5],
        "least_covered_topics": sorted_summaries[-5:][::-1],
        "critical_gaps": critical_gaps,
        "total_gaps": len(gaps),
    }


# ── Parameterized routes ───────────────────────────────────────────────────────

@router.get("/topics/{topic_id}")
def get_topic_coverage(topic_id: str) -> dict[str, Any]:
    """Full coverage stats for a topic including region breakdown and data gaps."""
    coverage = _get_coverage()
    cov = coverage.get(topic_id)

    if cov is None:
        cov = _build_live_coverage(topic_id)

    if cov is None:
        raise HTTPException(status_code=404, detail=f"Topic {topic_id!r} not found")

    topic_gaps = [g for g in _get_gaps() if g.get("topic_id") == topic_id]
    return {**cov, "gaps": topic_gaps}


@router.get("/topics/{topic_id}/gaps")
def get_topic_gaps(topic_id: str) -> list[dict[str, Any]]:
    """Research gaps identified for a topic."""
    taxonomy = _get_taxonomy()
    known_ids = {t["id"] for t in (taxonomy.get("topics") or [])}
    if topic_id not in known_ids:
        raise HTTPException(status_code=404, detail=f"Topic {topic_id!r} not found")
    return [g for g in _get_gaps() if g.get("topic_id") == topic_id]


@router.get("/topics/{topic_id}/subgraph")
def get_topic_subgraph(topic_id: str) -> dict[str, Any]:
    """Rich topic subgraph for visualization. Falls back to live build if artifact missing."""
    subgraphs = _get_subgraphs()
    sg = subgraphs.get(topic_id)
    if sg is not None:
        return sg

    from neural_search.graph.subgraph_builder import build_topic_subgraph

    taxonomy = _get_taxonomy()
    topics_by_id = {t["id"]: t for t in (taxonomy.get("topics") or [])}
    if topic_id not in topics_by_id:
        raise HTTPException(status_code=404, detail=f"Topic {topic_id!r} not found")

    papers, datasets = _load_live_corpus()
    findings = _load_live_findings()

    citations_path = REPO_ROOT / "artifacts/citations/citation_edges.jsonl"
    citations: list[dict] = []
    if citations_path.exists():
        with citations_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        citations.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    result = build_topic_subgraph(topic_id, taxonomy, papers, findings, datasets, citations)
    return {
        "topic_id": result.topic_id,
        "topic_label": result.topic_label,
        "nodes": [asdict(n) for n in result.nodes],
        "links": [asdict(lnk) for lnk in result.links],
        "meta": result.meta,
    }
