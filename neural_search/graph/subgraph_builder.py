"""Build rich per-topic subgraphs for knowledge graph visualization."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TopicNode:
    id: str
    type: str  # 'paper', 'dataset', 'region', 'task', 'method', 'finding'
    label: str
    year: int | None
    color: str
    size: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TopicLink:
    source: str
    target: str
    type: str  # 'paper_studies_region', 'paper_uses_method', etc.
    weight: float
    color: str


@dataclass
class TopicSubgraph:
    topic_id: str
    topic_label: str
    nodes: list[TopicNode]
    links: list[TopicLink]
    meta: dict[str, Any]


def _paper_matches_topic(paper: dict, topic: dict) -> bool:
    """Return True if a paper is relevant to the topic."""
    topic_tasks = {t.lower() for t in (topic.get("tasks") or [])}
    topic_regions = {r.lower() for r in (topic.get("regions") or [])}
    topic_label = topic.get("label", "").lower()
    topic_id = topic.get("id", "").replace("_", " ").lower()

    text = " ".join(filter(None, [
        paper.get("title", ""),
        paper.get("abstract", ""),
        " ".join(paper.get("concepts", []) if isinstance(paper.get("concepts"), list) else []),
        " ".join(paper.get("tasks", []) if isinstance(paper.get("tasks"), list) else []),
    ])).lower()

    if topic_label in text or topic_id in text:
        return True
    if any(task in text for task in topic_tasks):
        return True
    if any(region in text for region in topic_regions):
        return True
    return False


def _dataset_matches_topic(dataset: dict, topic: dict) -> bool:
    """Return True if a dataset is relevant to the topic."""
    topic_modalities = {m.lower() for m in (topic.get("modalities") or [])}
    topic_regions = {r.lower() for r in (topic.get("regions") or [])}

    ds_modalities = {m.lower() for m in (dataset.get("modalities") or [])}
    ds_regions = {r.lower() for r in (dataset.get("brain_regions") or [])}

    if ds_modalities & topic_modalities:
        return True
    if ds_regions & topic_regions:
        return True
    return False


def _finding_matches_topic(finding: dict, topic: dict) -> bool:
    """Return True if a finding is relevant to the topic."""
    topic_regions = {r.lower() for r in (topic.get("regions") or [])}
    topic_tasks = {t.lower() for t in (topic.get("tasks") or [])}

    text = " ".join(filter(None, [
        finding.get("finding_text", ""),
        " ".join(finding.get("regions", []) if isinstance(finding.get("regions"), list) else []),
    ])).lower()

    if any(r in text for r in topic_regions):
        return True
    if any(t in text for t in topic_tasks):
        return True
    return False


def build_topic_subgraph(
    topic_id: str,
    topic_taxonomy: dict,
    papers: list[dict],
    findings: list[dict],
    datasets: list[dict],
    citation_edges: list[dict],
) -> TopicSubgraph:
    """Build a rich subgraph for a topic."""
    topics_by_id = {t["id"]: t for t in (topic_taxonomy.get("topics") or [])}
    topic = topics_by_id.get(topic_id, {})
    topic_color = topic.get("color", "#6366f1")
    topic_label = topic.get("label", topic_id)
    muted_region = "#94a3b8"
    muted_method = "#cbd5e1"

    # Filter data by topic relevance
    topic_papers = [p for p in papers if _paper_matches_topic(p, topic)]
    topic_datasets = [d for d in datasets if _dataset_matches_topic(d, topic)]
    topic_findings = [f for f in findings if _finding_matches_topic(f, topic)]

    nodes: dict[str, TopicNode] = {}
    links: dict[str, TopicLink] = {}

    # Paper nodes (size proportional to citation_count)
    paper_ids_in_subgraph: set[str] = set()
    for paper in topic_papers[:80]:
        pid = paper.get("id") or paper.get("openalex_id") or paper.get("doi") or ""
        if not pid:
            continue
        paper_ids_in_subgraph.add(pid)
        cit_count = float(paper.get("citation_count") or paper.get("cited_by_count") or 1)
        size = max(4.0, min(20.0, 4.0 + (cit_count ** 0.4)))
        nodes[f"paper:{pid}"] = TopicNode(
            id=f"paper:{pid}",
            type="paper",
            label=paper.get("title", pid)[:80],
            year=paper.get("publication_year") or paper.get("year"),
            color=topic_color,
            size=size,
            meta={"citation_count": int(cit_count), "doi": paper.get("doi")},
        )

        # Paper -> region links
        for region in (paper.get("brain_regions") or paper.get("regions") or []):
            rkey = f"region:{region.lower().replace(' ', '_')}"
            if rkey not in nodes:
                nodes[rkey] = TopicNode(
                    id=rkey, type="region", label=region, year=None,
                    color=muted_region, size=6.0,
                )
            lkey = f"paper:{pid}|>{rkey}"
            if lkey not in links:
                links[lkey] = TopicLink(
                    source=f"paper:{pid}", target=rkey,
                    type="paper_studies_region", weight=0.8, color=topic_color + "88",
                )

    # Dataset nodes
    for ds in topic_datasets[:30]:
        did = ds.get("id") or ds.get("source_id") or ""
        if not did:
            continue
        year_raw = ds.get("year") or (ds.get("date_published", "") or "")[:4]
        ds_year = int(year_raw) if str(year_raw).isdigit() else None
        nodes[f"dataset:{did}"] = TopicNode(
            id=f"dataset:{did}",
            type="dataset",
            label=ds.get("title", did)[:80],
            year=ds_year,
            color=muted_method,
            size=8.0,
            meta={"source": ds.get("source"), "modalities": ds.get("modalities", [])},
        )
        for region in (ds.get("brain_regions") or []):
            rkey = f"region:{region.lower().replace(' ', '_')}"
            if rkey not in nodes:
                nodes[rkey] = TopicNode(
                    id=rkey, type="region", label=region, year=None,
                    color=muted_region, size=6.0,
                )
            lkey = f"dataset:{did}|>{rkey}"
            if lkey not in links:
                links[lkey] = TopicLink(
                    source=f"dataset:{did}", target=rkey,
                    type="dataset_covers_region", weight=0.9, color=muted_region + "88",
                )

    # Finding nodes (cluster by region, limit to 20)
    for finding in topic_findings[:20]:
        fid = finding.get("finding_id") or finding.get("id") or ""
        if not fid:
            continue
        fregions = finding.get("regions") or []
        nodes[f"finding:{fid}"] = TopicNode(
            id=f"finding:{fid}",
            type="finding",
            label=(finding.get("finding_text") or fid)[:80],
            year=None,
            color=topic_color + "bb",
            size=5.0,
            meta={
                "direction": finding.get("result_direction"),
                "confidence": finding.get("confidence"),
            },
        )
        for region in fregions[:2]:
            rkey = f"region:{region.lower().replace(' ', '_')}"
            if rkey not in nodes:
                nodes[rkey] = TopicNode(
                    id=rkey, type="region", label=region, year=None,
                    color=muted_region, size=6.0,
                )
            lkey = f"finding:{fid}|>{rkey}"
            if lkey not in links:
                links[lkey] = TopicLink(
                    source=f"finding:{fid}", target=rkey,
                    type="finding_in_region", weight=0.7, color=topic_color + "55",
                )

    # Citation edges (paper -> paper, both must be in subgraph)
    for edge in citation_edges:
        src = edge.get("source") or edge.get("citing_id") or ""
        tgt = edge.get("target") or edge.get("cited_id") or ""
        if src in paper_ids_in_subgraph and tgt in paper_ids_in_subgraph:
            lkey = f"paper:{src}|cites|paper:{tgt}"
            if lkey not in links:
                links[lkey] = TopicLink(
                    source=f"paper:{src}", target=f"paper:{tgt}",
                    type="paper_cites", weight=0.5, color="#e2e8f0",
                )

    nodes_list = list(nodes.values())
    links_list = list(links.values())

    years = [n.year for n in nodes_list if n.year and isinstance(n.year, int)]
    year_range = (min(years), max(years)) if years else None

    region_counts: Counter[str] = Counter()
    for lnk in links_list:
        if lnk.target.startswith("region:"):
            region_counts[lnk.target] += 1
    dominant_regions = [k.removeprefix("region:") for k, _ in region_counts.most_common(5)]

    meta: dict[str, Any] = {
        "node_count": len(nodes_list),
        "link_count": len(links_list),
        "year_range": list(year_range) if year_range else None,
        "dominant_regions": dominant_regions,
        "paper_count": len(topic_papers),
        "dataset_count": len(topic_datasets),
        "finding_count": len(topic_findings),
    }

    return TopicSubgraph(
        topic_id=topic_id,
        topic_label=topic_label,
        nodes=nodes_list,
        links=links_list,
        meta=meta,
    )


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def save_all_subgraphs(
    topic_taxonomy_path: Path,
    papers: list[dict],
    findings_path: Path,
    datasets: list[dict],
    citations_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Build and save all topic subgraphs to artifacts/topics/subgraphs.json."""
    if not topic_taxonomy_path.exists():
        return {"error": "topic_taxonomy.yaml not found", "topics_built": 0}

    with topic_taxonomy_path.open(encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f)

    findings = _load_jsonl(findings_path)
    citations = _load_jsonl(citations_path)

    result: dict[str, Any] = {}
    for topic in taxonomy.get("topics") or []:
        tid = topic["id"]
        sg = build_topic_subgraph(tid, taxonomy, papers, findings, datasets, citations)
        result[tid] = {
            "topic_id": sg.topic_id,
            "topic_label": sg.topic_label,
            "nodes": [asdict(n) for n in sg.nodes],
            "links": [asdict(lnk) for lnk in sg.links],
            "meta": sg.meta,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return {
        "topics_built": len(result),
        "output_path": str(output_path),
    }
