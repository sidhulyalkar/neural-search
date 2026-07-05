"""Timeline utilities for the KG: topic-based temporal discovery views."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TimelineEntry:
    year: int
    papers: list[dict]            # {id, title, doi, citation_count}
    findings: list[dict]          # {text, region, direction, confidence}
    datasets: list[dict]          # {id, title, source}
    methods_introduced: list[str] # modalities/methods newly appearing this year


@dataclass
class TopicTimeline:
    topic_id: str
    topic_label: str
    entries: list[TimelineEntry]  # sorted by year
    total_papers: int
    total_findings: int
    total_datasets: int
    year_range: tuple[int, int]
    key_regions: list[str]        # top 5 most mentioned
    key_methods: list[str]        # top 5 modalities


def _topic_matches(item_tags: list[str], topic_tasks: list[str], topic_regions: list[str]) -> bool:
    """Return True if item_tags overlap with topic tasks or regions (case-insensitive)."""
    tags_lower = {t.lower() for t in item_tags}
    for task in topic_tasks:
        if task.lower() in tags_lower:
            return True
    for region in topic_regions:
        if region.lower() in tags_lower:
            return True
    return False


def build_topic_timeline(
    topic_id: str,
    topic_taxonomy: dict,
    papers: list[dict],
    findings: list[dict],
    datasets: list[dict],
) -> TopicTimeline:
    """Build a temporal view of a topic from corpus and findings data."""
    # Locate this topic in the taxonomy
    topic_def: dict[str, Any] = {}
    for t in topic_taxonomy.get("topics", []):
        if t.get("id") == topic_id:
            topic_def = t
            break

    topic_label = topic_def.get("label", topic_id)
    topic_tasks: list[str] = topic_def.get("tasks", [])
    topic_regions: list[str] = topic_def.get("regions", [])

    # Filter papers
    topic_papers = []
    for paper in papers:
        paper_tasks = paper.get("tasks") or paper.get("task_labels") or []
        paper_regions = paper.get("brain_regions") or paper.get("regions") or []
        all_tags = list(paper_tasks) + list(paper_regions)
        if _topic_matches(all_tags, topic_tasks, topic_regions):
            topic_papers.append(paper)

    # Filter findings
    topic_findings = []
    for f in findings:
        f_tasks = [f.get("task") or ""]
        f_regions = f.get("regions") or [f.get("region") or ""]
        all_tags = f_tasks + f_regions
        if _topic_matches(all_tags, topic_tasks, topic_regions):
            topic_findings.append(f)

    # Filter datasets
    topic_datasets = []
    for ds in datasets:
        ds_tasks = ds.get("tasks") or []
        ds_regions = ds.get("brain_regions") or []
        all_tags = list(ds_tasks) + list(ds_regions)
        if _topic_matches(all_tags, topic_tasks, topic_regions):
            topic_datasets.append(ds)

    # Group by year
    by_year_papers: dict[int, list[dict]] = defaultdict(list)
    by_year_findings: dict[int, list[dict]] = defaultdict(list)
    by_year_datasets: dict[int, list[dict]] = defaultdict(list)

    for paper in topic_papers:
        year = paper.get("publication_year") or paper.get("year")
        if year and isinstance(year, int):
            by_year_papers[year].append({
                "id": paper.get("openalex_id") or paper.get("id", ""),
                "title": paper.get("title", ""),
                "doi": paper.get("doi"),
                "citation_count": paper.get("cited_by_count") or paper.get("citation_count") or 0,
            })

    for f in topic_findings:
        year = f.get("year") or f.get("publication_year")
        if year and isinstance(year, int):
            by_year_findings[year].append({
                "text": f.get("finding_text") or f.get("text", ""),
                "region": (f.get("regions") or [f.get("region", "")])[0],
                "direction": f.get("result_direction") or f.get("direction", ""),
                "confidence": f.get("confidence", 0.0),
            })

    for ds in topic_datasets:
        year = ds.get("year") or ds.get("publication_year")
        if year and isinstance(year, int):
            by_year_datasets[year].append({
                "id": ds.get("source_id") or ds.get("id", ""),
                "title": ds.get("title", ""),
                "source": ds.get("source", ""),
            })

    # Track methods per year to detect "first appearance"
    all_years = sorted(
        set(by_year_papers) | set(by_year_findings) | set(by_year_datasets)
    )
    seen_methods: set[str] = set()
    entries: list[TimelineEntry] = []
    for year in all_years:
        year_methods: set[str] = set()
        for paper in topic_papers:
            if (paper.get("publication_year") or paper.get("year")) == year:
                for mod in paper.get("modalities") or []:
                    year_methods.add(mod)
        new_methods = sorted(year_methods - seen_methods)
        seen_methods |= year_methods
        entries.append(TimelineEntry(
            year=year,
            papers=by_year_papers.get(year, []),
            findings=by_year_findings.get(year, []),
            datasets=by_year_datasets.get(year, []),
            methods_introduced=new_methods,
        ))

    # Compute summary stats
    region_counter: Counter[str] = Counter()
    method_counter: Counter[str] = Counter()
    for paper in topic_papers:
        for r in paper.get("brain_regions") or []:
            region_counter[r] += 1
        for m in paper.get("modalities") or []:
            method_counter[m] += 1

    year_range = (min(all_years), max(all_years)) if all_years else (0, 0)

    return TopicTimeline(
        topic_id=topic_id,
        topic_label=topic_label,
        entries=entries,
        total_papers=len(topic_papers),
        total_findings=len(topic_findings),
        total_datasets=len(topic_datasets),
        year_range=year_range,
        key_regions=[r for r, _ in region_counter.most_common(5)],
        key_methods=[m for m, _ in method_counter.most_common(5)],
    )


def build_all_topic_timelines(
    topic_taxonomy_path: Path,
    corpus_papers: list[dict],
    findings: list[dict],
    corpus_datasets: list[dict],
    output_path: Path,
) -> None:
    """Build and save all topic timelines to artifacts/topics/timelines.json."""
    with topic_taxonomy_path.open(encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f)

    topic_ids = [t["id"] for t in taxonomy.get("topics", [])]
    logger.info("Building timelines for %d topics", len(topic_ids))

    timelines: dict[str, Any] = {}
    for topic_id in topic_ids:
        tl = build_topic_timeline(topic_id, taxonomy, corpus_papers, findings, corpus_datasets)
        timelines[topic_id] = {
            **asdict(tl),
            "year_range": list(tl.year_range),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(timelines, indent=2), encoding="utf-8")
    logger.info("Saved timelines to %s", output_path)


def compute_citation_depth(
    paper_id: str,
    citation_edges: list[dict],
    max_depth: int = 3,
) -> dict[str, int]:
    """BFS from paper_id through citation_edges. Returns {paper_id: depth}."""
    # Build adjacency: citing → [cited]
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in citation_edges:
        citing = edge.get("citing_paper_id", "")
        cited = edge.get("cited_paper_id", "")
        if citing and cited:
            adjacency[citing].append(cited)

    visited: dict[str, int] = {paper_id: 0}
    queue = [paper_id]
    while queue:
        current = queue.pop(0)
        depth = visited[current]
        if depth >= max_depth:
            continue
        for neighbor in adjacency.get(current, []):
            if neighbor not in visited:
                visited[neighbor] = depth + 1
                queue.append(neighbor)

    return visited


def get_intellectual_ancestors(
    topic_id: str,
    papers: list[dict],
    citation_edges: list[dict],
) -> list[dict]:
    """
    Find foundational papers for a topic.

    Papers cited by 3+ topic papers (highest in-degree within topic).
    Returns top 10 sorted by in-degree.
    """
    # Collect all openalex IDs for topic papers (simplified: all papers passed in)
    topic_paper_ids: set[str] = set()
    for paper in papers:
        oa_id = paper.get("openalex_id") or paper.get("id")
        if oa_id:
            topic_paper_ids.add(str(oa_id).replace("https://openalex.org/", ""))

    # Count in-degree: how many topic papers cite each candidate
    in_degree: Counter[str] = Counter()
    for edge in citation_edges:
        citing = edge.get("citing_paper_id", "")
        cited = edge.get("cited_paper_id", "")
        if citing in topic_paper_ids and cited:
            in_degree[cited] += 1

    # Select papers cited by 3+ topic papers
    paper_index: dict[str, dict] = {}
    for paper in papers:
        oa_id = paper.get("openalex_id") or paper.get("id")
        if oa_id:
            short_id = str(oa_id).replace("https://openalex.org/", "")
            paper_index[short_id] = paper

    ancestors = []
    for pid, count in in_degree.most_common(20):
        if count < 3:
            break
        paper = paper_index.get(pid)
        if paper:
            ancestors.append({
                "id": pid,
                "title": paper.get("title", ""),
                "doi": paper.get("doi"),
                "year": paper.get("publication_year"),
                "cited_by_topic_papers": count,
                "citation_count": paper.get("cited_by_count", 0),
            })
        if len(ancestors) >= 10:
            break

    return ancestors
