"""Coverage statistics per topic — how much of what's known do we have?"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RegionCoverage:
    region_id: str
    region_label: str
    n_datasets: int
    n_papers: int
    n_findings: int
    is_understudied: bool  # n_datasets < 3 and region is in topic


@dataclass
class MethodCoverage:
    method_id: str
    method_label: str
    n_datasets_using: int
    first_year_seen: int | None
    last_year_seen: int | None


@dataclass
class TopicCoverage:
    topic_id: str
    total_papers: int
    total_datasets: int
    total_findings: int
    region_coverage: list[RegionCoverage]
    method_coverage: list[MethodCoverage]
    species_coverage: dict[str, int]
    year_range: tuple[int, int] | None
    coverage_score: float  # 0-1 composite
    data_gaps: list[str]


def _paper_matches_topic(paper: dict, topic: dict) -> bool:
    topic_tasks = {t.lower() for t in (topic.get("tasks") or [])}
    topic_regions = {r.lower() for r in (topic.get("regions") or [])}
    topic_label = topic.get("label", "").lower()
    topic_id = topic.get("id", "").replace("_", " ").lower()

    text = " ".join(filter(None, [
        paper.get("title", ""),
        paper.get("abstract", ""),
    ])).lower()

    if topic_label in text or topic_id in text:
        return True
    if any(task in text for task in topic_tasks):
        return True
    if any(region in text for region in topic_regions):
        return True
    return False


def _dataset_matches_topic(dataset: dict, topic: dict) -> bool:
    topic_modalities = {m.lower() for m in (topic.get("modalities") or [])}
    topic_regions = {r.lower() for r in (topic.get("regions") or [])}
    ds_modalities = {m.lower() for m in (dataset.get("modalities") or [])}
    ds_regions = {r.lower() for r in (dataset.get("brain_regions") or [])}
    return bool(ds_modalities & topic_modalities) or bool(ds_regions & topic_regions)


def _finding_matches_topic(finding: dict, topic: dict) -> bool:
    topic_regions = {r.lower() for r in (topic.get("regions") or [])}
    topic_tasks = {t.lower() for t in (topic.get("tasks") or [])}
    text = " ".join(filter(None, [
        finding.get("finding_text", ""),
        " ".join(finding.get("regions", []) if isinstance(finding.get("regions"), list) else []),
    ])).lower()
    return any(r in text for r in topic_regions) or any(t in text for t in topic_tasks)


def compute_topic_coverage(
    topic_id: str,
    topic: dict,
    papers: list[dict],
    findings: list[dict],
    datasets: list[dict],
) -> TopicCoverage:
    """Compute coverage statistics for a single topic."""
    topic_regions = topic.get("regions") or []
    topic_modalities = topic.get("modalities") or []

    topic_papers = [p for p in papers if _paper_matches_topic(p, topic)]
    topic_datasets = [d for d in datasets if _dataset_matches_topic(d, topic)]
    topic_findings = [f for f in findings if _finding_matches_topic(f, topic)]

    # Region coverage
    region_coverage_list: list[RegionCoverage] = []
    for region in topic_regions:
        region_lower = region.lower()
        n_ds = sum(
            1 for d in topic_datasets
            if any(r.lower() == region_lower for r in (d.get("brain_regions") or []))
        )
        n_papers = sum(
            1 for p in topic_papers
            if any(r.lower() == region_lower for r in (p.get("brain_regions") or p.get("regions") or []))
        )
        n_findings = sum(
            1 for f in topic_findings
            if any(r.lower() == region_lower for r in (f.get("regions") or []))
        )
        region_coverage_list.append(RegionCoverage(
            region_id=region.lower().replace(" ", "_"),
            region_label=region,
            n_datasets=n_ds,
            n_papers=n_papers,
            n_findings=n_findings,
            is_understudied=(n_ds < 3),
        ))

    # Method/modality coverage
    method_coverage_list: list[MethodCoverage] = []
    for modality in topic_modalities:
        modality_lower = modality.lower()
        using_datasets = [
            d for d in topic_datasets
            if any(m.lower() == modality_lower for m in (d.get("modalities") or []))
        ]
        years: list[int] = []
        for d in using_datasets:
            raw = d.get("year") or (d.get("date_published", "") or "")[:4]
            if str(raw).isdigit():
                years.append(int(raw))
        method_coverage_list.append(MethodCoverage(
            method_id=modality.lower().replace(" ", "_"),
            method_label=modality,
            n_datasets_using=len(using_datasets),
            first_year_seen=min(years) if years else None,
            last_year_seen=max(years) if years else None,
        ))

    # Species coverage
    species_counts: dict[str, int] = {}
    for d in topic_datasets:
        for sp in (d.get("species") or []):
            species_counts[sp.lower()] = species_counts.get(sp.lower(), 0) + 1

    # Year range
    all_years: list[int] = []
    for p in topic_papers:
        yr = p.get("publication_year") or p.get("year")
        if isinstance(yr, int):
            all_years.append(yr)
    year_range = (min(all_years), max(all_years)) if all_years else None

    # Coverage score components
    n_regions = len(topic_regions)
    n_modalities = len(topic_modalities)

    region_ratio = (
        sum(1 for rc in region_coverage_list if rc.n_datasets >= 1) / n_regions
        if n_regions > 0 else 0.0
    )
    method_diversity = (
        sum(1 for mc in method_coverage_list if mc.n_datasets_using >= 1) / n_modalities
        if n_modalities > 0 else 0.0
    )
    temporal_span = (
        min(1.0, (year_range[1] - year_range[0]) / 20.0)
        if year_range else 0.0
    )
    findings_density = min(1.0, len(topic_findings) / 50.0)

    coverage_score = round(
        0.35 * region_ratio
        + 0.25 * method_diversity
        + 0.20 * temporal_span
        + 0.20 * findings_density,
        3,
    )

    # Data gaps
    data_gaps: list[str] = []
    for rc in region_coverage_list:
        if rc.n_datasets == 0:
            data_gaps.append(f"No datasets for {rc.region_label} (topic region)")
        elif rc.n_datasets < 3:
            data_gaps.append(
                f"Only {rc.n_datasets} dataset(s) covering {rc.region_label}"
            )

    for mc in method_coverage_list:
        if mc.n_datasets_using == 0:
            data_gaps.append(f"No datasets using {mc.method_label} for this topic")
        elif mc.n_datasets_using == 1:
            data_gaps.append(f"Only 1 dataset using {mc.method_label} for this topic")

    if "human" not in species_counts and "homo_sapiens" not in species_counts:
        data_gaps.append("No human species data")

    if year_range and year_range[1] < 2020:
        data_gaps.append(f"No data after {year_range[1]}")

    if not topic_papers:
        data_gaps.append("No matching papers found in corpus")

    return TopicCoverage(
        topic_id=topic_id,
        total_papers=len(topic_papers),
        total_datasets=len(topic_datasets),
        total_findings=len(topic_findings),
        region_coverage=region_coverage_list,
        method_coverage=method_coverage_list,
        species_coverage=species_counts,
        year_range=year_range,
        coverage_score=coverage_score,
        data_gaps=data_gaps,
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


def compute_all_coverage(
    topic_taxonomy_path: Path,
    papers: list[dict],
    findings_path: Path,
    datasets: list[dict],
    output_path: Path,
) -> dict[str, Any]:
    """Compute and save coverage for all topics to artifacts/topics/coverage.json."""
    if not topic_taxonomy_path.exists():
        return {"error": "topic_taxonomy.yaml not found", "topics_computed": 0}

    with topic_taxonomy_path.open(encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f)

    findings = _load_jsonl(findings_path)

    result: dict[str, Any] = {}
    for topic in taxonomy.get("topics") or []:
        tid = topic["id"]
        cov = compute_topic_coverage(tid, topic, papers, findings, datasets)
        cov_dict = asdict(cov)
        # year_range is a tuple, JSON needs a list
        if cov_dict.get("year_range"):
            cov_dict["year_range"] = list(cov_dict["year_range"])
        result[tid] = cov_dict

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return {
        "topics_computed": len(result),
        "output_path": str(output_path),
    }
