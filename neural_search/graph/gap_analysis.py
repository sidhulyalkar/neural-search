"""Gap analysis — identify research opportunities and missing coverage per topic."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ResearchGap:
    gap_id: str
    title: str
    description: str
    topic_id: str
    gap_type: str  # 'missing_region', 'missing_species', 'missing_method', 'analysis_opportunity', 'cross_topic'
    severity: str  # 'critical', 'high', 'medium'
    suggested_action: str
    supporting_datasets: list[str]


def _dataset_matches_topic(dataset: dict, topic: dict) -> bool:
    topic_modalities = {m.lower() for m in (topic.get("modalities") or [])}
    topic_regions = {r.lower() for r in (topic.get("regions") or [])}
    ds_modalities = {m.lower() for m in (dataset.get("modalities") or [])}
    ds_regions = {r.lower() for r in (dataset.get("brain_regions") or [])}
    return bool(ds_modalities & topic_modalities) or bool(ds_regions & topic_regions)


def find_topic_gaps(
    topic_id: str,
    topic: dict,
    coverage: dict,
    papers: list[dict],
    datasets: list[dict],
    all_topics: list[dict],
) -> list[ResearchGap]:
    """Identify research gaps for a topic."""
    gaps: list[ResearchGap] = []
    topic_datasets = [d for d in datasets if _dataset_matches_topic(d, topic)]
    topic_modalities = topic.get("modalities") or []
    topic_regions = topic.get("regions") or []
    companion_topics = topic.get("companion_topics") or []

    # 1. Missing region gaps
    region_coverage = coverage.get("region_coverage") or []
    for rc in region_coverage:
        region_id = rc.get("region_id", "")
        region_label = rc.get("region_label", region_id)
        n_datasets = rc.get("n_datasets", 0)

        if n_datasets == 0:
            severity = "critical" if region_label in (topic.get("regions") or [])[:3] else "high"
            supporting = [
                d.get("id") or d.get("source_id", "")
                for d in topic_datasets
                if any(m.lower() in (topic.get("modalities") or []) for m in (d.get("modalities") or []))
            ][:3]
            gaps.append(ResearchGap(
                gap_id=f"{topic_id}:missing_region:{region_id}",
                title=f"No data for {region_label}",
                description=(
                    f"No datasets cover {region_label}, which is a key region for {topic.get('label', topic_id)}. "
                    f"This leaves a significant blind spot in our corpus."
                ),
                topic_id=topic_id,
                gap_type="missing_region",
                severity=severity,
                suggested_action=f"Search for datasets recording from {region_label} using relevant modalities.",
                supporting_datasets=supporting,
            ))

    # 2. Missing species gaps — modality present but only one species
    species_coverage = coverage.get("species_coverage") or {}
    all_species = list(species_coverage.keys())
    if len(all_species) == 1 and topic_modalities:
        sole_species = all_species[0]
        gaps.append(ResearchGap(
            gap_id=f"{topic_id}:missing_species:single_species",
            title=f"Single-species coverage ({sole_species} only)",
            description=(
                f"All {topic.get('label', topic_id)} datasets use only {sole_species}. "
                "Cross-species validation is missing, limiting generalizability."
            ),
            topic_id=topic_id,
            gap_type="missing_species",
            severity="high",
            suggested_action=(
                f"Find datasets from additional species (e.g., macaque or human) "
                f"using {', '.join(topic_modalities[:2])}."
            ),
            supporting_datasets=[d.get("id") or d.get("source_id", "") for d in topic_datasets[:2]],
        ))

    if not any(sp in species_coverage for sp in ["human", "homo_sapiens", "human (homo sapiens)"]):
        gaps.append(ResearchGap(
            gap_id=f"{topic_id}:missing_species:no_human",
            title="No human data",
            description=(
                f"No human datasets found for {topic.get('label', topic_id)}. "
                "Translational relevance cannot be assessed from this corpus."
            ),
            topic_id=topic_id,
            gap_type="missing_species",
            severity="medium",
            suggested_action=f"Ingest human datasets covering {', '.join(topic_regions[:3])}.",
            supporting_datasets=[],
        ))

    # 3. Analysis opportunities: datasets with right modalities but no findings yet
    method_coverage = coverage.get("method_coverage") or []
    total_findings = coverage.get("total_findings", 0)
    for mc in method_coverage:
        n_using = mc.get("n_datasets_using", 0)
        method_label = mc.get("method_label", mc.get("method_id", ""))
        if n_using >= 2 and total_findings == 0:
            supporting = [
                d.get("id") or d.get("source_id", "")
                for d in topic_datasets
                if any(m.lower() == method_label.lower() for m in (d.get("modalities") or []))
            ][:4]
            gaps.append(ResearchGap(
                gap_id=f"{topic_id}:analysis_opportunity:{mc.get('method_id', 'unknown')}",
                title=f"Untapped {method_label} datasets",
                description=(
                    f"{n_using} datasets use {method_label} for {topic.get('label', topic_id)} "
                    f"but no findings have been extracted yet. These datasets could yield new insights."
                ),
                topic_id=topic_id,
                gap_type="analysis_opportunity",
                severity="medium",
                suggested_action=f"Run finding extraction pipeline on {method_label} datasets for this topic.",
                supporting_datasets=supporting,
            ))

    # 4. Cross-topic bridges: companion topics sharing regions but no joint papers
    all_topics_by_id = {t["id"]: t for t in all_topics}
    topic_region_set = {r.lower() for r in topic_regions}
    for comp_id in companion_topics:
        comp_topic = all_topics_by_id.get(comp_id)
        if not comp_topic:
            continue
        comp_regions = {r.lower() for r in (comp_topic.get("regions") or [])}
        shared_regions = topic_region_set & comp_regions
        if not shared_regions:
            continue
        # Check for papers studying both topics
        both_keywords = {
            topic.get("label", "").lower(),
            comp_topic.get("label", "").lower(),
        }
        joint_papers = [
            p for p in papers
            if all(
                kw in (p.get("title", "") + " " + p.get("abstract", "")).lower()
                for kw in both_keywords
            )
        ]
        if not joint_papers:
            gaps.append(ResearchGap(
                gap_id=f"{topic_id}:cross_topic:{comp_id}",
                title=f"No joint {topic.get('label', topic_id)} + {comp_topic.get('label', comp_id)} studies",
                description=(
                    f"{topic.get('label', topic_id)} and {comp_topic.get('label', comp_id)} share "
                    f"regions ({', '.join(list(shared_regions)[:3])}) but no papers study both simultaneously. "
                    "Cross-topic analysis could reveal new circuit-level insights."
                ),
                topic_id=topic_id,
                gap_type="cross_topic",
                severity="medium",
                suggested_action=(
                    f"Search for datasets that can support simultaneous analysis of "
                    f"{topic.get('label', topic_id)} and {comp_topic.get('label', comp_id)}."
                ),
                supporting_datasets=[
                    d.get("id") or d.get("source_id", "")
                    for d in topic_datasets[:2]
                ],
            ))

    return gaps


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


def find_all_gaps(
    topic_taxonomy_path: Path,
    coverage_path: Path,
    papers: list[dict],
    datasets: list[dict],
    output_path: Path,
) -> list[dict]:
    """Find and save all gaps to artifacts/topics/gaps.json."""
    if not topic_taxonomy_path.exists():
        return []

    with topic_taxonomy_path.open(encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f)

    all_topics = taxonomy.get("topics") or []

    coverage_by_topic: dict[str, dict] = {}
    if coverage_path.exists():
        coverage_by_topic = json.loads(coverage_path.read_text(encoding="utf-8"))

    all_gaps: list[dict] = []
    for topic in all_topics:
        tid = topic["id"]
        coverage = coverage_by_topic.get(tid, {})
        topic_gaps = find_topic_gaps(
            tid, topic, coverage, papers, datasets, all_topics
        )
        all_gaps.extend(asdict(g) for g in topic_gaps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_gaps, indent=2), encoding="utf-8")

    return all_gaps
