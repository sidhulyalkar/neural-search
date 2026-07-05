"""Frontmatter dataclasses and Markdown renderers for the Obsidian vault."""
from __future__ import annotations

from datetime import UTC, date

import yaml


def render_frontmatter(data: dict) -> str:
    """Render a dict as a YAML frontmatter block.

    None values are kept as 'null' so human-owned fields (label, audit_status)
    are visible in the file and can be filled in without searching for them.
    """
    return "---\n" + yaml.dump(data, default_flow_style=False, allow_unicode=True) + "---\n"


def dataset_card_frontmatter(ev) -> dict:
    """Build frontmatter dict from a DatasetEvidence object."""
    today = date.today().isoformat()
    return {
        "type": "dataset",
        "dataset_id": ev.record_id,
        "source": ev.source,
        "species": ev.species or None,
        "modalities": ev.modalities or None,
        "data_levels": ev.data_levels or None,
        "tasks": ev.tasks or None,
        "regions": ev.regions or None,
        "license": ev.license,
        "doi": ev.doi,
        "url": ev.url,
        "raw_data_available": ev.raw_data_available,
        "metadata_completeness": round(ev.metadata_completeness, 2),
        "last_synced": today,
        "tags": ["dataset", ev.source],
    }


def dataset_card_body(ev) -> str:
    desc = ev.description or "_No description available._"
    return f"# {ev.title or ev.record_id}\n\n{desc}\n"


def query_card_frontmatter(spec) -> dict:
    return {
        "type": "query",
        "query_id": spec.query_id,
        "intent": spec.intent,
        "required_modalities": spec.required_modalities or None,
        "required_data_levels": spec.data_level_requirements or None,
        "species_constraints": spec.required_species or None,
        "region_constraints": spec.brain_regions or None,
        "task_constraints": spec.task_constraints or None,
        "hard_negatives": spec.hard_negatives or None,
        "status": "active",
        "tags": ["query", spec.intent.lower()],
    }


def query_card_body(spec) -> str:
    hn_lines = "\n".join(f"- {h}" for h in spec.hard_negatives) or "_None defined_"
    return (
        f"# {spec.query_text}\n\n"
        f"**Scientific goal:** {spec.scientific_goal}\n\n"
        f"## Hard Negatives\n{hn_lines}\n"
    )


def annotation_card_frontmatter(
    annotation_id: str,
    query_id: str,
    record_id: str,
    label: int | None,
    confidence: float | None,
    source: str,
    audit_status: str = "pending",
    judge_version: str = "lf_v1",
) -> dict:
    from datetime import datetime
    return {
        "type": "annotation",
        "annotation_id": annotation_id,
        "query_id": query_id,
        "dataset_id": record_id,
        "label": label,
        "confidence": confidence,
        "source": source,
        "audit_status": audit_status,
        "judge_version": judge_version,
        "created": datetime.now(UTC).date().isoformat(),
        "tags": ["annotation", "audit"],
    }


def annotation_card_body(
    query_text: str,
    scientific_goal: str,
    hard_negatives: list[str],
    dataset_title: str,
    dataset_desc: str | None,
    lf_votes: list[dict],
    ensemble_label: int | None,
    ensemble_confidence: float | None,
    llm_judgment: dict | None = None,
) -> str:
    hn_lines = "\n".join(f"- {h}" for h in hard_negatives) or "_None_"
    vote_lines = "\n".join(
        f"- **{v['lf_name']}**: label={v['label']}, conf={v['confidence']:.2f} — {v['rationale']}"
        for v in lf_votes if not v.get("abstain")
    ) or "_No active votes_"

    llm_section = ""
    if llm_judgment:
        llm_section = (
            f"\n## LLM Judgment\n"
            f"- Label: {llm_judgment.get('label')}, Confidence: {llm_judgment.get('confidence')}\n"
            f"- Rationale: {llm_judgment.get('rationale', '')}\n"
        )

    return (
        f"## Query\n**{query_text}**\n\n"
        f"**Scientific goal:** {scientific_goal}\n\n"
        f"## Hard Negatives\n{hn_lines}\n\n"
        f"## Dataset\n**{dataset_title}**\n\n{dataset_desc or '_No description_'}\n"
        f"\n## Rule Votes\n{vote_lines}\n"
        f"{llm_section}"
        f"\n## Ensemble\n"
        f"- Label: **{ensemble_label}**, Confidence: {ensemble_confidence}\n\n"
        f"## Human Audit Checklist\n"
        f"- [ ] Reviewed query intent and hard negatives\n"
        f"- [ ] Checked dataset modality / species\n"
        f"- [ ] Verified or corrected label in frontmatter\n"
        f"- [ ] Set `audit_status: done` in frontmatter\n\n"
        f"> **Edit in frontmatter:** `label`, `confidence`, `audit_status`\n"
    )


def paper_card_frontmatter(record: dict) -> dict:
    """Build frontmatter dict for a 09_Literature paper card."""
    return {
        "type": "paper",
        "paper_id": record.get("paper_id"),
        "doi": record.get("doi"),
        "title": record.get("title"),
        "authors": record.get("authors") or [],
        "year": record.get("year"),
        "n_findings": record.get("n_findings", 0),
        "finding_ids": record.get("finding_ids") or [],
        "linked_datasets": record.get("linked_datasets") or [],
        "modalities": record.get("modalities") or [],
        "regions": record.get("regions") or [],
        "species": record.get("species") or [],
        "extraction_model": record.get("extraction_model"),
        "extraction_prompt_version": record.get("extraction_prompt_version"),
        "tags": ["paper", "literature"],
    }


def paper_card_body(record: dict) -> str:
    title = record.get("title") or record.get("paper_id") or "Unknown Paper"
    authors = ", ".join(record.get("authors") or []) or "_Unknown_"
    year = record.get("year") or ""
    finding_ids = record.get("finding_ids") or []
    datasets = record.get("linked_datasets") or []

    findings_section = "\n".join(f"- finding_{fid}" for fid in finding_ids) or "_None extracted._"
    datasets_section = "\n".join(f"- [[{d}]]" for d in datasets) or "_None linked._"

    return (
        f"# {title}\n\n"
        f"**Authors:** {authors}  \n"
        f"**Year:** {year}\n\n"
        f"## Findings\n{findings_section}\n\n"
        f"## Linked Datasets\n{datasets_section}\n"
    )


def claim_card_frontmatter(claim: dict) -> dict:
    """Build frontmatter dict for a 10_Claims claim card."""
    return {
        "type": "claim",
        "claim_id": claim.get("claim_id"),
        "statement": claim.get("statement"),
        "direction": claim.get("direction"),
        "regions": claim.get("regions") or [],
        "species": claim.get("species") or [],
        "consensus_confidence": claim.get("consensus_confidence"),
        "n_supporting_findings": claim.get("n_supporting_findings", 0),
        "n_contradicting_findings": claim.get("n_contradicting_findings", 0),
        "magnitude_summary": claim.get("magnitude_summary"),
        "timescale": claim.get("timescale"),
        "evidence_strength": claim.get("evidence_strength"),
        "status": claim.get("status", "active"),
        "supporting_datasets": claim.get("supporting_datasets") or [],
        "supporting_papers": claim.get("supporting_papers") or [],
        "contradicted_by": claim.get("contradicted_by") or [],
        "synthesis_model": claim.get("synthesis_model"),
        "synthesis_prompt_version": claim.get("synthesis_prompt_version"),
        "synthesized_at": claim.get("synthesized_at"),
        "tags": ["claim", claim.get("direction", "other")],
    }


def claim_card_body(claim: dict) -> str:
    statement = claim.get("statement") or "_No statement._"
    agent_digest = claim.get("agent_digest") or "_Not yet generated._"
    supporting = claim.get("supporting_papers") or []
    contradicted = claim.get("contradicted_by") or []
    datasets = claim.get("supporting_datasets") or []

    supporting_section = "\n".join(f"- [[{p}]]" for p in supporting[:20]) or "_None._"
    contradicted_section = "\n".join(f"- [[{c}]]" for c in contradicted) or "_None._"
    datasets_section = "\n".join(f"- [[{d}]]" for d in datasets[:20]) or "_None._"

    return (
        f"# {statement}\n\n"
        f"## Agent Digest\n{agent_digest}\n\n"
        f"## Supporting Datasets\n{datasets_section}\n\n"
        f"## Supporting Papers\n{supporting_section}\n\n"
        f"## Contradicted By\n{contradicted_section}\n"
    )
