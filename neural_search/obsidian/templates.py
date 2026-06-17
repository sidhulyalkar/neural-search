"""Frontmatter dataclasses and Markdown renderers for the Obsidian vault."""
from __future__ import annotations

from datetime import date

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
    from datetime import datetime, timezone
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
        "created": datetime.now(timezone.utc).date().isoformat(),
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
