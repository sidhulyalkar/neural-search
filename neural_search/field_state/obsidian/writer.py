"""Export field-state artifacts to a local Obsidian memory mirror."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neural_search.field_state.memory.index import write_memory_index
from neural_search.field_state.obsidian.dashboard import write_dashboard
from neural_search.field_state.obsidian.frontmatter import (
    compose_note,
    parse_frontmatter,
)
from neural_search.field_state.obsidian.paths import (
    ensure_vault_structure,
    field_state_id,
    note_path,
    vault_relative,
)
from neural_search.field_state.obsidian.reader import extract_human_block
from neural_search.field_state.obsidian.sync import append_sync_log
from neural_search.field_state.obsidian.templates import (
    default_human_block,
    render_benchmark_gap_note,
    render_claim_note,
    render_opportunity_note,
    render_snapshot_note,
)
from neural_search.field_state.reports import weak_claims
from neural_search.field_state.scoring import rank_opportunities
from neural_search.field_state.store import (
    read_benchmark_gaps,
    read_claims,
    read_opportunities,
)

SAFE_HUMAN_FRONTMATTER_FIELDS = {
    "status",
    "review_status",
    "human_reviewer",
    "reviewed_at",
    "human_priority",
    "human_tags",
}


@dataclass
class ObsidianExportResult:
    """Result summary for an Obsidian export."""

    vault_path: Path
    memory_index_path: Path
    notes_created: int = 0
    notes_updated: int = 0
    notes_skipped: int = 0
    warnings: list[str] = dataclass_field(default_factory=list)


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write text by replacing from a temp file in the same folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def _existing_note_state(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, None
    text = path.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(text)
    return frontmatter, extract_human_block(body) or None


def _merge_frontmatter(
    generated: dict[str, Any],
    existing: dict[str, Any],
    warnings: list[str],
    path: Path,
) -> dict[str, Any]:
    merged = dict(generated)
    for key in SAFE_HUMAN_FRONTMATTER_FIELDS:
        if key in existing and existing[key] != generated.get(key):
            merged[key] = existing[key]
            if key in {"status", "review_status"}:
                warnings.append(f"preserved human {key} in {path}")
    if existing.get("created_at"):
        merged["created_at"] = existing["created_at"]
    return merged


def _write_generated_note(
    *,
    path: Path,
    frontmatter: dict[str, Any],
    body: str,
    warnings: list[str],
) -> str:
    existing_frontmatter, human_block = _existing_note_state(path)
    merged_frontmatter = _merge_frontmatter(frontmatter, existing_frontmatter, warnings, path)
    if human_block is None:
        human_block = default_human_block(str(frontmatter["type"]))
    # Body renderers already include a human block. Re-render with preserved block
    # by replacing the default block at call time, not by post-hoc string surgery.
    if default_human_block(str(frontmatter["type"])) in body:
        body = body.replace(default_human_block(str(frontmatter["type"])), human_block)
    text = compose_note(merged_frontmatter, body)
    existed = path.exists()
    if existed and path.read_text(encoding="utf-8") == text:
        return "skipped"
    atomic_write_text(path, text)
    return "updated" if existed else "created"


def _base_frontmatter(
    *,
    note_type: str,
    item_id: str,
    field: str,
    title: str,
    status: str,
    source_artifacts: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    frontmatter: dict[str, Any] = {
        "type": note_type,
        "field_state_id": field_state_id(note_type, item_id, title),
        "field": field,
        "title": title,
        "status": status,
        "review_status": "unreviewed",
        "created_at": now,
        "updated_at": now,
        "generated_at": now,
        "source_artifacts": source_artifacts,
        "schema_version": "0.2",
        "tags": ["field-state", note_type],
    }
    if extra:
        frontmatter.update(extra)
    return frontmatter


def _record_write(result: ObsidianExportResult, outcome: str) -> None:
    if outcome == "created":
        result.notes_created += 1
    elif outcome == "updated":
        result.notes_updated += 1
    else:
        result.notes_skipped += 1


def _export_claims(
    vault_path: Path,
    field: str,
    result: ObsidianExportResult,
    root: Path | None,
) -> None:
    claims = read_claims(root)
    gaps = read_benchmark_gaps(root)
    opportunities = read_opportunities(root)
    gaps_by_claim = {
        claim.claim_id: [gap.gap_id for gap in gaps if claim.claim_id in gap.related_claim_ids]
        for claim in claims
    }
    opportunities_by_claim = {
        claim.claim_id: [
            opportunity.opportunity_id
            for opportunity in opportunities
            if claim.claim_id in opportunity.linked_claim_ids
        ]
        for claim in claims
    }
    for claim in claims:
        title = claim.claim_text.rstrip(".")
        status = str(claim.status)
        path = note_path(vault_path, "claim", title, status)
        frontmatter = _base_frontmatter(
            note_type="claim",
            item_id=claim.claim_id,
            field=field,
            title=title,
            status=status,
            source_artifacts=claim.related_artifacts,
            extra={
                "claim_id": claim.claim_id,
                "evidence_level": str(claim.evidence_level),
                "confidence": claim.confidence,
                "missing_tests": claim.missing_tests,
                "related_benchmark_gaps": gaps_by_claim.get(claim.claim_id, []),
                "related_opportunities": opportunities_by_claim.get(claim.claim_id, []),
            },
        )
        body = render_claim_note(
            title=title,
            claim_text=claim.claim_text,
            claim_type="scientific_retrieval_claim",
            evidence_level=str(claim.evidence_level),
            confidence=claim.confidence,
            status=status,
            missing_tests=claim.missing_tests,
            supporting_artifacts=claim.related_artifacts,
            related_benchmark_gaps=gaps_by_claim.get(claim.claim_id, []),
            related_opportunities=opportunities_by_claim.get(claim.claim_id, []),
        )
        _record_write(
            result,
            _write_generated_note(
                path=path,
                frontmatter=frontmatter,
                body=body,
                warnings=result.warnings,
            ),
        )


def _export_gaps(
    vault_path: Path,
    field: str,
    result: ObsidianExportResult,
    root: Path | None,
) -> None:
    gaps = read_benchmark_gaps(root)
    opportunities = read_opportunities(root)
    opportunities_by_gap = {
        gap.gap_id: [
            opportunity.opportunity_id
            for opportunity in opportunities
            if gap.gap_id in opportunity.linked_gap_ids
        ]
        for gap in gaps
    }
    for gap in gaps:
        status = str(gap.status)
        path = note_path(vault_path, "benchmark_gap", gap.title, status)
        frontmatter = _base_frontmatter(
            note_type="benchmark_gap",
            item_id=gap.gap_id,
            field=field,
            title=gap.title,
            status=status,
            source_artifacts=gap.available_artifacts,
            extra={
                "gap_id": gap.gap_id,
                "severity": gap.severity,
                "why_it_matters": gap.why_it_matters,
                "related_claims": gap.related_claim_ids,
                "related_opportunities": opportunities_by_gap.get(gap.gap_id, []),
            },
        )
        body = render_benchmark_gap_note(
            title=gap.title,
            description=gap.description,
            gap_type="benchmark_gap",
            why_it_matters=gap.why_it_matters,
            required_validation=gap.blocking_questions,
            minimum_viable_benchmark="Create the smallest reviewable artifact that closes this gap.",
            related_claims=gap.related_claim_ids,
            related_opportunities=opportunities_by_gap.get(gap.gap_id, []),
            source_artifacts=gap.available_artifacts,
        )
        _record_write(
            result,
            _write_generated_note(
                path=path,
                frontmatter=frontmatter,
                body=body,
                warnings=result.warnings,
            ),
        )


def _export_opportunities(
    vault_path: Path,
    field: str,
    result: ObsidianExportResult,
    root: Path | None,
) -> None:
    for opportunity in rank_opportunities(read_opportunities(root)):
        status = str(opportunity.status)
        path = note_path(vault_path, "opportunity", opportunity.title, status)
        frontmatter = _base_frontmatter(
            note_type="opportunity",
            item_id=opportunity.opportunity_id,
            field=field,
            title=opportunity.title,
            status=status,
            source_artifacts=[],
            extra={
                "opportunity_id": opportunity.opportunity_id,
                "linked_claim_ids": opportunity.linked_claim_ids,
                "linked_gap_ids": opportunity.linked_gap_ids,
                "novelty_score": opportunity.novelty_score,
                "feasibility_score": opportunity.feasibility_score,
                "impact_score": opportunity.impact_score,
                "uncertainty_reduction_score": opportunity.uncertainty_reduction_score,
                "personal_fit_score": opportunity.personal_fit_score,
                "risk_score": opportunity.risk_score,
                "total_score": opportunity.total_score,
            },
        )
        body = render_opportunity_note(
            title=opportunity.title,
            hypothesis=opportunity.description,
            opportunity_type="field_state_opportunity",
            rationale=opportunity.rationale or "No rationale recorded.",
            novelty_score=opportunity.novelty_score,
            feasibility_score=opportunity.feasibility_score,
            impact_score=opportunity.impact_score,
            uncertainty_reduction_score=opportunity.uncertainty_reduction_score,
            personal_fit_score=opportunity.personal_fit_score,
            risk_score=opportunity.risk_score,
            total_score=opportunity.total_score,
            minimum_viable_experiment=opportunity.next_step,
            next_action=opportunity.next_step,
            related_claims=opportunity.linked_claim_ids,
            related_benchmark_gaps=opportunity.linked_gap_ids,
            related_artifacts=[],
            codex_task_stub_or_link=(
                f"`python -m neural_search.field_state.cli export-task "
                f"--opportunity-id {opportunity.opportunity_id} --vault <vault>`"
            ),
        )
        _record_write(
            result,
            _write_generated_note(
                path=path,
                frontmatter=frontmatter,
                body=body,
                warnings=result.warnings,
            ),
        )


def _export_snapshots(
    vault_path: Path,
    field: str,
    result: ObsidianExportResult,
    root: Path | None,
) -> None:
    claims = read_claims(root)
    gaps = read_benchmark_gaps(root)
    opportunities = rank_opportunities(read_opportunities(root))
    date = datetime.now(UTC).date().isoformat()
    top_opportunities = [f"{item.title} ({item.total_score:.3f})" for item in opportunities[:5]]
    weak = [claim.claim_text for claim in weak_claims(claims)[:5]]
    open_gaps = [gap.title for gap in gaps if str(gap.status) != "addressed"]
    actions = [item.next_step for item in opportunities[:3]]
    for title, path, snapshot_id in [
        (
            "latest_snapshot",
            note_path(vault_path, "field_snapshot", "latest_snapshot", "current", latest_snapshot=True),
            "latest",
        ),
        (
            f"{field} {date}",
            note_path(vault_path, "field_snapshot", f"{field} {date}", "current"),
            date,
        ),
    ]:
        frontmatter = _base_frontmatter(
            note_type="field_snapshot",
            item_id=f"{field}-{snapshot_id}",
            field=field,
            title=title,
            status="current",
            source_artifacts=[
                "artifacts/field_state/claims.jsonl",
                "artifacts/field_state/benchmark_gaps.jsonl",
                "artifacts/field_state/opportunities.jsonl",
                "reports/field_state/latest_snapshot.md",
            ],
            extra={"snapshot_id": f"{field}-{snapshot_id}"},
        )
        body = render_snapshot_note(
            field=field,
            date=date,
            summary="Current field-state snapshot generated from local JSONL artifacts.",
            top_opportunities=top_opportunities,
            weak_claims=weak,
            benchmark_gaps=open_gaps,
            recommended_next_actions=actions,
            snapshot_diff="Use `memory-diff` to compare against the memory index.",
        )
        _record_write(
            result,
            _write_generated_note(
                path=path,
                frontmatter=frontmatter,
                body=body,
                warnings=result.warnings,
            ),
        )


def export_obsidian_memory(
    vault_path: Path,
    field: str = "neuroscience_dataset_reuse",
    artifacts_dir: Path = Path("artifacts/field_state"),
    reports_dir: Path = Path("reports/field_state"),
    root: Path | None = None,
) -> ObsidianExportResult:
    """Export field-state artifacts to an Obsidian-compatible Markdown vault."""
    del artifacts_dir, reports_dir
    ensure_vault_structure(vault_path)
    result = ObsidianExportResult(
        vault_path=vault_path,
        memory_index_path=vault_path / "Field-State/90_System/memory_index.json",
    )
    _export_claims(vault_path, field, result, root)
    _export_gaps(vault_path, field, result, root)
    _export_opportunities(vault_path, field, result, root)
    _export_snapshots(vault_path, field, result, root)
    result.memory_index_path = write_memory_index(vault_path, field)
    write_dashboard(vault_path, field)
    append_sync_log(
        vault_path,
        operation="export-obsidian",
        field=field,
        notes_created=result.notes_created,
        notes_updated=result.notes_updated,
        notes_skipped=result.notes_skipped,
        warnings=result.warnings,
    )
    # Keep this import visible to mypy as a used path helper through export summaries.
    _ = vault_relative(vault_path, result.memory_index_path)
    return result
