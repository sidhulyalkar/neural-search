"""Markdown report generation for field-state artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from neural_search.field_state.memory.review_overlay import (
    OVERLAY_PATHS,
    ReviewOverlay,
)
from neural_search.field_state.schemas import (
    BenchmarkGap,
    EvidenceLevel,
    FieldClaim,
    FieldOpportunity,
)
from neural_search.field_state.scoring import (
    OPPORTUNITY_SCORE_WEIGHTS,
    rank_opportunities,
)
from neural_search.field_state.seeds import KNOWN_INPUTS
from neural_search.field_state.store import (
    BENCHMARK_GAPS_REPORT,
    LATEST_SNAPSHOT_REPORT,
    REPORT_DIR,
    TOP_OPPORTUNITIES_REPORT,
    WEAK_CLAIMS_REPORT,
    read_benchmark_gaps,
    read_claims,
    read_jsonl,
    read_opportunities,
    resolve_path,
)


def artifact_inventory(root: Path | None = None) -> dict[str, bool]:
    """Report whether known field-state inputs exist."""
    base = root or Path.cwd()
    return {item: (base / item).exists() for item in KNOWN_INPUTS}


def weak_claims(claims: list[FieldClaim]) -> list[FieldClaim]:
    """Return claims that most need validation."""
    weak_levels = {EvidenceLevel.HYPOTHESIS, EvidenceLevel.PLAUSIBLE}
    return sorted(
        [
            claim
            for claim in claims
            if claim.evidence_level in weak_levels
            or claim.confidence < 0.7
            or claim.missing_tests
        ],
        key=lambda claim: (claim.confidence, claim.claim_id),
    )


def _bullet_list(items: list[str], empty: str = "none") -> list[str]:
    if not items:
        return [f"- {empty}"]
    return [f"- {item}" for item in items]


def read_review_overlays(
    note_type: str,
    root: Path | None = None,
) -> list[ReviewOverlay]:
    """Read review overlays for one note type."""
    path = OVERLAY_PATHS.get(note_type)
    if path is None:
        return []
    return read_jsonl(path, ReviewOverlay, root)


def overlay_by_record_id(overlays: list[ReviewOverlay]) -> dict[str, ReviewOverlay]:
    """Index overlays by generated record ID and field-state ID."""
    indexed: dict[str, ReviewOverlay] = {}
    for overlay in overlays:
        indexed[overlay.field_state_id] = overlay
        if overlay.source_record_id:
            indexed[overlay.source_record_id] = overlay
    return indexed


def _review_lines(overlay: ReviewOverlay | None) -> list[str]:
    if overlay is None:
        return ["- Human review: none"]
    lines = [
        f"- Human review: `{overlay.review_status}`",
        f"- Human status: `{overlay.status}`",
        f"- Source note: `{overlay.source_note_path}`",
    ]
    if overlay.human_priority:
        lines.append(f"- Human priority: `{overlay.human_priority}`")
    if overlay.human_tags:
        lines.append(f"- Human tags: {', '.join(overlay.human_tags)}")
    return lines


def render_weak_claims_report(
    claims: list[FieldClaim],
    overlays: list[ReviewOverlay] | None = None,
) -> str:
    """Render the weak-claims Markdown report."""
    overlay_index = overlay_by_record_id(overlays or [])
    lines = [
        "# Weak Claims",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "Claims are listed when they have low evidence, low confidence, or unresolved tests.",
        "",
    ]
    for claim in weak_claims(claims):
        lines.extend(
            [
                f"## {claim.claim_text}",
                "",
                f"- ID: `{claim.claim_id}`",
                f"- Evidence level: `{claim.evidence_level}`",
                f"- Confidence: {claim.confidence:.2f}",
                f"- Status: `{claim.status}`",
                *_review_lines(overlay_index.get(claim.claim_id)),
                "- Related artifacts:",
                *_bullet_list(claim.related_artifacts),
                "- Missing tests:",
                *_bullet_list(claim.missing_tests),
                "",
            ]
        )
    return "\n".join(lines)


def render_benchmark_gaps_report(
    gaps: list[BenchmarkGap],
    overlays: list[ReviewOverlay] | None = None,
) -> str:
    """Render the benchmark-gaps Markdown report."""
    overlay_index = overlay_by_record_id(overlays or [])
    sorted_gaps = sorted(gaps, key=lambda gap: (-gap.severity, gap.gap_id))
    lines = [
        "# Benchmark Gaps",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
    ]
    for gap in sorted_gaps:
        lines.extend(
            [
                f"## {gap.title}",
                "",
                gap.description,
                "",
                f"- ID: `{gap.gap_id}`",
                f"- Severity: {gap.severity:.2f}",
                f"- Status: `{gap.status}`",
                f"- Why it matters: {gap.why_it_matters}",
                *_review_lines(overlay_index.get(gap.gap_id)),
                "- Expected artifacts:",
                *_bullet_list(gap.expected_artifacts),
                "- Available artifacts:",
                *_bullet_list(gap.available_artifacts),
                "- Blocking questions:",
                *_bullet_list(gap.blocking_questions),
                "",
            ]
        )
    return "\n".join(lines)


def render_top_opportunities_report(
    opportunities: list[FieldOpportunity],
    limit: int | None = None,
    overlays: list[ReviewOverlay] | None = None,
) -> str:
    """Render the ranked opportunities Markdown report."""
    ranked = rank_opportunities(opportunities)
    overlay_index = overlay_by_record_id(overlays or [])
    if limit is not None:
        ranked = ranked[:limit]
    lines = [
        "# Top Opportunities",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "Scoring formula:",
        "",
        "`total_score = 0.20 * novelty_score + 0.25 * feasibility_score + 0.20 * impact_score + 0.15 * uncertainty_reduction_score + 0.15 * personal_fit_score - 0.10 * risk_score`",
        "",
        "| Rank | Opportunity | Total | Review | Priority | Novelty | Feasibility | Impact | Uncertainty | Fit | Risk |",
        "| ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, opportunity in enumerate(ranked, start=1):
        overlay = overlay_index.get(opportunity.opportunity_id)
        review_status = overlay.review_status if overlay else "unreviewed"
        priority = overlay.human_priority if overlay and overlay.human_priority else ""
        lines.append(
            f"| {index} | {opportunity.title} | {opportunity.total_score:.3f} | "
            f"{review_status} | {priority} | "
            f"{opportunity.novelty_score:.1f} | {opportunity.feasibility_score:.1f} | "
            f"{opportunity.impact_score:.1f} | {opportunity.uncertainty_reduction_score:.1f} | "
            f"{opportunity.personal_fit_score:.1f} | {opportunity.risk_score:.1f} |"
        )
    lines.append("")
    lines.append("## Rationale")
    lines.append("")
    for opportunity in ranked:
        lines.extend(
            [
                f"### {opportunity.title}",
                "",
                opportunity.description,
                "",
                f"- ID: `{opportunity.opportunity_id}`",
                f"- Next step: {opportunity.next_step}",
                f"- Rationale: {opportunity.rationale or 'none'}",
                *_review_lines(overlay_index.get(opportunity.opportunity_id)),
                "",
            ]
        )
    lines.append("Weights:")
    for field_name, weight in OPPORTUNITY_SCORE_WEIGHTS.items():
        lines.append(f"- `{field_name}`: {weight:+.2f}")
    return "\n".join(lines)


def render_latest_snapshot_report(
    claims: list[FieldClaim],
    gaps: list[BenchmarkGap],
    opportunities: list[FieldOpportunity],
    root: Path | None = None,
    claim_overlays: list[ReviewOverlay] | None = None,
    gap_overlays: list[ReviewOverlay] | None = None,
    opportunity_overlays: list[ReviewOverlay] | None = None,
) -> str:
    """Render a compact field-state snapshot."""
    inventory = artifact_inventory(root)
    ranked = rank_opportunities(opportunities)
    open_gaps = [gap for gap in gaps if gap.status != "addressed"]
    weakest = weak_claims(claims)[:3]
    claim_overlay_index = overlay_by_record_id(claim_overlays or [])
    gap_overlay_index = overlay_by_record_id(gap_overlays or [])
    opportunity_overlay_index = overlay_by_record_id(opportunity_overlays or [])
    claim_overlay_count = sum(1 for claim in claims if claim_overlay_index.get(claim.claim_id))
    gap_overlay_count = sum(1 for gap in gaps if gap_overlay_index.get(gap.gap_id))
    opportunity_overlay_count = sum(
        1
        for opportunity in opportunities
        if opportunity_overlay_index.get(opportunity.opportunity_id)
    )
    reviewed_claims = sum(
        1
        for claim in claims
        if (overlay := claim_overlay_index.get(claim.claim_id))
        and overlay.review_status != "unreviewed"
    )
    reviewed_gaps = sum(
        1
        for gap in gaps
        if (overlay := gap_overlay_index.get(gap.gap_id))
        and overlay.review_status != "unreviewed"
    )
    reviewed_opportunities = sum(
        1
        for opportunity in opportunities
        if (overlay := opportunity_overlay_index.get(opportunity.opportunity_id))
        and overlay.review_status != "unreviewed"
    )
    lines = [
        "# Latest Field-State Snapshot",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Counts",
        "",
        f"- Claims: {len(claims)}",
        f"- Benchmark gaps: {len(gaps)}",
        f"- Open benchmark gaps: {len(open_gaps)}",
        f"- Opportunities: {len(opportunities)}",
        f"- Claim review overlays: {claim_overlay_count}",
        f"- Benchmark gap review overlays: {gap_overlay_count}",
        f"- Opportunity review overlays: {opportunity_overlay_count}",
        f"- Reviewed claims: {reviewed_claims}",
        f"- Reviewed benchmark gaps: {reviewed_gaps}",
        f"- Reviewed opportunities: {reviewed_opportunities}",
        "",
        "## Input Artifacts",
        "",
    ]
    for artifact, exists in inventory.items():
        marker = "present" if exists else "missing"
        lines.append(f"- `{artifact}`: {marker}")
    lines.extend(["", "## Top Opportunities", ""])
    for index, opportunity in enumerate(ranked[:5], start=1):
        overlay = opportunity_overlay_index.get(opportunity.opportunity_id)
        review = f", review={overlay.review_status}" if overlay else ""
        lines.append(
            f"{index}. {opportunity.title} ({opportunity.total_score:.3f}{review})"
        )
    lines.extend(["", "## Weakest Claims", ""])
    for claim in weakest:
        lines.append(
            f"- `{claim.claim_id}`: {claim.claim_text} "
            f"({claim.evidence_level}, confidence {claim.confidence:.2f})"
        )
    return "\n".join(lines)


def write_report(path: Path, markdown: str, root: Path | None = None) -> Path:
    """Write a Markdown report and return its resolved path."""
    output_path = resolve_path(path, root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def generate_reports(root: Path | None = None) -> dict[str, Path]:
    """Generate all field-state Markdown reports."""
    claims = read_claims(root)
    gaps = read_benchmark_gaps(root)
    opportunities = read_opportunities(root)
    claim_overlays = read_review_overlays("claim", root)
    gap_overlays = read_review_overlays("benchmark_gap", root)
    opportunity_overlays = read_review_overlays("opportunity", root)
    return {
        "weak_claims": write_report(
            WEAK_CLAIMS_REPORT,
            render_weak_claims_report(claims, claim_overlays),
            root,
        ),
        "benchmark_gaps": write_report(
            BENCHMARK_GAPS_REPORT,
            render_benchmark_gaps_report(gaps, gap_overlays),
            root,
        ),
        "top_opportunities": write_report(
            TOP_OPPORTUNITIES_REPORT,
            render_top_opportunities_report(opportunities, overlays=opportunity_overlays),
            root,
        ),
        "latest_snapshot": write_report(
            LATEST_SNAPSHOT_REPORT,
            render_latest_snapshot_report(
                claims,
                gaps,
                opportunities,
                root,
                claim_overlays=claim_overlays,
                gap_overlays=gap_overlays,
                opportunity_overlays=opportunity_overlays,
            ),
            root,
        ),
    }


def generate_opportunities_report(root: Path | None = None) -> Path:
    """Generate only the ranked opportunities report."""
    opportunities = read_opportunities(root)
    return write_report(
        TOP_OPPORTUNITIES_REPORT,
        render_top_opportunities_report(
            opportunities,
            overlays=read_review_overlays("opportunity", root),
        ),
        root,
    )


def generate_snapshot_report(root: Path | None = None) -> Path:
    """Generate only the latest snapshot report."""
    return write_report(
        LATEST_SNAPSHOT_REPORT,
        render_latest_snapshot_report(
            read_claims(root),
            read_benchmark_gaps(root),
            read_opportunities(root),
            root,
            claim_overlays=read_review_overlays("claim", root),
            gap_overlays=read_review_overlays("benchmark_gap", root),
            opportunity_overlays=read_review_overlays("opportunity", root),
        ),
        root,
    )


def ensure_report_dir(root: Path | None = None) -> Path:
    """Create the report directory."""
    report_dir = resolve_path(REPORT_DIR, root)
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir
