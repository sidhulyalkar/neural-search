"""Markdown templates for field-state Obsidian notes."""

from __future__ import annotations

from collections.abc import Sequence

GENERATED_BEGIN = "<!-- FIELDSTATE:BEGIN generated -->"
GENERATED_END = "<!-- FIELDSTATE:END generated -->"
HUMAN_BEGIN = "<!-- FIELDSTATE:BEGIN human -->"
HUMAN_END = "<!-- FIELDSTATE:END human -->"


def markdown_list(items: Sequence[str], empty: str = "none") -> str:
    """Render a compact Markdown list."""
    if not items:
        return f"- {empty}"
    return "\n".join(f"- {item}" for item in items)


def default_human_block(kind: str) -> str:
    """Return the default human-editable block for a note kind."""
    if kind == "claim":
        review_items = [
            "Reviewed",
            "Accepted",
            "Needs revision",
            "Rejected",
        ]
        heading = "Human review"
    elif kind == "benchmark_gap":
        review_items = [
            "Reviewed",
            "This is a real gap",
            "This gap is already resolved",
            "Needs refinement",
            "Reject for now",
        ]
        heading = "Human review"
    elif kind == "opportunity":
        review_items = [
            "Reviewed",
            "Worth pursuing",
            "Needs more evidence",
            "Reject for now",
            "Convert to Codex task",
        ]
        heading = "Human review"
    else:
        review_items = []
        heading = "Human notes" if kind == "field_snapshot" else "Notes"
    checks = "\n".join(f"- [ ] {item}" for item in review_items)
    review = f"\n{checks}\n" if checks else ""
    return (
        f"{HUMAN_BEGIN}\n\n"
        f"## {heading}\n"
        f"{review}\n"
        "## Reviewer notes\n\n"
        "<!-- Add notes here. -->\n\n"
        f"{HUMAN_END}"
    )


def wrap_blocks(generated: str, human: str | None, kind: str) -> str:
    """Wrap generated content and human content in preservation markers."""
    human_block = human if human is not None else default_human_block(kind)
    return (
        f"{GENERATED_BEGIN}\n"
        f"{generated.strip()}\n"
        f"{GENERATED_END}\n\n"
        f"{human_block.strip()}\n"
    )


def render_claim_note(
    *,
    title: str,
    claim_text: str,
    claim_type: str,
    evidence_level: str,
    confidence: float,
    status: str,
    missing_tests: Sequence[str],
    supporting_artifacts: Sequence[str],
    related_benchmark_gaps: Sequence[str],
    related_opportunities: Sequence[str],
    human_block: str | None = None,
) -> str:
    """Render a claim note body."""
    generated = f"""# Claim: {title}

## Claim statement

{claim_text}

## Claim type

{claim_type}

## Evidence level

{evidence_level}

## Confidence

{confidence:.2f}

## Current status

{status}

## Missing tests

{markdown_list(missing_tests)}

## Supporting artifacts

{markdown_list(supporting_artifacts)}

## Related benchmark gaps

{markdown_list(related_benchmark_gaps)}

## Related opportunities

{markdown_list(related_opportunities)}
"""
    return wrap_blocks(generated, human_block, "claim")


def render_benchmark_gap_note(
    *,
    title: str,
    description: str,
    gap_type: str,
    why_it_matters: str,
    required_validation: Sequence[str],
    minimum_viable_benchmark: str,
    related_claims: Sequence[str],
    related_opportunities: Sequence[str],
    source_artifacts: Sequence[str],
    human_block: str | None = None,
) -> str:
    """Render a benchmark-gap note body."""
    generated = f"""# Benchmark Gap: {title}

## Gap

{description}

## Gap type

{gap_type}

## Why this matters

{why_it_matters}

## Required validation

{markdown_list(required_validation)}

## Minimum viable benchmark

{minimum_viable_benchmark}

## Related claims

{markdown_list(related_claims)}

## Related opportunities

{markdown_list(related_opportunities)}

## Source artifacts

{markdown_list(source_artifacts)}
"""
    return wrap_blocks(generated, human_block, "benchmark_gap")


def render_opportunity_note(
    *,
    title: str,
    hypothesis: str,
    opportunity_type: str,
    rationale: str,
    novelty_score: float,
    feasibility_score: float,
    impact_score: float,
    uncertainty_reduction_score: float,
    personal_fit_score: float,
    risk_score: float,
    total_score: float,
    minimum_viable_experiment: str,
    next_action: str,
    related_claims: Sequence[str],
    related_benchmark_gaps: Sequence[str],
    related_artifacts: Sequence[str],
    codex_task_stub_or_link: str,
    human_block: str | None = None,
) -> str:
    """Render an opportunity note body."""
    generated = f"""# Opportunity: {title}

## Core hypothesis

{hypothesis}

## Opportunity type

{opportunity_type}

## Rationale

{rationale}

## Score breakdown

| Dimension | Score |
|---|---:|
| Novelty | {novelty_score:.1f} |
| Feasibility | {feasibility_score:.1f} |
| Impact | {impact_score:.1f} |
| Uncertainty reduction | {uncertainty_reduction_score:.1f} |
| Personal fit | {personal_fit_score:.1f} |
| Risk | {risk_score:.1f} |
| Total | {total_score:.3f} |

## Minimum viable experiment

{minimum_viable_experiment}

## Next action

{next_action}

## Related claims

{markdown_list(related_claims)}

## Related benchmark gaps

{markdown_list(related_benchmark_gaps)}

## Related artifacts

{markdown_list(related_artifacts)}

## Suggested Codex task

{codex_task_stub_or_link}
"""
    return wrap_blocks(generated, human_block, "opportunity")


def render_snapshot_note(
    *,
    field: str,
    date: str,
    summary: str,
    top_opportunities: Sequence[str],
    weak_claims: Sequence[str],
    benchmark_gaps: Sequence[str],
    recommended_next_actions: Sequence[str],
    snapshot_diff: str,
    human_block: str | None = None,
) -> str:
    """Render a field snapshot note body."""
    generated = f"""# Field Snapshot: {field} - {date}

## Summary

{summary}

## Top opportunities

{markdown_list(top_opportunities)}

## Weak claims

{markdown_list(weak_claims)}

## Benchmark gaps

{markdown_list(benchmark_gaps)}

## Recommended next actions

{markdown_list(recommended_next_actions)}

## Changes since previous snapshot

{snapshot_diff}
"""
    return wrap_blocks(generated, human_block, "field_snapshot")


def render_codex_task_note(
    *,
    title: str,
    goal: str,
    context: str,
    related_opportunity: str,
    files_to_inspect: Sequence[str],
    implementation_requirements: Sequence[str],
    acceptance_criteria: Sequence[str],
    tests: Sequence[str],
    risks: Sequence[str],
    human_block: str | None = None,
) -> str:
    """Render a Codex task note body."""
    generated = f"""# Codex Task: {title}

## Goal

{goal}

## Context

{context}

## Related opportunity

{related_opportunity}

## Files to inspect

{markdown_list(files_to_inspect)}

## Implementation requirements

{markdown_list(implementation_requirements)}

## Acceptance criteria

{markdown_list(acceptance_criteria)}

## Tests

{markdown_list(tests)}

## Risks

{markdown_list(risks)}
"""
    return wrap_blocks(generated, human_block, "codex_task")


def render_decision_note(
    *,
    title: str,
    decision: str,
    rationale: str,
    evidence: str,
    consequences: str,
    revisit_criteria: str,
    human_block: str | None = None,
) -> str:
    """Render a decision-log note body."""
    generated = f"""# Decision: {title}

## Decision

{decision}

## Why

{rationale}

## Evidence

{evidence}

## Consequences

{consequences}

## Revisit criteria

{revisit_criteria}
"""
    return wrap_blocks(generated, human_block, "decision_log")
