"""Export concept memory notes to an Obsidian vault."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from neural_search.field_state.concept_memory.schema import (
    ConceptBasis,
    ConceptNode,
    EvidenceLink,
)
from neural_search.field_state.obsidian.frontmatter import parse_frontmatter
from neural_search.field_state.obsidian.reader import extract_human_block
from neural_search.field_state.obsidian.templates import (
    GENERATED_BEGIN,
    GENERATED_END,
    HUMAN_BEGIN,
    HUMAN_END,
    markdown_list,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONCEPT_MEMORY_ROOT = "Field-State/55_Concept_Memory"

SAFE_HUMAN_FRONTMATTER_FIELDS = {
    "status",
    "review_status",
    "human_reviewer",
    "reviewed_at",
    "human_priority",
    "human_tags",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ConceptExportResult:
    vault_path: Path
    notes_created: int = 0
    notes_updated: int = 0
    notes_skipped: int = 0
    warnings: list[str] = dataclass_field(default_factory=list)


# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------


def _concept_slug(concept_id: str) -> str:
    """Convert concept_id to a filesystem-safe slug."""
    return concept_id.replace(":", "--").replace(" ", "-")


def _concept_note_filename(concept: ConceptNode) -> str:
    slug = _concept_slug(concept.concept_id)
    return f"concept--{concept.concept_type}--{slug}.md"


def _basis_note_filename(concept: ConceptNode) -> str:
    slug = _concept_slug(concept.concept_id)
    return f"basis--{slug}.md"


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Frontmatter serialization
# ---------------------------------------------------------------------------


def _render_frontmatter(data: dict[str, Any]) -> str:
    rendered = yaml.safe_dump(
        data,
        allow_unicode=False,
        default_flow_style=False,
        sort_keys=False,
    ).strip()
    return f"---\n{rendered}\n---\n"


def _aliases_yaml(aliases: list[str]) -> list[str]:
    return list(aliases)


# ---------------------------------------------------------------------------
# Human-block helpers
# ---------------------------------------------------------------------------


def _default_concept_human_block() -> str:
    return (
        f"{HUMAN_BEGIN}\n\n"
        "## Human notes\n\n"
        "<!-- Add concept notes here. -->\n\n"
        f"{HUMAN_END}"
    )


def _extract_safe_frontmatter(existing: dict[str, Any]) -> dict[str, Any]:
    return {k: existing[k] for k in SAFE_HUMAN_FRONTMATTER_FIELDS if k in existing}


# ---------------------------------------------------------------------------
# Concept note renderer
# ---------------------------------------------------------------------------


def _render_concept_note_body(
    concept: ConceptNode,
    basis: ConceptBasis | None,
    human_block: str | None,
) -> str:
    description = concept.description or "No description available."

    if basis is not None:
        evidence_basis_text = basis.summary or "No basis summary recorded."
        supporting_claims = basis.supporting_claim_ids
        supporting_datasets = basis.supporting_dataset_ids
        supporting_papers = basis.supporting_paper_ids
        related_opportunities = basis.related_opportunity_ids
        related_benchmark_gaps = basis.related_benchmark_gap_ids
        uncertainty_notes = basis.uncertainty_notes
        next_actions = basis.next_validation_actions
    else:
        evidence_basis_text = "No basis record generated."
        supporting_claims = []
        supporting_datasets = []
        supporting_papers = []
        related_opportunities = []
        related_benchmark_gaps = []
        uncertainty_notes = []
        next_actions = []

    generated_content = f"""# Concept: {concept.canonical_name}

## Definition / description

{description}

## Evidence basis

{evidence_basis_text}

## Supporting claims

{markdown_list(supporting_claims)}

## Supporting datasets

{markdown_list(supporting_datasets)}

## Supporting papers

{markdown_list(supporting_papers)}

## Related opportunities

{markdown_list(related_opportunities)}

## Related benchmark gaps

{markdown_list(related_benchmark_gaps)}

## Uncertainty notes

{markdown_list(uncertainty_notes)}

## Next validation actions

{markdown_list(next_actions)}"""

    resolved_human_block = human_block if human_block is not None else _default_concept_human_block()

    return (
        f"{GENERATED_BEGIN}\n"
        f"{generated_content.strip()}\n"
        f"{GENERATED_END}\n\n"
        f"{resolved_human_block.strip()}\n"
    )


# ---------------------------------------------------------------------------
# Concept note frontmatter builder
# ---------------------------------------------------------------------------


def _concept_frontmatter(
    concept: ConceptNode,
    basis: ConceptBasis | None,
    field: str,
) -> dict[str, Any]:
    evidence_strength = basis.evidence_strength if basis is not None else "unknown"
    extra_tags = [t for t in concept.tags if t not in {"field-state", "concept-memory"}]
    tags: list[str] = ["field-state", "concept-memory"] + extra_tags

    return {
        "type": "concept",
        "field_state_id": concept.concept_id,
        "field": field,
        "concept_id": concept.concept_id,
        "concept_type": concept.concept_type,
        "canonical_name": concept.canonical_name,
        "aliases": _aliases_yaml(concept.aliases),
        "evidence_strength": evidence_strength,
        "review_status": concept.review_status,
        "evidence_count": concept.evidence_count,
        "claim_count": concept.claim_count,
        "dataset_count": concept.dataset_count,
        "paper_count": concept.paper_count,
        "schema_version": "0.4",
        "tags": tags,
    }


# ---------------------------------------------------------------------------
# Note write helper
# ---------------------------------------------------------------------------


def _write_note(
    path: Path,
    frontmatter: dict[str, Any],
    body: str,
    warnings: list[str],
) -> str:
    """Write a note atomically. Returns 'created', 'updated', or 'skipped'."""
    existing_frontmatter: dict[str, Any] = {}
    human_block: str | None = None

    if path.exists():
        text = path.read_text(encoding="utf-8")
        try:
            existing_frontmatter, existing_body = parse_frontmatter(text)
            extracted = extract_human_block(existing_body)
            human_block = extracted if extracted else None
        except (ValueError, Exception) as exc:
            warnings.append(f"Could not parse existing note {path.name}: {exc}")

    # Merge safe human frontmatter fields
    merged_frontmatter = dict(frontmatter)
    for key in SAFE_HUMAN_FRONTMATTER_FIELDS:
        if key in existing_frontmatter and existing_frontmatter[key] != frontmatter.get(key):
            merged_frontmatter[key] = existing_frontmatter[key]
            if key in {"status", "review_status"}:
                warnings.append(f"preserved human {key} in {path.name}")

    # Rebuild body with preserved or default human block
    if human_block is not None:
        # Replace the default human block in the rendered body
        default_block = _default_concept_human_block()
        if default_block in body:
            body = body.replace(default_block, human_block)
        elif HUMAN_BEGIN not in body:
            # Append preserved human block at end
            body = body.rstrip("\n") + f"\n\n{human_block.strip()}\n"

    full_text = f"{_render_frontmatter(merged_frontmatter)}\n{body.lstrip()}"

    existed = path.exists()
    if existed and path.read_text(encoding="utf-8") == full_text:
        return "skipped"

    _atomic_write(path, full_text)
    return "updated" if existed else "created"


def _record_write(result: ConceptExportResult, outcome: str) -> None:
    if outcome == "created":
        result.notes_created += 1
    elif outcome == "updated":
        result.notes_updated += 1
    else:
        result.notes_skipped += 1


# ---------------------------------------------------------------------------
# Map notes
# ---------------------------------------------------------------------------


def _write_dataset_method_map(
    concepts: list[ConceptNode],
    evidence_links: list[EvidenceLink],
    maps_dir: Path,
    result: ConceptExportResult,
) -> None:
    concepts_by_id = {c.concept_id: c for c in concepts}

    # Build dataset → linked methods/modalities/tasks/brain_regions
    dataset_links: dict[str, dict[str, list[str]]] = {}
    for link in evidence_links:
        src = concepts_by_id.get(link.source_concept_id)
        tgt_id = link.target_concept_id
        if src is None or tgt_id is None:
            continue
        tgt = concepts_by_id.get(tgt_id)
        if tgt is None:
            continue

        # Dataset as source with linked methods/modalities/tasks/brain_regions
        if src.concept_type == "dataset":
            entry = dataset_links.setdefault(src.concept_id, {"modalities": [], "tasks": [], "brain_regions": []})
            if tgt.concept_type == "modality" and tgt.canonical_name not in entry["modalities"]:
                entry["modalities"].append(tgt.canonical_name)
            elif tgt.concept_type == "task" and tgt.canonical_name not in entry["tasks"]:
                entry["tasks"].append(tgt.canonical_name)
            elif tgt.concept_type == "brain_region" and tgt.canonical_name not in entry["brain_regions"]:
                entry["brain_regions"].append(tgt.canonical_name)

    rows = ["| Dataset | Modalities | Tasks | Brain Regions |", "|---|---|---|---|"]
    for ds_id, attrs in sorted(dataset_links.items()):
        ds_node = concepts_by_id.get(ds_id)
        name = ds_node.canonical_name if ds_node else ds_id
        modalities = ", ".join(attrs["modalities"]) or "—"
        tasks = ", ".join(attrs["tasks"]) or "—"
        regions = ", ".join(attrs["brain_regions"]) or "—"
        rows.append(f"| {name} | {modalities} | {tasks} | {regions} |")

    if len(rows) <= 2:
        rows.append("| — | — | — | — |")

    body = "# Dataset-Method Map\n\n" + "\n".join(rows) + "\n"
    path = maps_dir / "dataset_method_map.md"
    existed = path.exists()
    _atomic_write(path, body)
    outcome = "updated" if existed else "created"
    _record_write(result, outcome)


def _write_claim_concept_map(
    concepts: list[ConceptNode],
    evidence_links: list[EvidenceLink],
    maps_dir: Path,
    result: ConceptExportResult,
) -> None:
    concepts_by_id = {c.concept_id: c for c in concepts}

    claim_types: dict[str, list[str]] = {}
    for link in evidence_links:
        src = concepts_by_id.get(link.source_concept_id)
        tgt_id = link.target_concept_id
        if src is None or tgt_id is None:
            continue
        tgt = concepts_by_id.get(tgt_id)
        if tgt is None:
            continue
        if src.concept_type == "claim":
            entry = claim_types.setdefault(src.concept_id, [])
            if tgt.concept_type not in entry:
                entry.append(tgt.concept_type)

    rows = ["| Claim | Linked Concept Types |", "|---|---|"]
    for claim_id, types in sorted(claim_types.items()):
        claim_node = concepts_by_id.get(claim_id)
        name = claim_node.canonical_name if claim_node else claim_id
        rows.append(f"| {name} | {', '.join(sorted(types))} |")

    if len(rows) <= 2:
        rows.append("| — | — |")

    body = "# Claim-Concept Map\n\n" + "\n".join(rows) + "\n"
    path = maps_dir / "claim_concept_map.md"
    existed = path.exists()
    _atomic_write(path, body)
    outcome = "updated" if existed else "created"
    _record_write(result, outcome)


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------


def export_concept_memory_to_obsidian(
    concepts: list[ConceptNode],
    evidence_links: list[EvidenceLink],
    bases: list[ConceptBasis],
    vault_path: Path,
    field: str = "neuroscience_dataset_reuse",
) -> ConceptExportResult:
    """Export concept notes to vault_path/Field-State/55_Concept_Memory/.

    Preserves existing human blocks. Returns a summary result.
    """
    # Validate vault_path is absolute and exists (or can be created)
    vault_path = vault_path.resolve()

    result = ConceptExportResult(vault_path=vault_path)

    # Build index structures
    bases_by_id: dict[str, ConceptBasis] = {b.concept_id: b for b in bases}

    # Set up directories
    concept_memory_dir = vault_path / CONCEPT_MEMORY_ROOT
    concepts_dir = concept_memory_dir / "Concepts"
    basis_dir = concept_memory_dir / "Basis"
    maps_dir = concept_memory_dir / "Maps"
    for d in (concepts_dir, basis_dir, maps_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Export concept notes
    for concept in concepts:
        basis = bases_by_id.get(concept.concept_id)

        # -- Concept note --
        concept_path = concepts_dir / _concept_note_filename(concept)
        frontmatter = _concept_frontmatter(concept, basis, field)
        body = _render_concept_note_body(concept, basis, human_block=None)
        try:
            outcome = _write_note(concept_path, frontmatter, body, result.warnings)
        except Exception as exc:
            result.warnings.append(f"Failed to write concept note {concept.concept_id}: {exc}")
            continue
        _record_write(result, outcome)

        # -- Basis note --
        if basis is not None:
            basis_path = basis_dir / _basis_note_filename(concept)
            basis_frontmatter: dict[str, Any] = {
                "type": "concept_basis",
                "field_state_id": f"basis--{concept.concept_id}",
                "field": field,
                "concept_id": concept.concept_id,
                "canonical_name": concept.canonical_name,
                "concept_type": concept.concept_type,
                "evidence_strength": basis.evidence_strength,
                "schema_version": "0.4",
                "tags": ["field-state", "concept-memory", "basis"],
            }
            basis_body = _render_basis_body(basis)
            try:
                basis_outcome = _write_note(basis_path, basis_frontmatter, basis_body, result.warnings)
            except Exception as exc:
                result.warnings.append(f"Failed to write basis note {concept.concept_id}: {exc}")
                continue
            _record_write(result, basis_outcome)

    # Export map notes
    _write_dataset_method_map(concepts, evidence_links, maps_dir, result)
    _write_claim_concept_map(concepts, evidence_links, maps_dir, result)

    return result


def _render_basis_body(basis: ConceptBasis) -> str:
    """Render a basis note body (no human block — informational only)."""
    content = f"""# Basis: {basis.canonical_name}

## Summary

{basis.summary or "No summary available."}

## Evidence strength

{basis.evidence_strength}

## Supporting claims

{markdown_list(basis.supporting_claim_ids)}

## Supporting datasets

{markdown_list(basis.supporting_dataset_ids)}

## Supporting papers

{markdown_list(basis.supporting_paper_ids)}

## Related opportunities

{markdown_list(basis.related_opportunity_ids)}

## Related benchmark gaps

{markdown_list(basis.related_benchmark_gap_ids)}

## Uncertainty notes

{markdown_list(basis.uncertainty_notes)}

## Next validation actions

{markdown_list(basis.next_validation_actions)}
"""
    return content
