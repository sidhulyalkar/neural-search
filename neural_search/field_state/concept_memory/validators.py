"""Validate integrity of Graph-Indexed Concept Memory artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from neural_search.field_state.concept_memory.basis import read_concept_basis
from neural_search.field_state.concept_memory.graph_builder import (
    _CONCEPT_GRAPH_JSON,
    _CONCEPT_INDEX,
    _repo_root,
    read_concept_artifacts,
)
from neural_search.field_state.concept_memory.schema import (
    VALID_CONCEPT_TYPES,
    VALID_EVIDENCE_STRENGTHS,
    VALID_RELATION_TYPES,
    ConceptBasis,
    ConceptNode,
    EvidenceLink,
)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

_VALIDATION_JSON = Path("artifacts/field_state/concept_memory/concept_validation.json")
_VALIDATION_MD = Path("reports/field_state/concept_memory/concept_validation.md")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------


def validate_concept_memory(
    concepts: list[ConceptNode],
    evidence_links: list[EvidenceLink],
    bases: list[ConceptBasis],
    obsidian_export_path: Path | None = None,
) -> ValidationResult:
    """Run all integrity checks and return a ValidationResult."""
    errors: list[str] = []
    warnings: list[str] = []

    # --- Concept checks -------------------------------------------------------

    concept_ids: set[str] = set()
    duplicate_concept_ids = 0

    for concept in concepts:
        # 1. Duplicate concept_ids
        if concept.concept_id in concept_ids:
            errors.append(f"Duplicate concept_id: {concept.concept_id}")
            duplicate_concept_ids += 1
        else:
            concept_ids.add(concept.concept_id)

        # 2. Non-empty concept_id
        if not concept.concept_id.strip():
            errors.append("Concept has empty concept_id")

        # 3. Valid concept_type
        if concept.concept_type not in VALID_CONCEPT_TYPES:
            errors.append(
                f"Concept {concept.concept_id}: invalid concept_type"
                f" '{concept.concept_type}'"
            )

        # 4. Confidence in range
        if not (0.0 <= concept.confidence <= 1.0):
            errors.append(
                f"Concept {concept.concept_id}: confidence {concept.confidence}"
                " out of range [0.0, 1.0]"
            )

        # 5. Non-negative counts
        for count_name in ("evidence_count", "claim_count", "dataset_count", "paper_count"):
            val = getattr(concept, count_name)
            if val < 0:
                errors.append(
                    f"Concept {concept.concept_id}: {count_name} is negative ({val})"
                )

    # --- Evidence link checks -------------------------------------------------

    evidence_ids: set[str] = set()
    duplicate_evidence_ids = 0

    for link in evidence_links:
        eid = link.evidence_id

        # 6. Duplicate evidence_ids
        if eid in evidence_ids:
            errors.append(f"Duplicate evidence_id: {eid}")
            duplicate_evidence_ids += 1
        else:
            evidence_ids.add(eid)

        # 7. Non-empty evidence_id
        if not eid.strip():
            errors.append("EvidenceLink has empty evidence_id")

        # 8. source_concept_id references existing concept
        if link.source_concept_id not in concept_ids:
            errors.append(
                f"EvidenceLink {eid}: source_concept_id"
                f" {link.source_concept_id} not found"
            )

        # 9. target_concept_id references existing concept (when not None)
        if link.target_concept_id is not None and link.target_concept_id not in concept_ids:
            errors.append(
                f"EvidenceLink {eid}: target_concept_id"
                f" {link.target_concept_id} not found"
            )

        # 10. Valid relation_type
        if link.relation_type not in VALID_RELATION_TYPES:
            errors.append(
                f"EvidenceLink {eid}: invalid relation_type '{link.relation_type}'"
            )

        # 11. Confidence in range
        if not (0.0 <= link.confidence <= 1.0):
            errors.append(
                f"EvidenceLink {eid}: confidence {link.confidence}"
                " out of range [0.0, 1.0]"
            )

    # --- Basis checks ---------------------------------------------------------

    for basis in bases:
        # 12. concept_id in bases references existing concept
        if basis.concept_id not in concept_ids:
            errors.append(
                f"ConceptBasis concept_id {basis.concept_id} not found in concepts"
            )

        # 13. evidence_strength valid
        if basis.evidence_strength not in VALID_EVIDENCE_STRENGTHS:
            errors.append(
                f"ConceptBasis {basis.concept_id}: invalid evidence_strength"
                f" '{basis.evidence_strength}'"
            )

    # --- Graph file checks (warnings) -----------------------------------------

    repo = _repo_root()

    graph_json_path = repo / _CONCEPT_GRAPH_JSON
    if not graph_json_path.exists():
        warnings.append(f"concept_graph.json not found at {graph_json_path}")
    else:
        try:
            json.loads(graph_json_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"concept_graph.json is not parseable: {exc}")

    index_path = repo / _CONCEPT_INDEX
    if not index_path.exists():
        warnings.append(f"concept_index.json not found at {index_path}")
    else:
        try:
            json.loads(index_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"concept_index.json is not parseable: {exc}")

    # --- Obsidian export path traversal check (error) -------------------------

    if obsidian_export_path is not None:
        export_resolved = obsidian_export_path.resolve()
        # Check any note paths embedded in concepts
        for concept in concepts:
            for note_path_str in concept.source_note_paths:
                note_path = Path(note_path_str)
                if note_path.is_absolute():
                    try:
                        note_path.resolve().relative_to(export_resolved)
                    except ValueError:
                        errors.append(
                            f"Concept {concept.concept_id}: note path"
                            f" '{note_path_str}' is outside obsidian_export_path"
                            " (path traversal)"
                        )

    # --- Stats ----------------------------------------------------------------

    orphan_concepts = sum(1 for c in concepts if c.evidence_count == 0)
    reviewed_concepts = sum(1 for c in concepts if c.review_status != "unreviewed")

    stats: dict[str, int] = {
        "total_concepts": len(concepts),
        "total_evidence_links": len(evidence_links),
        "total_bases": len(bases),
        "duplicate_concept_ids": duplicate_concept_ids,
        "duplicate_evidence_ids": duplicate_evidence_ids,
        "orphan_concepts": orphan_concepts,
        "reviewed_concepts": reviewed_concepts,
    }

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_validation_artifacts(
    result: ValidationResult,
    root: Path | None = None,
) -> dict[str, Path]:
    """Write concept_validation.json and concept_validation.md."""
    base = root if root is not None else _repo_root()

    validated_at = datetime.now(UTC).isoformat()

    # 1. JSON artifact
    json_payload = {
        "is_valid": result.is_valid,
        "error_count": result.error_count,
        "warning_count": result.warning_count,
        "errors": result.errors,
        "warnings": result.warnings,
        "stats": result.stats,
        "validated_at": validated_at,
    }
    json_path = base / _VALIDATION_JSON
    _atomic_write(json_path, json.dumps(json_payload, indent=2))

    # 2. Markdown report
    status = "VALID" if result.is_valid else "INVALID"
    lines: list[str] = [
        "# Concept Memory Validation Report",
        "",
        f"**Status:** {status}  ",
        f"**Errors:** {result.error_count}  ",
        f"**Warnings:** {result.warning_count}  ",
        f"**Validated at:** {validated_at}",
        "",
        "## Stats",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key, val in result.stats.items():
        lines.append(f"| {key} | {val} |")

    lines += [
        "",
        "## Errors",
        "",
    ]
    if result.errors:
        for err in result.errors:
            lines.append(f"- {err}")
    else:
        lines.append("_No errors._")

    lines += [
        "",
        "## Warnings",
        "",
    ]
    if result.warnings:
        for warn in result.warnings:
            lines.append(f"- {warn}")
    else:
        lines.append("_No warnings._")

    md_path = base / _VALIDATION_MD
    _atomic_write(md_path, "\n".join(lines) + "\n")

    return {"json": json_path, "report": md_path}


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def run_concept_validation(root: Path | None = None) -> ValidationResult:
    """Read artifacts, validate, write outputs, return result."""
    base = root if root is not None else _repo_root()

    concepts, evidence_links = read_concept_artifacts(base)
    bases = read_concept_basis(base)

    result = validate_concept_memory(concepts, evidence_links, bases)
    write_validation_artifacts(result, base)

    return result
