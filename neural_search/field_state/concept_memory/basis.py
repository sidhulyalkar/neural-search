"""Generate ConceptBasis records for every concept in the graph."""

from __future__ import annotations

from pathlib import Path

from neural_search.field_state.concept_memory.evidence import (
    EvidenceSummary,
    evidence_strength_from_count,
    summarize_evidence,
)
from neural_search.field_state.concept_memory.schema import (
    ConceptBasis,
    ConceptNode,
    EvidenceLink,
)

# ---------------------------------------------------------------------------
# Repo root + artifact path
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]

_BASIS_JSONL = Path("artifacts/field_state/concept_memory/concept_basis.jsonl")


def _repo_root() -> Path:
    return _REPO_ROOT


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Linked concept type helpers
# ---------------------------------------------------------------------------

_BENCHMARK_GAP_TYPE = "benchmark_gap"
_OPPORTUNITY_TYPE = "opportunity"


def _concept_ids_by_type(
    evidence_summary: EvidenceSummary,
    concepts_by_id: dict[str, ConceptNode],
    concept_type: str,
) -> list[str]:
    """Return concept IDs from outgoing links that match the given type."""
    # For claim/dataset/paper/opportunity we already have them in the summary;
    # benchmark_gap is not pre-collected so we look it up via concepts_by_id.
    if concept_type == "claim":
        return evidence_summary["linked_claims"]
    if concept_type == "dataset":
        return evidence_summary["linked_datasets"]
    if concept_type == "paper":
        return evidence_summary["linked_papers"]
    if concept_type == "opportunity":
        return evidence_summary["linked_opportunities"]
    # fallback for benchmark_gap or any other type
    return []


def _collect_benchmark_gap_ids(
    concept: ConceptNode,
    evidence_links: list[EvidenceLink],
    concepts_by_id: dict[str, ConceptNode],
) -> list[str]:
    """Return concept IDs of outgoing-linked benchmark_gap concepts."""
    results: list[str] = []
    for lnk in evidence_links:
        if lnk.source_concept_id != concept.concept_id:
            continue
        tgt_id = lnk.target_concept_id
        if tgt_id is None:
            continue
        node = concepts_by_id.get(tgt_id)
        if node is not None and node.concept_type == _BENCHMARK_GAP_TYPE:
            results.append(tgt_id)
    return results


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def generate_concept_basis(
    concept: ConceptNode,
    evidence_links: list[EvidenceLink],
    concepts_by_id: dict[str, ConceptNode],
) -> ConceptBasis:
    """Build a ConceptBasis for a single concept."""
    ev_summary = summarize_evidence(concept, evidence_links, concepts_by_id)

    total_link_count = ev_summary["outgoing_links"] + ev_summary["incoming_links"]
    reviewed_count = ev_summary["reviewed_link_count"]

    evidence_strength = evidence_strength_from_count(reviewed_count, total_link_count)

    # Summary — no hallucination; use only existing fields
    if concept.description:
        summary = f"Concept: {concept.canonical_name}. {concept.description}"
    else:
        summary = f"Concept: {concept.canonical_name} (type: {concept.concept_type})"

    if concept.source_artifacts:
        summary += f" Evidence from: {concept.source_artifacts[0]}"

    # Collect all evidence_ids from outgoing + incoming links (up to 20)
    all_links_for_concept = [
        lnk for lnk in evidence_links
        if lnk.source_concept_id == concept.concept_id
        or lnk.target_concept_id == concept.concept_id
    ]
    evidence_link_ids = [lnk.evidence_id for lnk in all_links_for_concept[:20]]

    # Uncertainty notes
    uncertainty_notes: list[str] = []
    if concept.evidence_count == 0:
        uncertainty_notes.append(
            "No evidence links found — purely derived from artifact names."
        )
    if concept.review_status == "unreviewed":
        uncertainty_notes.append(
            "Not yet reviewed by a human — treat as speculative."
        )
    if evidence_strength in ("none", "weak"):
        uncertainty_notes.append(
            "Weak or no supporting evidence — verify before citing."
        )

    # Next validation actions
    ctype = concept.concept_type
    if ctype == "claim":
        next_validation_actions = ["Validate claim against dataset corpus benchmarks."]
    elif ctype == "benchmark_gap":
        next_validation_actions = [
            "Identify or create artifacts that address this gap."
        ]
    elif ctype == "dataset":
        next_validation_actions = [
            "Check dataset availability and access policy."
        ]
    elif ctype == "method":
        next_validation_actions = [
            "Confirm method is applicable to available datasets."
        ]
    else:
        next_validation_actions = ["Review linked evidence before use."]

    return ConceptBasis(
        concept_id=concept.concept_id,
        canonical_name=concept.canonical_name,
        concept_type=concept.concept_type,
        summary=summary,
        supporting_claim_ids=_concept_ids_by_type(ev_summary, concepts_by_id, "claim"),
        supporting_dataset_ids=_concept_ids_by_type(ev_summary, concepts_by_id, "dataset"),
        supporting_paper_ids=_concept_ids_by_type(ev_summary, concepts_by_id, "paper"),
        supporting_note_paths=concept.source_note_paths,
        related_opportunity_ids=_concept_ids_by_type(
            ev_summary, concepts_by_id, "opportunity"
        ),
        related_benchmark_gap_ids=_collect_benchmark_gap_ids(
            concept, evidence_links, concepts_by_id
        ),
        evidence_links=evidence_link_ids,
        evidence_strength=evidence_strength,
        uncertainty_notes=uncertainty_notes,
        next_validation_actions=next_validation_actions,
    )


def generate_all_bases(
    concepts: list[ConceptNode],
    evidence_links: list[EvidenceLink],
) -> list[ConceptBasis]:
    """Generate ConceptBasis for every concept."""
    concepts_by_id: dict[str, ConceptNode] = {c.concept_id: c for c in concepts}
    return [
        generate_concept_basis(concept, evidence_links, concepts_by_id)
        for concept in concepts
    ]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_concept_basis(
    bases: list[ConceptBasis],
    root: Path | None = None,
) -> Path:
    """Write concept_basis.jsonl. Returns the absolute path written."""
    base = root if root is not None else _repo_root()
    path = base / _BASIS_JSONL
    _atomic_write(path, "\n".join(b.to_jsonl() for b in bases))
    return path


def read_concept_basis(
    root: Path | None = None,
) -> list[ConceptBasis]:
    """Read concept_basis.jsonl. Returns list[ConceptBasis]."""
    base = root if root is not None else _repo_root()
    path = base / _BASIS_JSONL
    if not path.exists():
        return []
    results: list[ConceptBasis] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    results.append(ConceptBasis.from_jsonl(line))
                except Exception:
                    pass
    return results
