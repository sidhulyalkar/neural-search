"""Markdown report generation for Graph-Indexed Concept Memory."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from neural_search.field_state.concept_memory.artifact_utils import artifact_timestamp
from neural_search.field_state.concept_memory.retrieval import search_concepts
from neural_search.field_state.concept_memory.schema import (
    ConceptBasis,
    ConceptNode,
    EvidenceLink,
)

# ---------------------------------------------------------------------------
# Repo root
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]

_REPORT_DIR = Path("reports/field_state/concept_memory")

_REPORT_KEYS = [
    "concept_graph_summary",
    "concept_basis_summary",
    "top_concepts",
    "unsupported_concepts",
    "dataset_method_concept_map",
    "claim_basis_map",
    "retrieval_examples",
]

_REPORT_FILENAMES: dict[str, str] = {
    "concept_graph_summary": "concept_graph_summary.md",
    "concept_basis_summary": "concept_basis_summary.md",
    "top_concepts": "top_concepts.md",
    "unsupported_concepts": "unsupported_concepts.md",
    "dataset_method_concept_map": "dataset_method_concept_map.md",
    "claim_basis_map": "claim_basis_map.md",
    "retrieval_examples": "retrieval_examples.md",
}

_RETRIEVAL_QUERIES = [
    "Neuropixels spike sorting awake behaving",
    "calcium imaging cortex mouse",
    "human qrels benchmark evaluation",
    "metadata richness dataset reuse",
    "fMRI task decoding",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def atomic_write(path: Path, text: str) -> None:
    """Atomic write using temp file rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _now_utc() -> str:
    return artifact_timestamp()


def _table_row(*cells: str) -> str:
    return "| " + " | ".join(cells) + " |"


def _header_row(*cols: str) -> list[str]:
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    return [_table_row(*cols), sep]


def _claim_safety_notice() -> list[str]:
    return [
        "## Claim Safety Notice",
        "",
        "Concept Memory is structural/provenance infrastructure. Retrieval improvement claims require qrels-backed ablation results and are not established by these reports.",
        "",
    ]


# ---------------------------------------------------------------------------
# Report 1 — concept_graph_summary
# ---------------------------------------------------------------------------


def render_concept_graph_summary(
    concepts: list[ConceptNode],
    evidence_links: list[EvidenceLink],
) -> str:
    type_counts: Counter[str] = Counter(c.concept_type for c in concepts)
    relation_counts: Counter[str] = Counter(lnk.relation_type for lnk in evidence_links)
    reviewed = sum(1 for c in concepts if c.review_status != "unreviewed")
    unreviewed = len(concepts) - reviewed
    reviewed_support = sum(
        1 for lnk in evidence_links
        if lnk.relation_type == "supports" and lnk.review_status != "unreviewed"
    )
    reviewed_contradictions = sum(
        1 for lnk in evidence_links
        if lnk.relation_type == "contradicts" and lnk.review_status != "unreviewed"
    )
    metadata_links = sum(
        1 for lnk in evidence_links
        if lnk.relation_type not in {"supports", "contradicts"}
    )

    sorted_concepts = sorted(concepts, key=lambda c: -c.evidence_count)
    top10 = sorted_concepts[:10]
    orphans = [c for c in concepts if c.evidence_count == 0]
    reviewed_support_concepts = [
        c for c in concepts
        if any(
            lnk.review_status != "unreviewed"
            and lnk.relation_type == "supports"
            and (lnk.source_concept_id == c.concept_id or lnk.target_concept_id == c.concept_id)
            for lnk in evidence_links
        )
    ]

    lines = [
        "# Concept Graph Summary",
        "",
        f"Generated: {_now_utc()}",
        "",
        *_claim_safety_notice(),
        f"- **Total concepts**: {len(concepts)}",
        f"- **Total metadata/evidence links**: {len(evidence_links)}",
        f"- **Metadata-derived or neutral links**: {metadata_links}",
        f"- **Reviewed supporting evidence links**: {reviewed_support}",
        f"- **Reviewed contradictory evidence links**: {reviewed_contradictions}",
        f"- **Reviewed concepts**: {reviewed}",
        f"- **Unreviewed concepts**: {unreviewed}",
        f"- **Orphan concepts** (evidence_count == 0): {len(orphans)}",
        f"- **Concepts with reviewed supporting evidence**: {len(reviewed_support_concepts)}",
        "",
        "## Concepts by Type",
        "",
        *_header_row("type", "count"),
    ]
    for ctype, cnt in sorted(type_counts.items()):
        lines.append(_table_row(ctype, str(cnt)))

    lines += [
        "",
        "## Metadata/Evidence Links by Relation Type",
        "",
        *_header_row("relation_type", "count"),
    ]
    for rtype, cnt in sorted(relation_counts.items()):
        lines.append(_table_row(rtype, str(cnt)))

    lines += [
        "",
        "## Most Connected Metadata Concepts",
        "",
        *_header_row("concept_id", "canonical_name", "type", "evidence_count"),
    ]
    for c in top10:
        lines.append(_table_row(c.concept_id, c.canonical_name, c.concept_type, str(c.evidence_count)))

    lines += ["", "## Orphan Concepts (evidence_count == 0)", ""]
    if orphans:
        lines += [*_header_row("concept_id", "type")]
        for c in orphans:
            lines.append(_table_row(c.concept_id, c.concept_type))
    else:
        lines.append("_No orphan concepts._")

    lines += ["", "## Concepts With Reviewed Supporting Evidence", ""]
    if reviewed_support_concepts:
        for c in reviewed_support_concepts:
            lines.append(f"- `{c.concept_id}` — {c.canonical_name} ({c.concept_type})")
    else:
        lines.append("_None._")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report 2 — concept_basis_summary
# ---------------------------------------------------------------------------


def render_concept_basis_summary(bases: list[ConceptBasis]) -> str:
    lines = [
        "# Concept Basis Summary",
        "",
        f"Generated: {_now_utc()}",
        "",
        *_claim_safety_notice(),
        f"Total bases: {len(bases)}",
        "",
        *_header_row(
            "concept_id",
            "canonical_name",
            "type",
            "strength",
            "supporting_count",
            "contradicting_count",
            "metadata_count",
            "missing_count",
            "summary (120 chars)",
        ),
    ]
    for b in bases:
        summary_short = (b.summary[:117] + "...") if len(b.summary) > 120 else b.summary
        lines.append(_table_row(
            b.concept_id,
            b.canonical_name,
            b.concept_type,
            b.evidence_strength,
            str(b.supporting_count),
            str(b.contradicting_count),
            str(b.neutral_or_metadata_count),
            str(b.missing_count),
            summary_short,
        ))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report 3 — top_concepts
# ---------------------------------------------------------------------------


def render_top_concepts(concepts: list[ConceptNode], limit: int = 20) -> str:
    ranked = sorted(concepts, key=lambda c: (-c.evidence_count, -c.confidence))[:limit]
    lines = [
        "# Metadata-Supported Concepts",
        "",
        f"Generated: {_now_utc()}",
        "",
        *_claim_safety_notice(),
        f"Top {limit} concepts ranked by metadata/evidence link count, then confidence.",
        "",
        *_header_row("rank", "concept_id", "canonical_name", "type", "evidence_count", "confidence", "review_status"),
    ]
    for i, c in enumerate(ranked, start=1):
        lines.append(_table_row(
            str(i),
            c.concept_id,
            c.canonical_name,
            c.concept_type,
            str(c.evidence_count),
            f"{c.confidence:.3f}",
            c.review_status,
        ))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report 4 — unsupported_concepts
# ---------------------------------------------------------------------------


def _unsupported_reason(
    concept: ConceptNode,
    evidence_links: list[EvidenceLink],
) -> str | None:
    """Return reason string if concept is unsupported, else None."""
    if concept.evidence_count == 0:
        return "evidence_count == 0"
    # all links unreviewed
    all_links = [
        lnk for lnk in evidence_links
        if lnk.source_concept_id == concept.concept_id
        or lnk.target_concept_id == concept.concept_id
    ]
    if all_links and all(lnk.review_status == "unreviewed" for lnk in all_links):
        return "all evidence links unreviewed"
    if not concept.source_artifacts and not concept.source_note_paths:
        return "no source_artifacts and no source_note_paths"
    return None


def render_unsupported_concepts(
    concepts: list[ConceptNode],
    evidence_links: list[EvidenceLink],
) -> str:
    rows: list[tuple[ConceptNode, str]] = []
    for c in concepts:
        reason = _unsupported_reason(c, evidence_links)
        if reason is not None:
            rows.append((c, reason))

    lines = [
        "# Missing Reviewed Evidence Concepts",
        "",
        f"Generated: {_now_utc()}",
        "",
        *_claim_safety_notice(),
        f"Concepts lacking reviewed supporting evidence or source traceability: {len(rows)}",
        "",
    ]
    if rows:
        lines += [*_header_row("concept_id", "canonical_name", "type", "reason")]
        for c, reason in rows:
            lines.append(_table_row(c.concept_id, c.canonical_name, c.concept_type, reason))
    else:
        lines.append("_All concepts have at least some reviewed support or traceability._")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report 5 — dataset_method_concept_map
# ---------------------------------------------------------------------------


_DATASET_RELATION_FIELDS: dict[str, str] = {
    "has_modality": "Modalities",
    "has_task": "Tasks",
    "has_brain_region": "Brain Regions",
    "has_species": "Species",
    "linked_to_opportunity": "Related Opportunities",
}


def render_dataset_method_concept_map(
    concepts: list[ConceptNode],
    evidence_links: list[EvidenceLink],
) -> str:
    by_id: dict[str, ConceptNode] = {c.concept_id: c for c in concepts}
    datasets = [c for c in concepts if c.concept_type == "dataset"]

    lines = [
        "# Dataset–Method Concept Map",
        "",
        f"Generated: {_now_utc()}",
        "",
        *_claim_safety_notice(),
        f"Dataset concepts: {len(datasets)}",
        "",
    ]
    for ds in sorted(datasets, key=lambda c: c.canonical_name):
        lines += [f"## {ds.canonical_name}", f"_concept_id_: `{ds.concept_id}`", ""]
        # group outgoing links by relation_type
        rel_targets: dict[str, list[str]] = defaultdict(list)
        for lnk in evidence_links:
            if lnk.source_concept_id != ds.concept_id:
                continue
            tgt_id = lnk.target_concept_id
            if tgt_id is None:
                continue
            node = by_id.get(tgt_id)
            name = node.canonical_name if node else tgt_id
            rel_targets[lnk.relation_type].append(name)

        for rel_key, label in _DATASET_RELATION_FIELDS.items():
            items = rel_targets.get(rel_key, [])
            if items:
                lines.append(f"- **{label}**: {', '.join(sorted(set(items)))}")
            else:
                lines.append(f"- **{label}**: _none recorded_")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report 6 — claim_basis_map
# ---------------------------------------------------------------------------


def render_claim_basis_map(
    concepts: list[ConceptNode],
    bases: list[ConceptBasis],
) -> str:
    by_id: dict[str, ConceptNode] = {c.concept_id: c for c in concepts}
    basis_by_id: dict[str, ConceptBasis] = {b.concept_id: b for b in bases}
    claims = [c for c in concepts if c.concept_type == "claim"]

    lines = [
        "# Claim Basis Map",
        "",
        f"Generated: {_now_utc()}",
        "",
        *_claim_safety_notice(),
        f"Claim concepts: {len(claims)}",
        "",
    ]
    for c in sorted(claims, key=lambda x: x.canonical_name):
        b = basis_by_id.get(c.concept_id)
        strength = b.evidence_strength if b else "none"
        supporting = b.supporting_claim_ids + b.supporting_dataset_ids + b.supporting_paper_ids if b else []
        supporting_names = [
            by_id[sid].canonical_name if sid in by_id else sid
            for sid in supporting
        ]
        uncertainty = "; ".join(b.uncertainty_notes) if b and b.uncertainty_notes else "_none_"

        lines += [
            f"### {c.canonical_name}",
            "",
            f"- **Concept ID**: `{c.concept_id}`",
            f"- **Evidence strength**: {strength}",
            f"- **Reviewed supporting evidence**: {b.reviewed_supporting_count if b else 0}",
            f"- **Reviewed contradictory evidence**: {b.reviewed_contradicting_count if b else 0}",
            f"- **Metadata-derived links**: {b.neutral_or_metadata_count if b else 0}",
            f"- **Supporting concepts**: {', '.join(supporting_names) if supporting_names else '_none_'}",
            f"- **Contradictory evidence links**: {', '.join(b.contradicting_evidence_links) if b and b.contradicting_evidence_links else '_none_'}",
            f"- **Missing evidence**: {uncertainty}",
            f"- **Review status**: {c.review_status}",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report 7 — retrieval_examples
# ---------------------------------------------------------------------------


def render_retrieval_examples(
    concepts: list[ConceptNode],
    evidence_links: list[EvidenceLink],
    graph: Any | None = None,
) -> str:
    lines = [
        "# Retrieval Examples",
        "",
        f"Generated: {_now_utc()}",
        "",
        *_claim_safety_notice(),
        "Top-5 results for 5 example queries using lexical + graph-boost retrieval.",
        "",
    ]
    for query in _RETRIEVAL_QUERIES:
        results = search_concepts(
            query=query,
            concepts=concepts,
            evidence_links=evidence_links,
            graph=graph,
            limit=5,
        )
        lines += [
            f"## Query: `{query}`",
            "",
            *_header_row("rank", "concept_id", "canonical_name", "type", "score"),
        ]
        if results:
            for i, r in enumerate(results, start=1):
                lines.append(_table_row(
                    str(i),
                    r.concept_id,
                    r.canonical_name,
                    r.concept_type,
                    f"{r.score:.4f}",
                ))
        else:
            lines.append("_No results above threshold._")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def generate_all_reports(
    concepts: list[ConceptNode],
    evidence_links: list[EvidenceLink],
    bases: list[ConceptBasis],
    graph: Any | None = None,
    root: Path | None = None,
) -> dict[str, Path]:
    """Generate all 7 reports and return dict of label -> Path."""
    base = root if root is not None else _REPO_ROOT
    report_dir = base / _REPORT_DIR

    renderers: dict[str, str] = {
        "concept_graph_summary": render_concept_graph_summary(concepts, evidence_links),
        "concept_basis_summary": render_concept_basis_summary(bases),
        "top_concepts": render_top_concepts(concepts),
        "unsupported_concepts": render_unsupported_concepts(concepts, evidence_links),
        "dataset_method_concept_map": render_dataset_method_concept_map(concepts, evidence_links),
        "claim_basis_map": render_claim_basis_map(concepts, bases),
        "retrieval_examples": render_retrieval_examples(concepts, evidence_links, graph),
    }

    output: dict[str, Path] = {}
    for key, markdown in renderers.items():
        filename = _REPORT_FILENAMES[key]
        path = report_dir / filename
        atomic_write(path, markdown)
        output[key] = path

    return output
