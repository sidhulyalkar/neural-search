"""Load local artifacts into ConceptNode and EvidenceLink objects."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neural_search.field_state.concept_memory.ids import concept_id, evidence_id
from neural_search.field_state.concept_memory.normalize import normalize_concept_name
from neural_search.field_state.concept_memory.schema import ConceptNode, EvidenceLink
from neural_search.field_state.obsidian.frontmatter import parse_frontmatter

# ---------------------------------------------------------------------------
# Repo root resolution
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]  # neural_search/field_state/concept_memory/loaders.py → repo root


def _repo_root() -> Path:
    return _REPO_ROOT


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class LoadResult:
    concepts: list[ConceptNode] = field(default_factory=list)
    evidence_links: list[EvidenceLink] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    """Read a JSONL file, returning parsed dicts; skip malformed lines."""
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    records.append(json.loads(raw))
                except json.JSONDecodeError as exc:
                    warnings.append(f"{path}:{lineno}: JSON decode error — {exc}")
    except OSError as exc:
        warnings.append(f"Could not read {path}: {exc}")
    return records


def _make_link(
    src_id: str,
    tgt_id: str,
    relation: str,
    source_artifact: str | None = None,
    *,
    evidence_text: str | None = None,
    evidence_source_id: str | None = None,
    source_repository: str | None = None,
    source_record_id: str | None = None,
    source_field: str | None = None,
    extractor_name: str | None = None,
    extractor_version: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvidenceLink:
    eid = evidence_id(src_id, tgt_id, relation)
    return EvidenceLink(
        evidence_id=eid,
        source_concept_id=src_id,
        target_concept_id=tgt_id,
        evidence_type="derived_from_artifact",
        relation_type=relation,
        evidence_text=evidence_text,
        evidence_source_id=evidence_source_id,
        source_artifact=source_artifact,
        source_repository=source_repository,
        source_record_id=source_record_id,
        source_field=source_field,
        extractor_name=extractor_name,
        extractor_version=extractor_version,
        metadata=metadata or {},
    )


def _ensure_concept(
    concepts: dict[str, ConceptNode],
    cid: str,
    name: str,
    ctype: str,
    **kwargs: Any,
) -> None:
    """Insert concept if absent; no-op if already present."""
    if cid not in concepts:
        concepts[cid] = ConceptNode(
            concept_id=cid,
            canonical_name=name,
            concept_type=ctype,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Field-state artifact loader
# ---------------------------------------------------------------------------


def load_field_state_artifacts(root: Path | None = None) -> LoadResult:
    """Load claims, benchmark gaps, and opportunities from artifacts/field_state/.

    Also loads reviewed_* variants when present.
    Returns a LoadResult with deduplicated ConceptNode and EvidenceLink objects.
    """
    base = root if root is not None else _repo_root()
    art_dir = base / "artifacts" / "field_state"

    warnings: list[str] = []
    concepts: dict[str, ConceptNode] = {}
    links: dict[str, EvidenceLink] = {}

    # --- claims ---
    claims_path = art_dir / "claims.jsonl"
    rev_claims_path = art_dir / "reviewed_claims.jsonl"
    claim_ids_seen: set[str] = set()

    def _load_claims(path: Path) -> None:
        if not path.exists():
            if path == claims_path:
                warnings.append(f"Missing expected artifact: {path}")
            return
        art_key = str(path.relative_to(base))
        for rec in _read_jsonl(path, warnings):
            # Support both raw claims (claim_id) and reviewed claims (source_record_id / field_state_id)
            cid_raw = (
                rec.get("claim_id")
                or rec.get("source_record_id")
                or rec.get("field_state_id")
            )
            if not cid_raw:
                warnings.append(f"{path}: record missing claim_id, skipping")
                continue
            cid = concept_id("claim", cid_raw)
            if cid in claim_ids_seen:
                continue
            claim_ids_seen.add(cid)
            conf = float(rec.get("confidence", 0.5))
            conf = max(0.0, min(1.0, conf))
            text = rec.get("claim_text") or cid_raw
            concepts[cid] = ConceptNode(
                concept_id=cid,
                canonical_name=cid_raw,
                concept_type="claim",
                description=text,
                confidence=conf,
                claim_count=1,
                source_artifacts=[art_key],
            )

    _load_claims(claims_path)
    _load_claims(rev_claims_path)

    # --- benchmark gaps ---
    gaps_path = art_dir / "benchmark_gaps.jsonl"
    rev_gaps_path = art_dir / "reviewed_benchmark_gaps.jsonl"
    gap_claim_links: list[tuple[str, str]] = []
    gap_ids_seen: set[str] = set()

    def _load_gaps(path: Path) -> None:
        if not path.exists():
            if path == gaps_path:
                warnings.append(f"Missing expected artifact: {path}")
            return
        art_key = str(path.relative_to(base))
        for rec in _read_jsonl(path, warnings):
            gid_raw = (
                rec.get("gap_id")
                or rec.get("source_record_id")
                or rec.get("field_state_id")
            )
            if not gid_raw:
                warnings.append(f"{path}: record missing gap_id, skipping")
                continue
            gid = concept_id("benchmark_gap", gid_raw)
            if gid in gap_ids_seen:
                continue
            gap_ids_seen.add(gid)
            concepts[gid] = ConceptNode(
                concept_id=gid,
                canonical_name=rec.get("title") or gid_raw,
                concept_type="benchmark_gap",
                description=rec.get("description"),
                source_artifacts=[art_key],
            )
            for claim_raw in rec.get("related_claim_ids") or []:
                gap_claim_links.append((claim_raw, gid_raw))

    _load_gaps(gaps_path)
    _load_gaps(rev_gaps_path)

    # Build gap↔claim evidence links
    for claim_raw, gap_raw in gap_claim_links:
        src = concept_id("claim", claim_raw)
        tgt = concept_id("benchmark_gap", gap_raw)
        if src in concepts and tgt in concepts:
            lnk = _make_link(src, tgt, "linked_to_benchmark_gap")
            links[lnk.evidence_id] = lnk

    # --- opportunities ---
    opp_path = art_dir / "opportunities.jsonl"
    rev_opp_path = art_dir / "reviewed_opportunities.jsonl"
    opp_ids_seen: set[str] = set()

    def _load_opps(path: Path) -> None:
        if not path.exists():
            if path == opp_path:
                warnings.append(f"Missing expected artifact: {path}")
            return
        art_key = str(path.relative_to(base))
        for rec in _read_jsonl(path, warnings):
            oid_raw = (
                rec.get("opportunity_id")
                or rec.get("source_record_id")
                or rec.get("field_state_id")
            )
            if not oid_raw:
                warnings.append(f"{path}: record missing opportunity_id, skipping")
                continue
            oid = concept_id("opportunity", oid_raw)
            if oid in opp_ids_seen:
                continue
            opp_ids_seen.add(oid)
            concepts[oid] = ConceptNode(
                concept_id=oid,
                canonical_name=rec.get("title") or oid_raw,
                concept_type="opportunity",
                description=rec.get("description"),
                source_artifacts=[art_key],
            )
            # opp → gap links
            for gid_raw in rec.get("linked_gap_ids") or []:
                tgt = concept_id("benchmark_gap", gid_raw)
                if tgt in concepts:
                    lnk = _make_link(oid, tgt, "linked_to_benchmark_gap")
                    links[lnk.evidence_id] = lnk
            # claim → opp links
            for cid_raw in rec.get("linked_claim_ids") or []:
                src = concept_id("claim", cid_raw)
                if src in concepts:
                    lnk = _make_link(src, oid, "linked_to_opportunity")
                    links[lnk.evidence_id] = lnk

    _load_opps(opp_path)
    _load_opps(rev_opp_path)

    return LoadResult(
        concepts=list(concepts.values()),
        evidence_links=list(links.values()),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------------


def load_corpus(
    corpus_path: Path | None = None,
    max_datasets: int | None = None,
) -> LoadResult:
    """Load combined_corpus.jsonl and produce dataset/modality/task/etc. nodes."""
    if corpus_path is None:
        corpus_path = _repo_root() / "data" / "corpus" / "normalized" / "combined_corpus.jsonl"

    warnings: list[str] = []
    concepts: dict[str, ConceptNode] = {}
    links: dict[str, EvidenceLink] = {}

    if not corpus_path.exists():
        warnings.append(f"Corpus not found: {corpus_path}")
        return LoadResult(warnings=warnings)

    count = 0
    with corpus_path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            if max_datasets is not None and count >= max_datasets:
                break
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError as exc:
                warnings.append(f"{corpus_path}:{lineno}: JSON error — {exc}")
                continue

            source = rec.get("source") or "unknown"
            dataset_id = rec.get("source_id") or rec.get("id")
            if not dataset_id:
                warnings.append(f"{corpus_path}:{lineno}: no source_id/id, skipping")
                continue

            ds_cid = concept_id("dataset", f"{source}_{dataset_id}")
            raw_title = rec.get("title") or ""
            title = raw_title.strip() or str(dataset_id)

            if ds_cid in concepts:
                node = concepts[ds_cid]
                concepts[ds_cid] = node.model_copy(
                    update={
                        "dataset_count": node.dataset_count + 1,
                        "evidence_count": node.evidence_count + 1,
                        "source_ids": list(dict.fromkeys(node.source_ids + [str(dataset_id)])),
                    }
                )
            else:
                concepts[ds_cid] = ConceptNode(
                    concept_id=ds_cid,
                    canonical_name=title,
                    concept_type="dataset",
                    description=rec.get("description"),
                    dataset_count=1,
                    source_ids=[str(dataset_id)],
                )
            count += 1

            def _add_relation(
                source_cid: str,
                items: list[str],
                ctype: str,
                relation: str,
                source_field: str,
                *,
                normalise: bool = True,
                source_repository: str = str(source),
                source_record_id: str = str(dataset_id),
            ) -> None:
                for raw_val in items:
                    if not raw_val:
                        continue
                    raw_label = str(raw_val)
                    norm = normalize_concept_name(raw_label) if normalise else raw_label.lower().strip()
                    if not norm:
                        continue
                    tgt_cid = concept_id(ctype, norm)
                    if tgt_cid not in concepts:
                        concepts[tgt_cid] = ConceptNode(
                            concept_id=tgt_cid,
                            canonical_name=norm,
                            concept_type=ctype,
                        )
                    lnk = _make_link(
                        source_cid,
                        tgt_cid,
                        relation,
                        str(corpus_path),
                        evidence_text=f"{source_field}: {raw_label}",
                        evidence_source_id=source_record_id,
                        source_repository=source_repository,
                        source_record_id=source_record_id,
                        source_field=source_field,
                        extractor_name="concept_memory_corpus_loader",
                        extractor_version="0.4.1",
                        metadata={
                            "provenance_status": "complete",
                            "corpus_path": str(corpus_path),
                        },
                    )
                    links[lnk.evidence_id] = lnk

            _add_relation(
                ds_cid,
                rec.get("modalities") or [],
                "modality",
                "has_modality",
                "modalities",
            )
            _add_relation(ds_cid, rec.get("tasks") or [], "task", "has_task", "tasks")
            _add_relation(
                ds_cid,
                rec.get("brain_regions") or [],
                "brain_region",
                "has_brain_region",
                "brain_regions",
                normalise=False,
            )
            _add_relation(
                ds_cid,
                rec.get("species") or [],
                "species",
                "has_species",
                "species",
            )

    return LoadResult(
        concepts=list(concepts.values()),
        evidence_links=list(links.values()),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Obsidian notes loader
# ---------------------------------------------------------------------------

_NOTE_TYPE_TO_CONCEPT_TYPE: dict[str, str] = {
    "claim": "claim",
    "benchmark_gap": "benchmark_gap",
    "opportunity": "opportunity",
    "field_snapshot": "neuroscience_concept",
    "codex_task": "neuroscience_concept",
    "decision_log": "neuroscience_concept",
    "qrels_review": "neuroscience_concept",
    "eval_snapshot": "neuroscience_concept",
}

_VALID_OBSIDIAN_TYPES = set(_NOTE_TYPE_TO_CONCEPT_TYPE)


def load_obsidian_notes(
    vault_path: Path,
    field: str = "neuroscience_dataset_reuse",
) -> LoadResult:
    """Load Markdown notes from <vault_path>/Field-State/ into concept nodes."""
    warnings: list[str] = []
    concepts: dict[str, ConceptNode] = {}
    links: dict[str, EvidenceLink] = {}

    field_state_dir = vault_path / "Field-State"
    if not field_state_dir.exists():
        warnings.append(f"Obsidian vault Field-State directory not found: {field_state_dir}")
        return LoadResult(warnings=warnings)

    for note_path in sorted(field_state_dir.rglob("*.md")):
        try:
            text = note_path.read_text(encoding="utf-8")
            frontmatter, _ = parse_frontmatter(text)
        except (OSError, ValueError) as exc:
            warnings.append(f"{note_path}: could not parse — {exc}")
            continue

        note_type = str(frontmatter.get("type", ""))
        if note_type not in _VALID_OBSIDIAN_TYPES:
            continue

        ctype = _NOTE_TYPE_TO_CONCEPT_TYPE[note_type]
        source_id = str(frontmatter.get("field_state_id", ""))
        name = str(frontmatter.get("title", "") or source_id or note_path.stem)
        if not name:
            warnings.append(f"{note_path}: cannot determine name, skipping")
            continue

        cid = concept_id(ctype, name)
        review_status = str(frontmatter.get("review_status", "unreviewed"))

        concepts[cid] = ConceptNode(
            concept_id=cid,
            canonical_name=name,
            concept_type=ctype,
            source_note_paths=[str(note_path)],
            source_ids=[source_id] if source_id else [],
            review_status=review_status,
        )

        lnk = EvidenceLink(
            evidence_id=evidence_id(cid, None, "derived_from_note"),
            source_concept_id=cid,
            target_concept_id=None,
            evidence_type="derived_from_artifact",
            relation_type="derived_from_note",
            source_note_path=str(note_path),
        )
        links[lnk.evidence_id] = lnk

    return LoadResult(
        concepts=list(concepts.values()),
        evidence_links=list(links.values()),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Paper notes loader
# ---------------------------------------------------------------------------


def load_paper_notes(
    vault_path: Path | None = None,
    notes_dir: Path | None = None,
) -> LoadResult:
    """Load Markdown paper notes that have type: paper or doi+authors frontmatter."""
    warnings: list[str] = []
    concepts: dict[str, ConceptNode] = {}
    links: dict[str, EvidenceLink] = {}

    search_dirs: list[Path] = []
    if notes_dir is not None:
        search_dirs.append(notes_dir)
    elif vault_path is not None:
        search_dirs.append(vault_path)
    else:
        return LoadResult(warnings=["No vault_path or notes_dir provided for paper notes"])

    _PAPER_RELATION: dict[str, str] = {
        "methods": "uses_method",
        "datasets": "uses_dataset",
        "tasks": "has_task",
        "modalities": "has_modality",
        "brain_regions": "has_brain_region",
        "claims": "supports",
    }

    _FIELD_CTYPE: dict[str, str] = {
        "methods": "method",
        "datasets": "dataset",
        "tasks": "task",
        "modalities": "modality",
        "brain_regions": "brain_region",
        "claims": "claim",
    }

    for search_dir in search_dirs:
        for note_path in sorted(search_dir.rglob("*.md")):
            try:
                text = note_path.read_text(encoding="utf-8")
                frontmatter, _ = parse_frontmatter(text)
            except (OSError, ValueError) as exc:
                warnings.append(f"{note_path}: could not parse — {exc}")
                continue

            is_paper = (
                str(frontmatter.get("type", "")) == "paper"
                or (frontmatter.get("doi") and frontmatter.get("authors"))
            )
            if not is_paper:
                continue

            title = str(frontmatter.get("title", "") or note_path.stem)
            paper_cid = concept_id("paper", title)

            concepts[paper_cid] = ConceptNode(
                concept_id=paper_cid,
                canonical_name=title,
                concept_type="paper",
                source_note_paths=[str(note_path)],
                paper_count=1,
            )

            for fm_field, relation in _PAPER_RELATION.items():
                ctype = _FIELD_CTYPE[fm_field]
                items = frontmatter.get(fm_field) or []
                if isinstance(items, str):
                    items = [items]
                for raw_val in items:
                    norm = normalize_concept_name(str(raw_val))
                    if not norm:
                        continue
                    tgt_cid = concept_id(ctype, norm)
                    if tgt_cid not in concepts:
                        concepts[tgt_cid] = ConceptNode(
                            concept_id=tgt_cid,
                            canonical_name=norm,
                            concept_type=ctype,
                        )
                    lnk = _make_link(paper_cid, tgt_cid, relation, str(note_path))
                    links[lnk.evidence_id] = lnk

    return LoadResult(
        concepts=list(concepts.values()),
        evidence_links=list(links.values()),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------


def merge_load_results(results: list[LoadResult]) -> LoadResult:
    """Merge multiple LoadResults, deduplicating by concept_id and evidence_id.

    When a concept_id collision occurs: counts are summed and list fields are
    merged (deduplicated). The last writer wins for all scalar metadata fields.
    """
    merged_concepts: dict[str, ConceptNode] = {}
    merged_links: dict[str, EvidenceLink] = {}
    all_warnings: list[str] = []

    for result in results:
        all_warnings.extend(result.warnings)

        for link in result.evidence_links:
            merged_links[link.evidence_id] = link

        for node in result.concepts:
            if node.concept_id not in merged_concepts:
                merged_concepts[node.concept_id] = node
            else:
                existing = merged_concepts[node.concept_id]
                merged_concepts[node.concept_id] = existing.model_copy(
                    update={
                        "evidence_count": existing.evidence_count + node.evidence_count,
                        "dataset_count": existing.dataset_count + node.dataset_count,
                        "claim_count": existing.claim_count + node.claim_count,
                        "paper_count": existing.paper_count + node.paper_count,
                        "source_ids": list(
                            dict.fromkeys(existing.source_ids + node.source_ids)
                        ),
                        "source_artifacts": list(
                            dict.fromkeys(existing.source_artifacts + node.source_artifacts)
                        ),
                        "source_note_paths": list(
                            dict.fromkeys(existing.source_note_paths + node.source_note_paths)
                        ),
                        "aliases": list(
                            dict.fromkeys(existing.aliases + node.aliases)
                        ),
                        # Scalar overrides: last writer wins for description / name
                        "canonical_name": node.canonical_name or existing.canonical_name,
                        "description": node.description or existing.description,
                    }
                )

    return LoadResult(
        concepts=list(merged_concepts.values()),
        evidence_links=list(merged_links.values()),
        warnings=all_warnings,
    )
