"""Export qrels candidate pools to Obsidian review notes."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neural_search.field_state.eval_memory.qrels_schema import (
    QrelsCandidate,
    stable_qrels_candidate_id,
)
from neural_search.field_state.memory.index import write_memory_index
from neural_search.field_state.obsidian.dashboard import write_dashboard
from neural_search.field_state.obsidian.frontmatter import (
    compose_note,
    parse_frontmatter,
)
from neural_search.field_state.obsidian.paths import ensure_vault_structure, note_path
from neural_search.field_state.obsidian.reader import extract_human_block
from neural_search.field_state.obsidian.sync import append_sync_log
from neural_search.field_state.obsidian.templates import (
    HUMAN_BEGIN,
    HUMAN_END,
)
from neural_search.field_state.obsidian.writer import atomic_write_text
from neural_search.field_state.store import QRELS_CANDIDATES_PATH, write_jsonl


def _load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _record_id(record: dict[str, Any]) -> str:
    if record.get("record_id"):
        return str(record["record_id"])
    if record.get("dataset_id"):
        return str(record["dataset_id"])
    source = record.get("source")
    source_id = record.get("source_id")
    if source and source_id:
        return f"{source}:{source_id}"
    return str(record.get("id", ""))


def _query_text(record: dict[str, Any]) -> str:
    return str(record.get("query") or record.get("query_text") or record.get("text") or "")


def load_qrels_candidates(
    *,
    pool_path: Path | None,
    queries_path: Path | None,
    corpus_path: Path | None,
    field: str = "neuroscience_dataset_reuse",
    limit: int | None = None,
) -> list[QrelsCandidate]:
    """Load qrels candidates from pool, query, and corpus artifacts."""
    pool = _load_jsonl(pool_path)
    queries = {str(row.get("query_id")): row for row in _load_jsonl(queries_path)}
    corpus = {_record_id(row): row for row in _load_jsonl(corpus_path)}
    candidates: list[QrelsCandidate] = []
    for pool_row in pool[:limit]:
        query_id = str(pool_row.get("query_id", ""))
        dataset_id = _record_id(pool_row)
        query = queries.get(query_id, {})
        dataset = corpus.get(dataset_id, {})
        pooled_from = pool_row.get("pooled_from", [])
        retrieval_method: str | None
        if isinstance(pooled_from, list):
            retrieval_method = ",".join(str(item) for item in pooled_from)
        else:
            retrieval_method = str(pooled_from) if pooled_from else None
        source_artifacts = [
            str(path)
            for path in (pool_path, queries_path, corpus_path)
            if path is not None and path.exists()
        ]
        candidates.append(
            QrelsCandidate(
                id=stable_qrels_candidate_id(query_id, dataset_id),
                query_id=query_id,
                query_text=_query_text(query) or str(pool_row.get("query", "")),
                query_intent=query.get("intent") or pool_row.get("intent"),
                dataset_id=dataset_id,
                dataset_title=str(
                    dataset.get("title")
                    or pool_row.get("dataset_title")
                    or pool_row.get("title")
                    or dataset_id
                ),
                dataset_source=dataset.get("source") or pool_row.get("dataset_source"),
                dataset_description=dataset.get("description")
                or pool_row.get("dataset_description"),
                rank=pool_row.get("min_rank") or pool_row.get("rank"),
                retrieval_score=pool_row.get("retrieval_score") or pool_row.get("score"),
                retrieval_method=retrieval_method or pool_row.get("retrieval_method"),
                hard_negative_reason=pool_row.get("hard_negative_reason"),
                expected_relevance_hint=pool_row.get("status")
                or pool_row.get("expected_relevance_hint"),
                field=field,
                source_artifacts=source_artifacts,
                metadata={
                    "pool": pool_row,
                    "query_known_failure_modes": query.get("known_failure_modes", []),
                },
            )
        )
    return candidates


def _qrels_human_block() -> str:
    return (
        f"{HUMAN_BEGIN}\n\n"
        "## Human label\n\n"
        "relevance_score:\n"
        "usefulness_score:\n"
        "hard_negative_violation:\n"
        "label_confidence:\n"
        "annotator_id:\n\n"
        "## Rationale\n\n"
        "<!-- Explain the label here. -->\n\n"
        "## Reviewer notes\n\n"
        "<!-- Add notes here. -->\n\n"
        f"{HUMAN_END}"
    )


def render_qrels_review_body(candidate: QrelsCandidate, human_block: str | None = None) -> str:
    """Render one qrels review note body."""
    generated = f"""# Qrels Review: {candidate.query_id} x {candidate.dataset_title}

<!-- FIELDSTATE:BEGIN generated -->

## Query

{candidate.query_text}

## Query intent

{candidate.query_intent or "unknown"}

## Dataset

**Title:** {candidate.dataset_title}

**ID:** {candidate.dataset_id}

**Source:** {candidate.dataset_source or "unknown"}

## Dataset description

{candidate.dataset_description or "No description available."}

## Retrieval metadata

| Field | Value |
|---|---|
| Rank | {candidate.rank if candidate.rank is not None else ""} |
| Retrieval score | {candidate.retrieval_score if candidate.retrieval_score is not None else ""} |
| Retrieval method | {candidate.retrieval_method or ""} |
| Hard-negative reason | {candidate.hard_negative_reason or ""} |

## Labeling guide

Use the 0-3 graded scale:

- 0 = irrelevant / not reusable
- 1 = weakly related, unlikely useful
- 2 = relevant and potentially reusable
- 3 = highly relevant, directly reusable

Mark `hard_negative_violation: true` if the result violates an explicit negative constraint in the query.

<!-- FIELDSTATE:END generated -->
"""
    return f"{generated}\n{human_block or _qrels_human_block()}\n"


def _frontmatter(candidate: QrelsCandidate) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    return {
        "type": "qrels_review",
        "field_state_id": candidate.id,
        "field": candidate.field,
        "title": f"{candidate.query_id} x {candidate.dataset_title}",
        "query_id": candidate.query_id,
        "dataset_id": candidate.dataset_id,
        "query_intent": candidate.query_intent,
        "dataset_source": candidate.dataset_source,
        "status": "unreviewed",
        "review_status": "unreviewed",
        "relevance_score": None,
        "usefulness_score": None,
        "hard_negative_violation": None,
        "label_confidence": None,
        "annotator_id": None,
        "adjudication_status": None,
        "created_at": now,
        "updated_at": now,
        "generated_at": now,
        "schema_version": "0.3",
        "source_artifacts": candidate.source_artifacts,
        "tags": ["field-state", "qrels", "eval-memory"],
    }


def _write_qrels_note(vault_path: Path, candidate: QrelsCandidate) -> str:
    path = note_path(
        vault_path,
        "qrels_review",
        f"{candidate.query_id} {candidate.dataset_title}",
        "unreviewed",
    )
    body = render_qrels_review_body(candidate)
    frontmatter = _frontmatter(candidate)
    if path.exists():
        old_text = path.read_text(encoding="utf-8")
        old_frontmatter, old_body = parse_frontmatter(old_text)
        human_block = extract_human_block(old_body) or _qrels_human_block()
        body = render_qrels_review_body(candidate, human_block)
        for key in (
            "status",
            "review_status",
            "relevance_score",
            "usefulness_score",
            "hard_negative_violation",
            "label_confidence",
            "annotator_id",
            "adjudication_status",
        ):
            if key in old_frontmatter:
                frontmatter[key] = old_frontmatter[key]
        if old_frontmatter.get("created_at"):
            frontmatter["created_at"] = old_frontmatter["created_at"]
    text = compose_note(frontmatter, body)
    existed = path.exists()
    if existed and path.read_text(encoding="utf-8") == text:
        return "skipped"
    atomic_write_text(path, text)
    return "updated" if existed else "created"


def export_qrels_candidates(
    *,
    pool_path: Path | None,
    queries_path: Path | None,
    corpus_path: Path | None,
    vault_path: Path,
    field: str = "neuroscience_dataset_reuse",
    limit: int | None = None,
    root: Path | None = None,
) -> dict[str, int]:
    """Write qrels candidates JSONL and export Obsidian review notes."""
    ensure_vault_structure(vault_path)
    candidates = load_qrels_candidates(
        pool_path=pool_path,
        queries_path=queries_path,
        corpus_path=corpus_path,
        field=field,
        limit=limit,
    )
    write_jsonl(QRELS_CANDIDATES_PATH, candidates, root)
    counts = {"created": 0, "updated": 0, "skipped": 0}
    for candidate in candidates:
        counts[_write_qrels_note(vault_path, candidate)] += 1
    append_sync_log(
        vault_path,
        operation="qrels-export",
        field=field,
        notes_created=counts["created"],
        notes_updated=counts["updated"],
        notes_skipped=counts["skipped"],
    )
    write_memory_index(vault_path, field)
    write_dashboard(vault_path, field)
    return counts
