"""Import human qrels review notes from Obsidian."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from neural_search.field_state.eval_memory.qrels_schema import QrelsReview
from neural_search.field_state.obsidian.paths import vault_relative
from neural_search.field_state.obsidian.reader import ObsidianNote, read_obsidian_notes
from neural_search.field_state.obsidian.sync import append_sync_log
from neural_search.field_state.store import QRELS_REVIEWS_PATH, write_jsonl


def _coerce_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    try:
        parsed = int(str(value).strip())
    except ValueError:
        return None
    return parsed if parsed in {0, 1, 2, 3} else None


def _coerce_bool(value: object) -> bool | None:
    if value in {None, ""}:
        return None
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0"}:
        return False
    return None


def parse_human_label_fields(human_block: str) -> dict[str, str]:
    """Parse simple `key: value` fields from the human block."""
    fields: dict[str, str] = {}
    for line in human_block.splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split(":", 1)
        clean_key = key.strip()
        if clean_key in {
            "relevance_score",
            "usefulness_score",
            "hard_negative_violation",
            "label_confidence",
            "annotator_id",
            "adjudicated_relevance_score",
            "adjudicated_usefulness_score",
            "adjudicated_hard_negative_violation",
        }:
            fields[clean_key] = value.strip()
    return fields


def _section_after(human_block: str, heading: str) -> str | None:
    index = human_block.find(heading)
    if index == -1:
        return None
    text = human_block[index + len(heading) :].strip()
    next_heading = text.find("\n## ")
    if next_heading != -1:
        text = text[:next_heading]
    end_marker = "<!-- FIELDSTATE:END human -->"
    if end_marker in text:
        text = text[: text.index(end_marker)]
    text = text.strip()
    return text or None


def review_from_note(vault_path: Path, note: ObsidianNote) -> QrelsReview:
    """Build a QrelsReview from one qrels note."""
    frontmatter = note.frontmatter
    fields = parse_human_label_fields(note.human_block)
    relevance = _coerce_int(fields.get("relevance_score") or frontmatter.get("relevance_score"))
    usefulness = _coerce_int(fields.get("usefulness_score") or frontmatter.get("usefulness_score"))
    hard_negative = _coerce_bool(
        fields.get("hard_negative_violation") or frontmatter.get("hard_negative_violation")
    )
    review_status = str(frontmatter.get("review_status") or "unreviewed")
    if relevance is not None and review_status == "unreviewed":
        review_status = "reviewed"
    return QrelsReview(
        candidate_id=str(frontmatter.get("field_state_id", "")),
        query_id=str(frontmatter.get("query_id", "")),
        dataset_id=str(frontmatter.get("dataset_id", "")),
        annotator_id=fields.get("annotator_id") or frontmatter.get("annotator_id"),
        relevance_score=relevance,  # type: ignore[arg-type]
        usefulness_score=usefulness,  # type: ignore[arg-type]
        hard_negative_violation=hard_negative,
        label_confidence=fields.get("label_confidence")
        or frontmatter.get("label_confidence"),
        review_status=review_status,  # type: ignore[arg-type]
        rationale=_section_after(note.human_block, "## Rationale"),
        reviewer_notes=_section_after(note.human_block, "## Reviewer notes"),
        reviewed_at=frontmatter.get("reviewed_at") or datetime.now(UTC).isoformat()
        if relevance is not None
        else None,
        source_note_path=vault_relative(vault_path, note.path),
        adjudicated_relevance_score=_coerce_int(
            fields.get("adjudicated_relevance_score")
            or frontmatter.get("adjudicated_relevance_score")
        ),  # type: ignore[arg-type]
        adjudicated_usefulness_score=_coerce_int(
            fields.get("adjudicated_usefulness_score")
            or frontmatter.get("adjudicated_usefulness_score")
        ),  # type: ignore[arg-type]
        adjudicated_hard_negative_violation=_coerce_bool(
            fields.get("adjudicated_hard_negative_violation")
            or frontmatter.get("adjudicated_hard_negative_violation")
        ),
        adjudicator_notes=frontmatter.get("adjudicator_notes"),
    )


def import_qrels_reviews(
    *,
    vault_path: Path,
    field: str = "neuroscience_dataset_reuse",
    root: Path | None = None,
) -> list[QrelsReview]:
    """Import qrels reviews from an Obsidian vault."""
    notes = [
        note
        for note in read_obsidian_notes(vault_path, field)
        if note.note_type == "qrels_review"
    ]
    reviews = [review_from_note(vault_path, note) for note in notes]
    write_jsonl(QRELS_REVIEWS_PATH, reviews, root)
    append_sync_log(
        vault_path,
        operation="qrels-import",
        field=field,
        notes_created=len(reviews),
    )
    return reviews
