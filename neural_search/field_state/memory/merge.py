"""Apply human review overlays without mutating generated records."""

from __future__ import annotations

from typing import Any


def _record_id(record: dict[str, Any]) -> str:
    for key in ("field_state_id", "claim_id", "gap_id", "opportunity_id", "snapshot_id"):
        value = record.get(key)
        if value:
            return str(value)
    return ""


def apply_review_overlays(
    generated_records: list[dict[str, Any]],
    overlay_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return generated records annotated with human review overlay fields."""
    overlays: dict[str, dict[str, Any]] = {}
    for record in overlay_records:
        for key in ("field_state_id", "source_record_id"):
            value = record.get(key)
            if value:
                overlays[str(value)] = record
    merged: list[dict[str, Any]] = []
    for record in generated_records:
        copy = dict(record)
        overlay = overlays.get(_record_id(record))
        if overlay is not None:
            copy["review_status"] = overlay.get("review_status", copy.get("review_status"))
            copy["status"] = overlay.get("status", copy.get("status"))
            copy["human_reviewed"] = True
            copy["human_priority"] = overlay.get("human_priority")
            copy["human_tags"] = overlay.get("human_tags", [])
            copy["source_note_path"] = overlay.get("source_note_path")
        else:
            copy["human_reviewed"] = False
        merged.append(copy)
    return merged
