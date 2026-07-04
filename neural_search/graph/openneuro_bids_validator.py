"""Live BIDS structure validation against real OpenNeuro datasets.

Uses OpenNeuro's public GraphQL API (`https://openneuro.org/crn/graphql`,
same endpoint already used by `neural_search/ingestion/openneuro.py`) rather
than downloading dataset files. The API's `summary` field is server-computed
directly from the dataset's actual BIDS content (modalities, tasks, subject
list) — confirmed by direct query during development against ds000117 — so
this is real, live validation without needing per-subject file downloads.

Deeper checks (specific `*_events.tsv` presence/columns) would need the
nested file tree, which this API does not expose in a single query (`files`
on a snapshot only lists top-level entries; subdirectories like `sub-01/`
appear unexpanded). That is a known, documented limitation of this
validator, not a silent gap: `has_task_events` below is inferred from the
`summary.tasks` list being non-empty, not from directly reading an
events.tsv file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

OPENNEURO_API_URL = "https://openneuro.org/crn/graphql"
log = logging.getLogger(__name__)

_QUERY = """
query ($id: ID!) {
  dataset(id: $id) {
    id
    latestSnapshot {
      tag
      files { filename size }
      summary {
        modalities
        tasks
        subjects
      }
    }
  }
}
"""


@dataclass
class OpenNeuroValidation:
    dataset_id: str
    has_dataset_description: bool = False
    has_participants_tsv: bool = False
    modalities: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    n_subjects: int = 0
    error: str | None = None


def validate_openneuro_dataset(
    dataset_id: str,
    client: httpx.Client | None = None,
) -> OpenNeuroValidation:
    """Validate a dataset's declared BIDS structure via the OpenNeuro API.

    `dataset_id` is the OpenNeuro accession number (e.g. "ds000117").
    """

    result = OpenNeuroValidation(dataset_id=dataset_id)
    owns_client = client is None
    client = client or httpx.Client(timeout=20.0, follow_redirects=True)
    try:
        resp = client.post(
            OPENNEURO_API_URL, json={"query": _QUERY, "variables": {"id": dataset_id}}
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("errors"):
            result.error = str(payload["errors"])
            return result

        dataset = payload.get("data", {}).get("dataset")
        if not dataset:
            result.error = "dataset not found"
            return result

        snapshot = dataset.get("latestSnapshot") or {}
        filenames = {f["filename"] for f in snapshot.get("files", [])}
        result.has_dataset_description = "dataset_description.json" in filenames
        result.has_participants_tsv = "participants.tsv" in filenames

        summary = snapshot.get("summary") or {}
        result.modalities = summary.get("modalities") or []
        result.tasks = summary.get("tasks") or []
        result.n_subjects = len(summary.get("subjects") or [])
    except Exception as exc:  # noqa: BLE001 - record and continue, this validates untrusted remote data
        log.warning("OpenNeuro validation failed for %s: %s", dataset_id, exc)
        result.error = str(exc)
    finally:
        if owns_client:
            client.close()
    return result
