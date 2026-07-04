#!/usr/bin/env python3
"""Run live file validation against the top-confidence reanalysis suggestions.

Selects the top N candidate datasets (by confidence, deduplicated) across
`dataset_old_dataset_new_method_candidate` and `dataset_reanalysis_bridge_dataset`
edges in the production graph, and for each dataset whose source has a live
validator (currently DANDI and OpenNeuro — see
`neural_search/graph/dandi_nwb_validator.py` and
`neural_search/graph/openneuro_bids_validator.py`), fetches real metadata and
checks whether it confirms the suggested analysis family's core requirement
(`neural_search/graph/file_validation_requirements.py`).

Writes:
- `artifacts/validation/top_suggestions_file_validation.jsonl` — one row per
  validated dataset, machine-readable, consumed by
  `neural_search/graph/evidence_tier_upgrader.py` to upgrade matching edges'
  `evidence_tier` property on the next graph build.
- `reports/top_suggestions_validation_report.md` — human-readable summary.

This does not claim full coverage: datasets from sources with no validator
(figshare, gin, crcns, etc. as of this writing) are recorded with
`validator: "none"` rather than silently skipped, matching this project's
practice of reporting gaps explicitly.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from neural_search.graph.dandi_nwb_validator import validate_dandiset
from neural_search.graph.file_validation_requirements import (
    dandi_confirms_requirement,
    openneuro_confirms_requirement,
)
from neural_search.graph.openneuro_bids_validator import validate_openneuro_dataset
from neural_search.graph.schema import read_graph_json

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
GRAPH_PATH = PROJECT_ROOT / "data" / "graph" / "neural_search_graph.real_corpus.json"
OUTPUT_JSONL = PROJECT_ROOT / "artifacts" / "validation" / "top_suggestions_file_validation.jsonl"
OUTPUT_REPORT = PROJECT_ROOT / "reports" / "top_suggestions_validation_report.md"

SUGGESTION_EDGE_TYPES = {
    "dataset_old_dataset_new_method_candidate",
    "dataset_reanalysis_bridge_dataset",
}
VALIDATED_SOURCES = {"dandi", "openneuro"}


def select_top_suggestions(graph, top_n: int) -> list[Any]:
    """Highest-confidence suggestion edge per distinct candidate dataset."""

    best_by_dataset: dict[str, Any] = {}
    for edge in graph.edges.values():
        if edge.edge_type not in SUGGESTION_EDGE_TYPES:
            continue
        current = best_by_dataset.get(edge.source_node_id)
        if current is None or edge.confidence > current.confidence:
            best_by_dataset[edge.source_node_id] = edge
    ranked = sorted(
        best_by_dataset.values(), key=lambda e: (-e.confidence, e.source_node_id)
    )
    return ranked[:top_n]


def _parse_dataset_node_id(node_id: str) -> tuple[str, str]:
    # node:dataset:{source}:{source_id}
    parts = node_id.split(":", 3)
    return parts[2], parts[3]


def validate_one(edge, client: httpx.Client) -> dict[str, Any]:
    source, source_id = _parse_dataset_node_id(edge.source_node_id)
    analysis_family = edge.properties.get("analysis_family") or edge.properties.get("method")
    row: dict[str, Any] = {
        "dataset_node_id": edge.source_node_id,
        "source": source,
        "source_id": source_id,
        "suggested_edge_type": edge.edge_type,
        "analysis_family": analysis_family,
        "suggestion_confidence": edge.confidence,
    }

    if source == "dandi":
        results = validate_dandiset(source_id, max_assets=1, client=client)
        if not results:
            row["validator"] = "dandi"
            row["error"] = "no NWB assets found"
            row["confirmed"] = False
            return row
        result = results[0]
        row["validator"] = "dandi"
        row["raw_result"] = asdict(result)
        row["error"] = result.error
        row["confirmed"] = (
            dandi_confirms_requirement(analysis_family, result) if analysis_family else False
        )
    elif source == "openneuro":
        result = validate_openneuro_dataset(source_id, client=client)
        row["validator"] = "openneuro"
        row["raw_result"] = asdict(result)
        row["error"] = result.error
        row["confirmed"] = (
            openneuro_confirms_requirement(analysis_family, result) if analysis_family else False
        )
    else:
        row["validator"] = "none"
        row["error"] = f"no live validator available for source '{source}'"
        row["confirmed"] = False

    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None, help="cap for a quick test run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    graph = read_graph_json(GRAPH_PATH)
    top_edges = select_top_suggestions(graph, args.top_n)
    if args.limit:
        top_edges = top_edges[: args.limit]
    print(f"Selected {len(top_edges)} suggestions to validate")

    # Write incrementally (one row per completed validation, flushed
    # immediately) rather than buffering everything in memory and writing
    # once at the end. A transient network failure partway through a large
    # batch (confirmed live: httpx.ConnectTimeout on a DANDI request at
    # scale, previously uncaught) must not discard every already-completed
    # validation -- that's the same "one bad request corrupts the whole run"
    # failure mode this project has hit before with the literature-linking
    # scripts (see docs/memory literature_source_expansion). Any exception
    # from validate_one() (not just the ones its own try/except already
    # handles per-source) is recorded as an errored row and the run
    # continues, rather than crashing and losing prior progress.
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with (
        httpx.Client(timeout=30.0, follow_redirects=True) as client,
        OUTPUT_JSONL.open("w", encoding="utf-8") as fh,
    ):
        for i, edge in enumerate(top_edges, start=1):
            t0 = time.time()
            try:
                row = validate_one(edge, client)
            except Exception as exc:  # noqa: BLE001 - record and continue; see comment above
                source, source_id = _parse_dataset_node_id(edge.source_node_id)
                row = {
                    "dataset_node_id": edge.source_node_id,
                    "source": source,
                    "source_id": source_id,
                    "suggested_edge_type": edge.edge_type,
                    "analysis_family": edge.properties.get("analysis_family") or edge.properties.get("method"),
                    "suggestion_confidence": edge.confidence,
                    "validator": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "confirmed": False,
                }
                log.warning("Validation crashed for %s: %s", edge.source_node_id, exc)
            row["elapsed_s"] = round(time.time() - t0, 2)
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(
                f"  [{i}/{len(top_edges)}] {row['dataset_node_id']} "
                f"({row['validator']}) confirmed={row['confirmed']} "
                f"({row['elapsed_s']}s)"
            )
    print(f"Wrote {OUTPUT_JSONL}")

    _write_report(rows)
    return 0


def _write_report(rows: list[dict[str, Any]]) -> None:
    by_validator = Counter(r["validator"] for r in rows)
    confirmed = [r for r in rows if r["confirmed"]]
    errored = [r for r in rows if r.get("error")]

    lines = [
        "# Top Reanalysis Suggestions: Live File Validation Report",
        "",
        f"- Suggestions checked: {len(rows)}",
        f"- Confirmed by live file inspection (eligible for `file_validated` tier): {len(confirmed)}",
        f"- Errored or inconclusive: {len(errored)}",
        "",
        "## By validator",
        "",
    ]
    for validator, count in by_validator.most_common():
        lines.append(f"- {validator}: {count}")

    lines.extend(["", "## No validator available for this source (honest gap)", ""])
    no_validator = sorted({r["source"] for r in rows if r["validator"] == "none"})
    if no_validator:
        lines.extend(f"- {source}" for source in no_validator)
    else:
        lines.append("- none")

    lines.extend(["", "## Confirmed suggestions", ""])
    for r in confirmed:
        lines.append(
            f"- `{r['dataset_node_id']}` — {r['analysis_family']} confirmed via {r['validator']}"
        )

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_REPORT}")


if __name__ == "__main__":
    raise SystemExit(main())
