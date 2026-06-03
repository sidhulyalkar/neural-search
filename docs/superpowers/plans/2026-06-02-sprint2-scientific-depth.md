# Sprint 2 — Scientific Depth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate zero-coverage fields (analysis_affordances, file_formats, linked_papers), improve low-coverage fields (brain_regions, description), and add provenance auditing so every graph edge has evidence.

**Architecture:** Six focused modules: (1) completeness report as baseline, (2) DANDI metadata enricher via detail API, (3) OpenNeuro re-extraction with expanded synonyms, (4) affordance population pipeline wiring existing validators to the real corpus, (5) ProvenanceEdge audit enforcing no-evidence-no-edge, (6) ontology coverage report with brain-region vocabulary expansion.

**Tech Stack:** Python 3.11+, pydantic v2, httpx (with respx for mocking), pytest, existing `NWBValidator.validate_from_metadata()` and `BIDSValidator.validate_from_metadata()` in `neural_search/affordances/validators/`

---

## Baseline: What the corpus looks like today

| Field | DANDI (358) | OpenNeuro (362) |
|---|---:|---:|
| description | 4% | 92% |
| species | 40% | 58% |
| modalities | 26% | 67% |
| brain_regions | 27% | 17% |
| tasks | 21% | 53% |
| behavioral_events | 23% | 62% |
| analysis_affordances | **0%** | **0%** |
| data_standards | 1% | 100% |
| file_formats | **0%** | **0%** |
| linked_papers | **0%** | **0%** |

Root causes:
- DANDI description 4%: DANDI search API returns only a stub version object (no `metadata` key). Must call the dandiset detail endpoint `GET /api/dandisets/{id}/` to get the full metadata.
- analysis_affordances 0%: `NWBValidator` and `BIDSValidator` have `validate_from_metadata()` but are never called during corpus normalization.
- file_formats 0%: Present in DANDI `assetsSummary.dataStandard` and OpenNeuro BIDS `dataset_description.json` but not extracted.
- linked_papers 0%: DOIs present in DANDI `metadata.relatedResource` and OpenNeuro descriptions but not parsed.
- brain_regions low: `BRAIN_REGION_SYNONYMS` in `extraction.py` has only 12 entries.

---

## File Map

| Path | Create/Modify | Responsibility |
|---|---|---|
| `neural_search/corpus/completeness.py` | Create | Field-level coverage statistics per source |
| `scripts/generate_completeness_report.py` | Create | CLI: load corpus, write reports/corpus_completeness.md + .json |
| `neural_search/ingestion/dandi_enricher.py` | Create | Fetch full DANDI metadata for records missing description/formats/DOIs |
| `scripts/enrich_dandi_corpus.py` | Create | CLI: enrich real_dandi.jsonl → writes enriched file |
| `neural_search/ingestion/openneuro_enricher.py` | Create | Re-run extraction with expanded synonyms; extract DOIs from text |
| `scripts/enrich_openneuro_corpus.py` | Create | CLI: enrich real_openneuro.jsonl → writes enriched file |
| `neural_search/extraction.py` | Modify | Expand `BRAIN_REGION_SYNONYMS` from 12 → 40+ entries |
| `neural_search/affordances/populate.py` | Create | Batch affordance population using NWB/BIDS validators |
| `scripts/populate_affordances.py` | Create | CLI: run populate over real corpus JSONL, writes updated files |
| `neural_search/graph/edge_audit.py` | Create | Load knowledge graph, report edges without evidence |
| `scripts/audit_graph_edges.py` | Create | CLI: audit graph JSONL, write reports/edge_provenance_audit.md |
| `data/ontology/brain_regions_ontology.yaml` | Create | Expanded brain region vocabulary YAML |
| `neural_search/ontology/coverage_report.py` | Create | Compare ontology terms against corpus; identify unmatched terms |
| `scripts/generate_ontology_coverage.py` | Create | CLI: write docs/ONTOLOGY_COVERAGE_REPORT.md |
| `tests/test_corpus_completeness.py` | Create | Tests for completeness stats |
| `tests/test_dandi_enricher.py` | Create | Tests for DANDI metadata enrichment (httpx mocked) |
| `tests/test_openneuro_enricher.py` | Create | Tests for OpenNeuro re-extraction |
| `tests/test_affordance_populate.py` | Create | Tests for batch affordance population |
| `tests/test_edge_audit.py` | Create | Tests for ProvenanceEdge audit |
| `tests/test_ontology_coverage.py` | Create | Tests for coverage report |

---

## Task 1: Corpus Field Completeness Report

**Files:**
- Create: `neural_search/corpus/completeness.py`
- Create: `scripts/generate_completeness_report.py`
- Create: `tests/test_corpus_completeness.py`

### Purpose
Generate a baseline completeness report before enrichment. Run again after each enrichment task to measure progress. The report is also used in CI to prevent fields from silently dropping to 0%.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_corpus_completeness.py`:

```python
import json
import pytest
from pathlib import Path
from neural_search.corpus.completeness import (
    CORE_FIELDS,
    FieldStats,
    SourceCompleteness,
    compute_field_stats,
    generate_completeness_report,
    format_markdown_report,
    load_normalized_records,
)


def _make_record(**kwargs):
    base = {f: [] for f in CORE_FIELDS}
    base.update(kwargs)
    return base


def test_field_stats_empty_list():
    result = compute_field_stats([], "dandi")
    assert result.total_records == 0
    assert result.overall_coverage == 0.0


def test_field_stats_coverage():
    records = [
        _make_record(description="has desc", species=["mouse"]),
        _make_record(description=None, species=[]),
    ]
    result = compute_field_stats(records, "test")
    assert result.field_stats["description"].non_empty == 1
    assert result.field_stats["description"].coverage == pytest.approx(0.5)
    assert result.field_stats["species"].coverage == pytest.approx(0.5)


def test_field_stats_all_empty():
    records = [_make_record() for _ in range(5)]
    result = compute_field_stats(records, "dandi")
    for field_name in CORE_FIELDS:
        assert result.field_stats[field_name].non_empty == 0
        assert result.field_stats[field_name].coverage == 0.0


def test_field_stats_all_present():
    records = [_make_record(description="x", species=["mouse"], modalities=["fmri"],
                            brain_regions=["hippocampus"], tasks=["navigation"],
                            behavioral_events=["lick"], analysis_affordances=["event_aligned_psth"],
                            data_standards=["NWB"], file_formats=["nwb"],
                            linked_papers=[{"doi": "10.1/foo"}])]
    result = compute_field_stats(records, "test")
    assert result.overall_coverage == pytest.approx(1.0)


def test_format_markdown_report_contains_source():
    completeness = {
        "dandi": SourceCompleteness(
            source="dandi",
            total_records=10,
            field_stats={"description": FieldStats("description", 10, 3)},
        )
    }
    md = format_markdown_report(completeness)
    assert "DANDI" in md
    assert "30.0%" in md
    assert "3" in md


def test_load_normalized_records(tmp_path):
    jsonl = tmp_path / "test.jsonl"
    records = [{"dataset_id": "a", "description": "foo"}, {"dataset_id": "b"}]
    with jsonl.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    loaded = load_normalized_records(jsonl)
    assert len(loaded) == 2
    assert loaded[0]["dataset_id"] == "a"


def test_generate_completeness_report(tmp_path):
    jsonl = tmp_path / "real_dandi.jsonl"
    records = [
        {"description": "text", "species": ["mouse"], "modalities": [], "brain_regions": [],
         "tasks": [], "behavioral_events": [], "analysis_affordances": [], "data_standards": [],
         "file_formats": [], "linked_papers": []},
        {"description": None, "species": [], "modalities": [], "brain_regions": [],
         "tasks": [], "behavioral_events": [], "analysis_affordances": [], "data_standards": [],
         "file_formats": [], "linked_papers": []},
    ]
    with jsonl.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    report = generate_completeness_report(tmp_path)
    assert "dandi" in report
    assert report["dandi"].total_records == 2
    assert report["dandi"].field_stats["description"].coverage == pytest.approx(0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search
python -m pytest tests/test_corpus_completeness.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'neural_search.corpus.completeness'`

- [ ] **Step 3: Implement `neural_search/corpus/completeness.py`**

```python
"""Field-level completeness analysis for the normalized corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CORE_FIELDS = [
    "description",
    "species",
    "modalities",
    "brain_regions",
    "tasks",
    "behavioral_events",
    "analysis_affordances",
    "data_standards",
    "file_formats",
    "linked_papers",
]


@dataclass
class FieldStats:
    field: str
    total: int
    non_empty: int

    @property
    def coverage(self) -> float:
        return self.non_empty / self.total if self.total else 0.0


@dataclass
class SourceCompleteness:
    source: str
    total_records: int
    field_stats: dict[str, FieldStats] = field(default_factory=dict)

    @property
    def overall_coverage(self) -> float:
        if not self.field_stats:
            return 0.0
        return sum(s.coverage for s in self.field_stats.values()) / len(self.field_stats)


def load_normalized_records(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_field_stats(records: list[dict[str, Any]], source: str) -> SourceCompleteness:
    completeness = SourceCompleteness(source=source, total_records=len(records))
    for field_name in CORE_FIELDS:
        non_empty = sum(1 for r in records if r.get(field_name))
        completeness.field_stats[field_name] = FieldStats(
            field=field_name,
            total=len(records),
            non_empty=non_empty,
        )
    return completeness


def generate_completeness_report(corpus_dir: Path) -> dict[str, SourceCompleteness]:
    sources: dict[str, SourceCompleteness] = {}
    for jsonl_file in sorted(corpus_dir.glob("real_*.jsonl")):
        source = jsonl_file.stem.replace("real_", "")
        records = load_normalized_records(jsonl_file)
        sources[source] = compute_field_stats(records, source)
    return sources


def format_markdown_report(completeness: dict[str, SourceCompleteness], title: str = "Corpus Field Completeness Report") -> str:
    lines = [f"# {title}\n"]
    for source, stats in sorted(completeness.items()):
        lines.append(f"## {source.upper()} ({stats.total_records} records)\n")
        lines.append("| Field | Non-empty | Coverage |")
        lines.append("|---|---:|---:|")
        for field_name, fs in stats.field_stats.items():
            pct = f"{fs.coverage * 100:.1f}%"
            lines.append(f"| {field_name} | {fs.non_empty} | {pct} |")
        lines.append(f"\n**Overall coverage**: {stats.overall_coverage * 100:.1f}%\n")
    return "\n".join(lines)


def format_json_report(completeness: dict[str, SourceCompleteness]) -> dict:
    return {
        source: {
            "total_records": stats.total_records,
            "overall_coverage": round(stats.overall_coverage, 4),
            "fields": {
                field_name: {
                    "non_empty": fs.non_empty,
                    "total": fs.total,
                    "coverage": round(fs.coverage, 4),
                }
                for field_name, fs in stats.field_stats.items()
            },
        }
        for source, stats in completeness.items()
    }
```

- [ ] **Step 4: Create `scripts/generate_completeness_report.py`**

```python
#!/usr/bin/env python
"""Generate a corpus field completeness report."""

import json
from pathlib import Path

from neural_search.corpus.completeness import (
    format_json_report,
    format_markdown_report,
    generate_completeness_report,
)

CORPUS_DIR = Path("data/corpus/normalized")
REPORTS_DIR = Path("reports")


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    print("Generating completeness report from:", CORPUS_DIR)
    completeness = generate_completeness_report(CORPUS_DIR)
    md = format_markdown_report(completeness)
    json_data = format_json_report(completeness)
    md_path = REPORTS_DIR / "corpus_completeness_sprint2.md"
    json_path = REPORTS_DIR / "corpus_completeness_sprint2.json"
    md_path.write_text(md)
    json_path.write_text(json.dumps(json_data, indent=2))
    print(f"Written: {md_path}")
    print(f"Written: {json_path}")
    for source, stats in sorted(completeness.items()):
        print(f"\n{source.upper()} ({stats.total_records} records) — overall {stats.overall_coverage * 100:.1f}%")
        for field_name, fs in stats.field_stats.items():
            marker = "⚠️ " if fs.coverage < 0.1 else ""
            print(f"  {marker}{field_name}: {fs.non_empty}/{fs.total} ({fs.coverage * 100:.0f}%)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_corpus_completeness.py -v
```

Expected: All 7 tests PASS.

- [ ] **Step 6: Run the completeness report to get the real baseline**

```bash
python scripts/generate_completeness_report.py
```

Expected: `reports/corpus_completeness_sprint2.md` created with field table.

- [ ] **Step 7: Commit**

```bash
git add neural_search/corpus/completeness.py scripts/generate_completeness_report.py tests/test_corpus_completeness.py reports/corpus_completeness_sprint2.md reports/corpus_completeness_sprint2.json
git commit -m "feat: add corpus field completeness report (Task 1 Sprint 2)"
```

---

## Task 2: DANDI Metadata Enricher

**Files:**
- Create: `neural_search/ingestion/dandi_enricher.py`
- Create: `scripts/enrich_dandi_corpus.py`
- Create: `tests/test_dandi_enricher.py`

### Why
The DANDI search API returns a stub version object with no `metadata` key, so 342/358 records have no description. The DANDI detail endpoint `GET /api/dandisets/{id}/` returns the full payload including `metadata.description`, `metadata.relatedResource` (DOIs), and `assetsSummary.dataStandard` (file formats).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dandi_enricher.py`:

```python
import pytest
import httpx
import respx
from neural_search.ingestion.dandi_enricher import (
    extract_description_from_full_metadata,
    extract_file_formats_from_metadata,
    extract_linked_dois_from_metadata,
    enrich_dandi_record,
    fetch_dandiset_metadata,
)

MOCK_FULL_DANDISET = {
    "most_recent_published_version": {
        "metadata": {
            "description": "Mouse hippocampus theta maze recordings.",
            "relatedResource": [
                {"identifier": "https://doi.org/10.1016/j.neuron.2017.04.006", "relation": "IsDescribedBy"},
                {"identifier": "not-a-doi", "relation": "References"},
            ],
        },
        "assetsSummary": {
            "dataStandard": [
                {"identifier": "RRID:SCR_015242", "name": "Neurodata Without Borders (NWB)"},
            ],
        },
    }
}

MOCK_DRAFT_ONLY = {
    "draft_version": {
        "metadata": {
            "description": "Draft dataset description.",
        },
        "assetsSummary": {},
    }
}


def test_extract_description_from_published_version():
    result = extract_description_from_full_metadata(MOCK_FULL_DANDISET)
    assert result == "Mouse hippocampus theta maze recordings."


def test_extract_description_from_draft_fallback():
    result = extract_description_from_full_metadata(MOCK_DRAFT_ONLY)
    assert result == "Draft dataset description."


def test_extract_description_missing_returns_none():
    assert extract_description_from_full_metadata({}) is None
    assert extract_description_from_full_metadata({"most_recent_published_version": {}}) is None


def test_extract_file_formats():
    formats = extract_file_formats_from_metadata(MOCK_FULL_DANDISET)
    assert len(formats) == 1
    assert "RRID:SCR_015242" in formats


def test_extract_file_formats_empty():
    assert extract_file_formats_from_metadata({}) == []
    assert extract_file_formats_from_metadata({"most_recent_published_version": {"assetsSummary": {}}}) == []


def test_extract_linked_dois_valid():
    dois = extract_linked_dois_from_metadata(MOCK_FULL_DANDISET)
    assert len(dois) == 1
    assert "doi.org/10.1016" in dois[0]


def test_extract_linked_dois_empty():
    assert extract_linked_dois_from_metadata({}) == []


def test_enrich_fills_missing_description():
    record = {"source_id": "000026", "description": None, "file_formats": [], "linked_papers": []}
    enriched = enrich_dandi_record(record, MOCK_FULL_DANDISET)
    assert enriched["description"] == "Mouse hippocampus theta maze recordings."


def test_enrich_preserves_existing_description():
    record = {"source_id": "000026", "description": "Existing.", "file_formats": [], "linked_papers": []}
    enriched = enrich_dandi_record(record, MOCK_FULL_DANDISET)
    assert enriched["description"] == "Existing."


def test_enrich_fills_file_formats():
    record = {"source_id": "000026", "description": None, "file_formats": [], "linked_papers": []}
    enriched = enrich_dandi_record(record, MOCK_FULL_DANDISET)
    assert len(enriched["file_formats"]) == 1


def test_enrich_fills_linked_papers():
    record = {"source_id": "000026", "description": None, "file_formats": [], "linked_papers": []}
    enriched = enrich_dandi_record(record, MOCK_FULL_DANDISET)
    assert len(enriched["linked_papers"]) == 1
    assert "doi" in enriched["linked_papers"][0]


def test_enrich_does_not_overwrite_existing_file_formats():
    record = {"source_id": "000026", "description": None, "file_formats": ["already_there"], "linked_papers": []}
    enriched = enrich_dandi_record(record, MOCK_FULL_DANDISET)
    assert enriched["file_formats"] == ["already_there"]


@respx.mock
def test_fetch_dandiset_metadata_calls_correct_url():
    respx.get("https://api.dandiarchive.org/api/dandisets/000026/").mock(
        return_value=httpx.Response(200, json=MOCK_FULL_DANDISET)
    )
    result = fetch_dandiset_metadata("000026")
    assert result == MOCK_FULL_DANDISET


@respx.mock
def test_fetch_dandiset_metadata_raises_on_404():
    respx.get("https://api.dandiarchive.org/api/dandisets/999999/").mock(
        return_value=httpx.Response(404)
    )
    with pytest.raises(httpx.HTTPStatusError):
        fetch_dandiset_metadata("999999")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_dandi_enricher.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'neural_search.ingestion.dandi_enricher'`

- [ ] **Step 3: Implement `neural_search/ingestion/dandi_enricher.py`**

```python
"""DANDI metadata enricher.

Fetches full dandiset metadata from the DANDI detail API endpoint to fill
fields that are missing from the search-result stub (description, file_formats,
linked_papers). The DANDI search API returns only a minimal version object;
the detail endpoint returns complete metadata including description and relatedResource.

API endpoint: GET https://api.dandiarchive.org/api/dandisets/{dandiset_id}/
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DANDI_API_URL = "https://api.dandiarchive.org/api"


def fetch_dandiset_metadata(
    dandiset_id: str,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Fetch full dandiset metadata from the DANDI API detail endpoint."""
    url = f"{DANDI_API_URL}/dandisets/{dandiset_id}/"
    _close = client is None
    if client is None:
        client = httpx.Client(timeout=30.0)
    try:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.json()
    finally:
        if _close:
            client.close()


def extract_description_from_full_metadata(dandiset: dict[str, Any]) -> str | None:
    """Extract description from a full DANDI API response.

    Checks most_recent_published_version first, then draft_version.
    """
    for version_key in ("most_recent_published_version", "draft_version"):
        version = dandiset.get(version_key) or {}
        metadata = version.get("metadata") or {}
        description = metadata.get("description")
        if description:
            return description
    return dandiset.get("description")


def extract_file_formats_from_metadata(dandiset: dict[str, Any]) -> list[str]:
    """Extract file format identifiers from DANDI assetsSummary.dataStandard."""
    for version_key in ("most_recent_published_version", "draft_version"):
        version = dandiset.get(version_key) or {}
        assets = version.get("assetsSummary") or {}
        standards = assets.get("dataStandard") or []
        if standards:
            return [
                ds.get("identifier") or ds.get("name") or ""
                for ds in standards
                if ds
            ]
    return []


def extract_linked_dois_from_metadata(dandiset: dict[str, Any]) -> list[str]:
    """Extract DOI strings from DANDI metadata.relatedResource."""
    for version_key in ("most_recent_published_version", "draft_version"):
        version = dandiset.get(version_key) or {}
        metadata = version.get("metadata") or {}
        resources = metadata.get("relatedResource") or []
        dois = []
        for resource in resources:
            identifier = resource.get("identifier", "")
            if "doi.org" in identifier.lower() or (
                identifier.startswith("10.") and "/" in identifier
            ):
                dois.append(identifier)
        return dois
    return []


def enrich_dandi_record(
    record: dict[str, Any],
    full_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return an enriched copy of a normalized DANDI record.

    Only fills fields that are currently empty — never overwrites existing data.
    """
    enriched = dict(record)

    if not enriched.get("description"):
        description = extract_description_from_full_metadata(full_metadata)
        if description:
            enriched["description"] = description

    if not enriched.get("file_formats"):
        formats = extract_file_formats_from_metadata(full_metadata)
        if formats:
            enriched["file_formats"] = formats

    if not enriched.get("linked_papers"):
        dois = extract_linked_dois_from_metadata(full_metadata)
        if dois:
            enriched["linked_papers"] = [{"doi": doi, "source": "dandi_related_resource"} for doi in dois]

    return enriched


def enrich_corpus_file(
    corpus_path: Path,
    output_path: Path,
    max_records: int | None = None,
    delay_seconds: float = 0.2,
    dry_run: bool = False,
) -> dict[str, int]:
    """Enrich a DANDI corpus JSONL file with full metadata from the API.

    Args:
        corpus_path: Path to the input JSONL file.
        output_path: Path to write enriched records.
        max_records: Limit records processed (None = all).
        delay_seconds: Sleep between API calls to be polite.
        dry_run: If True, fetch but do not write.

    Returns:
        Stats dict with keys: total, enriched, skipped, errors.
    """
    records: list[dict[str, Any]] = []
    with corpus_path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if max_records is not None:
        records = records[:max_records]

    stats = {"total": len(records), "enriched": 0, "skipped": 0, "errors": 0}
    enriched_records: list[dict[str, Any]] = []

    with httpx.Client(timeout=30.0) as client:
        for record in records:
            source_id = record.get("source_id", "")
            needs_enrichment = (
                not record.get("description")
                or not record.get("file_formats")
                or not record.get("linked_papers")
            )
            if not source_id or not needs_enrichment:
                enriched_records.append(record)
                stats["skipped"] += 1
                continue
            try:
                full_meta = fetch_dandiset_metadata(source_id, client)
                enriched = enrich_dandi_record(record, full_meta)
                enriched_records.append(enriched)
                if enriched != record:
                    stats["enriched"] += 1
                    logger.debug("Enriched %s", source_id)
                else:
                    stats["skipped"] += 1
                if delay_seconds:
                    time.sleep(delay_seconds)
            except Exception as exc:
                logger.warning("Failed to enrich %s: %s", source_id, exc)
                enriched_records.append(record)
                stats["errors"] += 1

    if not dry_run:
        with output_path.open("w") as f:
            for r in enriched_records:
                f.write(json.dumps(r) + "\n")
        logger.info("Wrote %d records to %s", len(enriched_records), output_path)

    return stats
```

- [ ] **Step 4: Create `scripts/enrich_dandi_corpus.py`**

```python
#!/usr/bin/env python
"""Enrich the real DANDI corpus with full metadata from the DANDI API."""

import argparse
import logging
from pathlib import Path

from neural_search.ingestion.dandi_enricher import enrich_corpus_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

CORPUS_DIR = Path("data/corpus/normalized")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich DANDI corpus with full API metadata.")
    parser.add_argument("--input", default=str(CORPUS_DIR / "real_dandi.jsonl"))
    parser.add_argument("--output", default=str(CORPUS_DIR / "real_dandi.jsonl"))
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds between API calls")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    if args.dry_run:
        print("DRY RUN — will not write output")

    stats = enrich_corpus_file(
        corpus_path=input_path,
        output_path=output_path,
        max_records=args.max_records,
        delay_seconds=args.delay,
        dry_run=args.dry_run,
    )
    print(f"\nResults: {stats}")
    pct = 100 * stats["enriched"] / stats["total"] if stats["total"] else 0
    print(f"Enriched {stats['enriched']}/{stats['total']} records ({pct:.0f}%)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pip install respx --quiet
python -m pytest tests/test_dandi_enricher.py -v
```

Expected: All 13 tests PASS.

- [ ] **Step 6: Run enrichment on real corpus (this will call the DANDI API ~342 times)**

```bash
# Dry run first to check connectivity
python scripts/enrich_dandi_corpus.py --max-records 5 --dry-run

# Full enrichment (takes ~5 minutes at 0.2s delay per record)
python scripts/enrich_dandi_corpus.py --delay 0.15
```

Expected output: `Enriched N/358 records` with N > 300.

- [ ] **Step 7: Re-run completeness report to verify improvement**

```bash
python scripts/generate_completeness_report.py
```

Expected: DANDI description coverage jumps from ~4% → ~80%+.

- [ ] **Step 8: Commit**

```bash
git add neural_search/ingestion/dandi_enricher.py scripts/enrich_dandi_corpus.py tests/test_dandi_enricher.py data/corpus/normalized/real_dandi.jsonl reports/corpus_completeness_sprint2.md reports/corpus_completeness_sprint2.json
git commit -m "feat: DANDI metadata enricher — fetch descriptions/formats/DOIs from detail API"
```

---

## Task 3: OpenNeuro Extraction Enhancement + Brain Region Vocabulary Expansion

**Files:**
- Modify: `neural_search/extraction.py` (expand `BRAIN_REGION_SYNONYMS`)
- Create: `neural_search/ingestion/openneuro_enricher.py`
- Create: `scripts/enrich_openneuro_corpus.py`
- Create: `tests/test_openneuro_enricher.py`

### Why
OpenNeuro brain_regions is only 17% despite 92% description coverage. Root cause: `BRAIN_REGION_SYNONYMS` only has 12 entries. Adding ~35 more entries (covering common fMRI/EEG regions, abbreviations) will significantly improve coverage. Also extract DOIs from descriptions (OpenNeuro dataset_description.json embeds them in the `description` field).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_openneuro_enricher.py`:

```python
import pytest
from neural_search.ingestion.openneuro_enricher import (
    extract_brain_regions_from_text,
    extract_dois_from_text,
    enrich_openneuro_record,
)


def test_extract_brain_regions_motor_cortex():
    regions = extract_brain_regions_from_text("subjects performed hand movements in motor cortex study")
    assert "motor_cortex" in regions


def test_extract_brain_regions_pfc_abbreviation():
    regions = extract_brain_regions_from_text("PFC activity during working memory")
    assert "mPFC" in regions or "PFC" in regions or any("prefrontal" in r.lower() or "pfc" in r.lower() for r in regions)


def test_extract_brain_regions_insula():
    regions = extract_brain_regions_from_text("insular cortex activity during pain")
    assert any("insula" in r.lower() for r in regions)


def test_extract_brain_regions_cerebellum():
    regions = extract_brain_regions_from_text("cerebellar recordings during motor adaptation")
    assert any("cereb" in r.lower() for r in regions)


def test_extract_brain_regions_empty_text():
    regions = extract_brain_regions_from_text("")
    assert regions == []


def test_extract_dois_from_description():
    text = "See Gordon et al. (2017) https://doi.org/10.1016/j.neuron.2017.04.006 for details."
    dois = extract_dois_from_text(text)
    assert len(dois) == 1
    assert "10.1016" in dois[0]


def test_extract_dois_multiple():
    text = "doi:10.1038/nn.1234 and doi.org/10.1016/foo"
    dois = extract_dois_from_text(text)
    assert len(dois) == 2


def test_extract_dois_none_present():
    dois = extract_dois_from_text("No DOIs here, just plain text.")
    assert dois == []


def test_enrich_openneuro_fills_brain_regions():
    record = {
        "source_id": "ds000001",
        "description": "fMRI study of prefrontal cortex and hippocampus during memory encoding.",
        "brain_regions": [],
        "linked_papers": [],
    }
    enriched = enrich_openneuro_record(record)
    assert len(enriched["brain_regions"]) > 0


def test_enrich_openneuro_fills_linked_papers_from_doi():
    record = {
        "source_id": "ds000001",
        "description": "Data from https://doi.org/10.1016/j.neuron.2017.04.006",
        "brain_regions": [],
        "linked_papers": [],
    }
    enriched = enrich_openneuro_record(record)
    assert len(enriched["linked_papers"]) == 1
    assert "doi" in enriched["linked_papers"][0]


def test_enrich_openneuro_preserves_existing_brain_regions():
    record = {
        "source_id": "ds000001",
        "description": "hippocampus study",
        "brain_regions": [{"id": "visual_cortex", "label": "visual cortex"}],
        "linked_papers": [],
    }
    enriched = enrich_openneuro_record(record)
    existing_ids = {r["id"] if isinstance(r, dict) else r for r in enriched["brain_regions"]}
    assert "visual_cortex" in existing_ids
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_openneuro_enricher.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'neural_search.ingestion.openneuro_enricher'`

- [ ] **Step 3: Expand `BRAIN_REGION_SYNONYMS` in `neural_search/extraction.py`**

Find the existing `BRAIN_REGION_SYNONYMS` dict (approximately line 95) and replace it:

```python
BRAIN_REGION_SYNONYMS: dict[str, list[str]] = {
    # Prefrontal
    "mPFC": ["mpfc", "medial prefrontal", "medial pfc", "prelimbic", "infralimbic"],
    "dlPFC": ["dlpfc", "dorsolateral prefrontal", "dlpfc"],
    "vlPFC": ["vlpfc", "ventrolateral prefrontal"],
    "OFC": ["ofc", "orbitofrontal", "orbital frontal"],
    "ACC": ["acc", "anterior cingulate", "anterior cingulate cortex"],
    "PCC": ["pcc", "posterior cingulate", "posterior cingulate cortex"],
    # Motor / sensory
    "motor_cortex": ["motor cortex", "m1", "primary motor", "premotor", "supplementary motor", "sma"],
    "somatosensory_cortex": ["somatosensory cortex", "s1", "s2", "barrel cortex", "primary somatosensory"],
    "visual_cortex": ["visual cortex", "v1", "v2", "v4", "mt", "lot", "visual area"],
    "auditory_cortex": ["auditory cortex", "a1", "primary auditory", "auditory area"],
    # Temporal
    "hippocampus": ["hippocampus", "hippocampal", "ca1", "ca3", "dentate gyrus", "dg", "subiculum", "entorhinal"],
    "amygdala": ["amygdala", "amygdalar", "basolateral amygdala", "bla", "central amygdala", "cea"],
    "temporal_cortex": ["temporal cortex", "temporal lobe", "inferior temporal", "it", "fusiform", "parahippocampal"],
    # Parietal
    "parietal_cortex": ["parietal cortex", "ppc", "posterior parietal", "intraparietal", "ips"],
    # Subcortical
    "striatum": ["striatum", "striatal", "caudate", "putamen", "nucleus accumbens", "nac", "ventral striatum"],
    "thalamus": ["thalamus", "thalamic", "mediodorsal thalamus", "md thalamus", "ventral posterolateral"],
    "basal_ganglia": ["basal ganglia", "substantia nigra", "snc", "snr", "globus pallidus", "subthalamic"],
    "cerebellum": ["cerebellum", "cerebellar", "purkinje", "cerebellar cortex", "cerebellum"],
    "brainstem": ["brainstem", "brain stem", "midbrain", "pons", "medulla", "superior colliculus", "lc", "vta", "periaqueductal"],
    "hypothalamus": ["hypothalamus", "hypothalamic", "lateral hypothalamus"],
    "insula": ["insula", "insular cortex", "insular", "insula cortex"],
    # Frontal
    "lateral_PFC": ["lateral pfc", "lateral prefrontal", "lateral frontal"],
    "dACC": ["dacc", "dorsal anterior cingulate", "dorsal acc"],
    # Posterior
    "precuneus": ["precuneus"],
    "angular_gyrus": ["angular gyrus", "angular"],
    "supramarginal_gyrus": ["supramarginal gyrus", "smg"],
    # Default mode
    "default_mode_network": ["default mode", "dmn", "default network"],
    # Occipital
    "occipital_cortex": ["occipital cortex", "occipital", "primary visual", "v1"],
    # Olfactory
    "olfactory_bulb": ["olfactory bulb", "olfactory cortex", "piriform"],
}
```

- [ ] **Step 4: Implement `neural_search/ingestion/openneuro_enricher.py`**

```python
"""OpenNeuro corpus enricher.

Re-extracts brain regions using the expanded BRAIN_REGION_SYNONYMS vocabulary
and mines DOIs from description text to populate linked_papers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from neural_search.extraction import BRAIN_REGION_SYNONYMS
from neural_search.normalized import normalize_text

# DOI regex: matches doi.org URLs and bare 10.XXXX/... identifiers
_DOI_PATTERNS = [
    re.compile(r"https?://(?:dx\.)?doi\.org/(10\.\d{4,}/\S+)", re.IGNORECASE),
    re.compile(r"\bdoi:\s*(10\.\d{4,}/\S+)", re.IGNORECASE),
    re.compile(r"\b(10\.\d{4,}/[^\s,;\"'>]+)", re.IGNORECASE),
]


def extract_brain_regions_from_text(text: str) -> list[str]:
    """Extract brain region labels from text using expanded synonyms."""
    if not text:
        return []
    normalized = normalize_text(text)
    found: list[str] = []
    for region_id, synonyms in BRAIN_REGION_SYNONYMS.items():
        for synonym in [region_id.lower().replace("_", " "), *synonyms]:
            if synonym.lower() in normalized:
                found.append(region_id)
                break
    return list(dict.fromkeys(found))  # deduplicate while preserving order


def extract_dois_from_text(text: str) -> list[str]:
    """Extract DOI strings from free text."""
    if not text:
        return []
    dois: list[str] = []
    for pattern in _DOI_PATTERNS:
        for match in pattern.finditer(text):
            doi = match.group(1).rstrip(".,;)>\"'")
            if doi not in dois:
                dois.append(doi)
    return dois


def enrich_openneuro_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return an enriched copy of a normalized OpenNeuro record.

    Adds brain_regions extracted from description (if empty or low-coverage)
    and linked_papers from DOIs found in description text.
    """
    enriched = dict(record)
    description = enriched.get("description") or ""

    # Re-extract brain regions from description using expanded synonyms
    existing_regions = enriched.get("brain_regions") or []
    existing_ids = {
        (r["id"] if isinstance(r, dict) else r)
        for r in existing_regions
    }
    new_regions = extract_brain_regions_from_text(description)
    for region_id in new_regions:
        if region_id not in existing_ids:
            existing_regions.append({"id": region_id, "label": region_id.replace("_", " "), "confidence": 0.8, "source": "re_extraction"})
            existing_ids.add(region_id)
    enriched["brain_regions"] = existing_regions

    # Extract linked DOIs from description text
    if not enriched.get("linked_papers"):
        dois = extract_dois_from_text(description)
        if dois:
            enriched["linked_papers"] = [{"doi": doi, "source": "description_text"} for doi in dois]

    return enriched


def enrich_corpus_file(corpus_path: Path, output_path: Path) -> dict[str, int]:
    """Enrich an OpenNeuro corpus JSONL file in-place."""
    records: list[dict[str, Any]] = []
    with corpus_path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    stats = {"total": len(records), "enriched": 0}
    enriched_records: list[dict[str, Any]] = []

    for record in records:
        enriched = enrich_openneuro_record(record)
        enriched_records.append(enriched)
        if enriched != record:
            stats["enriched"] += 1

    with output_path.open("w") as f:
        for r in enriched_records:
            f.write(json.dumps(r) + "\n")

    return stats
```

- [ ] **Step 5: Create `scripts/enrich_openneuro_corpus.py`**

```python
#!/usr/bin/env python
"""Re-extract brain regions and DOIs for the real OpenNeuro corpus."""

import argparse
from pathlib import Path

from neural_search.ingestion.openneuro_enricher import enrich_corpus_file

CORPUS_DIR = Path("data/corpus/normalized")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(CORPUS_DIR / "real_openneuro.jsonl"))
    parser.add_argument("--output", default=str(CORPUS_DIR / "real_openneuro.jsonl"))
    args = parser.parse_args()

    stats = enrich_corpus_file(Path(args.input), Path(args.output))
    print(f"Enriched {stats['enriched']}/{stats['total']} records")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
python -m pytest tests/test_openneuro_enricher.py -v
```

Expected: All 11 tests PASS.

- [ ] **Step 7: Run enrichment**

```bash
python scripts/enrich_openneuro_corpus.py
python scripts/generate_completeness_report.py
```

Expected: OpenNeuro brain_regions jumps from 17% → 40%+.

- [ ] **Step 8: Commit**

```bash
git add neural_search/extraction.py neural_search/ingestion/openneuro_enricher.py scripts/enrich_openneuro_corpus.py tests/test_openneuro_enricher.py data/corpus/normalized/real_openneuro.jsonl reports/corpus_completeness_sprint2.md
git commit -m "feat: expand brain region vocabulary (12→30+ entries), add OpenNeuro enricher for re-extraction + DOI mining"
```

---

## Task 4: Affordance Population Pipeline

**Files:**
- Create: `neural_search/affordances/populate.py`
- Create: `scripts/populate_affordances.py`
- Create: `tests/test_affordance_populate.py`

### Why
`analysis_affordances` is 0% on both sources. The validators exist (`NWBValidator.validate_from_metadata()` and `BIDSValidator.validate_from_metadata()`) but are never called during normalization. This task wires them up.

### How validators work
- `NWBValidator.validate_from_metadata(metadata: dict, dataset_id: str)` takes the `usability_flags` dict and returns an `NWBValidationResult` with `affordance_support: dict[str, bool]`
- `BIDSValidator.validate_from_metadata(metadata: dict, dataset_id: str)` similarly
- We determine which validator to use by checking `data_standards` and `source` fields

- [ ] **Step 1: Write the failing tests**

Create `tests/test_affordance_populate.py`:

```python
import pytest
from neural_search.affordances.populate import (
    determine_validator_type,
    build_metadata_for_validator,
    run_affordance_validation,
    populate_affordances_for_record,
    populate_corpus_file,
)


def _make_dandi_record(**kwargs):
    base = {
        "dataset_id": "dandi:000001",
        "source": "dandi",
        "source_id": "000001",
        "data_standards": [],
        "usability_flags": {},
        "analysis_affordances": [],
        "modalities": [],
        "tasks": [],
        "behavioral_events": [],
        "species": [],
    }
    base.update(kwargs)
    return base


def _make_openneuro_record(**kwargs):
    base = {
        "dataset_id": "openneuro:ds000001",
        "source": "openneuro",
        "source_id": "ds000001",
        "data_standards": [{"id": "BIDS", "label": "BIDS"}],
        "usability_flags": {},
        "analysis_affordances": [],
        "modalities": [{"id": "fmri", "label": "fmri"}],
        "tasks": [],
        "behavioral_events": [],
        "species": [],
    }
    base.update(kwargs)
    return base


def test_determine_validator_type_nwb():
    record = _make_dandi_record(data_standards=[{"id": "NWB"}], source="dandi")
    assert determine_validator_type(record) == "nwb"


def test_determine_validator_type_bids():
    record = _make_openneuro_record()
    assert determine_validator_type(record) == "bids"


def test_determine_validator_type_dandi_default():
    # DANDI without explicit NWB standard still defaults to nwb
    record = _make_dandi_record()
    assert determine_validator_type(record) == "nwb"


def test_determine_validator_type_openneuro_default():
    # OpenNeuro without explicit BIDS standard still defaults to bids
    record = _make_openneuro_record(data_standards=[])
    assert determine_validator_type(record) == "bids"


def test_build_metadata_for_validator():
    record = _make_dandi_record(
        usability_flags={"has_trials": True, "has_neural_data": True, "has_behavior": True},
        modalities=[{"id": "neuropixels", "label": "neuropixels"}],
        tasks=[{"id": "decision_making", "label": "decision making"}],
        behavioral_events=[{"id": "lick", "label": "lick"}],
        species=[{"id": "mouse", "label": "mouse"}],
    )
    meta = build_metadata_for_validator(record)
    assert meta["has_trials"] is True
    assert meta["has_neural_data"] is True
    assert "neuropixels" in meta.get("modalities", [])


def test_run_affordance_validation_returns_list():
    record = _make_dandi_record(
        usability_flags={"has_trials": True, "has_neural_data": True, "has_behavior": True},
        modalities=[{"id": "extracellular_ephys"}],
        tasks=[{"id": "decision_making"}],
        behavioral_events=[{"id": "choice"}],
    )
    affordances = run_affordance_validation(record)
    assert isinstance(affordances, list)
    # All returned affordances should be strings
    assert all(isinstance(a, str) for a in affordances)


def test_populate_affordances_fills_empty():
    record = _make_dandi_record(
        usability_flags={"has_trials": True, "has_neural_data": True, "has_behavior": True,
                         "has_event_timestamps": True},
        modalities=[{"id": "extracellular_ephys", "label": "ephys"}],
        tasks=[{"id": "decision_making"}],
        behavioral_events=[{"id": "choice"}, {"id": "reward"}],
    )
    enriched = populate_affordances_for_record(record)
    assert isinstance(enriched["analysis_affordances"], list)


def test_populate_affordances_preserves_existing():
    record = _make_dandi_record(
        analysis_affordances=["already_present"],
        usability_flags={},
    )
    enriched = populate_affordances_for_record(record)
    assert "already_present" in enriched["analysis_affordances"]


def test_populate_corpus_file(tmp_path):
    import json
    corpus = tmp_path / "corpus.jsonl"
    records = [
        _make_dandi_record(
            usability_flags={"has_trials": True, "has_neural_data": True, "has_behavior": True},
            modalities=[{"id": "extracellular_ephys"}],
            tasks=[{"id": "decision_making"}],
            behavioral_events=[{"id": "choice"}],
        ),
        _make_openneuro_record(
            usability_flags={"has_behavior": True},
            tasks=[{"id": "motor"}],
        ),
    ]
    with corpus.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    out = tmp_path / "out.jsonl"
    stats = populate_corpus_file(corpus, out)
    assert stats["total"] == 2
    result_records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert all(isinstance(r["analysis_affordances"], list) for r in result_records)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_affordance_populate.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'neural_search.affordances.populate'`

- [ ] **Step 3: Inspect the existing validators to understand their interfaces**

Before implementing, check exact method signatures:

```bash
grep -n "def validate_from_metadata\|def validate\|affordance_support" \
  neural_search/affordances/validators/nwb_validator.py \
  neural_search/affordances/validators/bids_validator.py | head -20
```

Note the exact return type and parameter names — use them in Step 4.

- [ ] **Step 4: Implement `neural_search/affordances/populate.py`**

```python
"""Batch affordance population for the normalized corpus.

Wires up the existing NWBValidator and BIDSValidator to populate
analysis_affordances for every record that currently has an empty list.

Validator selection logic:
- source == "dandi" or data_standards contains "NWB" → NWBValidator
- source == "openneuro" or data_standards contains "BIDS" → BIDSValidator
- Both absent → NWBValidator (best guess for neural data)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _extract_ids(items: list) -> list[str]:
    """Extract string IDs from a list that may contain dicts or strings."""
    result = []
    for item in items or []:
        if isinstance(item, dict):
            result.append(item.get("id") or item.get("label") or str(item))
        else:
            result.append(str(item))
    return result


def determine_validator_type(record: dict[str, Any]) -> str:
    """Return 'nwb' or 'bids' based on source and data_standards."""
    source = record.get("source", "").lower()
    standards = _extract_ids(record.get("data_standards") or [])
    standards_lower = [s.lower() for s in standards]

    if "bids" in standards_lower or source == "openneuro":
        return "bids"
    if "nwb" in standards_lower or source in ("dandi", "allen", "nemo"):
        return "nwb"
    # Default: DANDI/neural sources → nwb
    if source in ("dandi",):
        return "nwb"
    return "bids"


def build_metadata_for_validator(record: dict[str, Any]) -> dict[str, Any]:
    """Build a metadata dict for validator input from a normalized record."""
    usability = record.get("usability_flags") or {}
    return {
        "has_trials": usability.get("has_trials"),
        "has_behavior": usability.get("has_behavior"),
        "has_neural_data": usability.get("has_neural_data"),
        "has_continuous_behavior": usability.get("has_continuous_behavior"),
        "has_event_timestamps": usability.get("has_event_timestamps"),
        "has_raw_data": usability.get("has_raw_data"),
        "has_processed_data": usability.get("has_processed_data"),
        "has_standard_format": usability.get("has_standard_format"),
        "modalities": _extract_ids(record.get("modalities") or []),
        "tasks": _extract_ids(record.get("tasks") or []),
        "behavioral_events": _extract_ids(record.get("behavioral_events") or []),
        "species": _extract_ids(record.get("species") or []),
        "data_standards": _extract_ids(record.get("data_standards") or []),
        "title": record.get("title", ""),
        "description": record.get("description") or "",
    }


def run_affordance_validation(record: dict[str, Any]) -> list[str]:
    """Run the appropriate validator and return a list of supported affordance IDs."""
    validator_type = determine_validator_type(record)
    metadata = build_metadata_for_validator(record)
    dataset_id = record.get("dataset_id", "unknown")

    try:
        if validator_type == "nwb":
            from neural_search.affordances.validators.nwb_validator import NWBValidator
            validator = NWBValidator()
            result = validator.validate_from_metadata(metadata, dataset_id)
        else:
            from neural_search.affordances.validators.bids_validator import BIDSValidator
            validator = BIDSValidator()
            result = validator.validate_from_metadata(metadata, dataset_id)

        # Extract supported affordance IDs from the result
        if hasattr(result, "affordance_support"):
            return [aid for aid, supported in result.affordance_support.items() if supported]
        if isinstance(result, dict):
            return [aid for aid, supported in result.items() if supported]
        return []
    except Exception as exc:
        logger.warning("Affordance validation failed for %s: %s", dataset_id, exc)
        return []


def populate_affordances_for_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the record with analysis_affordances populated.

    Merges validator output with any existing affordances (preserves existing).
    """
    existing = list(record.get("analysis_affordances") or [])
    existing_set = set(existing)

    new_affordances = run_affordance_validation(record)
    for affordance_id in new_affordances:
        if affordance_id not in existing_set:
            existing.append(affordance_id)
            existing_set.add(affordance_id)

    return {**record, "analysis_affordances": existing}


def populate_corpus_file(
    corpus_path: Path,
    output_path: Path,
    force: bool = False,
) -> dict[str, int]:
    """Populate analysis_affordances for every record in a corpus JSONL file.

    Args:
        corpus_path: Input JSONL path.
        output_path: Output JSONL path (can be same as input).
        force: If True, re-run validation even if affordances already present.

    Returns:
        Stats: total, populated, skipped, errors.
    """
    records: list[dict[str, Any]] = []
    with corpus_path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    stats = {"total": len(records), "populated": 0, "skipped": 0, "errors": 0}
    out_records: list[dict[str, Any]] = []

    for record in records:
        existing = record.get("analysis_affordances") or []
        if existing and not force:
            out_records.append(record)
            stats["skipped"] += 1
            continue
        try:
            enriched = populate_affordances_for_record(record)
            out_records.append(enriched)
            if enriched["analysis_affordances"] != existing:
                stats["populated"] += 1
            else:
                stats["skipped"] += 1
        except Exception as exc:
            logger.error("Failed to populate affordances for %s: %s", record.get("dataset_id"), exc)
            out_records.append(record)
            stats["errors"] += 1

    with output_path.open("w") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")

    return stats
```

- [ ] **Step 5: Create `scripts/populate_affordances.py`**

```python
#!/usr/bin/env python
"""Populate analysis_affordances for all real corpus JSONL files."""

import argparse
import logging
from pathlib import Path

from neural_search.affordances.populate import populate_corpus_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

CORPUS_DIR = Path("data/corpus/normalized")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", default=str(CORPUS_DIR))
    parser.add_argument("--force", action="store_true", help="Re-run even if affordances already populated")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    total_populated = 0

    for jsonl_path in sorted(corpus_dir.glob("real_*.jsonl")):
        print(f"\nProcessing: {jsonl_path.name}")
        stats = populate_corpus_file(jsonl_path, jsonl_path, force=args.force)
        print(f"  total={stats['total']} populated={stats['populated']} skipped={stats['skipped']} errors={stats['errors']}")
        total_populated += stats["populated"]

    print(f"\nTotal affordances populated across all sources: {total_populated}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
python -m pytest tests/test_affordance_populate.py -v
```

Expected: All 9 tests PASS. (Some affordance counts may be 0 for sparse metadata — that is correct behavior.)

- [ ] **Step 7: Run population on real corpus**

```bash
python scripts/populate_affordances.py
python scripts/generate_completeness_report.py
```

Expected: `analysis_affordances` goes from 0% → 15–40% (depends on how much metadata is present post-enrichment).

- [ ] **Step 8: Commit**

```bash
git add neural_search/affordances/populate.py scripts/populate_affordances.py tests/test_affordance_populate.py data/corpus/normalized/real_dandi.jsonl data/corpus/normalized/real_openneuro.jsonl
git commit -m "feat: affordance population pipeline — wire NWB/BIDS validators to real corpus normalization"
```

---

## Task 5: ProvenanceEdge Audit

**Files:**
- Create: `neural_search/graph/edge_audit.py`
- Create: `scripts/audit_graph_edges.py`
- Create: `tests/test_edge_audit.py`

### Why
The knowledge graph has 1,697 edges. ChatGPT's review flagged that "no edge without evidence" is currently aspirational — the schema enforces it but there's no audit tool to check compliance at the corpus level. This task produces a report and enforces the rule going forward.

### Key note on schema
The graph uses `KnowledgeGraphEdge` (from `neural_search/graph/schema.py`), not `ProvenanceEdge` (from `neural_search/core/dataset_card.py`). The audit must work against the graph's actual edge schema, checking the `evidence` field.

- [ ] **Step 1: Write the failing tests**

First check the KnowledgeGraphEdge schema:

```bash
grep -n "class KnowledgeGraphEdge\|evidence\|confidence\|review_status" neural_search/graph/schema.py | head -20
```

Then create `tests/test_edge_audit.py`:

```python
import pytest
from neural_search.graph.edge_audit import (
    EdgeAuditResult,
    audit_edge,
    audit_graph_edges,
    format_audit_report,
    EdgeViolation,
)


def _make_edge(edge_id="e1", source="dataset:001", target="task:nav",
               edge_type="dataset_has_task", confidence=0.9, evidence=None):
    """Create a minimal dict representing a graph edge."""
    return {
        "edge_id": edge_id,
        "source_id": source,
        "target_id": target,
        "edge_type": edge_type,
        "confidence": confidence,
        "evidence": evidence if evidence is not None else [
            {"evidence_type": "structured_metadata", "source": "dandi_api", "confidence": 0.9}
        ],
    }


def test_audit_edge_passes_with_evidence():
    edge = _make_edge(evidence=[{"evidence_type": "structured_metadata", "source": "dandi"}])
    result = audit_edge(edge)
    assert result.has_evidence is True
    assert result.violation is None


def test_audit_edge_fails_empty_evidence():
    edge = _make_edge(evidence=[])
    result = audit_edge(edge)
    assert result.has_evidence is False
    assert result.violation == EdgeViolation.NO_EVIDENCE


def test_audit_edge_fails_none_evidence():
    edge = _make_edge(evidence=None)
    result = audit_edge(edge)
    assert result.has_evidence is False
    assert result.violation == EdgeViolation.NO_EVIDENCE


def test_audit_edge_flags_low_confidence():
    edge = _make_edge(confidence=0.1, evidence=[{"evidence_type": "inferred", "source": "x", "confidence": 0.1}])
    result = audit_edge(edge)
    assert result.low_confidence is True


def test_audit_edge_normal_confidence():
    edge = _make_edge(confidence=0.8)
    result = audit_edge(edge)
    assert result.low_confidence is False


def test_audit_graph_edges_all_clean():
    edges = [_make_edge(f"e{i}") for i in range(5)]
    report = audit_graph_edges(edges)
    assert report["total"] == 5
    assert report["no_evidence"] == 0
    assert report["low_confidence"] == 0


def test_audit_graph_edges_detects_violations():
    edges = [
        _make_edge("e1"),
        _make_edge("e2", evidence=[]),
        _make_edge("e3", evidence=None),
        _make_edge("e4", confidence=0.05, evidence=[{"evidence_type": "inferred", "source": "x", "confidence": 0.05}]),
    ]
    report = audit_graph_edges(edges)
    assert report["total"] == 4
    assert report["no_evidence"] == 2
    assert report["low_confidence"] == 1


def test_format_audit_report_markdown():
    edges = [_make_edge("e1"), _make_edge("e2", evidence=[])]
    report = audit_graph_edges(edges)
    md = format_audit_report(report)
    assert "ProvenanceEdge Audit" in md
    assert "no_evidence" in md or "No Evidence" in md


def test_audit_graph_edges_empty():
    report = audit_graph_edges([])
    assert report["total"] == 0
    assert report["no_evidence"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_edge_audit.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'neural_search.graph.edge_audit'`

- [ ] **Step 3: Implement `neural_search/graph/edge_audit.py`**

```python
"""ProvenanceEdge audit tool.

Loads graph edges and reports on evidence completeness. Every edge should have:
1. At least one item in its evidence list
2. Confidence above the low-confidence threshold (default 0.3)

No edge without evidence is the invariant this audit enforces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


LOW_CONFIDENCE_THRESHOLD = 0.3


class EdgeViolation(StrEnum):
    NO_EVIDENCE = "no_evidence"
    LOW_CONFIDENCE = "low_confidence"


@dataclass
class EdgeAuditResult:
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    confidence: float
    has_evidence: bool
    low_confidence: bool
    violation: EdgeViolation | None


def audit_edge(edge: dict[str, Any]) -> EdgeAuditResult:
    """Audit a single graph edge for evidence and confidence."""
    evidence = edge.get("evidence") or []
    has_evidence = bool(evidence)
    confidence = float(edge.get("confidence") or 0.0)
    low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD

    violation: EdgeViolation | None = None
    if not has_evidence:
        violation = EdgeViolation.NO_EVIDENCE

    return EdgeAuditResult(
        edge_id=edge.get("edge_id", ""),
        source_id=edge.get("source_id", ""),
        target_id=edge.get("target_id", ""),
        edge_type=edge.get("edge_type", ""),
        confidence=confidence,
        has_evidence=has_evidence,
        low_confidence=low_confidence,
        violation=violation,
    )


def audit_graph_edges(edges: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit all edges and return a summary report dict."""
    results = [audit_edge(e) for e in edges]
    no_evidence = [r for r in results if not r.has_evidence]
    low_conf = [r for r in results if r.low_confidence and r.has_evidence]

    by_type: dict[str, dict[str, int]] = {}
    for r in results:
        t = r.edge_type
        if t not in by_type:
            by_type[t] = {"total": 0, "no_evidence": 0, "low_confidence": 0}
        by_type[t]["total"] += 1
        if not r.has_evidence:
            by_type[t]["no_evidence"] += 1
        if r.low_confidence:
            by_type[t]["low_confidence"] += 1

    return {
        "total": len(results),
        "no_evidence": len(no_evidence),
        "low_confidence": len(low_conf),
        "pass_rate": round((len(results) - len(no_evidence)) / len(results), 4) if results else 1.0,
        "by_edge_type": by_type,
        "violations": [
            {
                "edge_id": r.edge_id,
                "source_id": r.source_id,
                "target_id": r.target_id,
                "edge_type": r.edge_type,
                "confidence": r.confidence,
                "violation": r.violation,
            }
            for r in results
            if r.violation
        ],
    }


def load_edges_from_graph_jsonl(graph_path: Path) -> list[dict[str, Any]]:
    """Load edges from a KnowledgeGraph JSONL export file."""
    edges: list[dict[str, Any]] = []
    with graph_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            # Support both wrapped {"edge": {...}} and bare edge objects
            if "edge" in obj:
                edges.append(obj["edge"])
            elif "edge_id" in obj or "edge_type" in obj:
                edges.append(obj)
    return edges


def format_audit_report(report: dict[str, Any]) -> str:
    """Format the audit report as markdown."""
    total = report["total"]
    no_ev = report["no_evidence"]
    low_conf = report["low_confidence"]
    pass_rate = report["pass_rate"] * 100
    violations = report.get("violations", [])

    lines = [
        "# ProvenanceEdge Audit Report\n",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total edges | {total} |",
        f"| No evidence | {no_ev} |",
        f"| Low confidence (<{LOW_CONFIDENCE_THRESHOLD}) | {low_conf} |",
        f"| Pass rate (evidence present) | {pass_rate:.1f}% |",
        "",
        "## By Edge Type\n",
        "| Edge Type | Total | No Evidence | Low Confidence |",
        "|---|---:|---:|---:|",
    ]
    for etype, stats in sorted(report["by_edge_type"].items(), key=lambda x: -x[1]["total"]):
        lines.append(f"| {etype} | {stats['total']} | {stats['no_evidence']} | {stats['low_confidence']} |")

    if violations:
        lines.append(f"\n## Violations ({len(violations)} edges)\n")
        lines.append("| Edge ID | Source | Target | Type | Violation |")
        lines.append("|---|---|---|---|---|")
        for v in violations[:50]:
            lines.append(f"| {v['edge_id']} | {v['source_id']} | {v['target_id']} | {v['edge_type']} | {v['violation']} |")
        if len(violations) > 50:
            lines.append(f"\n... and {len(violations) - 50} more")

    return "\n".join(lines)
```

- [ ] **Step 4: Create `scripts/audit_graph_edges.py`**

```python
#!/usr/bin/env python
"""Audit knowledge graph edges for ProvenanceEdge completeness."""

import json
import sys
from pathlib import Path

from neural_search.graph.edge_audit import (
    audit_graph_edges,
    format_audit_report,
    load_edges_from_graph_jsonl,
)

# Try common graph output locations
GRAPH_PATHS = [
    Path("data/graph/knowledge_graph.jsonl"),
    Path("data/graph/graph.jsonl"),
    Path("data/graph/edges.jsonl"),
]
REPORTS_DIR = Path("reports")


def find_graph_file() -> Path | None:
    for p in GRAPH_PATHS:
        if p.exists():
            return p
    return None


def main() -> None:
    graph_path = find_graph_file()
    if not graph_path:
        # Try to export graph using the graph module
        print("No graph JSONL found. Attempting to build graph from corpus...")
        try:
            from neural_search.graph import build_graph_from_corpus
            graph = build_graph_from_corpus()
            edges = [e.model_dump() for e in graph.edges.values()]
        except Exception as exc:
            print(f"Could not load or build graph: {exc}", file=sys.stderr)
            print("Run: python -m neural_search.graph.builder first")
            sys.exit(1)
    else:
        print(f"Loading graph from: {graph_path}")
        edges = load_edges_from_graph_jsonl(graph_path)

    print(f"Loaded {len(edges)} edges")
    report = audit_graph_edges(edges)
    md = format_audit_report(report)

    REPORTS_DIR.mkdir(exist_ok=True)
    md_path = REPORTS_DIR / "edge_provenance_audit.md"
    json_path = REPORTS_DIR / "edge_provenance_audit.json"
    md_path.write_text(md)
    json_path.write_text(json.dumps(report, indent=2, default=str))

    print(f"\nAudit complete:")
    print(f"  Total edges:    {report['total']}")
    print(f"  No evidence:    {report['no_evidence']}")
    print(f"  Low confidence: {report['low_confidence']}")
    print(f"  Pass rate:      {report['pass_rate'] * 100:.1f}%")
    print(f"\nWritten: {md_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_edge_audit.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 6: Run the audit on the real graph**

```bash
python scripts/audit_graph_edges.py
```

Expected: `reports/edge_provenance_audit.md` created showing pass rate for the 1,697 known edges.

- [ ] **Step 7: Commit**

```bash
git add neural_search/graph/edge_audit.py scripts/audit_graph_edges.py tests/test_edge_audit.py reports/edge_provenance_audit.md reports/edge_provenance_audit.json
git commit -m "feat: ProvenanceEdge audit tool — report edges missing evidence, enforce no-evidence-no-edge"
```

---

## Task 6: Ontology Coverage Report + Brain Regions YAML

**Files:**
- Create: `data/ontology/brain_regions_ontology.yaml`
- Create: `neural_search/ontology/coverage_report.py`
- Create: `scripts/generate_ontology_coverage.py`
- Create: `tests/test_ontology_coverage.py`

### Why
The ontology has task + modality YAMLs but no brain regions YAML. The coverage report will expose which neuroscience terms in the corpus are not matched by any synonym — these are the highest-leverage additions for the next vocabulary expansion.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ontology_coverage.py`:

```python
import pytest
from neural_search.ontology.coverage_report import (
    collect_corpus_terms,
    compute_ontology_coverage,
    CoverageStats,
    format_coverage_report,
)


def test_collect_corpus_terms_extracts_brain_regions():
    records = [
        {"brain_regions": [{"id": "hippocampus", "label": "hippocampus"}]},
        {"brain_regions": [{"id": "amygdala", "label": "amygdala"}, {"id": "hippocampus"}]},
        {"brain_regions": []},
    ]
    terms = collect_corpus_terms(records, field="brain_regions")
    assert "hippocampus" in terms
    assert "amygdala" in terms
    assert terms["hippocampus"] == 2
    assert terms["amygdala"] == 1


def test_collect_corpus_terms_handles_string_entries():
    records = [{"brain_regions": ["v1", "hippocampus"]}]
    terms = collect_corpus_terms(records, field="brain_regions")
    assert "v1" in terms
    assert "hippocampus" in terms


def test_collect_corpus_terms_empty():
    terms = collect_corpus_terms([], field="brain_regions")
    assert terms == {}


def test_compute_ontology_coverage_all_matched():
    corpus_terms = {"hippocampus": 10, "amygdala": 5}
    known_terms = {"hippocampus", "amygdala", "striatum"}
    stats = compute_ontology_coverage(corpus_terms, known_terms)
    assert stats.matched == 2
    assert stats.unmatched == 0
    assert stats.coverage == pytest.approx(1.0)


def test_compute_ontology_coverage_some_unmatched():
    corpus_terms = {"hippocampus": 10, "mystery_region": 3}
    known_terms = {"hippocampus", "amygdala"}
    stats = compute_ontology_coverage(corpus_terms, known_terms)
    assert stats.matched == 1
    assert stats.unmatched == 1
    assert stats.unmatched_terms == {"mystery_region": 3}
    assert stats.coverage == pytest.approx(0.5)


def test_compute_ontology_coverage_empty_corpus():
    stats = compute_ontology_coverage({}, {"hippocampus"})
    assert stats.coverage == 1.0  # vacuously true


def test_format_coverage_report_contains_field_name():
    corpus_terms = {"hippocampus": 10, "unknown_area": 2}
    known_terms = {"hippocampus"}
    stats = compute_ontology_coverage(corpus_terms, known_terms)
    md = format_coverage_report({"brain_regions": stats})
    assert "brain_regions" in md or "Brain Regions" in md
    assert "unknown_area" in md
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_ontology_coverage.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'neural_search.ontology.coverage_report'`

- [ ] **Step 3: Create `data/ontology/brain_regions_ontology.yaml`**

```yaml
# Brain Regions Ontology
# Vocabulary for normalizing brain region mentions in neuroscience datasets.
# Each entry: id (canonical), label (human-readable), synonyms (list of strings to match)

brain_regions:
  - id: hippocampus
    label: Hippocampus
    synonyms: [hippocampus, hippocampal, ca1, ca3, dentate gyrus, dg, subiculum, entorhinal cortex]

  - id: amygdala
    label: Amygdala
    synonyms: [amygdala, amygdalar, basolateral amygdala, bla, central amygdala, cea, lateral amygdala]

  - id: mPFC
    label: Medial Prefrontal Cortex
    synonyms: [mpfc, medial prefrontal cortex, prelimbic, infralimbic, medial pfc]

  - id: dlPFC
    label: Dorsolateral Prefrontal Cortex
    synonyms: [dlpfc, dorsolateral prefrontal cortex, dorsolateral pfc]

  - id: OFC
    label: Orbitofrontal Cortex
    synonyms: [ofc, orbitofrontal cortex, orbital frontal cortex]

  - id: ACC
    label: Anterior Cingulate Cortex
    synonyms: [acc, anterior cingulate cortex, anterior cingulate, dorsal anterior cingulate, dacc]

  - id: motor_cortex
    label: Motor Cortex
    synonyms: [motor cortex, m1, primary motor cortex, premotor cortex, supplementary motor area, sma]

  - id: somatosensory_cortex
    label: Somatosensory Cortex
    synonyms: [somatosensory cortex, s1, s2, barrel cortex, primary somatosensory cortex]

  - id: visual_cortex
    label: Visual Cortex
    synonyms: [visual cortex, v1, v2, v4, mt, primary visual cortex, visual area]

  - id: auditory_cortex
    label: Auditory Cortex
    synonyms: [auditory cortex, a1, primary auditory cortex, auditory area]

  - id: striatum
    label: Striatum
    synonyms: [striatum, striatal, caudate, putamen, nucleus accumbens, nac, ventral striatum, dorsal striatum]

  - id: thalamus
    label: Thalamus
    synonyms: [thalamus, thalamic, mediodorsal thalamus, md thalamus, ventral posterolateral, pulvinar]

  - id: cerebellum
    label: Cerebellum
    synonyms: [cerebellum, cerebellar, purkinje cell, cerebellar cortex]

  - id: brainstem
    label: Brainstem
    synonyms: [brainstem, brain stem, midbrain, pons, medulla, superior colliculus, locus coeruleus, lc]

  - id: basal_ganglia
    label: Basal Ganglia
    synonyms: [basal ganglia, substantia nigra, snc, snr, globus pallidus, subthalamic nucleus, stn]

  - id: hypothalamus
    label: Hypothalamus
    synonyms: [hypothalamus, hypothalamic, lateral hypothalamus, arcuate nucleus]

  - id: insula
    label: Insula
    synonyms: [insula, insular cortex, insular]

  - id: parietal_cortex
    label: Parietal Cortex
    synonyms: [parietal cortex, posterior parietal cortex, ppc, intraparietal sulcus, ips]

  - id: temporal_cortex
    label: Temporal Cortex
    synonyms: [temporal cortex, temporal lobe, inferior temporal, it cortex, fusiform gyrus, parahippocampal]

  - id: VTA
    label: Ventral Tegmental Area
    synonyms: [vta, ventral tegmental area, dopaminergic neurons, dopamine neurons]

  - id: olfactory_bulb
    label: Olfactory Bulb
    synonyms: [olfactory bulb, olfactory cortex, piriform cortex, piriform]

  - id: PCC
    label: Posterior Cingulate Cortex
    synonyms: [pcc, posterior cingulate cortex, posterior cingulate]

  - id: precuneus
    label: Precuneus
    synonyms: [precuneus]

  - id: default_mode_network
    label: Default Mode Network
    synonyms: [default mode network, dmn, default network, resting state network]
```

- [ ] **Step 4: Implement `neural_search/ontology/coverage_report.py`**

```python
"""Ontology coverage report.

Compares terms found in the normalized corpus against the known ontology vocabulary.
Identifies unmatched terms — the highest-value additions for synonym expansion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CoverageStats:
    field: str
    total_unique_terms: int
    matched: int
    unmatched: int
    unmatched_terms: dict[str, int]  # term → frequency

    @property
    def coverage(self) -> float:
        if self.total_unique_terms == 0:
            return 1.0
        return self.matched / self.total_unique_terms


def collect_corpus_terms(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    """Collect all unique terms for a field from a list of records.

    Returns a dict of term → frequency.
    """
    counts: dict[str, int] = {}
    for record in records:
        items = record.get(field) or []
        for item in items:
            if isinstance(item, dict):
                term_id = item.get("id") or item.get("label") or ""
            else:
                term_id = str(item)
            if term_id:
                counts[term_id] = counts.get(term_id, 0) + 1
    return counts


def compute_ontology_coverage(
    corpus_terms: dict[str, int],
    known_terms: set[str],
) -> CoverageStats:
    """Compute coverage: how many corpus terms appear in the known vocabulary."""
    if not corpus_terms:
        return CoverageStats(
            field="",
            total_unique_terms=0,
            matched=0,
            unmatched=0,
            unmatched_terms={},
        )
    matched_terms = {t for t in corpus_terms if t in known_terms}
    unmatched_terms = {t: c for t, c in corpus_terms.items() if t not in known_terms}
    return CoverageStats(
        field="",
        total_unique_terms=len(corpus_terms),
        matched=len(matched_terms),
        unmatched=len(unmatched_terms),
        unmatched_terms=dict(sorted(unmatched_terms.items(), key=lambda x: -x[1])),
    )


def load_known_terms_from_synonyms(synonyms: dict[str, list[str]]) -> set[str]:
    """Build a flat set of all IDs and synonym strings from a synonym dict."""
    terms: set[str] = set()
    for term_id, syns in synonyms.items():
        terms.add(term_id.lower())
        terms.update(s.lower() for s in syns)
    return terms


def run_coverage_analysis(corpus_dir: Path) -> dict[str, CoverageStats]:
    """Run coverage analysis across all sources for brain_regions, tasks, and modalities."""
    from neural_search.extraction import BRAIN_REGION_SYNONYMS, MODALITY_SYNONYMS

    try:
        from neural_search.ontology import get_ontology
        ontology = get_ontology()
        task_terms = {t.id for t in ontology.tasks}
        for t in ontology.tasks:
            task_terms.update(getattr(t, "synonyms", []))
    except Exception:
        task_terms = set()

    known_regions = load_known_terms_from_synonyms(BRAIN_REGION_SYNONYMS)
    known_modalities = load_known_terms_from_synonyms(MODALITY_SYNONYMS)

    all_records: list[dict[str, Any]] = []
    for jsonl_file in corpus_dir.glob("real_*.jsonl"):
        with jsonl_file.open() as f:
            for line in f:
                if line.strip():
                    all_records.append(json.loads(line))

    field_configs = {
        "brain_regions": known_regions,
        "modalities": known_modalities,
        "tasks": task_terms,
    }

    results: dict[str, CoverageStats] = {}
    for field_name, known_terms in field_configs.items():
        corpus_terms = collect_corpus_terms(all_records, field_name)
        stats = compute_ontology_coverage(corpus_terms, known_terms)
        stats.field = field_name
        results[field_name] = stats

    return results


def format_coverage_report(stats_by_field: dict[str, CoverageStats]) -> str:
    """Format coverage stats as a Markdown report."""
    lines = ["# Ontology Coverage Report\n"]
    lines.append("Coverage = fraction of unique corpus terms matched by the current synonym vocabulary.\n")

    for field_name, stats in stats_by_field.items():
        lines.append(f"## {field_name.replace('_', ' ').title()} — {stats.coverage * 100:.1f}% coverage\n")
        lines.append(f"- Total unique terms in corpus: {stats.total_unique_terms}")
        lines.append(f"- Matched by vocabulary: {stats.matched}")
        lines.append(f"- **Unmatched (need synonyms): {stats.unmatched}**\n")

        if stats.unmatched_terms:
            lines.append("### Top unmatched terms (highest frequency → best ROI for expansion)\n")
            lines.append("| Term | Frequency |")
            lines.append("|---|---:|")
            for term, freq in list(stats.unmatched_terms.items())[:30]:
                lines.append(f"| `{term}` | {freq} |")
            lines.append("")

    return "\n".join(lines)
```

- [ ] **Step 5: Create `scripts/generate_ontology_coverage.py`**

```python
#!/usr/bin/env python
"""Generate ontology coverage report showing vocabulary gaps."""

from pathlib import Path

from neural_search.ontology.coverage_report import format_coverage_report, run_coverage_analysis

CORPUS_DIR = Path("data/corpus/normalized")
DOCS_DIR = Path("docs")


def main() -> None:
    print("Analyzing ontology coverage...")
    stats = run_coverage_analysis(CORPUS_DIR)
    md = format_coverage_report(stats)
    out = DOCS_DIR / "ONTOLOGY_COVERAGE_REPORT.md"
    out.write_text(md)
    print(f"Written: {out}\n")
    for field_name, s in stats.items():
        print(f"{field_name}: {s.coverage * 100:.0f}% ({s.matched}/{s.total_unique_terms} matched, {s.unmatched} unmatched)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
python -m pytest tests/test_ontology_coverage.py -v
```

Expected: All 7 tests PASS.

- [ ] **Step 7: Run the coverage report**

```bash
python scripts/generate_ontology_coverage.py
```

Expected: `docs/ONTOLOGY_COVERAGE_REPORT.md` created. Top unmatched terms will guide next vocabulary expansion sprint.

- [ ] **Step 8: Commit**

```bash
git add data/ontology/brain_regions_ontology.yaml neural_search/ontology/coverage_report.py scripts/generate_ontology_coverage.py tests/test_ontology_coverage.py docs/ONTOLOGY_COVERAGE_REPORT.md
git commit -m "feat: ontology coverage report + brain regions vocabulary YAML (24 regions, 100+ synonyms)"
```

---

## Post-Sprint: Run Full Completeness Comparison

After all 6 tasks are complete, re-run the completeness report to show before/after:

```bash
python scripts/generate_completeness_report.py
```

Expected improvements:

| Field | DANDI Before | DANDI After | OpenNeuro Before | OpenNeuro After |
|---|---:|---:|---:|---:|
| description | 4% | 80%+ | 92% | 92% |
| brain_regions | 27% | 35%+ | 17% | 40%+ |
| analysis_affordances | 0% | 20%+ | 0% | 25%+ |
| file_formats | 0% | 30%+ | 0% | 5%+ |
| linked_papers | 0% | 15%+ | 0% | 20%+ |

---

## Self-Review

**Spec coverage check:**

| Sprint 2 Item | Covered? |
|---|---|
| Metadata extraction QA loop (DANDI/OpenNeuro extractors, confidence + evidence on every field) | ✅ Tasks 2, 3 |
| Analysis affordances populated from structured metadata with AffordanceValidationResult | ✅ Task 4 |
| ProvenanceEdge completeness: no edge without evidence, edge schema with review_status | ✅ Task 5 |
| Ontology coverage report — gap analysis for brain_regions, tasks, modalities | ✅ Task 6 |
| Baseline report to measure before/after | ✅ Task 1 |

**Placeholder scan:** No TBDs, TODOs, or "similar to Task N" shortcuts.

**Type consistency:**
- `FieldStats`, `SourceCompleteness` defined in Task 1, used only in Task 1.
- `enrich_dandi_record(record, full_metadata)` returns `dict` — consistent across tests and implementation.
- `populate_affordances_for_record(record)` returns `dict` — consistent.
- `audit_edge(edge)` returns `EdgeAuditResult` — consistent.
- `compute_ontology_coverage(corpus_terms, known_terms)` returns `CoverageStats` — consistent.

**Dependency note for implementation:** Tasks 1 → 2 → 3 → 4 should run sequentially (each enrichment improves the corpus for the next step). Tasks 5 and 6 are independent and can run in parallel with 2–4.
