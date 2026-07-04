# Task 02 — Core Data Models (`neural_search/eval/`)

**Files:**
- Create: `neural_search/eval/__init__.py`
- Create: `neural_search/eval/evidence.py`
- Test: `tests/test_eval_evidence.py` (partial — just model construction here)

---

- [ ] **Step 1: Write the failing model tests**

Create `tests/test_eval_evidence.py`:

```python
"""Tests for evidence data models."""
from __future__ import annotations
import pytest
from neural_search.eval.evidence import (
    QuerySpec,
    DatasetEvidence,
    PairEvidence,
    LFVote,
    dataset_evidence_from_record,
    compute_metadata_completeness,
)


class TestQuerySpec:
    def test_default_lists_are_empty(self):
        q = QuerySpec(query_id="q1", query_text="test", intent="META_ANALYSIS", scientific_goal="x")
        assert q.required_modalities == []
        assert q.hard_negatives == []

    def test_to_dict_roundtrip(self):
        q = QuerySpec(
            query_id="q1", query_text="test", intent="META_ANALYSIS",
            scientific_goal="x", required_modalities=["fmri"], hard_negatives=["resting state"]
        )
        d = q.to_dict()
        assert d["query_id"] == "q1"
        assert d["required_modalities"] == ["fmri"]


class TestDatasetEvidence:
    def test_from_record_basic(self):
        record = {
            "source": "dandi", "source_id": "000004", "title": "A dataset",
            "description": "Human ephys", "species": ["human"],
            "modalities": ["extracellular_ephys"], "brain_regions": [],
            "tasks": [], "license": "CC-BY-4.0", "url": "https://example.com",
            "has_raw_data": True, "has_processed_data": False,
            "has_behavior": False, "has_trials": False,
            "data_standards": ["NWB"], "metadata_json": {},
        }
        ev = dataset_evidence_from_record(record)
        assert ev.record_id == "dandi:000004"
        assert ev.raw_data_available is True
        assert "raw" in ev.data_levels

    def test_metadata_completeness_all_present(self):
        record = {
            "source": "dandi", "source_id": "1", "title": "T",
            "description": "D", "species": ["human"], "modalities": ["fmri"],
            "brain_regions": ["prefrontal"], "tasks": ["reward"],
            "license": "CC-BY-4.0", "url": "https://x.com",
            "has_raw_data": True, "has_processed_data": False,
            "has_behavior": True, "has_trials": True,
            "data_standards": [], "metadata_json": {},
        }
        score = compute_metadata_completeness(record)
        assert score == 1.0

    def test_metadata_completeness_minimal(self):
        record = {
            "source": "dandi", "source_id": "2", "title": "T",
            "description": None, "species": [], "modalities": [],
            "brain_regions": [], "tasks": [], "license": None, "url": None,
            "has_raw_data": False, "has_processed_data": False,
            "has_behavior": False, "has_trials": False,
            "data_standards": [], "metadata_json": {},
        }
        score = compute_metadata_completeness(record)
        assert score < 0.3


class TestLFVote:
    def test_abstain_default_false(self):
        v = LFVote(lf_name="lf_test", label=2, confidence=0.8, rationale="ok")
        assert v.abstain is False

    def test_to_dict(self):
        v = LFVote(lf_name="lf_test", label=0, confidence=0.95, rationale="hard neg", abstain=False)
        d = v.to_dict()
        assert d["lf_name"] == "lf_test"
        assert d["label"] == 0
```

- [ ] **Step 2: Run — expect ImportError (module doesn't exist)**

```bash
pytest tests/test_eval_evidence.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'neural_search.eval'`

- [ ] **Step 3: Create `neural_search/eval/__init__.py`**

```python
"""Weak-supervision evaluation package."""
```

- [ ] **Step 4: Create `neural_search/eval/evidence.py`**

```python
"""Core evidence data models for weak-supervision labeling.

QuerySpec     — structured decomposition of a benchmark query
DatasetEvidence — normalized representation of a corpus record
PairEvidence  — a (query, dataset) pair ready for labeling functions
LFVote        — a single labeling function's output
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


# ---------------------------------------------------------------------------
# Metadata completeness
# ---------------------------------------------------------------------------

_COMPLETENESS_FIELDS = [
    "title", "description", "species", "modalities", "brain_regions",
    "tasks", "license", "url", "has_raw_data",
]


def compute_metadata_completeness(record: dict) -> float:
    """Return 0.0–1.0 fraction of key fields that are populated."""
    present = sum(
        1 for f in _COMPLETENESS_FIELDS
        if record.get(f) not in (None, [], "", {}, False)
    )
    return present / len(_COMPLETENESS_FIELDS)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class QuerySpec:
    query_id: str
    query_text: str
    intent: str
    scientific_goal: str
    required_modalities: list[str] = field(default_factory=list)
    preferred_modalities: list[str] = field(default_factory=list)
    required_species: list[str] = field(default_factory=list)
    preferred_species: list[str] = field(default_factory=list)
    brain_regions: list[str] = field(default_factory=list)
    task_constraints: list[str] = field(default_factory=list)
    data_level_requirements: list[str] = field(default_factory=list)
    hard_negatives: list[str] = field(default_factory=list)
    analysis_affordances: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DatasetEvidence:
    record_id: str
    source: str
    title: str
    description: str | None
    species: list[str] = field(default_factory=list)
    modalities: list[str] = field(default_factory=list)
    data_levels: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    license: str | None = None
    doi: str | None = None
    url: str | None = None
    raw_data_available: bool = False
    metadata_completeness: float = 0.0
    has_behavior: bool = False
    has_trials: bool = False
    data_standards: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PairEvidence:
    query_id: str
    record_id: str
    query: QuerySpec
    dataset: DatasetEvidence
    pooled_from: list[str] = field(default_factory=list)
    min_rank: int = 1000
    priority: str = "normal"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LFVote:
    lf_name: str
    label: int          # 0–3
    confidence: float   # 0.0–1.0
    rationale: str
    abstain: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _infer_data_levels(record: dict) -> list[str]:
    levels: list[str] = []
    if record.get("has_raw_data"):
        levels.append("raw")
    if record.get("has_processed_data"):
        levels.append("processed")
    return levels


def dataset_evidence_from_record(record: dict) -> DatasetEvidence:
    """Build a DatasetEvidence from a normalized corpus record dict."""
    source = record.get("source", "")
    source_id = record.get("source_id", "")
    record_id = f"{source}:{source_id}"

    doi: str | None = None
    metadata = record.get("metadata_json") or {}
    if isinstance(metadata, dict):
        doi = metadata.get("doi") or metadata.get("identifier")

    return DatasetEvidence(
        record_id=record_id,
        source=source,
        title=record.get("title") or "",
        description=record.get("description"),
        species=list(record.get("species") or []),
        modalities=list(record.get("modalities") or []),
        data_levels=_infer_data_levels(record),
        tasks=list(record.get("tasks") or []),
        regions=list(record.get("brain_regions") or []),
        license=record.get("license"),
        doi=doi,
        url=record.get("url"),
        raw_data_available=bool(record.get("has_raw_data")),
        metadata_completeness=compute_metadata_completeness(record),
        has_behavior=bool(record.get("has_behavior")),
        has_trials=bool(record.get("has_trials")),
        data_standards=list(record.get("data_standards") or []),
    )
```

- [ ] **Step 5: Run tests — expect green**

```bash
pytest tests/test_eval_evidence.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 6: Commit**

```bash
git add neural_search/eval/ tests/test_eval_evidence.py
git commit -m "feat(eval): add core data models — QuerySpec, DatasetEvidence, LFVote"
```
