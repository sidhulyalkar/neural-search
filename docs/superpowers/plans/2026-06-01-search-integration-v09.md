# Neural Search v0.9 — Search Integration & Real Graph Usefulness

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the v0.8 latent usefulness scorer into the live search pipeline so every `SearchResult` carries a `usefulness_score` field, replace the 0.3 graph-proximity placeholder with real PathSim when a knowledge graph is available, and add a human-labeling CLI for expanding the benchmark.

**Architecture:** Three integration points: (1) a new bridge module converts existing record/card dicts to `DatasetContext`; (2) `score_usefulness()` gains an optional `graph` parameter that computes real PathSim via `graph_usefulness_features()`; (3) `_augment_result_with_optional_scores()` in `core.py` calls the scorer and attaches results to `SearchResult.usefulness_score`. A CLI script handles interactive benchmark labeling.

**Tech Stack:** Python 3.10+, Pydantic v2, `neural_search.retrieval` (usefulness_scorer, graph_usefulness, query_intent), `neural_search.search.core`, `neural_search.schemas`, `neural_search.graph.schema.KnowledgeGraph`, `click` (already installed)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| **Create** | `neural_search/retrieval/dataset_context_bridge.py` | Convert raw record/card dict → `DatasetContext` |
| **Modify** | `neural_search/retrieval/usefulness_scorer.py` | Add optional `graph` parameter to `score_usefulness()` |
| **Modify** | `neural_search/retrieval/__init__.py` | Export `dataset_context_from_record` |
| **Modify** | `neural_search/schemas.py:340-360` | Add `usefulness_score: dict | None` to `SearchResult` |
| **Modify** | `neural_search/search/core.py` | Call scorer in `_augment_result_with_optional_scores()` |
| **Create** | `scripts/annotate_usefulness.py` | Interactive 4-level labeling CLI |
| **Create** | `tests/test_dataset_context_bridge.py` | Tests for bridge converter |
| **Create** | `tests/test_search_usefulness_integration.py` | End-to-end: search result carries usefulness_score |

---

## Context for Every Task

Before writing any code, read these two files to understand the current state:
- `neural_search/retrieval/usefulness_scorer.py` — DatasetContext dataclass, score_usefulness() signature
- `neural_search/retrieval/graph_usefulness.py` — graph_usefulness_features() function

Key types (already exist):
```python
# neural_search/retrieval/usefulness_scorer.py
@dataclass
class DatasetContext:
    dataset_id: str
    modalities: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    species: list[str] = field(default_factory=list)
    brain_regions: list[str] = field(default_factory=list)
    affordances: list[str] = field(default_factory=list)
    data_standards: list[str] = field(default_factory=list)
    session_count: int | None = None
    trial_count: int | None = None
    subject_count: int | None = None
    has_timestamps: bool = False
    quality_score: float = 0.0

# neural_search/retrieval/usefulness_scorer.py
def score_usefulness(
    query_context: DatasetContext,
    candidate: DatasetContext,
    intent: UsefulnessIntent | None = None,
) -> UsefulnessScore:

# neural_search/retrieval/graph_usefulness.py
def graph_usefulness_features(
    query_context: DatasetContext,
    candidate: DatasetContext,
    graph: dict | None = None,
) -> dict:  # keys: affordance_overlap, pipeline_overlap, complementarity, metapath_score
```

---

### Task 1: `DatasetContext` bridge from raw record

**Files:**
- Create: `neural_search/retrieval/dataset_context_bridge.py`
- Create: `tests/test_dataset_context_bridge.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dataset_context_bridge.py
from neural_search.retrieval.dataset_context_bridge import dataset_context_from_record
from neural_search.retrieval.usefulness_scorer import DatasetContext


def _make_record(
    dataset_id="test_ds",
    tasks=None,
    modalities=None,
    species=None,
    regions=None,
):
    return {
        "id": dataset_id,
        "tasks": [{"label": t} for t in (tasks or [])],
        "modalities": [{"label": m} for m in (modalities or [])],
        "species": [{"label": s} for s in (species or [])],
        "brain_regions": [{"label": r} for r in (regions or [])],
    }


def test_returns_dataset_context():
    record = _make_record("ds001", tasks=["decision-making"])
    ctx = dataset_context_from_record(record)
    assert isinstance(ctx, DatasetContext)


def test_dataset_id_extracted():
    record = _make_record("ds_abc")
    ctx = dataset_context_from_record(record)
    assert ctx.dataset_id == "ds_abc"


def test_tasks_extracted_from_label_dicts():
    record = _make_record(tasks=["reversal_learning", "go_nogo"])
    ctx = dataset_context_from_record(record)
    assert set(ctx.tasks) == {"reversal_learning", "go_nogo"}


def test_modalities_extracted():
    record = _make_record(modalities=["neuropixels", "calcium_imaging"])
    ctx = dataset_context_from_record(record)
    assert set(ctx.modalities) == {"neuropixels", "calcium_imaging"}


def test_empty_record_returns_default_context():
    ctx = dataset_context_from_record({})
    assert ctx.dataset_id == ""
    assert ctx.tasks == []
    assert ctx.modalities == []


def test_card_affordances_extracted():
    record = _make_record("ds_aff")
    card = {"analysis_affordances": [{"affordance_id": "choice_decoding"}, {"affordance_id": "glm_ready"}]}
    ctx = dataset_context_from_record(record, card)
    assert set(ctx.affordances) == {"choice_decoding", "glm_ready"}


def test_card_data_standards_extracted():
    record = _make_record("ds_std")
    card = {"data_standards": ["NWB", "BIDS"]}
    ctx = dataset_context_from_record(record, card)
    assert set(ctx.data_standards) == {"NWB", "BIDS"}


def test_session_count_from_record():
    record = _make_record("ds_sess")
    record["session_count"] = 42
    ctx = dataset_context_from_record(record)
    assert ctx.session_count == 42


def test_quality_score_from_card():
    record = _make_record("ds_q")
    card = {"quality_score": 0.87}
    ctx = dataset_context_from_record(record, card)
    assert abs(ctx.quality_score - 0.87) < 1e-6
```

- [ ] **Step 2: Run tests to confirm they all fail**

```bash
pytest tests/test_dataset_context_bridge.py -v
```
Expected: `ImportError` — module does not exist yet.

- [ ] **Step 3: Implement the bridge**

```python
# neural_search/retrieval/dataset_context_bridge.py
"""Bridge: convert raw record/card dicts to DatasetContext."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from neural_search.retrieval.usefulness_scorer import DatasetContext


def _extract_labels(items: Any) -> list[str]:
    """Extract label strings from a list of dicts or plain strings."""
    if not items:
        return []
    result = []
    for item in items:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, Mapping):
            label = item.get("label") or item.get("name") or item.get("id") or ""
            if label:
                result.append(str(label))
    return result


def dataset_context_from_record(
    record: Mapping[str, Any],
    card: Mapping[str, Any] | None = None,
) -> DatasetContext:
    """Convert a raw dataset record (and optional card) to DatasetContext.

    Args:
        record: Raw dataset dict as stored in the corpus.
        card: Optional dataset card dict (DatasetCardV1 serialized).

    Returns:
        DatasetContext ready for score_usefulness().
    """
    dataset_id = str(
        record.get("id")
        or record.get("source_id")
        or record.get("dataset_id")
        or ""
    )

    tasks = _extract_labels(record.get("tasks", []))
    modalities = _extract_labels(record.get("modalities", []))
    species = _extract_labels(record.get("species", []))
    brain_regions = _extract_labels(record.get("brain_regions", []))

    # Card fields (richer metadata)
    affordances: list[str] = []
    data_standards: list[str] = []
    quality_score: float = 0.0
    session_count: int | None = record.get("session_count")
    trial_count: int | None = record.get("trial_count")
    subject_count: int | None = record.get("subject_count")
    has_timestamps: bool = bool(record.get("has_timestamps", False))

    if card:
        aff_raw = card.get("analysis_affordances", [])
        for a in aff_raw:
            if isinstance(a, Mapping):
                aff_id = a.get("affordance_id") or a.get("id") or ""
                if aff_id:
                    affordances.append(str(aff_id))
            elif isinstance(a, str):
                affordances.append(a)

        std_raw = card.get("data_standards", [])
        for s in std_raw:
            if isinstance(s, str):
                data_standards.append(s)
            elif isinstance(s, Mapping):
                data_standards.append(str(s.get("name") or s.get("id") or ""))

        quality_score = float(card.get("quality_score") or 0.0)
        if session_count is None:
            session_count = card.get("session_count")
        if trial_count is None:
            trial_count = card.get("trial_count")
        if subject_count is None:
            subject_count = card.get("subject_count")

    return DatasetContext(
        dataset_id=dataset_id,
        modalities=modalities,
        tasks=tasks,
        species=species,
        brain_regions=brain_regions,
        affordances=affordances,
        data_standards=data_standards,
        session_count=session_count,
        trial_count=trial_count,
        subject_count=subject_count,
        has_timestamps=has_timestamps,
        quality_score=quality_score,
    )
```

- [ ] **Step 4: Run tests to confirm they all pass**

```bash
pytest tests/test_dataset_context_bridge.py -v
```
Expected: 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add neural_search/retrieval/dataset_context_bridge.py tests/test_dataset_context_bridge.py
git commit -m "feat: add DatasetContext bridge for record/card → scorer conversion"
```

---

### Task 2: Wire real graph proximity into `score_usefulness()`

**Files:**
- Modify: `neural_search/retrieval/usefulness_scorer.py`

The current `score_usefulness()` computes s9 as a constant 0.3 with a warning. This task wires in the real PathSim when a knowledge graph is provided.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_usefulness_scorer.py`:

```python
# Add at the bottom of tests/test_usefulness_scorer.py
from neural_search.graph.schema import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge, make_node_id, make_edge_id
from neural_search.retrieval.usefulness_scorer import score_usefulness, DatasetContext
from neural_search.retrieval.query_intent import UsefulnessIntent


def _make_minimal_graph() -> KnowledgeGraph:
    """Two dataset nodes sharing a task neighbor."""
    d1_id = make_node_id("dataset", "ds001")
    d2_id = make_node_id("dataset", "ds002")
    t1_id = make_node_id("task", "decision_making")
    nodes = {
        d1_id: KnowledgeGraphNode(node_id=d1_id, node_type="dataset", label="DS001"),
        d2_id: KnowledgeGraphNode(node_id=d2_id, node_type="dataset", label="DS002"),
        t1_id: KnowledgeGraphNode(node_id=t1_id, node_type="task", label="Decision Making"),
    }
    e1_id = make_edge_id(d1_id, "dataset_has_task", t1_id)
    e2_id = make_edge_id(d2_id, "dataset_has_task", t1_id)
    edges = {
        e1_id: KnowledgeGraphEdge(edge_id=e1_id, source_node_id=d1_id, target_node_id=t1_id, edge_type="dataset_has_task"),
        e2_id: KnowledgeGraphEdge(edge_id=e2_id, source_node_id=d2_id, target_node_id=t1_id, edge_type="dataset_has_task"),
    }
    return KnowledgeGraph(nodes=nodes, edges=edges)


def test_score_usefulness_accepts_graph_parameter():
    """score_usefulness must accept a 'graph' keyword argument."""
    qctx = DatasetContext(dataset_id="q", tasks=["decision-making"])
    cctx = DatasetContext(dataset_id="ds001", tasks=["decision-making"])
    graph = _make_minimal_graph()
    score = score_usefulness(qctx, cctx, UsefulnessIntent.REPLICATION, graph=graph)
    assert 0.0 <= score.total_score <= 1.0


def test_graph_proximity_no_longer_warns_when_graph_provided():
    """When a graph is provided and nodes exist, no graph_proximity warning."""
    qctx = DatasetContext(dataset_id="node:dataset:ds001", tasks=["decision-making"])
    cctx = DatasetContext(dataset_id="node:dataset:ds002", tasks=["decision-making"])
    graph = _make_minimal_graph()
    score = score_usefulness(qctx, cctx, UsefulnessIntent.REPLICATION, graph=graph)
    graph_warnings = [w for w in score.warnings if "graph_proximity" in w.lower()]
    assert graph_warnings == [], f"unexpected graph warning: {graph_warnings}"


def test_graph_proximity_still_warns_without_graph():
    """Without a graph, the 0.3 prior warning must still appear."""
    qctx = DatasetContext(dataset_id="q", tasks=["decision-making"])
    cctx = DatasetContext(dataset_id="c", tasks=["decision-making"])
    score = score_usefulness(qctx, cctx, UsefulnessIntent.REPLICATION, graph=None)
    assert any("graph_proximity" in w.lower() for w in score.warnings)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_usefulness_scorer.py::test_score_usefulness_accepts_graph_parameter tests/test_usefulness_scorer.py::test_graph_proximity_no_longer_warns_when_graph_provided tests/test_usefulness_scorer.py::test_graph_proximity_still_warns_without_graph -v
```
Expected: FAIL with `TypeError: score_usefulness() got an unexpected keyword argument 'graph'`.

- [ ] **Step 3: Modify `score_usefulness()` in `usefulness_scorer.py`**

Read the file first, then locate the `score_usefulness` function (approximately lines 100-170). Add the `graph` parameter and replace the s9 hardcoded block:

**Change 1** — add import at top of file (after existing imports):
```python
from __future__ import annotations
# existing imports...
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neural_search.graph.schema import KnowledgeGraph
```

**Change 2** — update the function signature (find current signature):
```python
# BEFORE:
def score_usefulness(
    query_context: DatasetContext,
    candidate: DatasetContext,
    intent: UsefulnessIntent | None = None,
) -> UsefulnessScore:

# AFTER:
def score_usefulness(
    query_context: DatasetContext,
    candidate: DatasetContext,
    intent: UsefulnessIntent | None = None,
    graph: "KnowledgeGraph | None" = None,
) -> UsefulnessScore:
```

**Change 3** — replace the s9 (graph_proximity) block. Find the block that looks like:
```python
    # s9: graph_proximity — Phase 3 placeholder
    graph_proximity = 0.3
    warnings.append("graph_proximity: using neutral prior 0.3 (graph not available)")
```

Replace with:
```python
    # s9: graph_proximity — use PathSim when graph is available
    if graph is not None:
        from neural_search.retrieval.graph_usefulness import graph_usefulness_features
        guf = graph_usefulness_features(query_context, candidate, graph.model_dump(mode="json"))
        graph_proximity = float(guf.get("metapath_score", 0.3))
    else:
        graph_proximity = 0.3
        warnings.append("graph_proximity: using neutral prior 0.3 (graph not available)")
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_usefulness_scorer.py -v
```
Expected: All existing + 3 new tests PASS (13 total).

- [ ] **Step 5: Commit**

```bash
git add neural_search/retrieval/usefulness_scorer.py tests/test_usefulness_scorer.py
git commit -m "feat: wire real graph proximity into score_usefulness when graph available"
```

---

### Task 3: Export bridge from `retrieval/__init__.py`

**Files:**
- Modify: `neural_search/retrieval/__init__.py`

- [ ] **Step 1: Read current `__init__.py`**

```bash
cat neural_search/retrieval/__init__.py
```

- [ ] **Step 2: Add `dataset_context_from_record` export**

Add to the import section:
```python
from neural_search.retrieval.dataset_context_bridge import (
    dataset_context_from_record,
)
```

Add to `__all__`:
```python
    "dataset_context_from_record",
```

- [ ] **Step 3: Verify import works**

```bash
python -c "from neural_search.retrieval import dataset_context_from_record; print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add neural_search/retrieval/__init__.py
git commit -m "feat: export dataset_context_from_record from retrieval package"
```

---

### Task 4: Add `usefulness_score` to `SearchResult`

**Files:**
- Modify: `neural_search/schemas.py:340-360`

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_search_usefulness_integration.py (create this file)
from neural_search.schemas import SearchResult


def test_search_result_has_usefulness_score_field():
    """SearchResult must have usefulness_score as optional dict."""
    result = SearchResult(dataset_id="ds_test", score=0.5)
    assert hasattr(result, "usefulness_score")
    assert result.usefulness_score is None


def test_search_result_usefulness_score_accepts_dict():
    result = SearchResult(
        dataset_id="ds_test",
        score=0.5,
        usefulness_score={"total_score": 0.75, "intent": "replication"},
    )
    assert result.usefulness_score["total_score"] == 0.75
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_search_usefulness_integration.py::test_search_result_has_usefulness_score_field tests/test_search_usefulness_integration.py::test_search_result_accepts_usefulness_score_dict -v
```
Expected: FAIL with `ValidationError` or `AttributeError`.

- [ ] **Step 3: Add field to `SearchResult`**

In `neural_search/schemas.py`, find the `SearchResult` class (line ~340). Add this field **after** `graph_context`:
```python
    usefulness_score: dict[str, Any] | None = None
```

The class should now look like:
```python
class SearchResult(BaseModel):
    dataset_id: UUID | str
    score: float = Field(ge=0.0)
    why_matched: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)
    inferred_concepts: list[str] = Field(default_factory=list)
    evidence_snippets: list[str] = Field(default_factory=list)
    missing_metadata_warnings: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    negative_constraint_matches: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    explanation: str | None = None
    reusable_reason: str | None = None
    dataset_card_preview: dict[str, Any] = Field(default_factory=dict)
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    graph_context: dict[str, Any] | None = None
    usefulness_score: dict[str, Any] | None = None   # NEW
    linked_papers: list[dict[str, Any]] = Field(default_factory=list)
    filtered_constraints: list[dict[str, Any]] = Field(default_factory=list)
    missing_metadata: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_search_usefulness_integration.py -v
```
Expected: 2 tests PASS.

- [ ] **Step 5: Run existing schema tests to check for regressions**

```bash
pytest tests/test_query_intent.py tests/test_relevance.py tests/test_search_intelligence.py -q
```
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add neural_search/schemas.py tests/test_search_usefulness_integration.py
git commit -m "feat: add usefulness_score field to SearchResult schema"
```

---

### Task 5: Wire usefulness scorer into search pipeline

**Files:**
- Modify: `neural_search/search/core.py`

This task adds usefulness scoring as an augmentation step in `_augment_result_with_optional_scores()`. It is purely additive — no existing scores change.

- [ ] **Step 1: Write the failing integration test**

Add to `tests/test_search_usefulness_integration.py`:

```python
from neural_search.search import search_datasets


def test_search_result_carries_usefulness_score():
    """search_datasets results must include usefulness_score dict."""
    response = search_datasets("mouse decision-making neuropixels")
    assert len(response.results) > 0
    first = response.results[0]
    assert first.usefulness_score is not None
    assert "total_score" in first.usefulness_score
    assert 0.0 <= first.usefulness_score["total_score"] <= 1.0


def test_usefulness_score_contains_intent():
    """usefulness_score must include the classified intent name."""
    response = search_datasets("replicate Steinmetz 2019 experiment")
    first = response.results[0]
    assert "intent" in first.usefulness_score


def test_usefulness_score_contains_dimension_scores():
    """usefulness_score must have a dimension_scores dict."""
    response = search_datasets("mouse hippocampus calcium imaging")
    first = response.results[0]
    assert "dimension_scores" in first.usefulness_score
    dims = first.usefulness_score["dimension_scores"]
    assert "modality_alignment" in dims or len(dims) > 0
```

- [ ] **Step 2: Run to confirm they fail**

```bash
pytest tests/test_search_usefulness_integration.py::test_search_result_carries_usefulness_score -v
```
Expected: FAIL — `usefulness_score` is None.

- [ ] **Step 3: Add imports to `core.py`**

At the top of `neural_search/search/core.py`, after the existing imports, add:

```python
from neural_search.retrieval.dataset_context_bridge import dataset_context_from_record
from neural_search.retrieval.query_intent import classify_query_intent as classify_usefulness_intent
from neural_search.retrieval.usefulness_scorer import DatasetContext, score_usefulness
```

- [ ] **Step 4: Build query context once per search call**

In `search_datasets()`, after the `classify_query_intent` call (around line 1127), add:

```python
    # Build query DatasetContext for usefulness scoring (once per search call)
    _query_usefulness_ctx = DatasetContext(
        dataset_id="__query__",
        modalities=list(parsed.get("modalities", [])),
        tasks=list(parsed.get("tasks", [])),
        species=list(parsed.get("species", [])),
        brain_regions=list(parsed.get("brain_regions", [])),
        affordances=list(parsed.get("affordances", [])),
    )
    _usefulness_intent_cls = classify_usefulness_intent(combined_query)
```

These two variables need to be accessible inside `_augment_result_with_optional_scores()`. The cleanest approach is to add them to the `parsed` dict so they flow through the existing pipeline:

```python
    parsed["_query_usefulness_ctx"] = _query_usefulness_ctx
    parsed["_usefulness_intent"] = _usefulness_intent_cls.intent
```

- [ ] **Step 5: Add usefulness scoring in `_augment_result_with_optional_scores()`**

Find the function `_augment_result_with_optional_scores` (around line 902). It already takes `result`, `query`, `parsed`, `config`, `graph`, etc. 

Add this block **at the end of the function body** (after the awareness scoring block, before the final return / end of function):

```python
    # Usefulness scoring — attach to result.usefulness_score
    try:
        query_ctx = parsed.get("_query_usefulness_ctx")
        usefulness_intent = parsed.get("_usefulness_intent")
        if query_ctx is not None:
            # Build DatasetContext from this result's underlying record
            raw_record = result.dataset_card_preview if result.dataset_card_preview else {}
            cand_ctx = dataset_context_from_record(raw_record)
            # Use the simple ID if dataset_card_preview has no structured data
            if not cand_ctx.dataset_id:
                cand_ctx = DatasetContext(
                    dataset_id=str(result.dataset_id),
                    modalities=list(result.score_breakdown.get("_modalities", [])),
                )
            u_score = score_usefulness(query_ctx, cand_ctx, usefulness_intent, graph=graph)
            result.usefulness_score = {
                "total_score": round(u_score.total_score, 4),
                "intent": u_score.intent.value,
                "dimension_scores": {k: round(v, 4) for k, v in u_score.dimension_scores.items()},
                "warnings": u_score.warnings,
            }
    except Exception:
        pass  # Usefulness scoring is always optional — never fail search
```

- [ ] **Step 6: Run tests to confirm they pass**

```bash
pytest tests/test_search_usefulness_integration.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 7: Run regression tests**

```bash
pytest tests/test_relevance.py tests/test_search_intelligence.py tests/test_search_artifact_integration.py -q
```
Expected: All pass.

- [ ] **Step 8: Commit**

```bash
git add neural_search/search/core.py tests/test_search_usefulness_integration.py
git commit -m "feat: wire latent usefulness scorer into search pipeline"
```

---

### Task 6: Human-labeling CLI for benchmark expansion

**Files:**
- Create: `scripts/annotate_usefulness.py`

This script enables a researcher to walk through query-candidate pairs and assign a 4-level usefulness label, appending to the JSONL seed file.

- [ ] **Step 1: Write the test**

```python
# tests/test_annotate_usefulness_script.py
import json
import subprocess
import sys
from pathlib import Path


def test_script_exists_and_is_importable():
    """Script must exist and have no syntax errors."""
    result = subprocess.run(
        [sys.executable, "-c", "import ast; ast.parse(open('scripts/annotate_usefulness.py').read())"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_script_dry_run_exits_cleanly(tmp_path):
    """With --dry-run, script should print pairs without writing anything."""
    # Create a minimal JSONL
    pairs_file = tmp_path / "pairs.jsonl"
    pairs_file.write_text(
        json.dumps({
            "query_id": "q001",
            "query": "mouse decision-making",
            "intent": "replication",
            "candidate_id": "dandi:000001",
            "usefulness_label": "useful",
            "label_type": "seed",
            "notes": "",
        }) + "\n"
    )
    result = subprocess.run(
        [sys.executable, "scripts/annotate_usefulness.py",
         "--file", str(pairs_file), "--dry-run"],
        capture_output=True, text=True, input="",
    )
    assert result.returncode == 0
    # File should not be modified
    content_after = pairs_file.read_text()
    assert "q001" in content_after
```

- [ ] **Step 2: Run to confirm it fails**

```bash
pytest tests/test_annotate_usefulness_script.py -v
```
Expected: FAIL — script does not exist.

- [ ] **Step 3: Implement the labeling CLI**

```python
#!/usr/bin/env python3
# scripts/annotate_usefulness.py
"""Interactive CLI for labeling query-candidate pairs with usefulness labels.

Usage:
    python scripts/annotate_usefulness.py --file data/eval/usefulness_seed_pairs.jsonl
    python scripts/annotate_usefulness.py --file data/eval/my_pairs.jsonl --start-from 10
    python scripts/annotate_usefulness.py --file data/eval/my_pairs.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VALID_LABELS = ["not_useful", "weakly_useful", "useful", "highly_useful"]
LABEL_SHORTCUTS = {"0": "not_useful", "1": "weakly_useful", "2": "useful", "3": "highly_useful"}


def _prompt_label(pair: dict) -> str | None:
    """Prompt the annotator to assign a label. Returns None to skip."""
    print("\n" + "=" * 60)
    print(f"Query [{pair['query_id']}]: {pair['query']}")
    print(f"Intent: {pair.get('intent', 'unknown')}")
    print(f"Candidate: {pair['candidate_id']}")
    print(f"Current label: {pair.get('usefulness_label', '(none)')}")
    print()
    print("Labels: 0=not_useful  1=weakly_useful  2=useful  3=highly_useful")
    print("Enter label (0-3), full name, 's' to skip, or 'q' to quit:")
    try:
        raw = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None
    if raw == "q":
        return None
    if raw == "s":
        return pair.get("usefulness_label", "not_useful")
    if raw in LABEL_SHORTCUTS:
        return LABEL_SHORTCUTS[raw]
    if raw in VALID_LABELS:
        return raw
    print(f"Invalid input '{raw}', skipping.")
    return pair.get("usefulness_label", "not_useful")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Label usefulness pairs interactively")
    parser.add_argument("--file", required=True, help="Path to JSONL pairs file")
    parser.add_argument("--start-from", type=int, default=0, help="Skip the first N pairs")
    parser.add_argument("--dry-run", action="store_true", help="Print pairs without prompting or writing")
    args = parser.parse_args(argv)

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    if args.dry_run:
        for i, pair in enumerate(pairs):
            print(f"[{i}] {pair.get('query_id')} — {pair.get('candidate_id')} — {pair.get('usefulness_label')}")
        return 0

    changed = 0
    for i, pair in enumerate(pairs):
        if i < args.start_from:
            continue
        new_label = _prompt_label(pair)
        if new_label is None:
            print("Quitting.")
            break
        if new_label != pair.get("usefulness_label"):
            pairs[i] = {**pair, "usefulness_label": new_label, "label_type": "human"}
            changed += 1

    # Write back
    with path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\nDone. {changed} label(s) changed. Saved to {path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_annotate_usefulness_script.py -v
```
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/annotate_usefulness.py tests/test_annotate_usefulness_script.py
git commit -m "feat: add interactive usefulness labeling CLI for benchmark expansion"
```

---

### Task 7: Expand seed pairs — cover METHOD_TRANSFER intent

**Files:**
- Modify: `data/eval/usefulness_seed_pairs.jsonl` (append 9 new pairs for q011, q012, q013)

The current 30 seed pairs (q001–q010) cover: pipeline_reuse, exploration, replication, cross_dataset_comparison, meta_analysis. `METHOD_TRANSFER` is under-represented. This task adds 3 queries × 3 candidates = 9 pairs.

- [ ] **Step 1: Write the test**

```python
# tests/test_seed_pairs_coverage.py
import json
from pathlib import Path
from collections import Counter


SEED_FILE = Path("data/eval/usefulness_seed_pairs.jsonl")


def test_seed_file_exists():
    assert SEED_FILE.exists(), "Seed JSONL file must exist"


def test_all_intents_covered():
    """Every UsefulnessIntent must have at least 2 query IDs."""
    pairs = [json.loads(l) for l in SEED_FILE.read_text().splitlines() if l.strip()]
    intent_queries: dict[str, set] = {}
    for p in pairs:
        intent = p["intent"]
        qid = p["query_id"]
        intent_queries.setdefault(intent, set()).add(qid)

    required_intents = {
        "strict_lookup", "replication", "meta_analysis",
        "pipeline_reuse", "cross_dataset_comparison",
        "exploration", "method_transfer",
    }
    for intent in required_intents:
        count = len(intent_queries.get(intent, set()))
        assert count >= 2, f"Intent '{intent}' has only {count} query(ies); need ≥2"


def test_minimum_30_pairs():
    pairs = [l for l in SEED_FILE.read_text().splitlines() if l.strip()]
    assert len(pairs) >= 30
```

- [ ] **Step 2: Run to confirm the METHOD_TRANSFER coverage check fails**

```bash
pytest tests/test_seed_pairs_coverage.py -v
```
Expected: `test_all_intents_covered` FAIL — method_transfer has 0 queries.

- [ ] **Step 3: Append 9 new METHOD_TRANSFER pairs to the JSONL**

Append these 9 lines to `data/eval/usefulness_seed_pairs.jsonl`:

```jsonl
{"query_id": "q011", "query": "apply choice decoding from rodent to primate data", "intent": "method_transfer", "candidate_id": "openneuro:ds003000", "usefulness_label": "highly_useful", "label_type": "seed", "notes": "macaque decision-making with trial labels, ideal for transferring rodent choice decoding pipeline"}
{"query_id": "q011", "query": "apply choice decoding from rodent to primate data", "intent": "method_transfer", "candidate_id": "dandi:000123", "usefulness_label": "useful", "label_type": "seed", "notes": "NHP electrophysiology during go/nogo — partial transfer feasibility"}
{"query_id": "q011", "query": "apply choice decoding from rodent to primate data", "intent": "method_transfer", "candidate_id": "dandi:000045", "usefulness_label": "not_useful", "label_type": "seed", "notes": "mouse EEG passive listening — no choice labels, wrong species for transfer"}
{"query_id": "q012", "query": "transfer GLM-HMM from 2-choice to multi-choice paradigm datasets", "intent": "method_transfer", "candidate_id": "dandi:000213", "usefulness_label": "highly_useful", "label_type": "seed", "notes": "multi-alternative forced-choice with trial outcomes — GLM-HMM ready"}
{"query_id": "q012", "query": "transfer GLM-HMM from 2-choice to multi-choice paradigm datasets", "intent": "method_transfer", "candidate_id": "dandi:000055", "usefulness_label": "weakly_useful", "label_type": "seed", "notes": "binary choice, different species — partial methodological transfer possible"}
{"query_id": "q012", "query": "transfer GLM-HMM from 2-choice to multi-choice paradigm datasets", "intent": "method_transfer", "candidate_id": "openneuro:ds002336", "usefulness_label": "not_useful", "label_type": "seed", "notes": "passive fMRI viewing — no trial choice structure, GLM-HMM inapplicable"}
{"query_id": "q013", "query": "use calcium imaging analysis pipeline on neuropixels data", "intent": "method_transfer", "candidate_id": "dandi:000402", "usefulness_label": "useful", "label_type": "seed", "notes": "neuropixels with clean spike sorting — analysis transferable with adaptation"}
{"query_id": "q013", "query": "use calcium imaging analysis pipeline on neuropixels data", "intent": "method_transfer", "candidate_id": "dandi:000114", "usefulness_label": "weakly_useful", "label_type": "seed", "notes": "mixed modality dataset — some signal overlap but requires reengineering"}
{"query_id": "q013", "query": "use calcium imaging analysis pipeline on neuropixels data", "intent": "method_transfer", "candidate_id": "openneuro:ds001604", "usefulness_label": "not_useful", "label_type": "seed", "notes": "fMRI only — completely different signal type, no transfer applicable"}
```

- [ ] **Step 4: Run tests to confirm all pass**

```bash
pytest tests/test_seed_pairs_coverage.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add data/eval/usefulness_seed_pairs.jsonl tests/test_seed_pairs_coverage.py
git commit -m "data: add METHOD_TRANSFER seed pairs for benchmark coverage"
```

---

### Task 8: Full test suite verification

**Files:**
- No code changes — verification only.

- [ ] **Step 1: Run all new v0.9 tests**

```bash
pytest tests/test_dataset_context_bridge.py tests/test_search_usefulness_integration.py tests/test_annotate_usefulness_script.py tests/test_seed_pairs_coverage.py -v
```
Expected: All pass with zero failures.

- [ ] **Step 2: Run full suite (excluding slow tests)**

```bash
pytest tests/ -q --tb=short --ignore=tests/test_search_quality.py 2>&1 | tail -10
```
Expected: `N passed` where N ≥ 958, 0 failed.

- [ ] **Step 3: If failures exist, investigate and fix**

Check which test module failed, read the error, fix root cause. Do not use `--ignore` to hide failures.

- [ ] **Step 4: Final cleanup commit if needed**

```bash
git add -p
git commit -m "chore: final v0.9 cleanup — all tests passing"
```

---

## Self-Review

### Spec Coverage Check
- [x] DatasetContext bridge (Task 1)
- [x] Real graph proximity wired (Task 2)
- [x] Bridge exported from package (Task 3)
- [x] `usefulness_score` on `SearchResult` (Task 4)
- [x] Search pipeline integration (Task 5)
- [x] Human-labeling CLI (Task 6)
- [x] METHOD_TRANSFER seed pairs + coverage test (Task 7)
- [x] Full suite verification (Task 8)

### Placeholder Scan
- All steps contain actual code, no "TBD" or "implement later".

### Type Consistency
- `DatasetContext` imported from `neural_search.retrieval.usefulness_scorer` consistently.
- `score_usefulness()` signature: `(query_context, candidate, intent, graph=None)` used consistently.
- `graph_usefulness_features()` accepts `graph: dict | None` — we pass `graph.model_dump(mode="json")`.
- `SearchResult.usefulness_score: dict[str, Any] | None` — `UsefulnessScore` is serialized with `.total_score`, `.intent.value`, `.dimension_scores`, `.warnings`.
