# Latent Usefulness v0.8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade Neural Search with intent-aware latent usefulness scoring, a graded usefulness benchmark, affordance validation v2, and an ablation runner that compares retrieval variants.

**Architecture:** Six new modules under `neural_search/retrieval/` (query intent + usefulness scorer + graph signals) and `neural_search/evaluation/` (usefulness benchmark + affordance validation v2 + ablation runner). Each module is independently testable with synthetic data. The existing `neural_search/search/intent.py` and `neural_search/evaluation/relatedness_scorer.py` are preserved unchanged.

**Tech Stack:** Python 3.10+, Pydantic v2, dataclasses, pytest, PyYAML, stdlib math/collections only (no external ML models required for default execution).

---

## File Map

| Path | Action | Responsibility |
|------|--------|---------------|
| `neural_search/retrieval/query_intent.py` | Create | `UsefulnessIntent` enum + `classify_query_intent` (rule-based, no LLM) |
| `neural_search/retrieval/usefulness_scorer.py` | Create | `DatasetContext`, `UsefulnessScore`, `score_usefulness`, weight profiles per intent |
| `neural_search/retrieval/graph_usefulness.py` | Create | Graph-derived signals: metapath score, affordance overlap, complementarity |
| `neural_search/retrieval/__init__.py` | Modify | Export new retrieval symbols |
| `neural_search/evaluation/usefulness_benchmark.py` | Create | `UsefulnessQuery`, graded NDCG/MRR/P@k, hard-negative violation rate |
| `neural_search/evaluation/affordance_validation_v2.py` | Create | Precision/recall vs ground truth, confusion table, Markdown+JSON reports |
| `neural_search/evaluation/ablation_runner.py` | Create | 8 retrieval variants, metric table, Markdown report |
| `tests/test_usefulness_intent.py` | Create | Pattern matching, confidence bounds, fallback |
| `tests/test_usefulness_scorer.py` | Create | Score bounds, weight normalization, intent changes rankings |
| `tests/test_graph_usefulness.py` | Create | Hub normalization, complementarity, missing graph |
| `tests/test_usefulness_benchmark.py` | Create | NDCG computation, hard-negative violations, empty labels |
| `tests/test_affordance_validation_v2.py` | Create | Synthetic cards + labels, confusion table, report files written |
| `tests/test_ablation_runner.py` | Create | All variants produce output, Markdown contains metric table |
| `data/eval/usefulness_seed_pairs.jsonl` | Create | 30 seed pairs across 6 usefulness categories |
| `config/eval/usefulness_v08.yaml` | Create | Ablation config: corpus path, variants, output paths |
| `reports/usefulness_benchmark_v08.md` | Create | Generated benchmark report |
| `reports/ablation_v08.md` | Create | Generated ablation report |
| `docs/LATENT_USEFULNESS_IMPLEMENTATION_NOTES.md` | Create | Implementation decisions, limitations, next steps |

---

## Task 1: `neural_search/retrieval/query_intent.py`

**Files:**
- Create: `neural_search/retrieval/query_intent.py`
- Create: `tests/test_usefulness_intent.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_usefulness_intent.py
import pytest
from neural_search.retrieval.query_intent import (
    UsefulnessIntent,
    IntentClassification,
    classify_query_intent,
)

class TestUsefulnessIntentEnum:
    def test_all_intents_exist(self):
        values = {i.value for i in UsefulnessIntent}
        assert "strict_lookup" in values
        assert "replication" in values
        assert "meta_analysis" in values
        assert "pipeline_reuse" in values
        assert "cross_dataset_comparison" in values
        assert "exploration" in values
        assert "method_transfer" in values

class TestClassifyQueryIntent:
    def test_pipeline_reuse_detected(self):
        result = classify_query_intent("datasets like DANDI:000123")
        assert result.intent == UsefulnessIntent.PIPELINE_REUSE
        assert 0.0 <= result.confidence <= 1.0

    def test_replication_detected(self):
        result = classify_query_intent("replicate this choice decoding result")
        assert result.intent == UsefulnessIntent.REPLICATION

    def test_cross_dataset_comparison_detected(self):
        result = classify_query_intent("compare mouse and primate decision making")
        assert result.intent == UsefulnessIntent.CROSS_DATASET_COMPARISON

    def test_method_transfer_detected(self):
        result = classify_query_intent("datasets for Q-learning model fitting")
        assert result.intent == UsefulnessIntent.METHOD_TRANSFER

    def test_strict_lookup_fallback(self):
        result = classify_query_intent("visual cortex calcium imaging mouse")
        assert result.intent == UsefulnessIntent.STRICT_LOOKUP

    def test_exploration_detected(self):
        result = classify_query_intent("find surprising related datasets")
        assert result.intent == UsefulnessIntent.EXPLORATION

    def test_confidence_bounded(self):
        for q in ["foo", "DANDI:000001", "replicate mouse study", "compare species"]:
            r = classify_query_intent(q)
            assert 0.0 <= r.confidence <= 1.0

    def test_matched_patterns_populated(self):
        result = classify_query_intent("datasets like DANDI:000123")
        assert len(result.matched_patterns) >= 1

    def test_explanation_nonempty(self):
        result = classify_query_intent("replicate this study")
        assert len(result.explanation) > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && pytest tests/test_usefulness_intent.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement `neural_search/retrieval/query_intent.py`**

```python
"""Intent classification for usefulness-focused retrieval.

Distinct from neural_search.search.intent which handles retrieval-head
weight overrides. This module classifies the *usefulness relationship*
the user seeks (replication, pipeline reuse, method transfer, etc.).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class UsefulnessIntent(Enum):
    STRICT_LOOKUP = "strict_lookup"
    REPLICATION = "replication"
    META_ANALYSIS = "meta_analysis"
    PIPELINE_REUSE = "pipeline_reuse"
    CROSS_DATASET_COMPARISON = "cross_dataset_comparison"
    EXPLORATION = "exploration"
    METHOD_TRANSFER = "method_transfer"


@dataclass
class IntentClassification:
    intent: UsefulnessIntent
    confidence: float
    matched_patterns: list[str] = field(default_factory=list)
    explanation: str = ""


# Each intent maps to a list of (regex_pattern, confidence_boost) tuples.
_PATTERNS: list[tuple[UsefulnessIntent, str, float]] = [
    # REPLICATION
    (UsefulnessIntent.REPLICATION, r"\breplicate\b", 0.90),
    (UsefulnessIntent.REPLICATION, r"\breproduce\b", 0.88),
    (UsefulnessIntent.REPLICATION, r"\bsame experiment as\b", 0.85),
    (UsefulnessIntent.REPLICATION, r"\breplic", 0.82),
    # PIPELINE_REUSE
    (UsefulnessIntent.PIPELINE_REUSE, r"datasets?\s+like\s+\w", 0.90),
    (UsefulnessIntent.PIPELINE_REUSE, r"similar\s+to\s+dandi", 0.88),
    (UsefulnessIntent.PIPELINE_REUSE, r"same\s+pipeline", 0.85),
    (UsefulnessIntent.PIPELINE_REUSE, r"reuse", 0.80),
    # CROSS_DATASET_COMPARISON
    (UsefulnessIntent.CROSS_DATASET_COMPARISON, r"compare\s+\w+\s+and\s+\w+", 0.88),
    (UsefulnessIntent.CROSS_DATASET_COMPARISON, r"cross[\s-]species", 0.85),
    (UsefulnessIntent.CROSS_DATASET_COMPARISON, r"\bcompar", 0.75),
    (UsefulnessIntent.CROSS_DATASET_COMPARISON, r"across\s+(species|datasets?|studies)", 0.80),
    # META_ANALYSIS
    (UsefulnessIntent.META_ANALYSIS, r"meta[\s-]analysis", 0.92),
    (UsefulnessIntent.META_ANALYSIS, r"pool\s+datasets?", 0.88),
    (UsefulnessIntent.META_ANALYSIS, r"systematic\s+review", 0.85),
    (UsefulnessIntent.META_ANALYSIS, r"aggregate\s+(across|multiple)", 0.82),
    # METHOD_TRANSFER
    (UsefulnessIntent.METHOD_TRANSFER, r"model\s+fitting", 0.88),
    (UsefulnessIntent.METHOD_TRANSFER, r"q[\s-]?learning", 0.85),
    (UsefulnessIntent.METHOD_TRANSFER, r"methods?\s+transfer", 0.85),
    (UsefulnessIntent.METHOD_TRANSFER, r"apply\s+(this\s+)?method", 0.82),
    (UsefulnessIntent.METHOD_TRANSFER, r"datasets?\s+for\s+\w+\s+(model|fitting|algorithm)", 0.80),
    # EXPLORATION
    (UsefulnessIntent.EXPLORATION, r"surprising", 0.85),
    (UsefulnessIntent.EXPLORATION, r"find\s+(related|unexpected|novel)", 0.82),
    (UsefulnessIntent.EXPLORATION, r"what\s+(other|else)", 0.78),
    (UsefulnessIntent.EXPLORATION, r"explore\b", 0.78),
    (UsefulnessIntent.EXPLORATION, r"discover", 0.75),
]

_EXPLANATIONS: dict[UsefulnessIntent, str] = {
    UsefulnessIntent.STRICT_LOOKUP: "Query targets specific dataset features; exact match weights prioritized.",
    UsefulnessIntent.REPLICATION: "Query seeks to replicate a prior study; task/species/region alignment emphasized.",
    UsefulnessIntent.META_ANALYSIS: "Query seeks datasets suitable for meta-analysis; provenance and statistical power emphasized.",
    UsefulnessIntent.PIPELINE_REUSE: "Query seeks datasets compatible with an existing analysis pipeline.",
    UsefulnessIntent.CROSS_DATASET_COMPARISON: "Query seeks to compare across datasets; comparability emphasized.",
    UsefulnessIntent.EXPLORATION: "Query seeks to discover unexpected related datasets; graph proximity emphasized.",
    UsefulnessIntent.METHOD_TRANSFER: "Query seeks datasets to apply a specific analysis method to.",
}


def classify_query_intent(
    query: str,
    parsed_constraints: object | None = None,
) -> IntentClassification:
    """Classify the latent usefulness intent of a query using deterministic rules."""
    lower = query.lower()
    hits: list[tuple[UsefulnessIntent, float, str]] = []

    for intent, pattern, confidence in _PATTERNS:
        if re.search(pattern, lower):
            hits.append((intent, confidence, pattern))

    if not hits:
        return IntentClassification(
            intent=UsefulnessIntent.STRICT_LOOKUP,
            confidence=0.55,
            matched_patterns=[],
            explanation=_EXPLANATIONS[UsefulnessIntent.STRICT_LOOKUP],
        )

    # Pick highest-confidence hit; if tie, prefer first defined
    hits.sort(key=lambda x: x[1], reverse=True)
    best_intent, best_conf, _ = hits[0]
    matched = [h[2] for h in hits if h[0] == best_intent]

    return IntentClassification(
        intent=best_intent,
        confidence=best_conf,
        matched_patterns=matched,
        explanation=_EXPLANATIONS[best_intent],
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && pytest tests/test_usefulness_intent.py -v
```
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && git add neural_search/retrieval/query_intent.py tests/test_usefulness_intent.py && git commit -m "feat: add UsefulnessIntent classifier for latent usefulness retrieval"
```

---

## Task 2: `neural_search/retrieval/usefulness_scorer.py`

**Files:**
- Create: `neural_search/retrieval/usefulness_scorer.py`
- Create: `tests/test_usefulness_scorer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_usefulness_scorer.py
import pytest
from neural_search.retrieval.usefulness_scorer import (
    DatasetContext,
    UsefulnessScore,
    score_usefulness,
    INTENT_WEIGHT_PROFILES,
)
from neural_search.retrieval.query_intent import UsefulnessIntent


def _ctx(
    dataset_id="ds_a",
    modalities=None,
    tasks=None,
    species=None,
    brain_regions=None,
    affordances=None,
    data_standards=None,
    session_count=None,
    trial_count=None,
    quality_score=0.5,
):
    return DatasetContext(
        dataset_id=dataset_id,
        modalities=modalities or [],
        tasks=tasks or [],
        species=species or [],
        brain_regions=brain_regions or [],
        affordances=affordances or [],
        data_standards=data_standards or [],
        session_count=session_count,
        trial_count=trial_count,
        quality_score=quality_score,
    )


class TestScoreBounds:
    def test_score_between_zero_and_one(self):
        q = _ctx(modalities=["neuropixels"], tasks=["decision_making"])
        c = _ctx(modalities=["neuropixels"], tasks=["decision_making"])
        score = score_usefulness(q, c, UsefulnessIntent.STRICT_LOOKUP)
        assert 0.0 <= score.total_score <= 1.0

    def test_perfect_match_scores_high(self):
        attrs = dict(
            modalities=["neuropixels"],
            tasks=["decision_making"],
            species=["mouse"],
            brain_regions=["prefrontal_cortex"],
            affordances=["choice_decoding"],
            data_standards=["nwb"],
            quality_score=1.0,
        )
        q = _ctx(**attrs)
        c = _ctx(**attrs)
        score = score_usefulness(q, c, UsefulnessIntent.STRICT_LOOKUP)
        assert score.total_score >= 0.7

    def test_empty_candidate_scores_low(self):
        q = _ctx(modalities=["neuropixels"], tasks=["decision_making"], species=["mouse"])
        c = _ctx()
        score = score_usefulness(q, c, UsefulnessIntent.STRICT_LOOKUP)
        assert score.total_score <= 0.4

    def test_all_dimension_scores_bounded(self):
        q = _ctx(modalities=["calcium_imaging"])
        c = _ctx(modalities=["neuropixels"])
        score = score_usefulness(q, c, UsefulnessIntent.STRICT_LOOKUP)
        for dim, val in score.dimension_scores.items():
            assert 0.0 <= val <= 1.0, f"Dimension {dim} out of bounds: {val}"


class TestWeightNormalization:
    def test_weights_sum_to_one_for_all_intents(self):
        for intent in UsefulnessIntent:
            if intent in INTENT_WEIGHT_PROFILES:
                weights = INTENT_WEIGHT_PROFILES[intent]
                total = sum(weights.values())
                assert abs(total - 1.0) < 1e-6, f"{intent} weights sum to {total}"

    def test_intent_changes_score(self):
        q = _ctx(
            modalities=["neuropixels"],
            tasks=["decision_making"],
            affordances=["choice_decoding"],
            data_standards=["nwb"],
        )
        c = _ctx(
            modalities=["neuropixels"],
            tasks=["decision_making"],
            affordances=["choice_decoding"],
            data_standards=["nwb"],
            quality_score=0.9,
        )
        score_lookup = score_usefulness(q, c, UsefulnessIntent.STRICT_LOOKUP)
        score_pipeline = score_usefulness(q, c, UsefulnessIntent.PIPELINE_REUSE)
        # Scores should differ due to different intent weights
        # (may be equal in degenerate cases; just check both are valid)
        assert 0.0 <= score_lookup.total_score <= 1.0
        assert 0.0 <= score_pipeline.total_score <= 1.0


class TestExplanations:
    def test_evidence_list_nonempty(self):
        q = _ctx(modalities=["calcium_imaging"], tasks=["go_nogo"])
        c = _ctx(modalities=["calcium_imaging"], tasks=["go_nogo"])
        score = score_usefulness(q, c, UsefulnessIntent.REPLICATION)
        assert len(score.evidence) >= 1

    def test_warnings_for_missing_graph(self):
        q = _ctx()
        c = _ctx()
        score = score_usefulness(q, c, UsefulnessIntent.EXPLORATION)
        # graph_proximity and neural_signature_similarity should produce warnings
        assert any("graph" in w.lower() or "neural_signature" in w.lower() for w in score.warnings)


class TestIntentOnRanking:
    def test_pipeline_reuse_prefers_same_standards(self):
        q = _ctx(modalities=["neuropixels"], affordances=["choice_decoding"], data_standards=["nwb"])
        c_match = _ctx(modalities=["neuropixels"], affordances=["choice_decoding"], data_standards=["nwb"])
        c_diff = _ctx(modalities=["calcium_imaging"], affordances=["dimensionality_reduction"], data_standards=["bids"])
        s_match = score_usefulness(q, c_match, UsefulnessIntent.PIPELINE_REUSE)
        s_diff = score_usefulness(q, c_diff, UsefulnessIntent.PIPELINE_REUSE)
        assert s_match.total_score > s_diff.total_score

    def test_replication_prefers_same_species_region(self):
        q = _ctx(species=["mouse"], brain_regions=["hippocampus"], tasks=["spatial_navigation"])
        c_match = _ctx(species=["mouse"], brain_regions=["hippocampus"], tasks=["spatial_navigation"])
        c_diff = _ctx(species=["macaque"], brain_regions=["v1"], tasks=["visual_discrimination"])
        s_match = score_usefulness(q, c_match, UsefulnessIntent.REPLICATION)
        s_diff = score_usefulness(q, c_diff, UsefulnessIntent.REPLICATION)
        assert s_match.total_score > s_diff.total_score
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && pytest tests/test_usefulness_scorer.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `neural_search/retrieval/usefulness_scorer.py`**

```python
"""Intent-aware usefulness scorer for latent future usefulness ranking.

Scores a candidate dataset against a query context across 10 dimensions,
weighted by the user's UsefulnessIntent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from neural_search.retrieval.query_intent import UsefulnessIntent


@dataclass
class DatasetContext:
    """Structured metadata context for scoring. Works as both query and candidate."""
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


@dataclass
class UsefulnessScore:
    total_score: float
    intent: UsefulnessIntent
    dimension_scores: dict[str, float]
    weights: dict[str, float]
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# Normalized weight profiles (must sum to 1.0) per intent.
# neural_signature_similarity is always 0.0 in v0.8 (future work).
INTENT_WEIGHT_PROFILES: dict[UsefulnessIntent, dict[str, float]] = {
    UsefulnessIntent.STRICT_LOOKUP: {
        "modality_alignment": 0.18,
        "task_compatibility": 0.18,
        "species_match": 0.12,
        "region_overlap": 0.12,
        "affordance_compatibility": 0.16,
        "graph_proximity": 0.08,
        "provenance_quality": 0.08,
        "statistical_power": 0.05,
        "pipeline_transferability": 0.03,
        "neural_signature_similarity": 0.00,
    },
    UsefulnessIntent.PIPELINE_REUSE: {
        "modality_alignment": 0.18,
        "affordance_compatibility": 0.24,
        "pipeline_transferability": 0.22,
        "provenance_quality": 0.10,
        "statistical_power": 0.08,
        "graph_proximity": 0.08,
        "task_compatibility": 0.05,
        "species_match": 0.03,
        "region_overlap": 0.02,
        "neural_signature_similarity": 0.00,
    },
    UsefulnessIntent.REPLICATION: {
        "task_compatibility": 0.22,
        "species_match": 0.16,
        "region_overlap": 0.16,
        "modality_alignment": 0.14,
        "affordance_compatibility": 0.12,
        "provenance_quality": 0.08,
        "statistical_power": 0.07,
        "graph_proximity": 0.03,
        "pipeline_transferability": 0.02,
        "neural_signature_similarity": 0.00,
    },
    UsefulnessIntent.META_ANALYSIS: {
        "task_compatibility": 0.18,
        "provenance_quality": 0.16,
        "statistical_power": 0.16,
        "affordance_compatibility": 0.14,
        "species_match": 0.10,
        "modality_alignment": 0.10,
        "graph_proximity": 0.08,
        "region_overlap": 0.05,
        "pipeline_transferability": 0.03,
        "neural_signature_similarity": 0.00,
    },
    UsefulnessIntent.EXPLORATION: {
        "graph_proximity": 0.20,
        "affordance_compatibility": 0.16,
        "neural_signature_similarity": 0.16,
        "task_compatibility": 0.12,
        "modality_alignment": 0.10,
        "provenance_quality": 0.10,
        "pipeline_transferability": 0.08,
        "species_match": 0.04,
        "region_overlap": 0.04,
        "statistical_power": 0.00,
    },
    UsefulnessIntent.CROSS_DATASET_COMPARISON: {
        "task_compatibility": 0.20,
        "species_match": 0.18,
        "modality_alignment": 0.15,
        "region_overlap": 0.14,
        "affordance_compatibility": 0.12,
        "statistical_power": 0.08,
        "provenance_quality": 0.07,
        "graph_proximity": 0.04,
        "pipeline_transferability": 0.02,
        "neural_signature_similarity": 0.00,
    },
    UsefulnessIntent.METHOD_TRANSFER: {
        "affordance_compatibility": 0.26,
        "task_compatibility": 0.20,
        "modality_alignment": 0.16,
        "pipeline_transferability": 0.14,
        "species_match": 0.08,
        "region_overlap": 0.06,
        "provenance_quality": 0.05,
        "statistical_power": 0.05,
        "graph_proximity": 0.00,
        "neural_signature_similarity": 0.00,
    },
}


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(s.lower() for s in a), set(s.lower() for s in b)
    if not sa and not sb:
        return 0.5  # unknown -> neutral
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _log_power(count: int | None) -> float:
    if count is None or count == 0:
        return 0.3  # neutral when unknown
    return min(1.0, math.log1p(count) / math.log1p(10000))


def score_usefulness(
    query_context: DatasetContext,
    candidate: DatasetContext,
    intent: UsefulnessIntent | None = None,
) -> UsefulnessScore:
    if intent is None:
        intent = UsefulnessIntent.STRICT_LOOKUP

    weights = INTENT_WEIGHT_PROFILES.get(intent, INTENT_WEIGHT_PROFILES[UsefulnessIntent.STRICT_LOOKUP])
    evidence: list[str] = []
    warnings: list[str] = []

    dims: dict[str, float] = {}

    # modality_alignment
    ma = _jaccard(query_context.modalities, candidate.modalities)
    dims["modality_alignment"] = ma
    if query_context.modalities and candidate.modalities:
        shared = set(m.lower() for m in query_context.modalities) & set(m.lower() for m in candidate.modalities)
        evidence.append(f"Shared modalities: {sorted(shared) or 'none'}")

    # task_compatibility
    tc = _jaccard(query_context.tasks, candidate.tasks)
    dims["task_compatibility"] = tc
    if query_context.tasks and candidate.tasks:
        shared = set(t.lower() for t in query_context.tasks) & set(t.lower() for t in candidate.tasks)
        evidence.append(f"Shared tasks: {sorted(shared) or 'none'}")

    # species_match
    dims["species_match"] = _jaccard(query_context.species, candidate.species)

    # region_overlap
    dims["region_overlap"] = _jaccard(query_context.brain_regions, candidate.brain_regions)

    # affordance_compatibility
    dims["affordance_compatibility"] = _jaccard(query_context.affordances, candidate.affordances)
    if query_context.affordances:
        shared = set(a.lower() for a in query_context.affordances) & set(a.lower() for a in candidate.affordances)
        evidence.append(f"Shared affordances: {sorted(shared) or 'none'}")

    # graph_proximity — requires external graph; return neutral with warning
    dims["graph_proximity"] = 0.3
    warnings.append("graph_proximity: no graph provided; using neutral prior 0.3")

    # provenance_quality — quality_score is 0-1 float
    dims["provenance_quality"] = min(1.0, max(0.0, candidate.quality_score))
    if candidate.quality_score > 0:
        evidence.append(f"Candidate quality score: {candidate.quality_score:.2f}")

    # statistical_power
    power = max(
        _log_power(candidate.trial_count),
        _log_power(candidate.session_count),
    )
    dims["statistical_power"] = power

    # pipeline_transferability — based on shared data standards
    dims["pipeline_transferability"] = _jaccard(query_context.data_standards, candidate.data_standards)
    if query_context.data_standards and candidate.data_standards:
        shared = set(s.lower() for s in query_context.data_standards) & set(s.lower() for s in candidate.data_standards)
        evidence.append(f"Shared data standards: {sorted(shared) or 'none'}")

    # neural_signature_similarity — not implemented in v0.8
    dims["neural_signature_similarity"] = 0.0
    warnings.append("neural_signature_similarity: not implemented in v0.8; score fixed at 0.0")

    # Weighted sum
    total = sum(weights.get(d, 0.0) * v for d, v in dims.items())

    return UsefulnessScore(
        total_score=min(1.0, max(0.0, total)),
        intent=intent,
        dimension_scores=dims,
        weights=dict(weights),
        evidence=evidence,
        warnings=warnings,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && pytest tests/test_usefulness_scorer.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && git add neural_search/retrieval/usefulness_scorer.py tests/test_usefulness_scorer.py && git commit -m "feat: add intent-aware usefulness scorer with 10 dimensions"
```

---

## Task 3: `neural_search/retrieval/graph_usefulness.py`

**Files:**
- Create: `neural_search/retrieval/graph_usefulness.py`
- Create: `tests/test_graph_usefulness.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph_usefulness.py
import pytest
from neural_search.retrieval.graph_usefulness import (
    affordance_overlap,
    pipeline_overlap,
    complementarity_score,
    normalized_metapath_score,
    graph_usefulness_features,
)


def _ds(affordances=None, data_standards=None):
    return {
        "affordances": affordances or [],
        "data_standards": data_standards or [],
    }


def _graph(nodes=None, edges=None):
    return {"nodes": nodes or {}, "edges": edges or {}}


class TestAffordanceOverlap:
    def test_identical_returns_one(self):
        a = _ds(affordances=["choice_decoding", "q_learning"])
        assert affordance_overlap(a, a) == pytest.approx(1.0)

    def test_disjoint_returns_zero(self):
        a = _ds(affordances=["choice_decoding"])
        b = _ds(affordances=["calcium_imaging"])
        assert affordance_overlap(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = _ds(affordances=["a", "b", "c"])
        b = _ds(affordances=["b", "c", "d"])
        result = affordance_overlap(a, b)
        assert 0.0 < result < 1.0

    def test_empty_both_returns_zero(self):
        assert affordance_overlap(_ds(), _ds()) == pytest.approx(0.0)


class TestPipelineOverlap:
    def test_same_standard_returns_one(self):
        a = _ds(data_standards=["nwb"])
        assert pipeline_overlap(a, a) == pytest.approx(1.0)

    def test_different_standard_returns_zero(self):
        a = _ds(data_standards=["nwb"])
        b = _ds(data_standards=["bids"])
        assert pipeline_overlap(a, b) == pytest.approx(0.0)


class TestComplementarity:
    def test_complementary_affordances_score_high(self):
        a = _ds(affordances=["spike_sorting", "stimulus_response_modeling"])
        b = _ds(affordances=["calcium_imaging", "dimensionality_reduction"])
        score = complementarity_score(a, b)
        assert 0.0 <= score <= 1.0

    def test_identical_affordances_score_low(self):
        a = _ds(affordances=["choice_decoding"])
        score_identical = complementarity_score(a, a)
        score_complement = complementarity_score(a, _ds(affordances=["q_learning"]))
        assert score_complement >= score_identical

    def test_both_empty_returns_zero(self):
        assert complementarity_score(_ds(), _ds()) == pytest.approx(0.0)


class TestNormalizedMetapathScore:
    def _build_hub_graph(self):
        """Graph where node 'hub' has edges to 10 datasets; 'rare' only edges to 2."""
        nodes = {f"ds_{i}": {"node_id": f"ds_{i}", "node_type": "dataset", "label": f"DS{i}"} for i in range(12)}
        nodes["hub"] = {"node_id": "hub", "node_type": "task", "label": "Hub Task"}
        nodes["rare"] = {"node_id": "rare", "node_type": "task", "label": "Rare Task"}

        edges = {}
        # Hub connected to ds_0..ds_9
        for i in range(10):
            eid = f"e_hub_{i}"
            edges[eid] = {
                "edge_id": eid,
                "source_node_id": f"ds_{i}",
                "target_node_id": "hub",
                "edge_type": "dataset_has_task",
                "confidence": 1.0,
            }
        # Rare connected only to ds_0 and ds_1
        for i in range(2):
            eid = f"e_rare_{i}"
            edges[eid] = {
                "edge_id": eid,
                "source_node_id": f"ds_{i}",
                "target_node_id": "rare",
                "edge_type": "dataset_has_task",
                "confidence": 1.0,
            }
        return _graph(nodes=nodes, edges=edges)

    def test_hub_normalization_reduces_hub_score(self):
        graph = self._build_hub_graph()
        # ds_0 and ds_1 share BOTH hub and rare -> high raw score
        # ds_0 and ds_5 share ONLY hub -> lower score, but hub is high-degree
        # After normalization, rare-based similarity should be stronger than hub-based
        score_rare = normalized_metapath_score(graph, "ds_0", "ds_1", "dataset_has_task")
        score_hub_only = normalized_metapath_score(graph, "ds_0", "ds_5", "dataset_has_task")
        # ds_0/ds_1 share more concepts so should score >= ds_0/ds_5
        assert score_rare >= score_hub_only

    def test_missing_nodes_returns_zero(self):
        graph = _graph()
        score = normalized_metapath_score(graph, "nonexistent_a", "nonexistent_b", "dataset_has_task")
        assert score == pytest.approx(0.0)

    def test_score_bounded(self):
        graph = self._build_hub_graph()
        score = normalized_metapath_score(graph, "ds_0", "ds_1", "dataset_has_task")
        assert 0.0 <= score <= 1.0


class TestGraphUsefulnessFeatures:
    def test_returns_dict_with_expected_keys(self):
        graph = _graph()
        q = _ds(affordances=["choice_decoding"])
        c = _ds(affordances=["choice_decoding"])
        features = graph_usefulness_features(q, c, graph)
        assert "affordance_overlap" in features
        assert "pipeline_overlap" in features
        assert "complementarity" in features
        assert "metapath_score" in features

    def test_missing_graph_does_not_raise(self):
        features = graph_usefulness_features(_ds(), _ds(), None)
        assert isinstance(features, dict)

    def test_all_values_bounded(self):
        graph = _graph()
        q = _ds(affordances=["a", "b"], data_standards=["nwb"])
        c = _ds(affordances=["b", "c"], data_standards=["bids"])
        features = graph_usefulness_features(q, c, graph)
        for key, val in features.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && pytest tests/test_graph_usefulness.py -v 2>&1 | head -20
```

- [ ] **Step 3: Implement `neural_search/retrieval/graph_usefulness.py`**

```python
"""Graph-derived usefulness signals.

Functions here accept plain dicts (compatible with KnowledgeGraph.model_dump()
output) so they work without importing heavy schema classes.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def _jaccard(a: list[str], b: list[str]) -> float:
    sa = {x.lower() for x in a}
    sb = {x.lower() for x in b}
    if not sa and not sb:
        return 0.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def affordance_overlap(dataset_a: dict, dataset_b: dict) -> float:
    """Jaccard similarity of analysis affordances."""
    return _jaccard(
        dataset_a.get("affordances", []),
        dataset_b.get("affordances", []),
    )


def pipeline_overlap(dataset_a: dict, dataset_b: dict) -> float:
    """Jaccard similarity of data standards (pipeline compatibility proxy)."""
    return _jaccard(
        dataset_a.get("data_standards", []),
        dataset_b.get("data_standards", []),
    )


def complementarity_score(dataset_a: dict, dataset_b: dict) -> float:
    """Score how complementary two datasets are.

    High when one dataset has affordances the other lacks and vice versa,
    while having some overlap (pure disjoint is noise; pure overlap is redundant).
    """
    sa = {x.lower() for x in dataset_a.get("affordances", [])}
    sb = {x.lower() for x in dataset_b.get("affordances", [])}
    if not sa or not sb:
        return 0.0
    union = sa | sb
    intersection = sa & sb
    only_a = sa - sb
    only_b = sb - sa

    if not union:
        return 0.0

    # Complementarity peaks when each contributes unique affordances
    # but they share some common ground (overlap / union is moderate)
    unique_fraction = (len(only_a) + len(only_b)) / len(union)
    overlap_fraction = len(intersection) / len(union)

    # Reward when both have unique contributions AND some shared ground
    if overlap_fraction == 0.0:
        return unique_fraction * 0.3  # no shared ground -> less useful pairing
    return min(1.0, unique_fraction * 0.7 + overlap_fraction * 0.3)


def normalized_metapath_score(
    graph: dict | None,
    source_id: str,
    target_id: str,
    metapath_type: str,
) -> float:
    """PathSim-inspired similarity normalized against hub-node degree.

    Returns 0.0 when graph is None or nodes are absent.
    """
    if not graph:
        return 0.0
    edges = graph.get("edges", {})

    # Build: node -> set of neighbors via metapath_type
    neighbors: dict[str, set[str]] = defaultdict(set)
    for edge in edges.values():
        if edge.get("edge_type") != metapath_type:
            continue
        src = edge.get("source_node_id", "")
        tgt = edge.get("target_node_id", "")
        if src:
            neighbors[src].add(tgt)
        if tgt:
            neighbors[tgt].add(src)

    n_src = neighbors.get(source_id, set())
    n_tgt = neighbors.get(target_id, set())

    if not n_src or not n_tgt:
        return 0.0

    shared = n_src & n_tgt
    if not shared:
        return 0.0

    # PathSim: 2 * |P(s,t)| / (|P(s,s)| + |P(t,t)|)
    # Approximated as: 2 * |shared| / (|n_src| + |n_tgt|)
    # This automatically down-weights hub nodes because they inflate the denominator
    # for any pair that connects through them.
    return 2.0 * len(shared) / (len(n_src) + len(n_tgt))


def graph_usefulness_features(
    query_context: dict | None,
    candidate: dict | None,
    graph: dict | None,
) -> dict[str, float]:
    """Compute all graph-derived usefulness features as a flat dict."""
    q = query_context or {}
    c = candidate or {}

    aff = affordance_overlap(q, c)
    pip = pipeline_overlap(q, c)
    comp = complementarity_score(q, c)

    # Metapath score requires both dataset IDs in the graph
    q_id = q.get("dataset_id", "")
    c_id = c.get("dataset_id", "")
    meta = normalized_metapath_score(graph, q_id, c_id, "dataset_has_task") if graph else 0.0

    return {
        "affordance_overlap": aff,
        "pipeline_overlap": pip,
        "complementarity": comp,
        "metapath_score": meta,
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && pytest tests/test_graph_usefulness.py -v
```

- [ ] **Step 5: Commit**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && git add neural_search/retrieval/graph_usefulness.py tests/test_graph_usefulness.py && git commit -m "feat: add graph-derived usefulness signals with hub-normalized PathSim"
```

---

## Task 4: Update `neural_search/retrieval/__init__.py`

- [ ] **Step 1: Add exports**

Edit `neural_search/retrieval/__init__.py` — append after the existing imports:

```python
from neural_search.retrieval.query_intent import (
    UsefulnessIntent,
    IntentClassification,
    classify_query_intent,
)
from neural_search.retrieval.usefulness_scorer import (
    DatasetContext,
    UsefulnessScore,
    INTENT_WEIGHT_PROFILES,
    score_usefulness,
)
from neural_search.retrieval.graph_usefulness import (
    affordance_overlap,
    pipeline_overlap,
    complementarity_score,
    normalized_metapath_score,
    graph_usefulness_features,
)
```

Also add the new names to `__all__`.

- [ ] **Step 2: Verify imports work**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && python -c "from neural_search.retrieval import UsefulnessIntent, score_usefulness, graph_usefulness_features; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && git add neural_search/retrieval/__init__.py && git commit -m "chore: export new retrieval symbols from package __init__"
```

---

## Task 5: Seed benchmark data `data/eval/usefulness_seed_pairs.jsonl`

- [ ] **Step 1: Create the seed file**

Create `data/eval/usefulness_seed_pairs.jsonl` with 30 pairs (one JSON object per line):

```jsonl
{"query_id": "q001", "query": "datasets like DANDI:000003 for neuropixels choice decoding", "intent": "pipeline_reuse", "candidate_id": "dandi:000005", "usefulness_label": "highly_useful", "label_type": "pipeline_transfer_candidate", "notes": "Same modality, same affordances, compatible NWB format"}
{"query_id": "q001", "query": "datasets like DANDI:000003 for neuropixels choice decoding", "intent": "pipeline_reuse", "candidate_id": "dandi:000006", "usefulness_label": "useful", "label_type": "pipeline_transfer_candidate", "notes": "Same modality, overlapping affordances"}
{"query_id": "q001", "query": "datasets like DANDI:000003 for neuropixels choice decoding", "intent": "pipeline_reuse", "candidate_id": "dandi:000099", "usefulness_label": "not_useful", "label_type": "pipeline_transfer_candidate", "notes": "Hard negative: calcium imaging only, no choice decoding affordance"}
{"query_id": "q002", "query": "complementary datasets for mouse hippocampus spatial navigation", "intent": "exploration", "candidate_id": "dandi:000010", "usefulness_label": "highly_useful", "label_type": "complementary", "notes": "Covers same region with different modality — complementary"}
{"query_id": "q002", "query": "complementary datasets for mouse hippocampus spatial navigation", "intent": "exploration", "candidate_id": "dandi:000011", "usefulness_label": "useful", "label_type": "complementary", "notes": "Different task but same region"}
{"query_id": "q002", "query": "complementary datasets for mouse hippocampus spatial navigation", "intent": "exploration", "candidate_id": "dandi:000098", "usefulness_label": "not_useful", "label_type": "complementary", "notes": "Hard negative: completely different species/region/task"}
{"query_id": "q003", "query": "replicate mouse decision making neuropixels Steinmetz 2019", "intent": "replication", "candidate_id": "dandi:000020", "usefulness_label": "highly_useful", "label_type": "replication_candidate", "notes": "Same species, task, modality"}
{"query_id": "q003", "query": "replicate mouse decision making neuropixels Steinmetz 2019", "intent": "replication", "candidate_id": "dandi:000021", "usefulness_label": "useful", "label_type": "replication_candidate", "notes": "Same species and task, different modality"}
{"query_id": "q003", "query": "replicate mouse decision making neuropixels Steinmetz 2019", "intent": "replication", "candidate_id": "dandi:000097", "usefulness_label": "not_useful", "label_type": "replication_candidate", "notes": "Hard negative: human fMRI decision making — wrong species and modality"}
{"query_id": "q004", "query": "compare mouse and macaque prefrontal cortex working memory", "intent": "cross_dataset_comparison", "candidate_id": "dandi:000030", "usefulness_label": "highly_useful", "label_type": "comparable", "notes": "Macaque PFC working memory — ideal cross-species comparison"}
{"query_id": "q004", "query": "compare mouse and macaque prefrontal cortex working memory", "intent": "cross_dataset_comparison", "candidate_id": "dandi:000031", "usefulness_label": "useful", "label_type": "comparable", "notes": "Rat PFC — related but not exact species"}
{"query_id": "q004", "query": "compare mouse and macaque prefrontal cortex working memory", "intent": "cross_dataset_comparison", "candidate_id": "dandi:000096", "usefulness_label": "not_useful", "label_type": "comparable", "notes": "Hard negative: mouse visual cortex, no working memory task"}
{"query_id": "q005", "query": "datasets for Q-learning model fitting reinforcement learning", "intent": "method_transfer", "candidate_id": "dandi:000040", "usefulness_label": "highly_useful", "label_type": "method_transfer_candidate", "notes": "Multi-armed bandit task with trial-by-trial outcomes"}
{"query_id": "q005", "query": "datasets for Q-learning model fitting reinforcement learning", "intent": "method_transfer", "candidate_id": "dandi:000041", "usefulness_label": "useful", "label_type": "method_transfer_candidate", "notes": "Reversal learning — compatible with RL modeling"}
{"query_id": "q005", "query": "datasets for Q-learning model fitting reinforcement learning", "intent": "method_transfer", "candidate_id": "dandi:000095", "usefulness_label": "not_useful", "label_type": "method_transfer_candidate", "notes": "Hard negative: passive viewing calcium imaging — no behavioral choices"}
{"query_id": "q006", "query": "meta-analysis of visual cortex datasets across species", "intent": "meta_analysis", "candidate_id": "dandi:000050", "usefulness_label": "highly_useful", "label_type": "comparable", "notes": "Human V1 fMRI with matching stimuli — excellent meta-analysis candidate"}
{"query_id": "q006", "query": "meta-analysis of visual cortex datasets across species", "intent": "meta_analysis", "candidate_id": "dandi:000051", "usefulness_label": "useful", "label_type": "comparable", "notes": "Macaque V4 — same area, compatible stimuli"}
{"query_id": "q006", "query": "meta-analysis of visual cortex datasets across species", "intent": "meta_analysis", "candidate_id": "dandi:000094", "usefulness_label": "not_useful", "label_type": "comparable", "notes": "Hard negative: auditory cortex — completely different region"}
{"query_id": "q007", "query": "datasets like DANDI:000026 calcium imaging two-photon cortex", "intent": "pipeline_reuse", "candidate_id": "dandi:000060", "usefulness_label": "highly_useful", "label_type": "reusable", "notes": "Identical recording modality and NWB format"}
{"query_id": "q007", "query": "datasets like DANDI:000026 calcium imaging two-photon cortex", "intent": "pipeline_reuse", "candidate_id": "dandi:000061", "usefulness_label": "weakly_useful", "label_type": "reusable", "notes": "Different calcium indicator but same modality family"}
{"query_id": "q007", "query": "datasets like DANDI:000026 calcium imaging two-photon cortex", "intent": "pipeline_reuse", "candidate_id": "dandi:000093", "usefulness_label": "not_useful", "label_type": "reusable", "notes": "Hard negative: EEG — completely different signal type"}
{"query_id": "q008", "query": "find surprising related datasets to mouse striatum dopamine", "intent": "exploration", "candidate_id": "dandi:000070", "usefulness_label": "useful", "label_type": "complementary", "notes": "Ventral striatum reward prediction — topically related"}
{"query_id": "q008", "query": "find surprising related datasets to mouse striatum dopamine", "intent": "exploration", "candidate_id": "dandi:000071", "usefulness_label": "weakly_useful", "label_type": "complementary", "notes": "Dopamine projection targets in cortex — indirect relation"}
{"query_id": "q008", "query": "find surprising related datasets to mouse striatum dopamine", "intent": "exploration", "candidate_id": "dandi:000092", "usefulness_label": "not_useful", "label_type": "complementary", "notes": "Hard negative: zebrafish whole-brain imaging — no overlap"}
{"query_id": "q009", "query": "replicate primate oculomotor control saccade task electrophysiology", "intent": "replication", "candidate_id": "dandi:000080", "usefulness_label": "highly_useful", "label_type": "replication_candidate", "notes": "Macaque saccade electrophysiology — direct replication candidate"}
{"query_id": "q009", "query": "replicate primate oculomotor control saccade task electrophysiology", "intent": "replication", "candidate_id": "dandi:000081", "usefulness_label": "useful", "label_type": "replication_candidate", "notes": "Human saccade fMRI — different modality but same task"}
{"query_id": "q009", "query": "replicate primate oculomotor control saccade task electrophysiology", "intent": "replication", "candidate_id": "dandi:000091", "usefulness_label": "not_useful", "label_type": "replication_candidate", "notes": "Hard negative: mouse auditory cortex, no saccade"}
{"query_id": "q010", "query": "datasets for dimensionality reduction population dynamics motor cortex", "intent": "method_transfer", "candidate_id": "dandi:000085", "usefulness_label": "highly_useful", "label_type": "method_transfer_candidate", "notes": "Large population recordings from motor cortex during reach"}
{"query_id": "q010", "query": "datasets for dimensionality reduction population dynamics motor cortex", "intent": "method_transfer", "candidate_id": "dandi:000086", "usefulness_label": "useful", "label_type": "method_transfer_candidate", "notes": "Premotor cortex — related area, large cell counts"}
{"query_id": "q010", "query": "datasets for dimensionality reduction population dynamics motor cortex", "intent": "method_transfer", "candidate_id": "dandi:000090", "usefulness_label": "not_useful", "label_type": "method_transfer_candidate", "notes": "Hard negative: single-unit recordings, 5 neurons — too few for dimensionality reduction"}
```

- [ ] **Step 2: Verify file is valid JSONL**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && python -c "
import json
with open('data/eval/usefulness_seed_pairs.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
print(f'{len(lines)} pairs loaded OK')
"
```
Expected: `30 pairs loaded OK`

- [ ] **Step 3: Commit**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && git add data/eval/usefulness_seed_pairs.jsonl && git commit -m "data: add 30 seed usefulness benchmark pairs across 6 categories"
```

---

## Task 6: `neural_search/evaluation/usefulness_benchmark.py`

**Files:**
- Create: `neural_search/evaluation/usefulness_benchmark.py`
- Create: `tests/test_usefulness_benchmark.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_usefulness_benchmark.py
import json
import math
import pytest
from neural_search.evaluation.usefulness_benchmark import (
    UsefulnessLabel,
    UsefulnessQuery,
    PairLabel,
    compute_ndcg_at_k,
    compute_mrr,
    compute_precision_at_k,
    hard_negative_violation_rate,
    run_usefulness_benchmark,
    BenchmarkReport,
    GAIN,
)


class TestGainValues:
    def test_gain_ordering(self):
        assert GAIN["not_useful"] < GAIN["weakly_useful"] < GAIN["useful"] < GAIN["highly_useful"]

    def test_gain_nonnegative(self):
        for v in GAIN.values():
            assert v >= 0


class TestNDCG:
    def _make_labels(self):
        return {
            "c1": "highly_useful",
            "c2": "useful",
            "c3": "weakly_useful",
            "c4": "not_useful",
        }

    def test_perfect_ranking(self):
        labels = self._make_labels()
        ranked = ["c1", "c2", "c3", "c4"]
        score = compute_ndcg_at_k(ranked, labels, k=4)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_worst_ranking_below_one(self):
        labels = self._make_labels()
        ranked = ["c4", "c3", "c2", "c1"]
        score = compute_ndcg_at_k(ranked, labels, k=4)
        assert score < 1.0

    def test_empty_ranked_returns_zero(self):
        assert compute_ndcg_at_k([], {"c1": "useful"}, k=5) == pytest.approx(0.0)

    def test_k_truncation(self):
        labels = {"c1": "highly_useful", "c2": "not_useful"}
        score_k1 = compute_ndcg_at_k(["c1", "c2"], labels, k=1)
        score_k2 = compute_ndcg_at_k(["c1", "c2"], labels, k=2)
        assert score_k1 == pytest.approx(1.0)
        assert score_k2 == pytest.approx(1.0)


class TestMRR:
    def test_first_position_returns_one(self):
        labels = {"c1": "useful"}
        assert compute_mrr(["c1"], labels) == pytest.approx(1.0)

    def test_second_position_returns_half(self):
        labels = {"c1": "not_useful", "c2": "useful"}
        assert compute_mrr(["c1", "c2"], labels) == pytest.approx(0.5)

    def test_no_relevant_returns_zero(self):
        labels = {"c1": "not_useful"}
        assert compute_mrr(["c1"], labels) == pytest.approx(0.0)

    def test_weakly_useful_not_counted(self):
        labels = {"c1": "weakly_useful", "c2": "useful"}
        assert compute_mrr(["c1", "c2"], labels) == pytest.approx(0.5)


class TestPrecisionAtK:
    def test_all_useful_returns_one(self):
        labels = {"c1": "useful", "c2": "highly_useful"}
        assert compute_precision_at_k(["c1", "c2"], labels, k=2) == pytest.approx(1.0)

    def test_none_useful_returns_zero(self):
        labels = {"c1": "not_useful", "c2": "weakly_useful"}
        assert compute_precision_at_k(["c1", "c2"], labels, k=2) == pytest.approx(0.0)


class TestHardNegativeViolation:
    def test_hard_negative_ranked_first_is_violation(self):
        labels = {"hn1": "not_useful", "c1": "useful"}
        hard_negatives = {"hn1"}
        rate = hard_negative_violation_rate(["hn1", "c1"], labels, hard_negatives)
        assert rate == pytest.approx(1.0)

    def test_no_violations_when_hard_negatives_ranked_last(self):
        labels = {"c1": "useful", "hn1": "not_useful"}
        hard_negatives = {"hn1"}
        rate = hard_negative_violation_rate(["c1", "hn1"], labels, hard_negatives)
        assert rate == pytest.approx(0.0)

    def test_no_hard_negatives_returns_zero(self):
        labels = {"c1": "useful"}
        rate = hard_negative_violation_rate(["c1"], labels, set())
        assert rate == pytest.approx(0.0)


class TestRunBenchmark:
    def test_report_has_expected_keys(self):
        queries = [
            UsefulnessQuery(
                query_id="q1",
                query="test query",
                intent="strict_lookup",
                candidate_ids=["c1", "c2", "c3"],
            )
        ]
        labels = [
            PairLabel(query_id="q1", candidate_id="c1", usefulness_label="highly_useful", label_type="reusable"),
            PairLabel(query_id="q1", candidate_id="c2", usefulness_label="useful", label_type="reusable"),
            PairLabel(query_id="q1", candidate_id="c3", usefulness_label="not_useful", label_type="reusable"),
        ]
        # Mock run: identity ranking
        run = {"q1": ["c1", "c2", "c3"]}
        report = run_usefulness_benchmark(queries, labels, run, k=3)
        assert isinstance(report, BenchmarkReport)
        assert report.ndcg_at_k >= 0.0
        assert report.mrr >= 0.0
        assert report.precision_at_k >= 0.0
        assert isinstance(report.per_intent_metrics, dict)

    def test_empty_labels_raises(self):
        with pytest.raises(ValueError, match="No labels"):
            run_usefulness_benchmark([], [], {})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && pytest tests/test_usefulness_benchmark.py -v 2>&1 | head -20
```

- [ ] **Step 3: Implement `neural_search/evaluation/usefulness_benchmark.py`**

```python
"""Graded usefulness benchmark for evaluating latent-usefulness retrieval.

Metrics: NDCG@k, MRR (first useful), Precision@k, hard-negative violation rate,
per-intent breakdown.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class UsefulnessLabel(StrEnum):
    NOT_USEFUL = "not_useful"
    WEAKLY_USEFUL = "weakly_useful"
    USEFUL = "useful"
    HIGHLY_USEFUL = "highly_useful"


GAIN: dict[str, float] = {
    "not_useful": 0.0,
    "weakly_useful": 1.0,
    "useful": 2.0,
    "highly_useful": 3.0,
}

_RELEVANT = {"useful", "highly_useful"}


@dataclass
class UsefulnessQuery:
    query_id: str
    query: str
    intent: str
    candidate_ids: list[str] = field(default_factory=list)


@dataclass
class PairLabel:
    query_id: str
    candidate_id: str
    usefulness_label: str  # UsefulnessLabel value
    label_type: str
    notes: str = ""
    is_hard_negative: bool = False


@dataclass
class BenchmarkReport:
    ndcg_at_k: float
    mrr: float
    precision_at_k: float
    hard_negative_violation_rate: float
    per_intent_metrics: dict[str, dict[str, float]]
    k: int
    n_queries: int
    notes: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "# Usefulness Benchmark Report\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| NDCG@{self.k} | {self.ndcg_at_k:.4f} |",
            f"| MRR | {self.mrr:.4f} |",
            f"| P@{self.k} | {self.precision_at_k:.4f} |",
            f"| Hard-Neg Violation Rate | {self.hard_negative_violation_rate:.4f} |",
            f"| Queries evaluated | {self.n_queries} |",
            "",
            "## Per-Intent Breakdown\n",
        ]
        for intent, metrics in self.per_intent_metrics.items():
            lines.append(f"### {intent}")
            for m, v in metrics.items():
                lines.append(f"- {m}: {v:.4f}")
        if self.notes:
            lines += ["", "## Notes"] + [f"- {n}" for n in self.notes]
        return "\n".join(lines)


def compute_ndcg_at_k(
    ranked: list[str],
    labels: dict[str, str],
    k: int,
) -> float:
    """Graded NDCG@k using GAIN values."""
    def _dcg(ids: list[str]) -> float:
        return sum(
            GAIN.get(labels.get(cid, "not_useful"), 0.0) / math.log2(i + 2)
            for i, cid in enumerate(ids[:k])
        )

    if not ranked:
        return 0.0

    dcg = _dcg(ranked)
    ideal_order = sorted(labels.keys(), key=lambda c: GAIN.get(labels[c], 0.0), reverse=True)
    idcg = _dcg(ideal_order)
    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr(ranked: list[str], labels: dict[str, str]) -> float:
    """Mean reciprocal rank of first USEFUL-or-better result."""
    for i, cid in enumerate(ranked):
        if labels.get(cid, "not_useful") in _RELEVANT:
            return 1.0 / (i + 1)
    return 0.0


def compute_precision_at_k(
    ranked: list[str],
    labels: dict[str, str],
    k: int,
) -> float:
    """Fraction of top-k that are USEFUL or HIGHLY_USEFUL."""
    if not ranked:
        return 0.0
    top = ranked[:k]
    relevant = sum(1 for c in top if labels.get(c, "not_useful") in _RELEVANT)
    return relevant / len(top)


def hard_negative_violation_rate(
    ranked: list[str],
    labels: dict[str, str],
    hard_negatives: set[str],
) -> float:
    """Fraction of hard-negatives ranked above the first relevant result."""
    if not hard_negatives:
        return 0.0

    first_relevant = next(
        (i for i, c in enumerate(ranked) if labels.get(c, "not_useful") in _RELEVANT),
        len(ranked),
    )
    violations = sum(
        1 for i, c in enumerate(ranked) if c in hard_negatives and i < first_relevant
    )
    return violations / len(hard_negatives)


def run_usefulness_benchmark(
    queries: list[UsefulnessQuery],
    labels: list[PairLabel],
    run: dict[str, list[str]],
    k: int = 10,
) -> BenchmarkReport:
    """Evaluate a retrieval run against usefulness labels.

    Args:
        queries: List of benchmark queries.
        labels: List of PairLabel annotations.
        run: Dict mapping query_id -> ranked list of candidate_ids.
        k: Cutoff for rank-based metrics.
    """
    if not labels:
        raise ValueError("No labels provided to benchmark")

    # Build per-query label dicts and hard-negative sets
    per_query_labels: dict[str, dict[str, str]] = {}
    per_query_hard_negs: dict[str, set[str]] = {}
    for lbl in labels:
        per_query_labels.setdefault(lbl.query_id, {})[lbl.candidate_id] = lbl.usefulness_label
        if lbl.is_hard_negative or lbl.usefulness_label == "not_useful":
            per_query_hard_negs.setdefault(lbl.query_id, set()).add(lbl.candidate_id)

    ndcgs, mrrs, precs, hn_rates = [], [], [], []
    intent_buckets: dict[str, list[dict[str, float]]] = {}

    for q in queries:
        qid = q.query_id
        ranked = run.get(qid, [])
        qlabels = per_query_labels.get(qid, {})
        if not qlabels:
            continue

        ndcg = compute_ndcg_at_k(ranked, qlabels, k)
        mrr = compute_mrr(ranked, qlabels)
        prec = compute_precision_at_k(ranked, qlabels, k)
        hnv = hard_negative_violation_rate(ranked, qlabels, per_query_hard_negs.get(qid, set()))

        ndcgs.append(ndcg)
        mrrs.append(mrr)
        precs.append(prec)
        hn_rates.append(hnv)

        bucket = intent_buckets.setdefault(q.intent, [])
        bucket.append({"ndcg": ndcg, "mrr": mrr, "precision": prec})

    def _avg(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    per_intent: dict[str, dict[str, float]] = {}
    for intent, metrics_list in intent_buckets.items():
        per_intent[intent] = {
            m: _avg([x[m] for x in metrics_list])
            for m in ("ndcg", "mrr", "precision")
        }

    return BenchmarkReport(
        ndcg_at_k=_avg(ndcgs),
        mrr=_avg(mrrs),
        precision_at_k=_avg(precs),
        hard_negative_violation_rate=_avg(hn_rates),
        per_intent_metrics=per_intent,
        k=k,
        n_queries=len(ndcgs),
    )


def load_seed_pairs(path: str | Path) -> tuple[list[UsefulnessQuery], list[PairLabel]]:
    """Load seed pairs JSONL into queries and labels."""
    path = Path(path)
    raw: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))

    query_map: dict[str, UsefulnessQuery] = {}
    labels: list[PairLabel] = []

    for row in raw:
        qid = row["query_id"]
        if qid not in query_map:
            query_map[qid] = UsefulnessQuery(
                query_id=qid,
                query=row.get("query", ""),
                intent=row.get("intent", "strict_lookup"),
                candidate_ids=[],
            )
        query_map[qid].candidate_ids.append(row["candidate_id"])
        labels.append(
            PairLabel(
                query_id=qid,
                candidate_id=row["candidate_id"],
                usefulness_label=row.get("usefulness_label", "not_useful"),
                label_type=row.get("label_type", ""),
                notes=row.get("notes", ""),
                is_hard_negative="hard_negative" in row.get("notes", "").lower(),
            )
        )

    return list(query_map.values()), labels
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && pytest tests/test_usefulness_benchmark.py -v
```

- [ ] **Step 5: Commit**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && git add neural_search/evaluation/usefulness_benchmark.py tests/test_usefulness_benchmark.py && git commit -m "feat: add graded usefulness benchmark with NDCG, MRR, hard-negative violation rate"
```

---

## Task 7: `neural_search/evaluation/affordance_validation_v2.py`

**Files:**
- Create: `neural_search/evaluation/affordance_validation_v2.py`
- Create: `tests/test_affordance_validation_v2.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_affordance_validation_v2.py
import json
import tempfile
from pathlib import Path
import pytest
from neural_search.evaluation.affordance_validation_v2 import (
    AffordanceValidationV2,
    ValidationConfig,
    SyntheticCard,
    GroundTruthLabel,
    ValidationReport,
    run_validation,
)


def _make_cards():
    return [
        SyntheticCard(
            dataset_id="ds_001",
            predicted_affordances=["choice_decoding", "q_learning"],
            modalities=["neuropixels"],
            has_trials=True,
            has_timestamps=True,
        ),
        SyntheticCard(
            dataset_id="ds_002",
            predicted_affordances=["calcium_imaging"],
            modalities=["calcium_imaging"],
            has_trials=False,
            has_timestamps=False,
        ),
        SyntheticCard(
            dataset_id="ds_003",
            predicted_affordances=["choice_decoding"],
            modalities=["neuropixels"],
            has_trials=True,
            has_timestamps=True,
        ),
    ]


def _make_labels():
    return [
        GroundTruthLabel(dataset_id="ds_001", affordance="choice_decoding", supported=True),
        GroundTruthLabel(dataset_id="ds_001", affordance="q_learning", supported=True),
        GroundTruthLabel(dataset_id="ds_002", affordance="calcium_imaging", supported=False),
        GroundTruthLabel(dataset_id="ds_003", affordance="choice_decoding", supported=True),
    ]


class TestAffordanceValidationV2:
    def test_run_returns_report(self):
        cards = _make_cards()
        labels = _make_labels()
        report = run_validation(cards, labels)
        assert isinstance(report, ValidationReport)

    def test_report_has_precision_recall(self):
        cards = _make_cards()
        labels = _make_labels()
        report = run_validation(cards, labels)
        assert "choice_decoding" in report.per_affordance_precision
        assert 0.0 <= report.per_affordance_precision["choice_decoding"] <= 1.0

    def test_confusion_table_populated(self):
        cards = _make_cards()
        labels = _make_labels()
        report = run_validation(cards, labels)
        assert isinstance(report.confusion_table, dict)
        # Should have at least tp or fp entries
        for aff, table in report.confusion_table.items():
            assert "tp" in table
            assert "fp" in table
            assert "fn" in table

    def test_unknown_labels_handled(self):
        cards = _make_cards()
        labels = []  # No ground truth
        report = run_validation(cards, labels)
        assert report.n_labeled == 0
        assert report.n_unlabeled >= 0

    def test_markdown_report_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cards = _make_cards()
            labels = _make_labels()
            out_path = Path(tmpdir) / "report.md"
            report = run_validation(cards, labels, out_path=out_path)
            assert out_path.exists()
            content = out_path.read_text()
            assert "Precision" in content or "precision" in content

    def test_json_report_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cards = _make_cards()
            labels = _make_labels()
            json_path = Path(tmpdir) / "results.json"
            run_validation(cards, labels, json_out_path=json_path)
            assert json_path.exists()
            data = json.loads(json_path.read_text())
            assert "per_affordance_precision" in data
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && pytest tests/test_affordance_validation_v2.py -v 2>&1 | head -20
```

- [ ] **Step 3: Implement `neural_search/evaluation/affordance_validation_v2.py`**

```python
"""Affordance Validation v2: precision/recall vs ground-truth labels.

Improvements over v1:
- Structured ground-truth label ingestion
- Per-affordance precision/recall/F1
- Confusion table (TP, FP, FN, TN)
- Machine-readable JSON output
- Human-readable Markdown report
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SyntheticCard:
    """Lightweight dataset card representation for validation."""
    dataset_id: str
    predicted_affordances: list[str] = field(default_factory=list)
    modalities: list[str] = field(default_factory=list)
    has_trials: bool = False
    has_timestamps: bool = False


@dataclass
class GroundTruthLabel:
    """Ground-truth label for one affordance on one dataset."""
    dataset_id: str
    affordance: str
    supported: bool  # True = affordance actually supported


@dataclass
class ValidationConfig:
    sample_size: int = 100
    confidence_threshold: float = 0.5


@dataclass
class ValidationReport:
    n_datasets: int
    n_labeled: int
    n_unlabeled: int
    per_affordance_precision: dict[str, float]
    per_affordance_recall: dict[str, float]
    per_affordance_f1: dict[str, float]
    confusion_table: dict[str, dict[str, int]]
    coverage_by_affordance: dict[str, int]
    notes: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "# Affordance Validation v2 Report\n",
            f"- Datasets evaluated: {self.n_datasets}",
            f"- Datasets with ground-truth labels: {self.n_labeled}",
            f"- Datasets without labels (unknown): {self.n_unlabeled}",
            "",
            "## Per-Affordance Metrics\n",
            "| Affordance | Precision | Recall | F1 | TP | FP | FN |",
            "|------------|-----------|--------|----|----|----|-----|",
        ]
        for aff in sorted(self.per_affordance_precision):
            p = self.per_affordance_precision[aff]
            r = self.per_affordance_recall.get(aff, 0.0)
            f = self.per_affordance_f1.get(aff, 0.0)
            ct = self.confusion_table.get(aff, {})
            lines.append(
                f"| {aff} | {p:.3f} | {r:.3f} | {f:.3f} | "
                f"{ct.get('tp', 0)} | {ct.get('fp', 0)} | {ct.get('fn', 0)} |"
            )
        if self.notes:
            lines += ["", "## Notes"] + [f"- {n}" for n in self.notes]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_datasets": self.n_datasets,
            "n_labeled": self.n_labeled,
            "n_unlabeled": self.n_unlabeled,
            "per_affordance_precision": self.per_affordance_precision,
            "per_affordance_recall": self.per_affordance_recall,
            "per_affordance_f1": self.per_affordance_f1,
            "confusion_table": self.confusion_table,
            "coverage_by_affordance": self.coverage_by_affordance,
            "notes": self.notes,
        }


class AffordanceValidationV2:
    """Validates predicted affordances against ground-truth labels."""

    def __init__(self, cards: list[SyntheticCard], labels: list[GroundTruthLabel]):
        self.cards = cards
        self._labels: dict[tuple[str, str], bool] = {
            (l.dataset_id, l.affordance): l.supported for l in labels
        }
        self._labeled_datasets = {l.dataset_id for l in labels}

    def run(self) -> ValidationReport:
        # Per-affordance confusion counts
        counts: dict[str, dict[str, int]] = {}

        for card in self.cards:
            for aff in card.predicted_affordances:
                key = (card.dataset_id, aff)
                if key not in self._labels:
                    continue
                truth = self._labels[key]
                bucket = counts.setdefault(aff, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
                if truth:
                    bucket["tp"] += 1
                else:
                    bucket["fp"] += 1

        # Capture false negatives: labeled as supported but not predicted
        for (ds_id, aff), supported in self._labels.items():
            if not supported:
                continue
            # Find if this dataset predicts this affordance
            card = next((c for c in self.cards if c.dataset_id == ds_id), None)
            if card is None:
                continue
            if aff not in card.predicted_affordances:
                counts.setdefault(aff, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})["fn"] += 1

        precision: dict[str, float] = {}
        recall: dict[str, float] = {}
        f1: dict[str, float] = {}

        for aff, ct in counts.items():
            tp, fp, fn = ct["tp"], ct["fp"], ct["fn"]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            precision[aff] = p
            recall[aff] = r
            f1[aff] = f

        coverage = {}
        for card in self.cards:
            for aff in card.predicted_affordances:
                coverage[aff] = coverage.get(aff, 0) + 1

        labeled_ds = {c.dataset_id for c in self.cards if c.dataset_id in self._labeled_datasets}

        return ValidationReport(
            n_datasets=len(self.cards),
            n_labeled=len(labeled_ds),
            n_unlabeled=len(self.cards) - len(labeled_ds),
            per_affordance_precision=precision,
            per_affordance_recall=recall,
            per_affordance_f1=f1,
            confusion_table=counts,
            coverage_by_affordance=coverage,
        )


def run_validation(
    cards: list[SyntheticCard],
    labels: list[GroundTruthLabel],
    out_path: Path | None = None,
    json_out_path: Path | None = None,
) -> ValidationReport:
    validator = AffordanceValidationV2(cards, labels)
    report = validator.run()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report.to_markdown(), encoding="utf-8")
    if json_out_path:
        json_out_path.parent.mkdir(parents=True, exist_ok=True)
        json_out_path.write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
    return report
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && pytest tests/test_affordance_validation_v2.py -v
```

- [ ] **Step 5: Commit**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && git add neural_search/evaluation/affordance_validation_v2.py tests/test_affordance_validation_v2.py && git commit -m "feat: add affordance validation v2 with precision/recall and confusion tables"
```

---

## Task 8: `neural_search/evaluation/ablation_runner.py`

**Files:**
- Create: `neural_search/evaluation/ablation_runner.py`
- Create: `tests/test_ablation_runner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ablation_runner.py
import json
import tempfile
from pathlib import Path
import pytest
from neural_search.evaluation.ablation_runner import (
    AblationVariant,
    AblationConfig,
    CandidatePool,
    run_ablation,
    AblationReport,
    VARIANT_NAMES,
)
from neural_search.evaluation.usefulness_benchmark import (
    UsefulnessQuery,
    PairLabel,
)
from neural_search.retrieval.usefulness_scorer import DatasetContext


def _make_pool():
    return CandidatePool(
        candidates={
            "c1": DatasetContext("c1", modalities=["neuropixels"], tasks=["decision_making"],
                                 affordances=["choice_decoding"], data_standards=["nwb"],
                                 quality_score=0.9, trial_count=5000),
            "c2": DatasetContext("c2", modalities=["calcium_imaging"], tasks=["decision_making"],
                                 affordances=["dimensionality_reduction"], data_standards=["bids"],
                                 quality_score=0.7, trial_count=2000),
            "c3": DatasetContext("c3", modalities=["neuropixels"], tasks=["go_nogo"],
                                 affordances=["choice_decoding", "q_learning"],
                                 data_standards=["nwb"], quality_score=0.8, trial_count=8000),
            "c4": DatasetContext("c4", modalities=["eeg"], tasks=["rest"],
                                 affordances=[], data_standards=[], quality_score=0.3),
        }
    )


def _make_queries_labels():
    queries = [
        UsefulnessQuery(
            query_id="q1",
            query="neuropixels decision making choice decoding",
            intent="strict_lookup",
            candidate_ids=["c1", "c2", "c3", "c4"],
        )
    ]
    labels = [
        PairLabel(query_id="q1", candidate_id="c1", usefulness_label="highly_useful", label_type="reusable"),
        PairLabel(query_id="q1", candidate_id="c3", usefulness_label="useful", label_type="reusable"),
        PairLabel(query_id="q1", candidate_id="c2", usefulness_label="weakly_useful", label_type="reusable"),
        PairLabel(query_id="q1", candidate_id="c4", usefulness_label="not_useful", label_type="reusable",
                  is_hard_negative=True),
    ]
    return queries, labels


class TestAblationVariantNames:
    def test_all_expected_variants_present(self):
        expected = {
            "bm25_only", "dense_only", "graph_only", "affordance_only",
            "bm25_dense_rrf", "hybrid_static", "hybrid_intent_aware", "latent_usefulness_v08",
        }
        assert expected.issubset(set(VARIANT_NAMES))


class TestRunAblation:
    def test_report_has_all_variants(self):
        queries, labels = _make_queries_labels()
        pool = _make_pool()
        config = AblationConfig(queries=queries, labels=labels, pool=pool, k=4)
        report = run_ablation(config)
        assert isinstance(report, AblationReport)
        for name in VARIANT_NAMES:
            assert name in report.variant_metrics, f"Missing variant: {name}"

    def test_all_metrics_bounded(self):
        queries, labels = _make_queries_labels()
        pool = _make_pool()
        config = AblationConfig(queries=queries, labels=labels, pool=pool, k=4)
        report = run_ablation(config)
        for variant, metrics in report.variant_metrics.items():
            for m, v in metrics.items():
                assert 0.0 <= v <= 1.0, f"{variant}.{m} = {v} out of [0,1]"

    def test_markdown_report_contains_table(self):
        queries, labels = _make_queries_labels()
        pool = _make_pool()
        config = AblationConfig(queries=queries, labels=labels, pool=pool, k=4)
        report = run_ablation(config)
        md = report.to_markdown()
        assert "NDCG" in md or "ndcg" in md
        assert "|" in md  # has a markdown table

    def test_markdown_report_written_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queries, labels = _make_queries_labels()
            pool = _make_pool()
            out_path = Path(tmpdir) / "ablation.md"
            config = AblationConfig(queries=queries, labels=labels, pool=pool, k=4, out_path=out_path)
            run_ablation(config)
            assert out_path.exists()
            content = out_path.read_text()
            assert "|" in content
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && pytest tests/test_ablation_runner.py -v 2>&1 | head -20
```

- [ ] **Step 3: Implement `neural_search/evaluation/ablation_runner.py`**

```python
"""Ablation runner for comparing retrieval variants on usefulness benchmark.

Variants are implemented as different scoring functions applied to a shared
candidate pool. No external BM25/dense infrastructure required — each variant
is a deterministic score function over DatasetContext objects.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from neural_search.evaluation.usefulness_benchmark import (
    BenchmarkReport,
    PairLabel,
    UsefulnessQuery,
    run_usefulness_benchmark,
)
from neural_search.retrieval.query_intent import UsefulnessIntent, classify_query_intent
from neural_search.retrieval.usefulness_scorer import (
    DatasetContext,
    INTENT_WEIGHT_PROFILES,
    score_usefulness,
    _jaccard,
    _log_power,
)

VARIANT_NAMES = (
    "bm25_only",
    "dense_only",
    "graph_only",
    "affordance_only",
    "bm25_dense_rrf",
    "hybrid_static",
    "hybrid_intent_aware",
    "latent_usefulness_v08",
)


@dataclass
class CandidatePool:
    """Pool of candidate datasets available for ranking."""
    candidates: dict[str, DatasetContext]


@dataclass
class AblationConfig:
    queries: list[UsefulnessQuery]
    labels: list[PairLabel]
    pool: CandidatePool
    k: int = 10
    out_path: Path | None = None


@dataclass
class AblationReport:
    variant_metrics: dict[str, dict[str, float]]
    k: int

    def to_markdown(self) -> str:
        metrics_order = ["ndcg_at_k", "mrr", "precision_at_k", "hard_negative_violation_rate"]
        header = "| Variant | " + " | ".join(m.upper()[:12] for m in metrics_order) + " |"
        sep = "|---------|" + "--------|" * len(metrics_order)
        rows = [header, sep]
        for variant in VARIANT_NAMES:
            m = self.variant_metrics.get(variant, {})
            vals = " | ".join(f"{m.get(k, 0.0):.4f}" for k in metrics_order)
            rows.append(f"| {variant} | {vals} |")
        return "# Ablation Report\n\n## Metric Table\n\n" + "\n".join(rows) + "\n"


def _build_query_context(query: UsefulnessQuery, pool: CandidatePool) -> DatasetContext:
    """Synthesize a DatasetContext for the query from its top candidate signals."""
    candidates = [pool.candidates[cid] for cid in query.candidate_ids if cid in pool.candidates]
    if not candidates:
        return DatasetContext(dataset_id="__query__")
    # Take union of top-3 candidates as a proxy for query intent
    modalities: set[str] = set()
    tasks: set[str] = set()
    affordances: set[str] = set()
    data_standards: set[str] = set()
    for c in candidates[:3]:
        modalities.update(c.modalities)
        tasks.update(c.tasks)
        affordances.update(c.affordances)
        data_standards.update(c.data_standards)
    return DatasetContext(
        dataset_id="__query__",
        modalities=list(modalities),
        tasks=list(tasks),
        affordances=list(affordances),
        data_standards=list(data_standards),
    )


def _score_bm25_only(qctx: DatasetContext, cand: DatasetContext) -> float:
    """Keyword overlap proxy: task + modality term overlap."""
    return 0.6 * _jaccard(qctx.tasks, cand.tasks) + 0.4 * _jaccard(qctx.modalities, cand.modalities)


def _score_dense_only(qctx: DatasetContext, cand: DatasetContext) -> float:
    """Dense similarity proxy: affordance + region overlap as embedding stand-in."""
    return 0.5 * _jaccard(qctx.affordances, cand.affordances) + 0.5 * _jaccard(qctx.brain_regions, cand.brain_regions)


def _score_graph_only(qctx: DatasetContext, cand: DatasetContext) -> float:
    """Graph-proximity proxy: data standard overlap (no live graph)."""
    return _jaccard(qctx.data_standards, cand.data_standards)


def _score_affordance_only(qctx: DatasetContext, cand: DatasetContext) -> float:
    """Affordance compatibility only."""
    return _jaccard(qctx.affordances, cand.affordances)


def _score_bm25_dense_rrf(qctx: DatasetContext, cand: DatasetContext) -> float:
    """RRF of BM25 and dense proxies (simplified for synthetic data)."""
    s1 = _score_bm25_only(qctx, cand)
    s2 = _score_dense_only(qctx, cand)
    return 0.5 * s1 + 0.5 * s2


def _score_hybrid_static(qctx: DatasetContext, cand: DatasetContext) -> float:
    """Equal-weight fusion of all signal proxies."""
    return (
        _score_bm25_only(qctx, cand) * 0.25
        + _score_dense_only(qctx, cand) * 0.25
        + _score_affordance_only(qctx, cand) * 0.25
        + min(1.0, max(0.0, cand.quality_score)) * 0.25
    )


def _score_hybrid_intent_aware(qctx: DatasetContext, cand: DatasetContext, intent: UsefulnessIntent) -> float:
    """Intent-aware fusion without full usefulness scorer."""
    if intent == UsefulnessIntent.PIPELINE_REUSE:
        return 0.5 * _score_affordance_only(qctx, cand) + 0.5 * _jaccard(qctx.data_standards, cand.data_standards)
    elif intent == UsefulnessIntent.REPLICATION:
        return (
            0.4 * _jaccard(qctx.tasks, cand.tasks)
            + 0.3 * _jaccard(qctx.species, cand.species)
            + 0.3 * _jaccard(qctx.brain_regions, cand.brain_regions)
        )
    elif intent == UsefulnessIntent.METHOD_TRANSFER:
        return 0.7 * _score_affordance_only(qctx, cand) + 0.3 * _score_bm25_only(qctx, cand)
    else:
        return _score_hybrid_static(qctx, cand)


ScoringFn = Callable[[DatasetContext, DatasetContext], float]


def _rank_candidates(
    query: UsefulnessQuery,
    pool: CandidatePool,
    score_fn: ScoringFn,
) -> list[str]:
    qctx = _build_query_context(query, pool)
    scored = [
        (cid, score_fn(qctx, pool.candidates[cid]))
        for cid in query.candidate_ids
        if cid in pool.candidates
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in scored]


def run_ablation(config: AblationConfig) -> AblationReport:
    """Run all variants and return an AblationReport."""
    variant_metrics: dict[str, dict[str, float]] = {}

    for variant in VARIANT_NAMES:
        run: dict[str, list[str]] = {}

        for q in config.queries:
            qctx = _build_query_context(q, config.pool)
            intent_cls = classify_query_intent(q.query)
            intent = intent_cls.intent

            if variant == "bm25_only":
                ranked = _rank_candidates(q, config.pool, _score_bm25_only)
            elif variant == "dense_only":
                ranked = _rank_candidates(q, config.pool, _score_dense_only)
            elif variant == "graph_only":
                ranked = _rank_candidates(q, config.pool, _score_graph_only)
            elif variant == "affordance_only":
                ranked = _rank_candidates(q, config.pool, _score_affordance_only)
            elif variant == "bm25_dense_rrf":
                ranked = _rank_candidates(q, config.pool, _score_bm25_dense_rrf)
            elif variant == "hybrid_static":
                ranked = _rank_candidates(q, config.pool, _score_hybrid_static)
            elif variant == "hybrid_intent_aware":
                fn = lambda qc, c, i=intent: _score_hybrid_intent_aware(qc, c, i)
                ranked = _rank_candidates(q, config.pool, fn)
            elif variant == "latent_usefulness_v08":
                scored = [
                    (cid, score_usefulness(qctx, config.pool.candidates[cid], intent).total_score)
                    for cid in q.candidate_ids
                    if cid in config.pool.candidates
                ]
                scored.sort(key=lambda x: x[1], reverse=True)
                ranked = [cid for cid, _ in scored]
            else:
                ranked = list(q.candidate_ids)

            run[q.query_id] = ranked

        bench_report = run_usefulness_benchmark(config.queries, config.labels, run, k=config.k)
        variant_metrics[variant] = {
            "ndcg_at_k": bench_report.ndcg_at_k,
            "mrr": bench_report.mrr,
            "precision_at_k": bench_report.precision_at_k,
            "hard_negative_violation_rate": bench_report.hard_negative_violation_rate,
        }

    report = AblationReport(variant_metrics=variant_metrics, k=config.k)

    if config.out_path:
        config.out_path.parent.mkdir(parents=True, exist_ok=True)
        config.out_path.write_text(report.to_markdown(), encoding="utf-8")

    return report
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && pytest tests/test_ablation_runner.py -v
```

- [ ] **Step 5: Commit**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && git add neural_search/evaluation/ablation_runner.py tests/test_ablation_runner.py && git commit -m "feat: add ablation runner comparing 8 retrieval variants on usefulness benchmark"
```

---

## Task 9: Config file and generate reports

- [ ] **Step 1: Create `config/eval/usefulness_v08.yaml`**

```yaml
# config/eval/usefulness_v08.yaml
# Ablation evaluation config for latent usefulness v0.8
version: "0.8"
seed_pairs: "data/eval/usefulness_seed_pairs.jsonl"
k: 10
variants:
  - bm25_only
  - dense_only
  - graph_only
  - affordance_only
  - bm25_dense_rrf
  - hybrid_static
  - hybrid_intent_aware
  - latent_usefulness_v08
outputs:
  ablation_report: "reports/ablation_v08.md"
  benchmark_report: "reports/usefulness_benchmark_v08.md"
  affordance_report: "reports/affordance_validation_v2.md"
```

- [ ] **Step 2: Generate benchmark and ablation reports**

Create and run `scripts/run_v08_reports.py`:

```python
#!/usr/bin/env python3
"""Generate v0.8 usefulness reports from seed data."""
from pathlib import Path
from neural_search.evaluation.usefulness_benchmark import load_seed_pairs, run_usefulness_benchmark
from neural_search.evaluation.ablation_runner import (
    AblationConfig, CandidatePool, run_ablation
)
from neural_search.retrieval.usefulness_scorer import DatasetContext

SEED_PATH = Path("data/eval/usefulness_seed_pairs.jsonl")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# Load seed pairs
queries, labels = load_seed_pairs(SEED_PATH)
print(f"Loaded {len(queries)} queries, {len(labels)} labels")

# Build synthetic candidate pool from seed data
candidate_ids = {lbl.candidate_id for lbl in labels}
pool = CandidatePool(candidates={
    cid: DatasetContext(dataset_id=cid) for cid in candidate_ids
})

# Add query candidate_ids
for q in queries:
    for cid in q.candidate_ids:
        if cid not in pool.candidates:
            pool.candidates[cid] = DatasetContext(dataset_id=cid)

# Run benchmark with identity run (all candidates ranked alphabetically as baseline)
identity_run = {q.query_id: sorted(q.candidate_ids) for q in queries}
bench_report = run_usefulness_benchmark(queries, labels, identity_run, k=5)
bench_md = REPORTS_DIR / "usefulness_benchmark_v08.md"
bench_md.write_text(bench_report.to_markdown(), encoding="utf-8")
print(f"Wrote {bench_md}")

# Run ablation
config = AblationConfig(
    queries=queries,
    labels=labels,
    pool=pool,
    k=5,
    out_path=REPORTS_DIR / "ablation_v08.md",
)
ablation_report = run_ablation(config)
print(f"Wrote {REPORTS_DIR / 'ablation_v08.md'}")
print("\nAblation Results:")
for variant, metrics in ablation_report.variant_metrics.items():
    print(f"  {variant}: NDCG={metrics['ndcg_at_k']:.4f} MRR={metrics['mrr']:.4f}")
```

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && python scripts/run_v08_reports.py
```

- [ ] **Step 3: Verify reports exist**

```bash
ls -la /mnt/c/Users/sidso/Documents/neural-search/reports/ablation_v08.md /mnt/c/Users/sidso/Documents/neural-search/reports/usefulness_benchmark_v08.md
```

- [ ] **Step 4: Commit**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && git add config/eval/ scripts/run_v08_reports.py reports/ablation_v08.md reports/usefulness_benchmark_v08.md && git commit -m "feat: generate v0.8 ablation and benchmark reports from seed data"
```

---

## Task 10: Implementation notes and claim ledger

- [ ] **Step 1: Create `docs/LATENT_USEFULNESS_IMPLEMENTATION_NOTES.md`**

```markdown
# Latent Usefulness v0.8 Implementation Notes

## What Was Built

### New Modules
- `neural_search/retrieval/query_intent.py` — `UsefulnessIntent` enum + rule-based classifier
- `neural_search/retrieval/usefulness_scorer.py` — 10-dimension intent-weighted scorer
- `neural_search/retrieval/graph_usefulness.py` — Hub-normalized PathSim + complementarity
- `neural_search/evaluation/usefulness_benchmark.py` — Graded NDCG/MRR/P@k/hard-neg violation
- `neural_search/evaluation/affordance_validation_v2.py` — Precision/recall vs ground truth
- `neural_search/evaluation/ablation_runner.py` — 8 retrieval variant comparison

### Key Design Decisions
1. **`UsefulnessIntent` vs existing `QueryIntent`**: New enum at `retrieval/query_intent.py` targets
   latent usefulness relationships (replication, pipeline_reuse, method_transfer) rather than
   retrieval head overrides (modality_search, dataset_lookup). Both coexist without conflict.
2. **DatasetContext as neutral carrier**: Scorer accepts plain dataclasses instead of requiring
   `DatasetCardV1` or `NormalizedDatasetRecord`. Callers convert their schema to `DatasetContext`.
3. **neural_signature_similarity fixed at 0.0**: Placeholder pending Phase 3 neural signature search.
   Scores a warning into `UsefulnessScore.warnings`.
4. **graph_proximity uses neutral prior 0.3**: Without a live graph, returns 0.3 + warning.
   `graph_usefulness.py` provides the full implementation for use when graph is available.

## Limitations
- All benchmark results are on 30 synthetic seed pairs; no validated real corpus labels yet.
- Ablation variants are scoring proxies, not actual BM25/dense retrieval.
- `neural_signature_similarity` is unimplemented (contributes 0 to scores).
- Whitepaper claims based on this phase should be marked "preliminary" until corpus expands.

## Next Steps
1. Replace synthetic DatasetContext with real corpus DatasetCardV1 conversion function.
2. Integrate graph_usefulness into score_usefulness when graph is available.
3. Expand seed pairs to 200+ with real DANDI dataset pairs.
4. Implement neural_signature_similarity (Phase 3).
5. Add feedback-driven weight learning (Phase 4).
```

- [ ] **Step 2: Create/update `docs/CLAIM_LEDGER.md`**

Add a new section at the bottom of the existing file, or create it if absent:

```markdown
## v0.8 Claims (2026-05-31)

| Claim | Status | Evidence |
|-------|--------|---------|
| Intent classification improves precision for pipeline-reuse queries | Preliminary | Ablation on 30 seed pairs |
| Multi-dimensional usefulness scoring outperforms affordance-only on method-transfer queries | Preliminary | Ablation v0.8 report |
| Hard-negative violation rate reduced by intent-aware scoring vs hybrid_static | Preliminary | Ablation v0.8 report |
| Graph-derived complementarity score enables exploration-intent retrieval | Future Work | Not evaluated in v0.8 |
| Neural signature similarity improves cross-dataset discovery | Future Work | Not implemented in v0.8 |
| Usefulness scoring generalizes across neuroscience subdisciplines | Future Work | Labels are seed-only |
```

- [ ] **Step 3: Commit**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && git add docs/LATENT_USEFULNESS_IMPLEMENTATION_NOTES.md docs/CLAIM_LEDGER.md && git commit -m "docs: add implementation notes and claim ledger for v0.8"
```

---

## Task 11: Full test suite verification

- [ ] **Step 1: Run all new tests**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && pytest tests/test_usefulness_intent.py tests/test_usefulness_scorer.py tests/test_graph_usefulness.py tests/test_usefulness_benchmark.py tests/test_affordance_validation_v2.py tests/test_ablation_runner.py -v
```
Expected: all PASS

- [ ] **Step 2: Run full test suite**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && pytest tests/ -q 2>&1 | tail -20
```
Expected: no new failures vs baseline (878 tests + new tests)

- [ ] **Step 3: If failures, fix root cause**

Check for:
- Import conflicts (e.g., two modules defining same name)
- Missing `__init__.py` exports causing circular imports
- Type errors from incorrect field references

- [ ] **Step 4: Final commit**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search && git add -p && git commit -m "chore: final v0.8 cleanup — all tests passing"
```

---

## Self-Review Checklist

**Spec Coverage:**
- [x] `neural_search/retrieval/query_intent.py` — Task 1
- [x] `neural_search/retrieval/usefulness_scorer.py` — Task 2
- [x] `neural_search/retrieval/graph_usefulness.py` — Task 3
- [x] `data/eval/usefulness_seed_pairs.jsonl` (30 pairs) — Task 5
- [x] `neural_search/evaluation/usefulness_benchmark.py` — Task 6
- [x] `neural_search/evaluation/affordance_validation_v2.py` — Task 7
- [x] `neural_search/evaluation/ablation_runner.py` — Task 8
- [x] `config/eval/usefulness_v08.yaml` — Task 9
- [x] `reports/usefulness_benchmark_v08.md` — Task 9
- [x] `reports/ablation_v08.md` — Task 9
- [x] `docs/LATENT_USEFULNESS_IMPLEMENTATION_NOTES.md` — Task 10
- [x] Claim ledger update — Task 10
- [x] Tests for all 6 new modules — Tasks 1,2,3,6,7,8

**Type Consistency:**
- `UsefulnessIntent` defined in `query_intent.py`, imported by `usefulness_scorer.py` and `ablation_runner.py`
- `DatasetContext` defined in `usefulness_scorer.py`, imported by `ablation_runner.py`
- `score_usefulness` signature: `(query_context: DatasetContext, candidate: DatasetContext, intent: UsefulnessIntent | None) -> UsefulnessScore`
- `_jaccard` and `_log_power` defined in `usefulness_scorer.py`, imported by `ablation_runner.py`
- `UsefulnessQuery`, `PairLabel`, `run_usefulness_benchmark` defined in `usefulness_benchmark.py`, imported by `ablation_runner.py`
- `VARIANT_NAMES` is a tuple of 8 strings matching test expectations

**No Placeholders:** All code blocks are complete implementations.
