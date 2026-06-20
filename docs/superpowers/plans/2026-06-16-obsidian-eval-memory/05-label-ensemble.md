# Task 05 — Label Ensemble + Qrels Generation

**Files:**
- Create: `neural_search/eval/label_ensemble.py`
- Create: `scripts/eval/build_qrels_from_votes.py`
- Test: `tests/test_label_ensemble.py`
- Test: `tests/test_qrels_tiers.py`

---

## Part A — Label Ensemble

- [ ] **Step 1: Create `tests/test_label_ensemble.py`**

```python
"""Tests for label ensemble and qrels tier assignment."""
from __future__ import annotations
import pytest
from neural_search.eval.evidence import LFVote
from neural_search.eval.label_ensemble import (
    aggregate_votes,
    assign_tier,
    compute_audit_priority,
    EnsembleResult,
)


def _vote(lf_name: str, label: int, conf: float, abstain: bool = False) -> LFVote:
    return LFVote(lf_name=lf_name, label=label, confidence=conf,
                  rationale="test", abstain=abstain)


class TestAggregateVotes:
    def test_hard_negative_override(self):
        votes = [
            _vote("lf_hard_negative", 0, 0.95),          # hard negative fires
            _vote("lf_required_modality", 3, 0.90),       # would say relevant
        ]
        result = aggregate_votes(votes)
        assert result.label == 0
        assert result.hard_negative_triggered is True

    def test_all_abstain_returns_bronze(self):
        votes = [_vote(f"lf_{i}", 0, 0.0, abstain=True) for i in range(5)]
        result = aggregate_votes(votes)
        assert result.tier == "bronze"
        assert result.label in (0, 1)

    def test_strong_agreement_gives_silver(self):
        votes = [
            _vote("lf_required_modality", 3, 0.90),
            _vote("lf_species_constraint", 3, 0.85),
            _vote("lf_raw_data_available", 3, 0.70),
            _vote("lf_license_reusable", 3, 0.85),
        ]
        result = aggregate_votes(votes)
        assert result.tier == "silver"
        assert result.label == 3

    def test_high_disagreement_gives_bronze(self):
        votes = [
            _vote("lf_a", 0, 0.80),
            _vote("lf_b", 3, 0.80),
            _vote("lf_c", 0, 0.80),
        ]
        result = aggregate_votes(votes)
        assert result.tier == "bronze"

    def test_weighted_average_rounds_correctly(self):
        # 2 votes at label=2 with conf=0.9 → weighted avg = 2.0 → label=2
        votes = [
            _vote("lf_a", 2, 0.90),
            _vote("lf_b", 2, 0.90),
            _vote("lf_c", 2, 0.90),
        ]
        result = aggregate_votes(votes)
        assert result.label == 2


class TestAssignTier:
    def test_gold_when_human_audited(self):
        result = EnsembleResult(label=2, confidence=0.9, tier="silver",
                                hard_negative_triggered=False, disagreement=0.1,
                                active_vote_count=3, audit_priority=0.0,
                                provenance=[])
        tier = assign_tier(result, human_audited=True)
        assert tier == "gold"

    def test_silver_requires_3_active_votes(self):
        result = EnsembleResult(label=2, confidence=0.9, tier="silver",
                                hard_negative_triggered=False, disagreement=0.1,
                                active_vote_count=2, audit_priority=0.0,
                                provenance=[])
        tier = assign_tier(result, human_audited=False)
        assert tier == "bronze"


class TestAuditPriority:
    def test_hard_neg_raises_priority(self):
        result_with_hn = EnsembleResult(label=0, confidence=0.5, tier="bronze",
                                        hard_negative_triggered=True, disagreement=1.0,
                                        active_vote_count=2, audit_priority=0.0,
                                        provenance=[])
        result_without_hn = EnsembleResult(label=2, confidence=0.8, tier="silver",
                                           hard_negative_triggered=False, disagreement=0.0,
                                           active_vote_count=4, audit_priority=0.0,
                                           provenance=[])
        p_with = compute_audit_priority(result_with_hn, min_rank=1)
        p_without = compute_audit_priority(result_without_hn, min_rank=100)
        assert p_with > p_without
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/test_label_ensemble.py -v 2>&1 | head -10
```

- [ ] **Step 3: Create `neural_search/eval/label_ensemble.py`**

```python
"""Confidence-weighted vote aggregation and gold/silver/bronze qrels tiers.

Tier definitions:
  gold   — human-audited label
  silver — ≥3 non-abstaining LFs, avg_confidence ≥ 0.75, variance < 0.5
  bronze — everything else
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from neural_search.eval.evidence import LFVote


@dataclass
class EnsembleResult:
    label: int
    confidence: float
    tier: str                       # "bronze" before human audit
    hard_negative_triggered: bool
    disagreement: float             # variance of active vote labels
    active_vote_count: int
    audit_priority: float
    provenance: list[str]           # lf_names that influenced label


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def aggregate_votes(votes: list[LFVote]) -> EnsembleResult:
    """Aggregate LF votes into a single EnsembleResult."""
    # 1. Hard-negative override
    hard_neg = [v for v in votes if v.lf_name == "lf_hard_negative" and not v.abstain]
    if hard_neg:
        return EnsembleResult(
            label=0, confidence=0.95, tier="bronze",
            hard_negative_triggered=True, disagreement=0.0,
            active_vote_count=len([v for v in votes if not v.abstain]),
            audit_priority=0.0,
            provenance=["lf_hard_negative"],
        )

    # 2. Active (non-abstaining) votes
    active = [v for v in votes if not v.abstain]
    if not active:
        return EnsembleResult(
            label=1, confidence=0.30, tier="bronze",
            hard_negative_triggered=False, disagreement=0.0,
            active_vote_count=0, audit_priority=0.0, provenance=[],
        )

    # 3. Weighted average
    total_weight = sum(v.confidence for v in active)
    weighted_label_f = sum(v.label * v.confidence for v in active) / total_weight
    label = round(weighted_label_f)
    avg_conf = total_weight / len(active)

    # 4. Disagreement (population variance of labels)
    variance = sum((v.label - weighted_label_f) ** 2 for v in active) / len(active)

    # 5. Tier assignment
    if len(active) >= 3 and avg_conf >= 0.75 and variance < 0.5:
        tier = "silver"
    else:
        tier = "bronze"

    provenance = [v.lf_name for v in active]

    return EnsembleResult(
        label=label,
        confidence=min(avg_conf, 1.0),
        tier=tier,
        hard_negative_triggered=False,
        disagreement=variance,
        active_vote_count=len(active),
        audit_priority=0.0,
        provenance=provenance,
    )


def assign_tier(result: EnsembleResult, human_audited: bool) -> str:
    """Upgrade tier to gold if human-audited; enforce silver requirements."""
    if human_audited:
        return "gold"
    # Re-enforce silver requirements (active_vote_count may have been computed before)
    if result.tier == "silver" and result.active_vote_count < 3:
        return "bronze"
    return result.tier


def compute_audit_priority(result: EnsembleResult, min_rank: int) -> float:
    """Higher = more urgent to audit.

    Factors: disagreement, hard-negative flag, proximity to top of rank list.
    """
    rank_boost = 1.0 / (math.log(min_rank + 1) + 1.0)
    hn_factor = 2.0 if result.hard_negative_triggered else 1.0
    return result.disagreement * hn_factor * rank_boost


# ---------------------------------------------------------------------------
# Qrel record builder
# ---------------------------------------------------------------------------

def make_qrel(
    query_id: str,
    record_id: str,
    result: EnsembleResult,
    tier: str,
) -> dict:
    """Return a JSONL-ready qrel dict."""
    from datetime import datetime, timezone
    return {
        "query_id": query_id,
        "record_id": record_id,
        "label": result.label,
        "confidence": round(result.confidence, 4),
        "source": tier,
        "provenance": result.provenance,
        "hard_negative_triggered": result.hard_negative_triggered,
        "disagreement": round(result.disagreement, 4),
        "created": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Hard-negative violation helper (used by metrics scripts)
# ---------------------------------------------------------------------------

def compute_hard_negative_violations(
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[tuple[str, float]]],
    cutoff: int = 10,
) -> dict[str, int]:
    """Count hard-negative (label=0) entries appearing in top-k of run.

    Returns {query_id: violation_count}.
    """
    violations: dict[str, int] = {}
    for qid, ranked in run.items():
        q_qrels = qrels.get(qid, {})
        hard_negatives = {rid for rid, lbl in q_qrels.items() if lbl == 0}
        top_k = [rid for rid, _ in ranked[:cutoff]]
        count = sum(1 for rid in top_k if rid in hard_negatives)
        if count:
            violations[qid] = count
    return violations
```

- [ ] **Step 4: Run ensemble tests — expect green**

```bash
pytest tests/test_label_ensemble.py -v
```

---

## Part B — build_qrels_from_votes.py

- [ ] **Step 5: Create `tests/test_qrels_tiers.py`**

```python
"""Tests for qrels tier generation."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import pytest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


class TestQrelsTierSeparation:
    """Integration test: the build script produces three separate tier files."""

    def test_tiers_are_non_overlapping_on_source_field(self, tmp_path):
        """Gold, silver, bronze must each have only their tier's source label."""
        gold = tmp_path / "qrels_gold.jsonl"
        silver = tmp_path / "qrels_silver.jsonl"
        bronze = tmp_path / "qrels_bronze.jsonl"

        # Run the builder with synthetic data
        import subprocess, sys
        pair_evidence = [
            {   # high-confidence pair → silver
                "query_id": "q1", "record_id": "dandi:1",
                "query": {
                    "query_id": "q1", "query_text": "human fmri",
                    "intent": "META_ANALYSIS", "scientific_goal": "x",
                    "required_modalities": ["fmri"], "preferred_modalities": [],
                    "required_species": ["human"], "preferred_species": [],
                    "brain_regions": [], "task_constraints": [],
                    "data_level_requirements": [], "hard_negatives": [],
                    "analysis_affordances": [],
                },
                "dataset": {
                    "record_id": "dandi:1", "source": "dandi", "title": "Human fMRI",
                    "description": "fMRI study", "species": ["human"],
                    "modalities": ["fmri"], "data_levels": ["raw"], "tasks": [],
                    "regions": [], "license": "CC-BY-4.0", "doi": None, "url": None,
                    "raw_data_available": True, "metadata_completeness": 0.9,
                    "has_behavior": True, "has_trials": True, "data_standards": ["NWB"],
                },
                "pooled_from": ["usefulness"], "min_rank": 1, "priority": "high",
            }
        ]
        _write_jsonl(tmp_path / "pair_evidence.jsonl", pair_evidence)
        _write_jsonl(tmp_path / "votes.jsonl", [])  # no pre-computed votes

        result = subprocess.run(
            [sys.executable, "scripts/eval/build_qrels_from_votes.py",
             "--evidence", str(tmp_path / "pair_evidence.jsonl"),
             "--votes", str(tmp_path / "votes.jsonl"),
             "--out-gold", str(gold),
             "--out-silver", str(silver),
             "--out-bronze", str(bronze),
             "--audit-queue", str(tmp_path / "audit_queue.jsonl"),
             ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

        silver_rows = _read_jsonl(silver)
        bronze_rows = _read_jsonl(bronze)
        gold_rows = _read_jsonl(gold)

        # Gold should be empty (no human audits provided)
        assert gold_rows == []
        # Silver or bronze must have the one pair
        total = len(silver_rows) + len(bronze_rows)
        assert total == 1

        for r in silver_rows:
            assert r["source"] == "silver"
        for r in bronze_rows:
            assert r["source"] == "bronze"
```

- [ ] **Step 6: Create `scripts/eval/build_qrels_from_votes.py`**

```python
#!/usr/bin/env python3
"""Aggregate LF votes (+ optional LLM judgments) into gold/silver/bronze qrels.

Usage:
    python scripts/eval/build_qrels_from_votes.py \
        --evidence artifacts/eval/pair_evidence.jsonl \
        --votes artifacts/eval/label_function_votes.jsonl \
        --out-gold artifacts/qrels_gold.jsonl \
        --out-silver artifacts/qrels_silver.jsonl \
        --out-bronze artifacts/qrels_bronze.jsonl \
        --audit-queue artifacts/eval/audit_queue.jsonl

    # With optional LLM judgments:
        --llm artifacts/eval/llm_judgments.jsonl

    # With human audits to promote to gold:
        --human-audits artifacts/eval/human_audits.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_search.eval.evidence import LFVote, PairEvidence, QuerySpec, DatasetEvidence
from neural_search.eval.label_ensemble import (
    EnsembleResult,
    aggregate_votes,
    assign_tier,
    compute_audit_priority,
    make_qrel,
)


def _load_jsonl(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _parse_votes(rows: list[dict]) -> dict[tuple[str, str], list[LFVote]]:
    """Return {(query_id, record_id): [LFVote, ...]}."""
    out: dict[tuple[str, str], list[LFVote]] = {}
    for row in rows:
        key = (row["query_id"], row["record_id"])
        out[key] = [LFVote(**v) for v in row.get("votes", [])]
    return out


def _parse_llm(rows: list[dict]) -> dict[tuple[str, str], LFVote]:
    """Convert LLM judgment rows to synthetic LFVotes (weight 1.5×)."""
    out: dict[tuple[str, str], LFVote] = {}
    for row in rows:
        key = (row["query_id"], row["record_id"])
        # LLM vote is treated as a weighted signal (confidence boosted 1.5×, capped at 1.0)
        boosted_conf = min(float(row.get("confidence", 0.5)) * 1.5, 1.0)
        out[key] = LFVote(
            lf_name="llm_judge",
            label=int(row.get("label", 1)),
            confidence=boosted_conf,
            rationale=row.get("rationale", ""),
            abstain=False,
        )
    return out


def _parse_human_audits(rows: list[dict]) -> dict[tuple[str, str], int]:
    """Return {(query_id, record_id): label} for human-confirmed labels."""
    return {
        (r["query_id"], r["record_id"]): int(r["label"])
        for r in rows
        if r.get("audit_status") == "done" and "label" in r
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--votes", required=True, type=Path)
    parser.add_argument("--llm", type=Path, default=None)
    parser.add_argument("--human-audits", type=Path, default=None)
    parser.add_argument("--out-gold", required=True, type=Path)
    parser.add_argument("--out-silver", required=True, type=Path)
    parser.add_argument("--out-bronze", required=True, type=Path)
    parser.add_argument("--audit-queue", required=True, type=Path)
    args = parser.parse_args()

    evidence_rows = _load_jsonl(args.evidence)
    votes_by_pair = _parse_votes(_load_jsonl(args.votes))
    llm_by_pair = _parse_llm(_load_jsonl(args.llm))
    human_labels = _parse_human_audits(_load_jsonl(getattr(args, "human_audits", None)))

    for p in (args.out_gold, args.out_silver, args.out_bronze, args.audit_queue):
        p.parent.mkdir(parents=True, exist_ok=True)

    counts = {"gold": 0, "silver": 0, "bronze": 0, "audit": 0}
    report_entries: list[dict] = []

    with (args.out_gold.open("w", encoding="utf-8") as gold_fh,
          args.out_silver.open("w", encoding="utf-8") as silver_fh,
          args.out_bronze.open("w", encoding="utf-8") as bronze_fh,
          args.audit_queue.open("w", encoding="utf-8") as aq_fh):

        for row in evidence_rows:
            qid = row["query_id"]
            rid = row["record_id"]
            key = (qid, rid)

            # Collect votes
            votes: list[LFVote] = list(votes_by_pair.get(key, []))
            if key in llm_by_pair:
                votes.append(llm_by_pair[key])

            # Run LFs inline if no pre-computed votes
            if not votes:
                from neural_search.eval.evidence import PairEvidence, QuerySpec, DatasetEvidence
                from neural_search.eval.labeling_functions import run_all_lfs
                try:
                    pair = PairEvidence(
                        query_id=qid, record_id=rid,
                        query=QuerySpec(**row["query"]),
                        dataset=DatasetEvidence(**row["dataset"]),
                        pooled_from=row.get("pooled_from", []),
                        min_rank=row.get("min_rank", 1000),
                        priority=row.get("priority", "normal"),
                    )
                    votes = run_all_lfs(pair)
                except Exception:
                    pass

            result = aggregate_votes(votes)
            human_audited = key in human_labels
            if human_audited:
                result = EnsembleResult(
                    label=human_labels[key],
                    confidence=1.0,
                    tier="gold",
                    hard_negative_triggered=result.hard_negative_triggered,
                    disagreement=result.disagreement,
                    active_vote_count=result.active_vote_count,
                    audit_priority=result.audit_priority,
                    provenance=result.provenance + ["human_audit"],
                )

            tier = assign_tier(result, human_audited=human_audited)
            result.audit_priority = compute_audit_priority(result, min_rank=row.get("min_rank", 1000))
            qrel = make_qrel(qid, rid, result, tier)

            if tier == "gold":
                gold_fh.write(json.dumps(qrel) + "\n")
                counts["gold"] += 1
            elif tier == "silver":
                silver_fh.write(json.dumps(qrel) + "\n")
                counts["silver"] += 1
            else:
                bronze_fh.write(json.dumps(qrel) + "\n")
                counts["bronze"] += 1

            # Audit queue: items with high priority or unresolved disagreement
            if not human_audited and (result.disagreement > 0.5 or result.hard_negative_triggered):
                aq_entry = {**qrel, "audit_priority": round(result.audit_priority, 4),
                            "pair_evidence": row}
                aq_fh.write(json.dumps(aq_entry) + "\n")
                counts["audit"] += 1

    print(f"Gold: {counts['gold']}, Silver: {counts['silver']}, "
          f"Bronze: {counts['bronze']}, Audit queue: {counts['audit']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run ensemble + qrels tests**

```bash
pytest tests/test_label_ensemble.py tests/test_qrels_tiers.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Run on real data**

```bash
python scripts/eval/build_qrels_from_votes.py \
    --evidence artifacts/eval/pair_evidence.jsonl \
    --votes artifacts/eval/label_function_votes.jsonl \
    --out-gold artifacts/qrels_gold.jsonl \
    --out-silver artifacts/qrels_silver.jsonl \
    --out-bronze artifacts/qrels_bronze.jsonl \
    --audit-queue artifacts/eval/audit_queue.jsonl
```

Expected: prints counts for each tier.

- [ ] **Step 9: Commit**

```bash
git add neural_search/eval/label_ensemble.py \
    scripts/eval/build_qrels_from_votes.py \
    tests/test_label_ensemble.py tests/test_qrels_tiers.py
git commit -m "feat(eval): label ensemble + gold/silver/bronze qrels generation"
```
