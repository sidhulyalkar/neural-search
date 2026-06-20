# Task 04 — Labeling Functions

**Files:**
- Create: `neural_search/eval/labeling_functions.py`
- Create: `scripts/eval/run_labeling_functions.py`
- Test: `tests/test_labeling_functions.py`

---

## Part A — Write the tests first

- [ ] **Step 1: Create `tests/test_labeling_functions.py`**

```python
"""Tests for all 13 deterministic labeling functions."""
from __future__ import annotations

import pytest
from neural_search.eval.evidence import DatasetEvidence, LFVote, PairEvidence, QuerySpec
from neural_search.eval.labeling_functions import (
    lf_analysis_affordance,
    lf_data_level_required,
    lf_hard_negative,
    lf_license_reusable,
    lf_meta_analysis_depth,
    lf_metadata_completeness,
    lf_partial_modality,
    lf_pipeline_reuse,
    lf_raw_data_available,
    lf_region_constraint,
    lf_required_modality,
    lf_species_constraint,
    lf_task_constraint,
    run_all_lfs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pair(
    required_modalities=None,
    preferred_modalities=None,
    required_species=None,
    hard_negatives=None,
    intent="META_ANALYSIS",
    analysis_affordances=None,
    task_constraints=None,
    data_level_requirements=None,
    brain_regions=None,
    # dataset fields
    d_species=None,
    d_modalities=None,
    d_tasks=None,
    d_regions=None,
    d_license=None,
    d_raw=False,
    d_completeness=0.5,
    d_data_levels=None,
    d_data_standards=None,
    d_has_behavior=False,
    d_title="A dataset",
    d_description="",
) -> PairEvidence:
    q = QuerySpec(
        query_id="q1", query_text="test", intent=intent, scientific_goal="x",
        required_modalities=required_modalities or [],
        preferred_modalities=preferred_modalities or [],
        required_species=required_species or [],
        hard_negatives=hard_negatives or [],
        analysis_affordances=analysis_affordances or [],
        task_constraints=task_constraints or [],
        data_level_requirements=data_level_requirements or [],
        brain_regions=brain_regions or [],
    )
    d = DatasetEvidence(
        record_id="dandi:1", source="dandi", title=d_title,
        description=d_description,
        species=d_species or [], modalities=d_modalities or [],
        tasks=d_tasks or [], regions=d_regions or [],
        license=d_license, raw_data_available=d_raw,
        metadata_completeness=d_completeness,
        data_levels=d_data_levels or [],
        data_standards=d_data_standards or [],
        has_behavior=d_has_behavior,
    )
    return PairEvidence(query_id="q1", record_id="dandi:1", query=q, dataset=d)


# ---------------------------------------------------------------------------
# lf_hard_negative
# ---------------------------------------------------------------------------

class TestLfHardNegative:
    def test_no_hard_negatives_abstains(self):
        pair = _make_pair(hard_negatives=[])
        vote = lf_hard_negative(pair)
        assert vote.abstain is True

    def test_matching_hard_negative_votes_zero(self):
        # dataset is resting-state fMRI
        pair = _make_pair(
            hard_negatives=["resting-state fMRI with reward words in description"],
            d_title="Resting state fMRI study",
            d_modalities=["fmri"],
        )
        vote = lf_hard_negative(pair)
        assert vote.abstain is False
        assert vote.label == 0
        assert vote.confidence >= 0.90

    def test_non_matching_hard_negative_abstains(self):
        pair = _make_pair(
            hard_negatives=["resting-state fMRI with reward words"],
            d_title="Mouse neuropixels visual cortex",
            d_modalities=["neuropixels"],
            d_species=["mouse"],
        )
        vote = lf_hard_negative(pair)
        assert vote.abstain is True


# ---------------------------------------------------------------------------
# lf_required_modality
# ---------------------------------------------------------------------------

class TestLfRequiredModality:
    def test_full_match_votes_3(self):
        pair = _make_pair(required_modalities=["fmri"], d_modalities=["fmri"])
        vote = lf_required_modality(pair)
        assert vote.label == 3
        assert vote.confidence >= 0.85

    def test_no_required_abstains(self):
        pair = _make_pair(required_modalities=[])
        vote = lf_required_modality(pair)
        assert vote.abstain is True

    def test_modality_mismatch_votes_zero(self):
        pair = _make_pair(
            required_modalities=["fmri"],
            d_modalities=["extracellular_ephys"],
        )
        vote = lf_required_modality(pair)
        assert vote.label == 0
        assert vote.confidence >= 0.80

    def test_partial_match_votes_2(self):
        pair = _make_pair(
            required_modalities=["fmri", "meg"],
            d_modalities=["fmri"],
        )
        vote = lf_required_modality(pair)
        assert vote.label == 2


# ---------------------------------------------------------------------------
# lf_species_constraint
# ---------------------------------------------------------------------------

class TestLfSpeciesConstraint:
    def test_species_match_votes_3(self):
        pair = _make_pair(required_species=["mouse"], d_species=["mouse"])
        vote = lf_species_constraint(pair)
        assert vote.label == 3

    def test_species_mismatch_votes_zero(self):
        pair = _make_pair(required_species=["human"], d_species=["mouse"])
        vote = lf_species_constraint(pair)
        assert vote.label == 0

    def test_no_constraint_abstains(self):
        pair = _make_pair(required_species=[])
        vote = lf_species_constraint(pair)
        assert vote.abstain is True


# ---------------------------------------------------------------------------
# lf_license_reusable
# ---------------------------------------------------------------------------

class TestLfLicenseReusable:
    def test_cc_by_votes_high(self):
        pair = _make_pair(d_license="CC-BY-4.0")
        vote = lf_license_reusable(pair)
        assert vote.label >= 2
        assert vote.abstain is False

    def test_no_license_abstains(self):
        pair = _make_pair(d_license=None)
        vote = lf_license_reusable(pair)
        assert vote.abstain is True

    def test_restrictive_license_votes_low(self):
        pair = _make_pair(d_license="All rights reserved")
        vote = lf_license_reusable(pair)
        assert vote.label <= 1


# ---------------------------------------------------------------------------
# lf_raw_data_available
# ---------------------------------------------------------------------------

class TestLfRawDataAvailable:
    def test_raw_available_votes_positive(self):
        pair = _make_pair(d_raw=True)
        vote = lf_raw_data_available(pair)
        assert vote.label >= 2

    def test_no_raw_votes_lower(self):
        pair = _make_pair(d_raw=False)
        vote = lf_raw_data_available(pair)
        assert vote.label <= 2


# ---------------------------------------------------------------------------
# lf_metadata_completeness
# ---------------------------------------------------------------------------

class TestLfMetadataCompleteness:
    def test_high_completeness_votes_high(self):
        pair = _make_pair(d_completeness=0.9)
        vote = lf_metadata_completeness(pair)
        assert vote.label >= 2

    def test_low_completeness_abstains_or_votes_low(self):
        pair = _make_pair(d_completeness=0.1)
        vote = lf_metadata_completeness(pair)
        assert vote.abstain or vote.label <= 1


# ---------------------------------------------------------------------------
# run_all_lfs
# ---------------------------------------------------------------------------

class TestRunAllLfs:
    def test_returns_13_votes(self):
        pair = _make_pair(
            required_modalities=["fmri"],
            required_species=["human"],
            d_modalities=["fmri"],
            d_species=["human"],
            d_license="CC-BY-4.0",
            d_raw=True,
            d_completeness=0.8,
        )
        votes = run_all_lfs(pair)
        assert len(votes) == 13

    def test_all_votes_are_lf_vote(self):
        pair = _make_pair()
        votes = run_all_lfs(pair)
        from neural_search.eval.evidence import LFVote
        assert all(isinstance(v, LFVote) for v in votes)
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/test_labeling_functions.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'neural_search.eval.labeling_functions'`

## Part B — Implement

- [ ] **Step 3: Create `neural_search/eval/labeling_functions.py`**

```python
"""13 deterministic labeling functions for weak-supervision qrels generation.

Each LF receives a PairEvidence and returns an LFVote.
LFs abstain (abstain=True) when they have no signal for the pair.
Hard-negative LF dominates: label=0, confidence=0.95 when a known
failure mode is matched.
"""
from __future__ import annotations

from neural_search.eval.evidence import LFVote, PairEvidence

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_OPEN_LICENSES = {
    "cc0", "cc-by", "cc by", "cc-by-4.0", "cc by 4.0",
    "pddl", "odc-by", "cc-by-sa", "open data commons",
    "mit", "apache", "bsd", "public domain",
}

_RESTRICTIVE_TERMS = {
    "all rights reserved", "proprietary", "not for redistribution",
    "restricted", "not open",
}

_STOP_WORDS = {
    "with", "that", "from", "this", "when", "have", "been",
    "will", "their", "they", "some", "into", "than", "then",
    "data", "dataset",
}

_PIPELINE_STANDARDS = {"NWB", "nwb", "BIDS", "bids", "NeuroData Without Borders"}

_META_ANALYSIS_MODALITIES = {"fmri", "eeg", "meg"}


def _candidate_text(pair: PairEvidence) -> str:
    d = pair.dataset
    parts = [
        d.title or "",
        d.description or "",
        " ".join(d.species),
        " ".join(d.modalities),
        " ".join(d.tasks),
        " ".join(d.regions),
    ]
    return " ".join(parts).lower()


def _hn_terms(hn_phrase: str) -> list[str]:
    return [
        t for t in hn_phrase.lower().split()
        if len(t) > 3 and t not in _STOP_WORDS
    ]


# ---------------------------------------------------------------------------
# LF 01 — Hard negative
# ---------------------------------------------------------------------------

def lf_hard_negative(pair: PairEvidence) -> LFVote:
    """Vote label=0 at high confidence when candidate matches a known failure mode."""
    if not pair.query.hard_negatives:
        return LFVote("lf_hard_negative", 0, 0.0, "No hard negatives defined.", abstain=True)

    candidate = _candidate_text(pair)
    for hn in pair.query.hard_negatives:
        terms = _hn_terms(hn)
        if not terms:
            continue
        matches = sum(1 for t in terms if t in candidate)
        if matches / len(terms) >= 0.5:
            return LFVote(
                "lf_hard_negative", 0, 0.95,
                f"Matches hard negative: '{hn}' ({matches}/{len(terms)} terms)"
            )
    return LFVote("lf_hard_negative", 0, 0.0, "No hard negative matched.", abstain=True)


# ---------------------------------------------------------------------------
# LF 02 — Required modality (full / partial / miss)
# ---------------------------------------------------------------------------

def lf_required_modality(pair: PairEvidence) -> LFVote:
    required = set(pair.query.required_modalities)
    if not required:
        return LFVote("lf_required_modality", 0, 0.0, "No required modalities.", abstain=True)

    dataset_mods = set(pair.dataset.modalities)
    matched = required & dataset_mods
    n_req, n_matched = len(required), len(matched)

    if n_matched == n_req:
        return LFVote("lf_required_modality", 3, 0.90, f"All required modalities present: {matched}")
    if n_matched > 0:
        return LFVote("lf_required_modality", 2, 0.70,
                      f"Partial modality match {n_matched}/{n_req}: {matched}")
    return LFVote("lf_required_modality", 0, 0.85,
                  f"Required modalities {required} absent from dataset {dataset_mods}")


# ---------------------------------------------------------------------------
# LF 03 — Preferred modality (nice-to-have)
# ---------------------------------------------------------------------------

def lf_partial_modality(pair: PairEvidence) -> LFVote:
    preferred = set(pair.query.preferred_modalities)
    if not preferred:
        return LFVote("lf_partial_modality", 0, 0.0, "No preferred modalities.", abstain=True)

    dataset_mods = set(pair.dataset.modalities)
    matched = preferred & dataset_mods
    if matched:
        return LFVote("lf_partial_modality", 1, 0.55, f"Preferred modalities present: {matched}")
    return LFVote("lf_partial_modality", 0, 0.40, "No preferred modalities matched.", abstain=True)


# ---------------------------------------------------------------------------
# LF 04 — Species constraint
# ---------------------------------------------------------------------------

def lf_species_constraint(pair: PairEvidence) -> LFVote:
    required = set(pair.query.required_species)
    if not required:
        return LFVote("lf_species_constraint", 0, 0.0, "No species constraint.", abstain=True)

    dataset_sp = set(pair.dataset.species)
    if required & dataset_sp:
        return LFVote("lf_species_constraint", 3, 0.90, f"Species match: {required & dataset_sp}")
    return LFVote("lf_species_constraint", 0, 0.85,
                  f"Species mismatch: required {required}, found {dataset_sp}")


# ---------------------------------------------------------------------------
# LF 05 — Task constraint
# ---------------------------------------------------------------------------

def lf_task_constraint(pair: PairEvidence) -> LFVote:
    constraints = set(pair.query.task_constraints)
    if not constraints:
        return LFVote("lf_task_constraint", 0, 0.0, "No task constraints.", abstain=True)

    dataset_tasks = set(pair.dataset.tasks)
    matched = constraints & dataset_tasks
    if matched:
        return LFVote("lf_task_constraint", 3, 0.80, f"Task match: {matched}")
    return LFVote("lf_task_constraint", 1, 0.55, "No task constraint matched.")


# ---------------------------------------------------------------------------
# LF 06 — Brain region constraint
# ---------------------------------------------------------------------------

def lf_region_constraint(pair: PairEvidence) -> LFVote:
    required = set(pair.query.brain_regions)
    if not required:
        return LFVote("lf_region_constraint", 0, 0.0, "No region constraints.", abstain=True)

    dataset_regions = set(pair.dataset.regions)
    if required & dataset_regions:
        return LFVote("lf_region_constraint", 3, 0.80,
                      f"Region match: {required & dataset_regions}")
    return LFVote("lf_region_constraint", 1, 0.55,
                  f"Region mismatch: required {required}, found {dataset_regions}")


# ---------------------------------------------------------------------------
# LF 07 — Data level required
# ---------------------------------------------------------------------------

def lf_data_level_required(pair: PairEvidence) -> LFVote:
    required = set(pair.query.data_level_requirements)
    if not required:
        return LFVote("lf_data_level_required", 0, 0.0, "No data level requirement.", abstain=True)

    dataset_levels = set(pair.dataset.data_levels)
    if required & dataset_levels:
        return LFVote("lf_data_level_required", 3, 0.80, f"Data level match: {required & dataset_levels}")
    return LFVote("lf_data_level_required", 0, 0.75,
                  f"Required data levels {required} not present; found {dataset_levels}")


# ---------------------------------------------------------------------------
# LF 08 — Raw data availability
# ---------------------------------------------------------------------------

def lf_raw_data_available(pair: PairEvidence) -> LFVote:
    if pair.dataset.raw_data_available:
        return LFVote("lf_raw_data_available", 3, 0.70, "Raw data available.")
    return LFVote("lf_raw_data_available", 2, 0.55, "Raw data not available — processed only.")


# ---------------------------------------------------------------------------
# LF 09 — License reusability
# ---------------------------------------------------------------------------

def lf_license_reusable(pair: PairEvidence) -> LFVote:
    lic = pair.dataset.license
    if not lic:
        return LFVote("lf_license_reusable", 0, 0.0, "License unknown.", abstain=True)

    lic_lower = lic.lower()
    if any(term in lic_lower for term in _OPEN_LICENSES):
        return LFVote("lf_license_reusable", 3, 0.85, f"Open license: {lic}")
    if any(term in lic_lower for term in _RESTRICTIVE_TERMS):
        return LFVote("lf_license_reusable", 0, 0.80, f"Restrictive license: {lic}")
    return LFVote("lf_license_reusable", 1, 0.50, f"Unknown license reusability: {lic}")


# ---------------------------------------------------------------------------
# LF 10 — Metadata completeness
# ---------------------------------------------------------------------------

def lf_metadata_completeness(pair: PairEvidence) -> LFVote:
    score = pair.dataset.metadata_completeness
    if score >= 0.8:
        return LFVote("lf_metadata_completeness", 3, 0.70, f"High metadata completeness: {score:.2f}")
    if score >= 0.5:
        return LFVote("lf_metadata_completeness", 2, 0.55, f"Moderate completeness: {score:.2f}")
    if score >= 0.3:
        return LFVote("lf_metadata_completeness", 1, 0.50, f"Low completeness: {score:.2f}")
    return LFVote("lf_metadata_completeness", 0, 0.40, f"Very low completeness: {score:.2f}", abstain=True)


# ---------------------------------------------------------------------------
# LF 11 — Analysis affordance overlap
# ---------------------------------------------------------------------------

def lf_analysis_affordance(pair: PairEvidence) -> LFVote:
    affordances = set(str(a).lower() for a in pair.query.analysis_affordances)
    if not affordances:
        return LFVote("lf_analysis_affordance", 0, 0.0, "No affordances specified.", abstain=True)

    candidate = _candidate_text(pair)
    matched = [a for a in affordances if a in candidate]
    if matched:
        return LFVote("lf_analysis_affordance", 2, 0.60, f"Affordance signals present: {matched}")
    return LFVote("lf_analysis_affordance", 1, 0.40, "No affordance signals detected.", abstain=True)


# ---------------------------------------------------------------------------
# LF 12 — Pipeline reuse signal
# ---------------------------------------------------------------------------

def lf_pipeline_reuse(pair: PairEvidence) -> LFVote:
    """For PIPELINE_REUSE intent: reward standardized formats (NWB/BIDS)."""
    if pair.query.intent != "PIPELINE_REUSE":
        return LFVote("lf_pipeline_reuse", 0, 0.0, "Not a pipeline-reuse query.", abstain=True)

    standards = set(pair.dataset.data_standards)
    if standards & _PIPELINE_STANDARDS:
        return LFVote("lf_pipeline_reuse", 3, 0.80,
                      f"Standardized format present: {standards & _PIPELINE_STANDARDS}")
    return LFVote("lf_pipeline_reuse", 1, 0.55, "No standardized data format detected.")


# ---------------------------------------------------------------------------
# LF 13 — Meta-analysis depth
# ---------------------------------------------------------------------------

def lf_meta_analysis_depth(pair: PairEvidence) -> LFVote:
    """For META_ANALYSIS intent: reward behavioral metadata + large-n + NWB."""
    if pair.query.intent != "META_ANALYSIS":
        return LFVote("lf_meta_analysis_depth", 0, 0.0, "Not a meta-analysis query.", abstain=True)

    score = 0
    rationale: list[str] = []
    if pair.dataset.has_behavior:
        score += 1
        rationale.append("has_behavior")
    if pair.dataset.has_trials:
        score += 1
        rationale.append("has_trials")
    if set(pair.dataset.data_standards) & _PIPELINE_STANDARDS:
        score += 1
        rationale.append("standardized_format")
    if pair.dataset.modalities and pair.dataset.modalities[0] in _META_ANALYSIS_MODALITIES:
        score += 1
        rationale.append("imaging_modality")

    label = min(score, 3)
    conf = 0.50 + 0.10 * score
    return LFVote("lf_meta_analysis_depth", label, min(conf, 0.85),
                  f"Meta-analysis signals: {', '.join(rationale) or 'none'}")


# ---------------------------------------------------------------------------
# Run all LFs
# ---------------------------------------------------------------------------

_ALL_LFS = [
    lf_hard_negative,
    lf_required_modality,
    lf_partial_modality,
    lf_species_constraint,
    lf_task_constraint,
    lf_region_constraint,
    lf_data_level_required,
    lf_raw_data_available,
    lf_license_reusable,
    lf_metadata_completeness,
    lf_analysis_affordance,
    lf_pipeline_reuse,
    lf_meta_analysis_depth,
]


def run_all_lfs(pair: PairEvidence) -> list[LFVote]:
    """Run all 13 labeling functions and return their votes."""
    return [lf(pair) for lf in _ALL_LFS]
```

- [ ] **Step 4: Run all LF tests — expect green**

```bash
pytest tests/test_labeling_functions.py -v
```

Expected: all tests pass.

## Part C — run_labeling_functions.py script

- [ ] **Step 5: Create `scripts/eval/run_labeling_functions.py`**

```python
#!/usr/bin/env python3
"""Run all labeling functions over pair_evidence.jsonl.

Usage:
    python scripts/eval/run_labeling_functions.py \
        --evidence artifacts/eval/pair_evidence.jsonl \
        --out artifacts/eval/label_function_votes.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_search.eval.evidence import PairEvidence, QuerySpec, DatasetEvidence
from neural_search.eval.labeling_functions import run_all_lfs


def _load_pair(row: dict) -> PairEvidence:
    q_data = row["query"]
    d_data = row["dataset"]
    return PairEvidence(
        query_id=row["query_id"],
        record_id=row["record_id"],
        query=QuerySpec(**q_data),
        dataset=DatasetEvidence(**d_data),
        pooled_from=row.get("pooled_from", []),
        min_rank=row.get("min_rank", 1000),
        priority=row.get("priority", "normal"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with args.evidence.open(encoding="utf-8") as in_fh, \
         args.out.open("w", encoding="utf-8") as out_fh:
        for line in in_fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pair = _load_pair(row)
            votes = run_all_lfs(pair)
            record = {
                "query_id": pair.query_id,
                "record_id": pair.record_id,
                "votes": [v.to_dict() for v in votes],
            }
            out_fh.write(json.dumps(record) + "\n")
            written += 1

    print(f"Labeling functions applied to {written} pairs → {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run on real data**

```bash
python scripts/eval/run_labeling_functions.py \
    --evidence artifacts/eval/pair_evidence.jsonl \
    --out artifacts/eval/label_function_votes.jsonl
```

Expected: `Labeling functions applied to N pairs → artifacts/eval/label_function_votes.jsonl`

- [ ] **Step 7: Commit**

```bash
git add neural_search/eval/labeling_functions.py \
    scripts/eval/run_labeling_functions.py \
    tests/test_labeling_functions.py
git commit -m "feat(eval): 13 deterministic labeling functions + run script"
```
