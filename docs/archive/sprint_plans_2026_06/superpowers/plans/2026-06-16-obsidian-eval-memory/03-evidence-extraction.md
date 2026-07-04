# Task 03 — Evidence Extraction

**Files:**
- Create: `neural_search/eval/query_decomposition.py`
- Create: `scripts/eval/build_evidence.py`
- Test: `tests/test_eval_evidence.py` (extend with decomposition + pair tests)

---

## Part A — Query Decomposition

- [ ] **Step 1: Extend test_eval_evidence.py with decomposition tests**

Append to `tests/test_eval_evidence.py`:

```python
from neural_search.eval.query_decomposition import decompose_query, load_query_specs


class TestQueryDecomposition:
    def test_extracts_fmri_modality(self):
        record = {
            "query_id": "q_0001",
            "intent": "META_ANALYSIS",
            "query": "human fMRI reward prediction error reinforcement learning task",
            "scientific_goal": "Find datasets for meta-analysis.",
            "required_evidence": ["species", "modality"],
            "nice_to_have": [],
            "known_failure_modes": ["resting-state fMRI with reward words in description"],
        }
        spec = decompose_query(record)
        assert "fmri" in spec.required_modalities
        assert "human" in spec.required_species

    def test_extracts_hard_negatives(self):
        record = {
            "query_id": "q_0002",
            "intent": "MODEL_VALIDATION",
            "query": "mouse visual cortex calcium imaging",
            "scientific_goal": "Find mouse calcium datasets.",
            "required_evidence": [],
            "nice_to_have": [],
            "known_failure_modes": [
                "mouse electrophysiology visual cortex — modality mismatch",
                "human visual fMRI — species mismatch",
            ],
        }
        spec = decompose_query(record)
        assert len(spec.hard_negatives) == 2
        assert "calcium_imaging" in spec.required_modalities
        assert "mouse" in spec.required_species

    def test_neuropixels_query(self):
        record = {
            "query_id": "q_0003",
            "intent": "PIPELINE_REUSE",
            "query": "extracellular electrophysiology spike sorting neuropixels single unit",
            "scientific_goal": "Find ephys datasets.",
            "required_evidence": ["modality"],
            "nice_to_have": [],
            "known_failure_modes": [],
        }
        spec = decompose_query(record)
        assert any(m in spec.required_modalities for m in ["neuropixels", "extracellular_ephys"])
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/test_eval_evidence.py::TestQueryDecomposition -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'neural_search.eval.query_decomposition'`

- [ ] **Step 3: Create `neural_search/eval/query_decomposition.py`**

```python
"""Parse benchmark query records into structured QuerySpec objects.

Uses keyword matching against modality/species ontology terms.
Does not call any external service — pure deterministic parsing.
"""
from __future__ import annotations

import json
from pathlib import Path

from neural_search.eval.evidence import QuerySpec

# ---------------------------------------------------------------------------
# Keyword maps  (lowercase key → canonical id)
# ---------------------------------------------------------------------------

_MODALITY_KW: dict[str, str] = {
    "fmri": "fmri",
    "functional mri": "fmri",
    "functional magnetic": "fmri",
    "bold": "fmri",
    "calcium imaging": "calcium_imaging",
    "two-photon": "calcium_imaging",
    "2-photon": "calcium_imaging",
    "gcamp": "calcium_imaging",
    "neuropixels": "neuropixels",
    "extracellular electrophysiology": "extracellular_ephys",
    "extracellular ephys": "extracellular_ephys",
    "electrophysiology": "extracellular_ephys",
    "single unit": "extracellular_ephys",
    "multi-unit": "extracellular_ephys",
    "spike sorting": "extracellular_ephys",
    "spike-sorting": "extracellular_ephys",
    "eeg": "eeg",
    "electroencephalography": "eeg",
    "meg": "meg",
    "magnetoencephalography": "meg",
    "patch clamp": "intracellular_ephys",
    "intracellular": "intracellular_ephys",
    "dti": "dti",
    "diffusion mri": "dti",
    "diffusion tensor": "dti",
    "lfp": "extracellular_ephys",
    "local field potential": "extracellular_ephys",
}

_SPECIES_KW: dict[str, str] = {
    "human": "human",
    "humans": "human",
    "patient": "human",
    "patients": "human",
    "homo sapiens": "human",
    "mouse": "mouse",
    "mice": "mouse",
    "murine": "mouse",
    "mus musculus": "mouse",
    "rat": "rat",
    "rats": "rat",
    "rodent": "rodent",
    "rodents": "rodent",
    "macaque": "macaque",
    "macaques": "macaque",
    "rhesus": "macaque",
    "primate": "macaque",
    "non-human primate": "macaque",
    "nhp": "macaque",
    "marmoset": "marmoset",
    "zebrafish": "zebrafish",
    "drosophila": "drosophila",
    "fly": "drosophila",
}

_DATA_LEVEL_KW: dict[str, str] = {
    "raw data": "raw",
    "raw recordings": "raw",
    "preprocessed": "preprocessed",
    "processed": "processed",
    "spike times": "processed",
    "spike-sorted": "processed",
}


def _extract_keywords(text: str, kw_map: dict[str, str]) -> list[str]:
    """Return canonical ids for all keywords found in text (no duplicates, order preserved)."""
    text_lower = text.lower()
    seen: set[str] = set()
    results: list[str] = []
    # Sort by length descending so longer phrases match before substrings
    for kw in sorted(kw_map, key=len, reverse=True):
        if kw in text_lower:
            canonical = kw_map[kw]
            if canonical not in seen:
                seen.add(canonical)
                results.append(canonical)
    return results


def decompose_query(record: dict) -> QuerySpec:
    """Parse a raw benchmark query dict into a structured QuerySpec."""
    query_text = record.get("query", "")
    nice_to_have: list[str] = record.get("nice_to_have", []) or []
    nice_text = " ".join(str(n) for n in nice_to_have)

    return QuerySpec(
        query_id=record["query_id"],
        query_text=query_text,
        intent=record.get("intent", ""),
        scientific_goal=record.get("scientific_goal", ""),
        required_modalities=_extract_keywords(query_text, _MODALITY_KW),
        preferred_modalities=_extract_keywords(nice_text, _MODALITY_KW),
        required_species=_extract_keywords(query_text, _SPECIES_KW),
        preferred_species=_extract_keywords(nice_text, _SPECIES_KW),
        brain_regions=[],   # kept empty — too noisy to parse reliably from free text
        task_constraints=[],
        data_level_requirements=_extract_keywords(query_text, _DATA_LEVEL_KW),
        hard_negatives=list(record.get("known_failure_modes", []) or []),
        analysis_affordances=nice_to_have,
    )


def load_query_specs(queries_path: Path) -> list[QuerySpec]:
    """Load all QuerySpecs from a JSONL benchmark queries file."""
    specs: list[QuerySpec] = []
    with queries_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            specs.append(decompose_query(json.loads(line)))
    return specs
```

- [ ] **Step 4: Run decomposition tests — expect green**

```bash
pytest tests/test_eval_evidence.py -v
```

Expected: all tests pass.

---

## Part B — build_evidence.py script

- [ ] **Step 5: Write the failing script integration test**

Append to `tests/test_eval_evidence.py`:

```python
import json
import tempfile
from pathlib import Path
from neural_search.eval.evidence import dataset_evidence_from_record, PairEvidence
from neural_search.eval.query_decomposition import decompose_query


class TestPairEvidenceConstruction:
    def test_pair_evidence_links_query_and_dataset(self):
        q_record = {
            "query_id": "q_0001",
            "intent": "META_ANALYSIS",
            "query": "human fMRI reward",
            "scientific_goal": "meta-analysis",
            "required_evidence": [],
            "nice_to_have": [],
            "known_failure_modes": ["resting state"],
        }
        d_record = {
            "source": "dandi", "source_id": "000004", "title": "Human ephys",
            "description": None, "species": ["human"], "modalities": ["extracellular_ephys"],
            "brain_regions": [], "tasks": [], "license": None, "url": None,
            "has_raw_data": True, "has_processed_data": False,
            "has_behavior": False, "has_trials": False,
            "data_standards": [], "metadata_json": {},
        }
        spec = decompose_query(q_record)
        evidence = dataset_evidence_from_record(d_record)
        pair = PairEvidence(
            query_id="q_0001",
            record_id="dandi:000004",
            query=spec,
            dataset=evidence,
            pooled_from=["usefulness"],
            min_rank=3,
            priority="high",
        )
        d = pair.to_dict()
        assert d["query_id"] == "q_0001"
        assert d["dataset"]["record_id"] == "dandi:000004"
        assert d["query"]["required_species"] == ["human"]
```

- [ ] **Step 6: Run — expect green (uses only already-created models)**

```bash
pytest tests/test_eval_evidence.py::TestPairEvidenceConstruction -v
```

- [ ] **Step 7: Create `scripts/eval/build_evidence.py`**

```python
#!/usr/bin/env python3
"""Build pair_evidence.jsonl from the benchmark pool, queries, and corpus.

Usage:
    python scripts/eval/build_evidence.py \
        --pool reports/eval/benchmark_pool.jsonl \
        --queries artifacts/benchmark_queries.jsonl \
        --corpus data/corpus/normalized/combined_corpus.jsonl \
        --out artifacts/eval/pair_evidence.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_search.eval.evidence import dataset_evidence_from_record, PairEvidence
from neural_search.eval.query_decomposition import load_query_specs


def _load_pool(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_corpus_index(path: Path) -> dict[str, dict]:
    """Return {record_id: record} for fast lookup."""
    index: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = f"{rec.get('source', '')}:{rec.get('source_id', '')}"
            index[rid] = rec
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pair evidence JSONL.")
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    specs = {s.query_id: s for s in load_query_specs(args.queries)}
    corpus = _load_corpus_index(args.corpus)
    pool_rows = _load_pool(args.pool)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    missing_queries = 0
    missing_corpus = 0
    written = 0

    with args.out.open("w", encoding="utf-8") as out_fh:
        for row in pool_rows:
            qid = row["query_id"]
            rid = row["record_id"]

            if qid not in specs:
                missing_queries += 1
                continue
            if rid not in corpus:
                missing_corpus += 1
                continue

            pair = PairEvidence(
                query_id=qid,
                record_id=rid,
                query=specs[qid],
                dataset=dataset_evidence_from_record(corpus[rid]),
                pooled_from=row.get("pooled_from") or [],
                min_rank=int(row.get("min_rank", 1000)),
                priority=str(row.get("priority", "normal")),
            )
            out_fh.write(json.dumps(pair.to_dict()) + "\n")
            written += 1

    print(f"Written: {written} pairs")
    if missing_queries:
        print(f"Warning: {missing_queries} pool rows had no matching query spec")
    if missing_corpus:
        print(f"Warning: {missing_corpus} pool rows had no matching corpus record")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Run the script (dry-run to check it imports cleanly)**

```bash
python scripts/eval/build_evidence.py --help
```

Expected: prints usage with `--pool`, `--queries`, `--corpus`, `--out` args.

- [ ] **Step 9: Run on real data**

```bash
mkdir -p artifacts/eval
python scripts/eval/build_evidence.py \
    --pool reports/eval/benchmark_pool.jsonl \
    --queries artifacts/benchmark_queries.jsonl \
    --corpus data/corpus/normalized/combined_corpus.jsonl \
    --out artifacts/eval/pair_evidence.jsonl
```

Expected: `Written: N pairs` (N > 0). Check `artifacts/eval/pair_evidence.jsonl` has records.

- [ ] **Step 10: Commit**

```bash
git add neural_search/eval/query_decomposition.py scripts/eval/build_evidence.py \
    tests/test_eval_evidence.py
git commit -m "feat(eval): query decomposition + build_evidence.py script"
```
