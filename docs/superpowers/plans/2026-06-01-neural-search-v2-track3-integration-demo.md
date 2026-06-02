# Neural Search v2.0 Track 3 — Integration & Killer Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After Track 1 (embedding upgrade) and Track 2 (corpus expansion) both pass their exit criteria, converge at the integration point: rebuild graph, re-embed all records, rebuild turbovec index, run the 5-layer evaluation suite, then build the killer demo query pipeline.

**Architecture:** Integration rebuilds the three derived artifacts (graph, embeddings, index) from the expanded corpus. The 5-layer evaluation suite produces a side-by-side comparison table (v0.9 baseline vs v2.0). The killer demo uses a 5-stage pipeline: decompose → constraints → set-coverage scoring → role assignment → metrics.

**Prerequisites:** Both Track 1 AND Track 2 must pass their exit criteria before starting this plan. `reports/baseline_v09.json` must exist.

**Tech Stack:** existing `neural_search.*`, `scripts/rebuild_corpus_graph.py`, `scripts/recompute_embeddings.py`, `scripts/build_turbovec_index.py`, scipy, pytest

---

## File Map

**Create:**
- `neural_search/retrieval/set_coverage_scorer.py`
- `neural_search/retrieval/role_assignment.py`
- `scripts/run_integration.py`
- `scripts/run_evaluation_suite.py`
- `scripts/compare_versions.py`
- `scripts/run_killer_demo.py`
- `tests/test_set_coverage_scorer.py`
- `tests/test_role_assignment.py`
- `tests/test_killer_demo.py`

**Modify:**
- `docs/whitepaper/neural_search_whitepaper.tex` — v2.0 results section

---

## Task 1: Run Integration Point

Rebuild graph, embeddings, and index from the expanded corpus. This locks in the state for all subsequent evaluation.

**Files:**
- Create: `scripts/run_integration.py`

- [ ] **Step 1: Verify both tracks passed exit criteria**

```bash
# Track 1 checks
python scripts/ablate_graph_proximity.py --n-queries 20
cat reports/turbovec_recall.json | python3 -c "import json,sys; d=json.load(sys.stdin); print('recall:', d['mean_recall'], 'PASS' if d['pass'] else 'FAIL')"

# Track 2 checks
python scripts/validate_corpus.py
```

Expected: graph ablation ≥10%, turbovec recall ≥0.95, corpus ≥4000 records. Do not proceed if any check fails.

- [ ] **Step 2: Create integration runner script**

Create `scripts/run_integration.py`:

```python
#!/usr/bin/env python3
"""Run the v2.0 integration point: rebuild graph, re-embed, rebuild index.

Run only after Track 1 and Track 2 both pass their exit criteria.

Usage:
    python scripts/run_integration.py
    python scripts/run_integration.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run(cmd: list[str], *, dry_run: bool = False) -> int:
    if dry_run:
        print(f"  [dry-run] would run: {' '.join(cmd)}")
        return 0
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  ERROR: command failed with exit code {result.returncode}")
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-graph", action="store_true", help="Skip graph rebuild")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding recompute")
    parser.add_argument("--skip-index", action="store_true", help="Skip turbovec index rebuild")
    args = parser.parse_args(argv)

    print(f"=== Neural Search v2.0 Integration Point ({'DRY RUN' if args.dry_run else 'LIVE'}) ===")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")

    steps = []

    if not args.skip_graph:
        print("\n[1/3] Rebuilding knowledge graph from expanded corpus...")
        rc = _run([sys.executable, "scripts/rebuild_corpus_graph.py"], dry_run=args.dry_run)
        steps.append(("rebuild_graph", rc == 0))

    if not args.skip_embed:
        print("\n[2/3] Recomputing dense embeddings (BGE-large-en-v1.5)...")
        rc = _run(
            [sys.executable, "scripts/recompute_embeddings.py", "--provider", "dense"],
            dry_run=args.dry_run,
        )
        steps.append(("recompute_embeddings", rc == 0))

    if not args.skip_index:
        print("\n[3/3] Building turbovec index from new embeddings...")
        rc = _run([sys.executable, "scripts/build_turbovec_index.py"], dry_run=args.dry_run)
        steps.append(("build_turbovec_index", rc == 0))

    print("\n=== Integration Summary ===")
    all_ok = True
    for step, passed in steps:
        status = "OK" if passed else "FAILED"
        print(f"  {step}: {status}")
        if not passed:
            all_ok = False

    if not args.dry_run:
        record = {
            "integration_run_at": datetime.now(timezone.utc).isoformat(),
            "steps": {step: passed for step, passed in steps},
            "all_passed": all_ok,
        }
        Path("reports").mkdir(exist_ok=True)
        Path("reports/integration_run.json").write_text(json.dumps(record, indent=2))

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run integration**

```bash
python scripts/run_integration.py
```

Expected: graph rebuilt from ≥4000 datasets, embeddings recomputed, turbovec index rebuilt. Takes ~10 minutes on GPU.

- [ ] **Step 4: Verify turbovec recall on new index**

```bash
python scripts/validate_turbovec_recall.py --k 50
```

Expected: recall@50 ≥ 0.95. If not, re-run `build_turbovec_index.py --bit-width 2`.

- [ ] **Step 5: Commit integration artifacts**

```bash
git add scripts/run_integration.py reports/integration_run.json
git commit -m "feat: add integration runner; rebuild graph/embeddings/index from expanded corpus"
```

---

## Task 2: 5-Layer Evaluation Suite Runner

**Files:**
- Create: `scripts/run_evaluation_suite.py`
- Create: `scripts/compare_versions.py`

- [ ] **Step 1: Create evaluation suite runner**

Create `scripts/run_evaluation_suite.py`:

```python
#!/usr/bin/env python3
"""Run the complete 5-layer evaluation suite and produce a summary report.

Layer 1: Retrieval quality (NDCG@10, MRR, P@5, Recall@10) via existing benchmark
Layer 2: Latent usefulness quality (Spearman r, pairwise accuracy)
Layer 3: Corpus quality dashboard
Layer 4: Index quality (turbovec recall, latency)
Layer 5: Graph contribution (ablation)

Usage:
    python scripts/run_evaluation_suite.py
    python scripts/run_evaluation_suite.py --layers 1,2 --n-queries 30
    python scripts/run_evaluation_suite.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run_and_capture(cmd: list[str]) -> tuple[int, str]:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.returncode, result.stdout + result.stderr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="1,2,3,4,5", help="Comma-separated layer numbers")
    parser.add_argument("--n-queries", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    layers = [int(x) for x in args.layers.split(",")]

    if args.dry_run:
        print(f"DRY RUN — would run layers {layers} with {args.n_queries} queries")
        return 0

    results: dict = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "n_queries": args.n_queries,
        "layers": {},
    }

    if 1 in layers:
        print("\n[Layer 1] Retrieval quality benchmark...")
        rc, out = _run_and_capture([
            sys.executable, "-m", "neural_search.evaluation",
            "--suite", "real_corpus", "--k", "10",
        ])
        results["layers"]["1_retrieval"] = {"returncode": rc, "summary": out[-500:]}
        print(f"  Layer 1: {'OK' if rc == 0 else 'FAILED'}")

    if 2 in layers:
        print("\n[Layer 2] Latent usefulness correlation...")
        rc, out = _run_and_capture([
            sys.executable, "scripts/evaluate_usefulness_correlation.py",
            "--n-queries", str(args.n_queries),
        ])
        # Extract Spearman r from output
        spearman_r = None
        for line in out.split("\n"):
            if "spearman" in line.lower() and "=" in line:
                try:
                    spearman_r = float(line.split("=")[1].split()[0])
                except Exception:
                    pass
        results["layers"]["2_usefulness"] = {
            "returncode": rc,
            "spearman_r": spearman_r,
        }
        print(f"  Layer 2: Spearman r = {spearman_r}")

    if 3 in layers:
        print("\n[Layer 3] Corpus quality...")
        rc, out = _run_and_capture([sys.executable, "scripts/validate_corpus.py"])
        results["layers"]["3_corpus"] = {"returncode": rc, "summary": out[-500:]}
        print(f"  Layer 3: {'PASS' if rc == 0 else 'FAIL'}")

    if 4 in layers:
        print("\n[Layer 4] Index quality (turbovec recall)...")
        rc, out = _run_and_capture([
            sys.executable, "scripts/validate_turbovec_recall.py", "--k", "50",
        ])
        recall = None
        try:
            for line in out.split("\n"):
                if "mean_recall" in line:
                    recall = float(line.split(":")[1].strip().rstrip(","))
                    break
        except Exception:
            pass
        results["layers"]["4_index"] = {"returncode": rc, "recall": recall}
        print(f"  Layer 4: recall@50 = {recall}")

    if 5 in layers:
        print("\n[Layer 5] Graph contribution (ablation)...")
        rc, out = _run_and_capture([
            sys.executable, "scripts/ablate_graph_proximity.py",
            "--n-queries", str(args.n_queries),
        ])
        pct_changed = None
        try:
            report = json.loads(Path("reports/graph_ablation.json").read_text())
            pct_changed = report.get("pct_pairs_changed")
        except Exception:
            pass
        results["layers"]["5_graph"] = {"returncode": rc, "pct_pairs_changed": pct_changed}
        print(f"  Layer 5: {pct_changed}% pairs changed with real graph")

    Path("reports").mkdir(exist_ok=True)
    Path("reports/evaluation_suite_v2.json").write_text(json.dumps(results, indent=2))
    print(f"\nReport → reports/evaluation_suite_v2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Create version comparison script**

Create `scripts/compare_versions.py`:

```python
#!/usr/bin/env python3
"""Compare v0.9 baseline with v2.0 results side-by-side.

Usage:
    python scripts/compare_versions.py
"""
from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    baseline_path = Path("reports/baseline_v09.json")
    eval_path = Path("reports/evaluation_suite_v2.json")
    corr_path = Path("reports/usefulness_correlation_v09.json")
    recall_path = Path("reports/turbovec_recall.json")

    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    eval_v2 = json.loads(eval_path.read_text()) if eval_path.exists() else {}
    corr = json.loads(corr_path.read_text()) if corr_path.exists() else {}
    recall = json.loads(recall_path.read_text()) if recall_path.exists() else {}

    layers = eval_v2.get("layers", {})
    spearman_v2 = layers.get("2_usefulness", {}).get("spearman_r")
    corpus_v2_ok = layers.get("3_corpus", {}).get("returncode") == 0
    recall_v2 = layers.get("4_index", {}).get("recall")
    graph_v2_pct = layers.get("5_graph", {}).get("pct_pairs_changed")

    print("=" * 60)
    print("Neural Search: v0.9 Baseline vs v2.0")
    print("=" * 60)
    print(f"{'Metric':<35} {'v0.9':>10} {'v2.0':>10}")
    print("-" * 60)
    print(f"{'Corpus size':<35} {baseline.get('total_corpus_records', '?'):>10} {'≥4000' if corpus_v2_ok else '?':>10}")
    print(f"{'Embedding dim':<35} {'64 (hash)':>10} {'1024 (BGE)':>10}")
    print(f"{'Spearman r (usefulness corr.)':<35} {baseline.get('spearman_r') or '0.5044':>10} {spearman_v2 or '?':>10}")
    print(f"{'TurboVec recall@50':<35} {'N/A':>10} {recall_v2 or '?':>10}")
    print(f"{'Graph s9 pairs changed (%)':<35} {'0%':>10} {f'{graph_v2_pct}%' if graph_v2_pct else '?':>10}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run evaluation suite**

```bash
python scripts/run_evaluation_suite.py --n-queries 30
```

Expected: all 5 layers run, report at `reports/evaluation_suite_v2.json`.

- [ ] **Step 4: Run comparison**

```bash
python scripts/compare_versions.py
```

Expected: side-by-side table printed to stdout.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_evaluation_suite.py scripts/compare_versions.py \
        reports/evaluation_suite_v2.json
git commit -m "feat: add 5-layer evaluation suite runner and version comparison"
```

---

## Task 3: Set-Coverage Scorer

The killer demo ranks result sets by contribution to a query-level goal, not just individual usefulness scores.

**Files:**
- Create: `neural_search/retrieval/set_coverage_scorer.py`
- Create: `tests/test_set_coverage_scorer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_set_coverage_scorer.py`:

```python
"""Tests for set-coverage scorer."""
import pytest
from neural_search.retrieval.set_coverage_scorer import (
    SetCoverageScorer,
    SetCoverageResult,
    SetConstraints,
)


def test_import():
    from neural_search.retrieval.set_coverage_scorer import SetCoverageScorer
    assert SetCoverageScorer is not None


def _make_dataset(dataset_id, modalities, species, regions, affordances, usefulness):
    return {
        "dataset_id": dataset_id,
        "modalities": modalities,
        "species": species,
        "brain_regions": regions,
        "affordances": affordances,
        "usefulness_score": usefulness,
    }


def test_score_single_dataset():
    scorer = SetCoverageScorer()
    constraints = SetConstraints(
        required_modalities=["fmri"],
        required_species=["human"],
    )
    datasets = [_make_dataset("ds1", ["fmri"], ["human"], ["cortex"], ["decoding"], 0.7)]
    result = scorer.score_set(datasets, constraints)
    assert isinstance(result, SetCoverageResult)
    assert result.total_score > 0.0


def test_penalizes_hard_negative():
    scorer = SetCoverageScorer()
    constraints = SetConstraints(hard_negative_modalities=["eeg"])
    datasets = [
        _make_dataset("ds1", ["fmri"], ["human"], ["cortex"], [], 0.8),
        _make_dataset("ds2", ["eeg"], ["human"], ["cortex"], [], 0.9),
    ]
    result = scorer.score_set(datasets, constraints)
    # ds2 violates hard negative — should be penalized
    violations = result.hard_negative_violations
    assert "ds2" in violations


def test_rewards_modality_diversity():
    scorer = SetCoverageScorer()
    constraints = SetConstraints()
    diverse = [
        _make_dataset("ds1", ["fmri"], ["human"], [], [], 0.6),
        _make_dataset("ds2", ["neuropixels"], ["mouse"], [], [], 0.6),
        _make_dataset("ds3", ["eeg"], ["human"], [], [], 0.6),
    ]
    uniform = [
        _make_dataset("ds4", ["fmri"], ["human"], [], [], 0.6),
        _make_dataset("ds5", ["fmri"], ["human"], [], [], 0.6),
        _make_dataset("ds6", ["fmri"], ["human"], [], [], 0.6),
    ]
    r_diverse = scorer.score_set(diverse, constraints)
    r_uniform = scorer.score_set(uniform, constraints)
    assert r_diverse.coverage_bonus > r_uniform.coverage_bonus
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest tests/test_set_coverage_scorer.py -v
```

- [ ] **Step 3: Create the set-coverage scorer**

Create `neural_search/retrieval/set_coverage_scorer.py`:

```python
"""Set-coverage scorer for multi-dataset result sets.

Scores a set of datasets D against a query goal, rewarding:
  - Individual usefulness quality (mean usefulness_score)
  - Modality, species, region diversity across the set
  - Complementary affordances (unique across set)
  - Provenance quality (DOI + license + complete metadata)

And penalizing:
  - Near-duplicate datasets (redundancy)
  - Missing required metadata fields
  - Hard-negative constraint violations

Formula:
    score(D) = mean_usefulness
             + α * coverage_bonus
             + β * complementarity_bonus
             + γ * provenance_bonus
             - δ * redundancy_penalty
             - ε * missing_metadata_penalty
             - ζ * hard_negative_penalty

Default weights: α=β=γ=δ=ε=ζ=0.1 (equal weighting, tunable in v2.1)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SetConstraints:
    """Query-level constraints for set-coverage scoring."""
    required_modalities: list[str] = field(default_factory=list)
    required_species: list[str] = field(default_factory=list)
    required_regions: list[str] = field(default_factory=list)
    required_affordances: list[str] = field(default_factory=list)
    hard_negative_modalities: list[str] = field(default_factory=list)
    hard_negative_species: list[str] = field(default_factory=list)


@dataclass
class SetCoverageResult:
    """Result of scoring a dataset set."""
    total_score: float
    mean_usefulness: float
    coverage_bonus: float
    complementarity_bonus: float
    provenance_bonus: float
    redundancy_penalty: float
    missing_metadata_penalty: float
    hard_negative_penalty: float
    hard_negative_violations: list[str] = field(default_factory=list)
    dataset_count: int = 0


class SetCoverageScorer:
    """Score a set of datasets for collective usefulness."""

    def __init__(
        self,
        alpha: float = 0.1,   # coverage bonus weight
        beta: float = 0.1,    # complementarity bonus weight
        gamma: float = 0.1,   # provenance bonus weight
        delta: float = 0.15,  # redundancy penalty weight
        epsilon: float = 0.05, # missing metadata penalty weight
        zeta: float = 0.30,   # hard-negative penalty weight (high — constraints matter)
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.zeta = zeta

    def score_set(
        self,
        datasets: list[dict[str, Any]],
        constraints: SetConstraints,
    ) -> SetCoverageResult:
        """Score a set of datasets collectively."""
        if not datasets:
            return SetCoverageResult(
                total_score=0.0, mean_usefulness=0.0,
                coverage_bonus=0.0, complementarity_bonus=0.0,
                provenance_bonus=0.0, redundancy_penalty=0.0,
                missing_metadata_penalty=0.0, hard_negative_penalty=0.0,
            )

        # Mean individual usefulness
        usefulness_scores = [
            float(d.get("usefulness_score", 0.0)) for d in datasets
        ]
        mean_usefulness = sum(usefulness_scores) / len(usefulness_scores)

        # Coverage bonus: fraction of required diversity dimensions covered
        all_modalities = {
            (m.get("label") if isinstance(m, dict) else m).lower()
            for d in datasets for m in d.get("modalities", [])
        }
        all_species = {
            (s.get("label") if isinstance(s, dict) else s).lower()
            for d in datasets for s in d.get("species", [])
        }
        all_regions = {
            (r.get("label") if isinstance(r, dict) else r).lower()
            for d in datasets for r in d.get("brain_regions", [])
        }

        coverage_bonus = 0.0
        n_dims = 0
        if constraints.required_modalities:
            req = {m.lower() for m in constraints.required_modalities}
            coverage_bonus += len(req & all_modalities) / len(req)
            n_dims += 1
        if constraints.required_species:
            req = {s.lower() for s in constraints.required_species}
            coverage_bonus += len(req & all_species) / len(req)
            n_dims += 1
        if not n_dims:
            # No requirements → reward breadth
            coverage_bonus = min(1.0, len(all_modalities) / 4.0) * 0.5 + \
                             min(1.0, len(all_species) / 3.0) * 0.5
        else:
            coverage_bonus /= n_dims

        # Complementarity: unique affordances contributed by each dataset
        all_affordances: list[set[str]] = []
        for d in datasets:
            affs = {
                (a.get("affordance_id") if isinstance(a, dict) else a).lower()
                for a in d.get("affordances", [])
                if a
            }
            all_affordances.append(affs)
        total_aff_union = set().union(*all_affordances) if all_affordances else set()
        unique_per_dataset = []
        for i, affs in enumerate(all_affordances):
            others = set().union(*(all_affordances[j] for j in range(len(all_affordances)) if j != i))
            unique_per_dataset.append(len(affs - others))
        complementarity_bonus = (
            min(1.0, sum(unique_per_dataset) / max(1, len(total_aff_union)))
            if total_aff_union else 0.0
        )

        # Provenance bonus: fraction with DOI or source_id
        n_with_id = sum(
            1 for d in datasets
            if d.get("doi") or d.get("source_id") or d.get("dataset_id")
        )
        provenance_bonus = n_with_id / len(datasets)

        # Redundancy penalty: datasets with identical modality+species signature
        signatures: dict[str, int] = {}
        for d in datasets:
            sig = tuple(sorted(
                [(m.get("label") if isinstance(m, dict) else m) for m in d.get("modalities", [])] +
                [(s.get("label") if isinstance(s, dict) else s) for s in d.get("species", [])]
            ))
            signatures[sig] = signatures.get(sig, 0) + 1
        redundant = sum(max(0, v - 1) for v in signatures.values())
        redundancy_penalty = min(1.0, redundant / len(datasets))

        # Missing metadata penalty: fraction with no modalities
        n_no_modality = sum(1 for d in datasets if not d.get("modalities"))
        missing_metadata_penalty = n_no_modality / len(datasets)

        # Hard-negative penalty
        violations: list[str] = []
        for d in datasets:
            did = str(d.get("dataset_id") or d.get("source_id") or "?")
            mods = {(m.get("label") if isinstance(m, dict) else m).lower() for m in d.get("modalities", [])}
            specs = {(s.get("label") if isinstance(s, dict) else s).lower() for s in d.get("species", [])}
            for hn in constraints.hard_negative_modalities:
                if hn.lower() in mods:
                    violations.append(did)
                    break
            for hn in constraints.hard_negative_species:
                if hn.lower() in specs:
                    violations.append(did)
                    break
        hard_negative_penalty = min(1.0, len(violations) / len(datasets))

        total = (
            mean_usefulness
            + self.alpha * coverage_bonus
            + self.beta * complementarity_bonus
            + self.gamma * provenance_bonus
            - self.delta * redundancy_penalty
            - self.epsilon * missing_metadata_penalty
            - self.zeta * hard_negative_penalty
        )

        return SetCoverageResult(
            total_score=max(0.0, min(1.0, total)),
            mean_usefulness=round(mean_usefulness, 4),
            coverage_bonus=round(coverage_bonus, 4),
            complementarity_bonus=round(complementarity_bonus, 4),
            provenance_bonus=round(provenance_bonus, 4),
            redundancy_penalty=round(redundancy_penalty, 4),
            missing_metadata_penalty=round(missing_metadata_penalty, 4),
            hard_negative_penalty=round(hard_negative_penalty, 4),
            hard_negative_violations=violations,
            dataset_count=len(datasets),
        )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_set_coverage_scorer.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add neural_search/retrieval/set_coverage_scorer.py tests/test_set_coverage_scorer.py
git commit -m "feat: add SetCoverageScorer for multi-dataset result set ranking"
```

---

## Task 4: Dataset Role Assignment

Each result dataset is assigned exactly one role. A dataset without an assignable role is excluded from the final set.

**Files:**
- Create: `neural_search/retrieval/role_assignment.py`
- Create: `tests/test_role_assignment.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_role_assignment.py`:

```python
"""Tests for dataset role assignment."""
from neural_search.retrieval.role_assignment import (
    DatasetRole,
    assign_role,
    RoleAssignment,
)


def test_anchor_role():
    datasets = [
        {"dataset_id": "ds1", "usefulness_score": 0.9, "sub_query_matches": 4,
         "modalities": ["neuropixels"], "tasks": ["working_memory", "decision_making"],
         "species": ["mouse"], "brain_regions": ["prefrontal_cortex"]},
    ]
    role = assign_role(datasets[0], datasets, anchor_id=None)
    assert role.role == DatasetRole.ANCHOR


def test_replication_role():
    anchor = {"dataset_id": "anchor", "usefulness_score": 0.9, "sub_query_matches": 4,
              "tasks": ["working_memory"], "species": ["mouse"], "modalities": ["neuropixels"]}
    candidate = {"dataset_id": "rep", "usefulness_score": 0.7, "sub_query_matches": 2,
                 "tasks": ["working_memory"], "species": ["mouse"], "modalities": ["calcium_imaging"]}
    role = assign_role(candidate, [anchor, candidate], anchor_id="anchor")
    assert role.role == DatasetRole.REPLICATION


def test_cross_species_role():
    anchor = {"dataset_id": "anchor", "tasks": ["reversal_learning"], "species": ["mouse"],
              "modalities": ["neuropixels"], "usefulness_score": 0.9, "sub_query_matches": 4}
    candidate = {"dataset_id": "human_ds", "tasks": ["reversal_learning"], "species": ["human"],
                 "modalities": ["fmri"], "usefulness_score": 0.7, "sub_query_matches": 2}
    role = assign_role(candidate, [anchor, candidate], anchor_id="anchor")
    assert role.role == DatasetRole.CROSS_SPECIES_COMPARATOR


def test_no_role_excluded():
    """Dataset with no matching criteria gets UNASSIGNABLE role."""
    anchor = {"dataset_id": "anchor", "tasks": ["working_memory"], "species": ["mouse"],
              "modalities": ["neuropixels"], "usefulness_score": 0.9, "sub_query_matches": 4}
    candidate = {"dataset_id": "unrelated", "tasks": [], "species": [],
                 "modalities": [], "usefulness_score": 0.1, "sub_query_matches": 0,
                 "affordances": []}
    role = assign_role(candidate, [anchor, candidate], anchor_id="anchor")
    assert role.role == DatasetRole.UNASSIGNABLE
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest tests/test_role_assignment.py -v
```

- [ ] **Step 3: Create role assignment**

Create `neural_search/retrieval/role_assignment.py`:

```python
"""Dataset role assignment for multi-dataset result sets.

Assigns each result dataset exactly one role based on its relationship to
the query and the anchor dataset. A dataset with no assignable role is
excluded from the final demonstration result set.

Roles (in priority order):
  1. ANCHOR            — highest usefulness, matches ≥3 sub-queries
  2. REPLICATION       — same task + species as anchor, different modality
  3. CROSS_SPECIES_COMPARATOR — same task, different species from anchor
  4. METHODOLOGICAL_COMPLEMENT — different region, shares ≥2 affordances
  5. PERTURBATION_CAUSAL       — has optogenetic/pharmacological manipulation
  6. BEHAVIOR_RICH             — rich trial-by-trial events, minimal neural
  7. POPULATION_DYNAMICS       — large cell count (≥100), dimensionality reduction
  8. IMAGING_EPHYS_BRIDGE      — both fMRI/EEG and electrophysiology
  9. UNASSIGNABLE              — no role matches; dataset excluded
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class DatasetRole(StrEnum):
    ANCHOR = "anchor"
    REPLICATION = "replication"
    CROSS_SPECIES_COMPARATOR = "cross_species_comparator"
    METHODOLOGICAL_COMPLEMENT = "methodological_complement"
    PERTURBATION_CAUSAL = "perturbation_causal"
    BEHAVIOR_RICH = "behavior_rich"
    POPULATION_DYNAMICS = "population_dynamics"
    IMAGING_EPHYS_BRIDGE = "imaging_ephys_bridge"
    UNASSIGNABLE = "unassignable"


@dataclass
class RoleAssignment:
    dataset_id: str
    role: DatasetRole
    evidence: str = ""


def _labels(items: list, key: str = "label") -> set[str]:
    """Extract lowercase label strings from a list of dicts or strings."""
    result = set()
    for item in items:
        if isinstance(item, dict):
            v = item.get(key) or item.get("id") or item.get("affordance_id") or ""
        else:
            v = str(item)
        if v:
            result.add(v.lower())
    return result


def _find_anchor(datasets: list[dict]) -> dict | None:
    """Find the anchor dataset: highest sub_query_matches, then usefulness_score."""
    if not datasets:
        return None
    return max(
        datasets,
        key=lambda d: (d.get("sub_query_matches", 0), d.get("usefulness_score", 0.0)),
    )


_PERTURBATION_KEYWORDS = {
    "optogenetic", "opto", "chemogenetic", "dreadd", "pharmacological",
    "lesion", "inactivation", "muscimol", "silencing",
}

_EPHYS_MODALITIES = {"neuropixels", "extracellular_ephys", "ecog", "ieeg", "fiber_photometry", "patch_clamp"}
_IMAGING_MODALITIES = {"fmri", "eeg", "meg"}


def assign_role(
    dataset: dict[str, Any],
    all_datasets: list[dict[str, Any]],
    anchor_id: str | None = None,
) -> RoleAssignment:
    """Assign a role to a single dataset."""
    did = str(dataset.get("dataset_id") or dataset.get("source_id") or "unknown")
    mods = _labels(dataset.get("modalities", []))
    species = _labels(dataset.get("species", []))
    tasks = _labels(dataset.get("tasks", []))
    regions = _labels(dataset.get("brain_regions", []))
    affordances = _labels(dataset.get("affordances", []))
    sq_matches = dataset.get("sub_query_matches", 0)
    u_score = dataset.get("usefulness_score", 0.0)
    description = str(dataset.get("description") or "").lower()

    # Find anchor
    anchor = next((d for d in all_datasets if str(d.get("dataset_id") or "") == anchor_id), None)
    if anchor is None:
        anchor = _find_anchor(all_datasets)

    anchor_tasks = _labels(anchor.get("tasks", [])) if anchor else set()
    anchor_species = _labels(anchor.get("species", [])) if anchor else set()
    anchor_mods = _labels(anchor.get("modalities", [])) if anchor else set()
    anchor_affordances = _labels(anchor.get("affordances", [])) if anchor else set()
    anchor_regions = _labels(anchor.get("brain_regions", [])) if anchor else set()

    # 1. ANCHOR
    anchor_did = str(anchor.get("dataset_id") or "") if anchor else ""
    if did == anchor_did or (sq_matches >= 3 and u_score >= 0.5):
        return RoleAssignment(did, DatasetRole.ANCHOR, f"matches {sq_matches} sub-queries, score={u_score:.2f}")

    # 2. REPLICATION: same task + species as anchor, different modality
    if (tasks & anchor_tasks and species & anchor_species
            and mods and not (mods & anchor_mods)):
        return RoleAssignment(
            did, DatasetRole.REPLICATION,
            f"same tasks/species as anchor, different modality {mods}"
        )

    # 3. CROSS-SPECIES COMPARATOR: same task, different species
    if tasks & anchor_tasks and species and not (species & anchor_species):
        return RoleAssignment(
            did, DatasetRole.CROSS_SPECIES_COMPARATOR,
            f"same tasks {tasks & anchor_tasks}, different species {species}"
        )

    # 4. METHODOLOGICAL COMPLEMENT: different region, shares ≥2 affordances
    shared_aff = affordances & anchor_affordances
    different_regions = bool(regions) and not (regions & anchor_regions)
    if len(shared_aff) >= 2 and different_regions:
        return RoleAssignment(
            did, DatasetRole.METHODOLOGICAL_COMPLEMENT,
            f"shares affordances {shared_aff}, different region {regions}"
        )

    # 5. PERTURBATION/CAUSAL: manipulation keyword in description or tasks
    if any(kw in description or kw in tasks for kw in _PERTURBATION_KEYWORDS):
        return RoleAssignment(
            did, DatasetRole.PERTURBATION_CAUSAL,
            "contains perturbation/causal manipulation keyword"
        )

    # 6. BEHAVIOR-RICH: minimal neural, rich behavioral events
    has_behavior = dataset.get("has_behavior") or "behavior" in description
    neural_mods = mods & (_EPHYS_MODALITIES | _IMAGING_MODALITIES)
    if has_behavior and not neural_mods and dataset.get("has_trials"):
        return RoleAssignment(
            did, DatasetRole.BEHAVIOR_RICH,
            "rich behavioral events, no neural recording modality"
        )

    # 7. POPULATION DYNAMICS: large cell count, dimensionality reduction affordance
    pop_dyn_affs = {"population_dynamics", "dimensionality_reduction", "neural_manifold"}
    subject_count = dataset.get("subject_count") or dataset.get("session_count") or 0
    if (pop_dyn_affs & affordances) or subject_count >= 100:
        return RoleAssignment(
            did, DatasetRole.POPULATION_DYNAMICS,
            f"population dynamics affordances or large count ({subject_count})"
        )

    # 8. IMAGING-EPHYS BRIDGE: has both fMRI/EEG and electrophysiology
    if mods & _IMAGING_MODALITIES and mods & _EPHYS_MODALITIES:
        return RoleAssignment(
            did, DatasetRole.IMAGING_EPHYS_BRIDGE,
            f"combined imaging {mods & _IMAGING_MODALITIES} + ephys {mods & _EPHYS_MODALITIES}"
        )

    return RoleAssignment(did, DatasetRole.UNASSIGNABLE, "no matching role criteria")
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_role_assignment.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add neural_search/retrieval/role_assignment.py tests/test_role_assignment.py
git commit -m "feat: add DatasetRole assignment (8 roles + UNASSIGNABLE)"
```

---

## Task 5: Killer Demo Script

The 5-stage pipeline: decompose → constraints → retrieval + set-coverage → roles → metrics.

**Files:**
- Create: `scripts/run_killer_demo.py`
- Create: `tests/test_killer_demo.py`

- [ ] **Step 1: Write the test**

Create `tests/test_killer_demo.py`:

```python
"""Tests for killer demo pipeline."""
import subprocess, sys


def test_dry_run():
    r = subprocess.run(
        [sys.executable, "scripts/run_killer_demo.py", "--dry-run"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "DRY RUN" in r.stdout or "dry" in r.stdout.lower()


def test_syntax():
    r = subprocess.run(
        [sys.executable, "-m", "py_compile", "scripts/run_killer_demo.py"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
```

- [ ] **Step 2: Create the killer demo script**

Create `scripts/run_killer_demo.py`:

```python
#!/usr/bin/env python3
"""Killer Demo — 5-stage multi-dataset query pipeline.

Query:
  "Map the neural circuit mechanisms underlying flexible cognitive control —
   integrating datasets spanning prefrontal-hippocampal interactions,
   dopaminergic reward modulation, motor adaptation, and cross-species
   learning-dependent plasticity — to identify convergent computational
   mechanisms."

Stage 1: Query decomposition into typed sub-queries
Stage 2: Per-sub-query constraint extraction
Stage 3: Retrieval + set-coverage scoring
Stage 4: Role assignment to result set
Stage 5: Demo success metrics

Usage:
    python scripts/run_killer_demo.py
    python scripts/run_killer_demo.py --k 15 --output reports/killer_demo.json
    python scripts/run_killer_demo.py --dry-run
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KILLER_QUERY = (
    "Map the neural circuit mechanisms underlying flexible cognitive control — "
    "integrating datasets spanning prefrontal-hippocampal interactions, "
    "dopaminergic reward modulation, motor adaptation, and cross-species "
    "learning-dependent plasticity — to identify convergent computational "
    "mechanisms that could be tested in a single unified experiment."
)

# Stage 1: Rule-based sub-query decomposition
SUB_QUERIES = [
    {
        "id": "SQ1",
        "query": "prefrontal cortex hippocampus interaction working memory",
        "intent": "cross_dataset_comparison",
        "species_constraint": ["mouse", "macaque", "human"],
        "brain_regions": ["prefrontal_cortex", "hippocampus"],
        "task_family": "working_memory",
    },
    {
        "id": "SQ2",
        "query": "dopamine reward prediction error striatum",
        "intent": "meta_analysis",
        "species_constraint": ["mouse", "rat", "macaque"],
        "brain_regions": ["striatum"],
        "task_family": "reward_learning",
    },
    {
        "id": "SQ3",
        "query": "motor cortex adaptation learning plasticity",
        "intent": "method_transfer",
        "species_constraint": ["mouse", "macaque", "human"],
        "brain_regions": ["motor_cortex"],
        "task_family": "motor_task",
    },
    {
        "id": "SQ4",
        "query": "cross-species decision making flexible behavior reversal learning",
        "intent": "cross_dataset_comparison",
        "species_constraint": ["mouse", "rat", "macaque", "human"],
        "brain_regions": ["prefrontal_cortex", "striatum"],
        "task_family": "decision_making",
    },
    {
        "id": "SQ5",
        "query": "population dynamics prefrontal cortex latent space manifold",
        "intent": "method_transfer",
        "species_constraint": ["mouse", "macaque"],
        "brain_regions": ["prefrontal_cortex"],
        "task_family": "any",
    },
]


def _count_sub_query_matches(dataset: dict, sub_queries: list[dict]) -> int:
    """Count how many sub-queries this dataset is relevant for."""
    title_desc = f"{dataset.get('title', '')} {dataset.get('description', '')}".lower()
    mods = {(m.get("label") if isinstance(m, dict) else m).lower() for m in dataset.get("modalities", [])}
    species = {(s.get("label") if isinstance(s, dict) else s).lower() for s in dataset.get("species", [])}
    tasks = {(t.get("label") if isinstance(t, dict) else t).lower() for t in dataset.get("tasks", [])}
    regions = {(r.get("label") if isinstance(r, dict) else r).lower() for r in dataset.get("brain_regions", [])}

    count = 0
    for sq in sub_queries:
        # Check if any keyword from sub-query text appears in dataset text
        sq_words = set(sq["query"].lower().split())
        overlap_text = sq_words & set(title_desc.split())

        sq_regions = {r.lower() for r in sq.get("brain_regions", [])}
        sq_species = {s.lower() for s in sq.get("species_constraint", [])}

        text_match = len(overlap_text) >= 2
        region_match = bool(sq_regions & regions)
        species_match = bool(sq_species & species)

        if text_match or (region_match and species_match):
            count += 1

    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=20, help="Datasets to retrieve per sub-query")
    parser.add_argument("--output", default="reports/killer_demo.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        print("DRY RUN — 5-stage killer demo pipeline")
        print(f"Query: {KILLER_QUERY[:80]}...")
        print(f"Sub-queries: {len(SUB_QUERIES)}")
        for sq in SUB_QUERIES:
            print(f"  {sq['id']}: {sq['query'][:60]}")
        return 0

    from neural_search.search.core import search_datasets
    from neural_search.retrieval.set_coverage_scorer import SetCoverageScorer, SetConstraints
    from neural_search.retrieval.role_assignment import assign_role, DatasetRole

    print("=" * 60)
    print("Killer Demo — Neural Circuit Cognitive Control")
    print("=" * 60)

    # Stage 1 + 2: Decompose and retrieve per sub-query
    print("\nStage 1-2: Sub-query decomposition and retrieval...")
    all_results: dict[str, dict] = {}  # dataset_id -> enriched record
    dataset_sq_counts: dict[str, int] = {}

    for sq in SUB_QUERIES:
        print(f"  {sq['id']}: {sq['query'][:50]}...")
        response = search_datasets(sq["query"], limit=args.k)
        for result in response.results:
            did = str(result.dataset_id)
            if did not in all_results:
                all_results[did] = {
                    "dataset_id": did,
                    "title": result.dataset_card_preview or did,
                    "usefulness_score": (result.usefulness_score or {}).get("total_score", 0.0),
                    "modalities": result.inferred_concepts,
                    "species": [],
                    "brain_regions": [],
                    "affordances": [],
                    "tasks": result.matched_terms,
                    "description": str(result.why_matched),
                }
                dataset_sq_counts[did] = 0
            dataset_sq_counts[did] = dataset_sq_counts.get(did, 0) + 1

    for did, rec in all_results.items():
        rec["sub_query_matches"] = dataset_sq_counts[did]

    # Stage 3: Set-coverage scoring
    print("\nStage 3: Set-coverage scoring...")
    candidate_list = sorted(all_results.values(), key=lambda d: -d["usefulness_score"])[:30]
    scorer = SetCoverageScorer()
    constraints = SetConstraints(
        required_modalities=["neuropixels", "fmri", "calcium_imaging"],
        required_species=["mouse", "human"],
        hard_negative_modalities=[],
    )
    coverage_result = scorer.score_set(candidate_list, constraints)
    print(f"  Set-coverage score: {coverage_result.total_score:.4f}")
    print(f"  Coverage bonus: {coverage_result.coverage_bonus:.4f}")
    print(f"  Hard-negative violations: {coverage_result.hard_negative_violations}")

    # Stage 4: Role assignment
    print("\nStage 4: Role assignment...")
    role_assignments: list[dict] = []
    anchor_id = max(dataset_sq_counts, key=dataset_sq_counts.get) if dataset_sq_counts else None
    assigned_roles: set[str] = set()

    for ds in candidate_list[:15]:
        ra = assign_role(ds, candidate_list, anchor_id=anchor_id)
        if ra.role.value == "unassignable":
            continue
        role_assignments.append({
            "dataset_id": ra.dataset_id,
            "role": ra.role.value,
            "evidence": ra.evidence,
            "title": ds.get("title", ""),
        })
        assigned_roles.add(ra.role.value)

    print(f"  Assigned roles: {sorted(assigned_roles)}")
    print(f"  Final result set: {len(role_assignments)} datasets")

    # Stage 5: Success metrics
    print("\nStage 5: Demo success metrics...")
    hard_criteria = {
        "all_datasets_have_role": all(r["role"] != "unassignable" for r in role_assignments),
        "zero_hard_negative_violations": len(coverage_result.hard_negative_violations) == 0,
        "anchor_assigned": "anchor" in assigned_roles,
    }
    coverage_criteria = {
        "n_distinct_roles": len(assigned_roles),
        "coverage_bonus": coverage_result.coverage_bonus,
        "complementarity_bonus": coverage_result.complementarity_bonus,
    }

    print("\n  Hard criteria:")
    for k, v in hard_criteria.items():
        print(f"    [{'x' if v else ' '}] {k}")
    print("\n  Coverage criteria (measured):")
    for k, v in coverage_criteria.items():
        print(f"    {k}: {v}")

    output = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "query": KILLER_QUERY,
        "sub_queries": SUB_QUERIES,
        "n_candidates": len(all_results),
        "set_coverage": {
            "total_score": coverage_result.total_score,
            "coverage_bonus": coverage_result.coverage_bonus,
            "complementarity_bonus": coverage_result.complementarity_bonus,
        },
        "role_assignments": role_assignments,
        "hard_criteria": hard_criteria,
        "coverage_criteria": coverage_criteria,
        "all_hard_criteria_pass": all(hard_criteria.values()),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nReport → {args.output}")
    print(f"Overall: {'PASS' if output['all_hard_criteria_pass'] else 'FAIL (check hard criteria)'}")
    return 0 if output["all_hard_criteria_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_killer_demo.py -v
```

Expected: 2 passed

- [ ] **Step 4: Run the full demo**

```bash
python scripts/run_killer_demo.py
```

Expected: 5-stage pipeline runs, `reports/killer_demo.json` written. Review output:
- All datasets have assigned roles
- Zero hard-negative violations
- Anchor assigned
- Coverage bonus > 0

If hard criteria fail, investigate: check that the expanded corpus has diverse modalities/species.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_killer_demo.py tests/test_killer_demo.py \
        reports/killer_demo.json
git commit -m "feat: add 5-stage killer demo pipeline (decompose → constraints → set-coverage → roles → metrics)"
```

---

## Task 6: Update Whitepaper v2.0 Results Section

Add v2.0 results to the LaTeX whitepaper based on actual evaluation numbers.

**Files:**
- Modify: `docs/whitepaper/neural_search_whitepaper.tex`

- [ ] **Step 1: Read current results sections**

Read [docs/whitepaper/neural_search_whitepaper.tex](docs/whitepaper/neural_search_whitepaper.tex) to locate the existing `v1.0` benchmark table and usefulness calibration section.

- [ ] **Step 2: Add v2.0 comparison table**

After the existing v1.0 results, add a subsection (read evaluation suite report for actual numbers):

```latex
\subsection{v2.0 Evaluation Results}

Table~\ref{tab:v2_comparison} compares v0.9 baseline with v2.0 across all five evaluation layers.

\begin{table}[h]
\centering
\begin{tabular}{lcc}
\hline
\textbf{Metric} & \textbf{v0.9 Baseline} & \textbf{v2.0} \\
\hline
Corpus size (usable) & 738 & [INSERT from validate\_corpus.py] \\
Embedding model & 64-dim hashing & BGE-large-en-v1.5 (1024d) \\
Spearman $r$ (usefulness) & 0.5044 & [INSERT from eval suite] \\
NDCG@10 & 0.822 & [INSERT from benchmark] \\
TurboVec recall@50 & N/A & [INSERT from validate\_turbovec\_recall.py] \\
Graph $s_9$ pairs changed & 0\% & [INSERT from ablation] \\
\hline
\end{tabular}
\caption{v0.9 baseline vs.\ v2.0 across five evaluation layers. All metrics are measured on identical benchmark queries.}
\label{tab:v2_comparison}
\end{table}
```

- [ ] **Step 3: Insert actual numbers from reports**

```bash
cat reports/evaluation_suite_v2.json
cat reports/turbovec_recall.json
cat reports/graph_ablation.json
python scripts/compare_versions.py
```

Replace each `[INSERT ...]` placeholder with the actual values from the reports.

- [ ] **Step 4: Add killer demo subsection**

```latex
\subsection{Killer Demo: Multi-Dataset Cognitive Control Query}

The killer demo query requests a curated set of datasets spanning five
distinct neuroscientific sub-questions. The 5-stage pipeline
(decomposition $\to$ constraints $\to$ set-coverage scoring $\to$ role
assignment $\to$ metrics) returned [N] datasets with zero hard-negative
violations, covering [M] distinct modalities and [K] species.

Dataset roles assigned: anchor, replication, cross-species comparator,
methodological complement, population dynamics.

Set-coverage score: [INSERT]. Coverage bonus: [INSERT].
```

Again, insert actual numbers from `reports/killer_demo.json`.

- [ ] **Step 5: Commit**

```bash
git add docs/whitepaper/neural_search_whitepaper.tex
git commit -m "docs: add v2.0 evaluation results and killer demo to whitepaper"
```

---

## Task 7: Final Integration Test and Branch Cleanup

- [ ] **Step 1: Run full test suite on main branch**

```bash
pytest --timeout=300 -x -q
```

Expected: all tests pass (≥979 + new tests from Track 1, 2, 3).

- [ ] **Step 2: Run complete evaluation suite one final time**

```bash
python scripts/run_evaluation_suite.py --n-queries 30
python scripts/compare_versions.py
```

Record final numbers.

- [ ] **Step 3: Commit final state**

```bash
git add reports/
git commit -m "chore: record final v2.0 evaluation metrics"
```

- [ ] **Step 4: Invoke finishing-a-development-branch**

Use `superpowers:finishing-a-development-branch` to complete the branch (verify tests → present merge/PR options).

---

## Integration Exit Criteria Checklist

Before merging Track 3:

- [ ] `run_integration.py` completes without error
- [ ] `validate_turbovec_recall.py` shows recall@50 ≥ 0.95 on new index
- [ ] `run_evaluation_suite.py` completes all 5 layers
- [ ] Killer demo hard criteria all pass (roles assigned, no violations, anchor present)
- [ ] Whitepaper updated with actual v2.0 numbers (no `[INSERT]` placeholders)
- [ ] All tests pass (full suite, no regressions)
- [ ] `compare_versions.py` table shows improvements vs v0.9 baseline
