# Vault Enrichment + Novel Claims Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the Obsidian vault into a grounded knowledge store by synthesizing 122K literature findings into machine-readable claim objects backed by KG nodes, with a FastAPI agent interface.

**Architecture:** Allen CCF normalizes regions across findings → rule-based clustering groups findings by (region, direction, species) → Claude Haiku synthesizes consensus claims per cluster → claim nodes + edges written to the KG → vault exports paper + claim cards → FastAPI `/api/claims/*` endpoints expose claims to the multi-agent system.

**Tech Stack:** Python 3.11, httpx (Allen CCF fetch), anthropic SDK (claim synthesis), FastAPI + Pydantic (API), PyYAML (config), pytest (tests), existing `neural_search.graph.schema` (KG), existing `neural_search.obsidian` (vault export).

## Global Constraints

- Python ≥ 3.11; all scripts runnable as `python scripts/literature/<name>.py` from repo root
- REPO_ROOT pattern: `REPO_ROOT = Path(__file__).parent.parent.parent; sys.path.insert(0, str(REPO_ROOT))`
- Artifacts output to `artifacts/claims/` and `data/ontology/` (create dirs if missing)
- `data/ontology/` content is gitignored (fetched at runtime, never committed)
- Synthesis model: `claude-haiku-4-5-20251001`; prompt templates live in `configs/literature/`
- All KG node/edge types must be added to `SUPPORTED_NODE_TYPES` / `SUPPORTED_EDGE_TYPES` before use
- Obsidian writes use `safe_write_note()` from `neural_search.obsidian.io` — never write raw files
- FastAPI router follows the module-level cache pattern from `apps/api/graph_router.py`
- All tests use `tmp_path` and synthetic fixture data — never load from the real 122K corpus
- Run full test suite: `pytest tests/ -x -q` from repo root

---

## Task 1: Schema Extension — Add `claim` Node and Edge Types

**Files:**
- Modify: `neural_search/graph/schema.py`
- Test: `tests/test_claim_schema.py`

**Interfaces:**
- Produces: `"claim" in SUPPORTED_NODE_TYPES`, five new edge type strings available for KG builder use in Task 4

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_claim_schema.py
from neural_search.graph.schema import SUPPORTED_NODE_TYPES, SUPPORTED_EDGE_TYPES


def test_claim_node_type_registered():
    assert "claim" in SUPPORTED_NODE_TYPES


def test_claim_edge_types_registered():
    for edge_type in (
        "claim_supports_finding",
        "claim_contradicts_claim",
        "claim_supported_by_dataset",
        "claim_supported_by_paper",
        "claim_derived_from_finding",
    ):
        assert edge_type in SUPPORTED_EDGE_TYPES, f"Missing edge type: {edge_type}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_claim_schema.py -v
```
Expected: FAIL — `assert "claim" in SUPPORTED_NODE_TYPES`

- [ ] **Step 3: Add `claim` to `SUPPORTED_NODE_TYPES` in `neural_search/graph/schema.py`**

Open `neural_search/graph/schema.py`. Find the line with `"finding",` in `SUPPORTED_NODE_TYPES` and add below it:

```python
    "claim",
```

- [ ] **Step 4: Add five new edge types to `SUPPORTED_EDGE_TYPES`**

Find the block ending with `"finding_involves_species",` in `SUPPORTED_EDGE_TYPES` and add:

```python
    "claim_supports_finding",
    "claim_contradicts_claim",
    "claim_supported_by_dataset",
    "claim_supported_by_paper",
    "claim_derived_from_finding",
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_claim_schema.py -v
```
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add neural_search/graph/schema.py tests/test_claim_schema.py
git commit -m "feat(kg): add claim node type and five claim edge types to schema"
```

---

## Task 2: Region Normalizer — Allen CCF + UBERON Fetch and Normalization

**Files:**
- Create: `neural_search/literature/region_normalizer.py`
- Test: `tests/test_region_normalizer.py`

**Interfaces:**
- Produces:
  - `fetch_allen_ccf(cache_path: Path) -> dict[str, dict]` — `{str(structure_id): {"name": str, "acronym": str, "parent_id": int | None}}`
  - `build_name_index(structures: dict) -> dict[str, int]` — `{lowercase_name_or_acronym: structure_id}`
  - `normalize_region(region: str, name_index: dict, structures: dict) -> str` — canonical lowercase name
  - `get_parent_chain(structure_id: int, structures: dict) -> list[str]` — ancestor names root→leaf
  - `normalize_finding(finding: dict, name_index: dict, structures: dict) -> dict` — adds `regions_normalized: list[str]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_region_normalizer.py
import json
import pytest
from pathlib import Path
from neural_search.literature.region_normalizer import (
    build_name_index,
    get_parent_chain,
    normalize_region,
    normalize_finding,
)

MOCK_STRUCTURES = {
    "997": {"name": "root", "acronym": "root", "parent_id": None},
    "1080": {"name": "hippocampal formation", "acronym": "hpf", "parent_id": 997},
    "407": {"name": "hippocampus", "acronym": "hc", "parent_id": 1080},
    "382": {"name": "cornu ammonis 1", "acronym": "ca1", "parent_id": 407},
}


def test_build_name_index_includes_names_and_acronyms():
    idx = build_name_index(MOCK_STRUCTURES)
    assert idx["hippocampus"] == 407
    assert idx["hc"] == 407
    assert idx["ca1"] == 382
    assert idx["cornu ammonis 1"] == 382


def test_normalize_region_known():
    idx = build_name_index(MOCK_STRUCTURES)
    assert normalize_region("CA1", idx, MOCK_STRUCTURES) == "cornu ammonis 1"
    assert normalize_region("hippocampus", idx, MOCK_STRUCTURES) == "hippocampus"


def test_normalize_region_unknown_passthrough():
    idx = build_name_index(MOCK_STRUCTURES)
    assert normalize_region("mystery_region", idx, MOCK_STRUCTURES) == "mystery_region"


def test_get_parent_chain():
    chain = get_parent_chain(382, MOCK_STRUCTURES)
    assert "cornu ammonis 1" in chain
    assert "hippocampus" in chain
    assert "hippocampal formation" in chain


def test_normalize_finding_adds_regions_normalized():
    idx = build_name_index(MOCK_STRUCTURES)
    finding = {"finding_id": "f1", "regions": ["CA1", "mystery_region"]}
    result = normalize_finding(finding, idx, MOCK_STRUCTURES)
    assert result["regions_normalized"] == ["cornu ammonis 1", "mystery_region"]
    assert result["finding_id"] == "f1"  # original fields preserved


def test_normalize_finding_empty_regions():
    idx = build_name_index(MOCK_STRUCTURES)
    finding = {"finding_id": "f2", "regions": []}
    result = normalize_finding(finding, idx, MOCK_STRUCTURES)
    assert result["regions_normalized"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_region_normalizer.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'neural_search.literature.region_normalizer'`

- [ ] **Step 3: Create `neural_search/literature/region_normalizer.py`**

```python
"""Allen CCF + UBERON region normalization for neuroscience finding records.

Fetches the Allen Mouse Brain Atlas structure tree on first call and caches
it to data/ontology/allen_ccf.json. Subsequent calls read the cache.

UBERON_BRIDGE provides cross-species canonical mappings for terms that appear
in human/primate findings with different terminology than the Allen CCF mouse
atlas (e.g., "medial temporal lobe" → "hippocampal formation").
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

ALLEN_API_URL = (
    "https://api.brain-map.org/api/v2/data/query.json"
    "?criteria=model::Structure,rma::criteria,[graph_id$eq1]"
    ",rma::options[num_rows$eqall]"
    "&only=[id,name,acronym,parent_structure_id]"
)
DEFAULT_CACHE = Path("data/ontology/allen_ccf.json")

# Cross-species canonical mappings (human/primate → Allen CCF mouse equivalent).
# Extend this dict as new species are encountered in findings.
UBERON_BRIDGE: dict[str, str] = {
    "medial temporal lobe": "hippocampal formation",
    "mtl": "hippocampal formation",
    "temporal lobe": "temporal lobe",
    "subiculum": "subiculum",
    "entorhinal cortex": "entorhinal area",
    "ec": "entorhinal area",
    "dlpfc": "prefrontal cortex",
    "dorsolateral prefrontal cortex": "prefrontal cortex",
    "vlpfc": "prefrontal cortex",
    "ofc": "orbital frontal cortex",
    "orbitofrontal cortex": "orbital frontal cortex",
    "acc": "anterior cingulate area",
    "anterior cingulate cortex": "anterior cingulate area",
    "v1": "primary visual cortex",
    "primary visual cortex": "primary visual cortex",
    "m1": "primary motor cortex",
    "primary motor cortex": "primary motor cortex",
    "s1": "primary somatosensory area",
    "primary somatosensory cortex": "primary somatosensory area",
    "basolateral amygdala": "basolateral amygdaloid nucleus",
    "bla": "basolateral amygdaloid nucleus",
    "central amygdala": "central amygdaloid nucleus",
    "cea": "central amygdaloid nucleus",
    "dorsal striatum": "caudoputamen",
    "caudate putamen": "caudoputamen",
    "nucleus accumbens": "nucleus accumbens",
    "nac": "nucleus accumbens",
    "vta": "ventral tegmental area",
    "ventral tegmental area": "ventral tegmental area",
    "substantia nigra": "substantia nigra",
    "snr": "substantia nigra, reticular part",
    "locus coeruleus": "locus ceruleus",
    "lc": "locus ceruleus",
    "raphe": "raphe nuclei",
}


def fetch_allen_ccf(cache_path: Path = DEFAULT_CACHE) -> dict[str, dict[str, Any]]:
    """Return Allen CCF structure map {str(id): {name, acronym, parent_id}}.

    Downloads from Allen API on first call; reads cache on subsequent calls.
    """
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    logger.info("Fetching Allen CCF from %s", ALLEN_API_URL)
    response = httpx.get(ALLEN_API_URL, timeout=60)
    response.raise_for_status()
    payload = response.json()

    structures: dict[str, dict[str, Any]] = {}
    for s in payload.get("msg", []):
        structures[str(s["id"])] = {
            "name": s["name"].lower().strip(),
            "acronym": s["acronym"].lower().strip(),
            "parent_id": s.get("parent_structure_id"),
        }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(structures, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Cached %d structures to %s", len(structures), cache_path)
    return structures


def build_name_index(structures: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Return {lowercase_name_or_acronym: structure_id} for fast lookup."""
    index: dict[str, int] = {}
    for sid, s in structures.items():
        index[s["name"]] = int(sid)
        index[s["acronym"]] = int(sid)
    return index


def normalize_region(
    region: str,
    name_index: dict[str, int],
    structures: dict[str, dict[str, Any]],
) -> str:
    """Return canonical CCF name for region, or UBERON bridge term, or original string.

    Lookup order: (1) Allen CCF exact match, (2) UBERON_BRIDGE cross-species map,
    (3) original string lowercased.
    """
    cleaned = region.lower().strip()
    if cleaned in name_index:
        sid = name_index[cleaned]
        return structures[str(sid)]["name"]
    if cleaned in UBERON_BRIDGE:
        return UBERON_BRIDGE[cleaned]
    return cleaned


def get_parent_chain(
    structure_id: int,
    structures: dict[str, dict[str, Any]],
) -> list[str]:
    """Return list of ancestor names from structure up to (and including) root."""
    chain: list[str] = []
    sid: int | None = structure_id
    visited: set[int] = set()
    while sid is not None and sid not in visited:
        s = structures.get(str(sid))
        if s is None:
            break
        chain.append(s["name"])
        visited.add(sid)
        sid = s["parent_id"]
    return chain


def normalize_finding(
    finding: dict[str, Any],
    name_index: dict[str, int],
    structures: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return finding dict with added `regions_normalized` list.

    Original fields are preserved unchanged. `regions_normalized` contains
    canonical Allen CCF names (or original strings for unrecognized regions).
    """
    normalized = [
        normalize_region(r, name_index, structures)
        for r in (finding.get("regions") or [])
    ]
    return {**finding, "regions_normalized": normalized}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_region_normalizer.py -v
```
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add neural_search/literature/region_normalizer.py tests/test_region_normalizer.py
git commit -m "feat(literature): add Allen CCF region normalizer with cache"
```

---

## Task 3: Claim Synthesizer — Cluster, Synthesize, Detect Contradictions

**Files:**
- Create: `neural_search/literature/claim_synthesizer.py`
- Create: `configs/literature/synthesis_v1.yaml`
- Test: `tests/test_claim_synthesizer.py`

**Interfaces:**
- Consumes: finding dicts with `regions_normalized: list[str]` field (from Task 2)
- Produces:
  - `cluster_findings(findings: list[dict], min_size: int) -> list[dict]` — list of cluster dicts
  - `synthesize_claim(cluster: dict, client: anthropic.Anthropic, config: dict) -> dict` — claim dict
  - `detect_contradictions(claims: list[dict]) -> list[dict]` — claims with `contradicted_by` populated
  - Cluster schema: `{cluster_id, regions, direction, species, n_findings, findings: [...]}`
  - Claim schema: `{claim_id, statement, direction, regions, species, consensus_confidence, n_supporting_findings, n_contradicting_findings, magnitude_summary, timescale, evidence_strength, status, supporting_papers, supporting_datasets, contradicted_by, synthesis_model, synthesis_prompt_version, synthesized_at}`

- [ ] **Step 1: Create synthesis config**

```yaml
# configs/literature/synthesis_v1.yaml
model: claude-haiku-4-5-20251001
max_tokens: 512
temperature: 0.0
prompt_version: synthesis_v1

system_prompt: |
  You are a neuroscience research assistant synthesizing a consensus claim
  from a cluster of related scientific findings.

  Given findings that share the same brain regions, result direction, and
  species, output ONE JSON object — no other text:
  {
    "statement": "One sentence empirical claim in present tense.",
    "magnitude_summary": "Concise effect size summary, e.g. 'r=0.6-0.8' or 'N/A'.",
    "timescale": "millisecond|second|minute|hour|day|chronic|unknown",
    "evidence_strength": "direct|indirect|computational|review"
  }

  Rules:
  - statement must be a specific, falsifiable empirical claim
  - do NOT include citations or paper counts in the statement
  - timescale: pick the most common across findings; use unknown if unclear
  - evidence_strength: direct = measured directly; indirect = inferred

user_template: |
  Brain regions: {regions}
  Species: {species}
  Result direction: {direction}
  Number of findings: {n_findings}

  Sample findings:
  {findings_text}

  Write the consensus claim JSON.
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_claim_synthesizer.py
from __future__ import annotations

import pytest
from neural_search.literature.claim_synthesizer import (
    cluster_findings,
    detect_contradictions,
    _claim_id_from_cluster,
    _opposite_directions,
)

FINDINGS = [
    {
        "finding_id": f"f{i}",
        "paper_id": f"paper:openalex:W{i}",
        "finding_text": "Theta oscillations increase during spatial navigation",
        "result_direction": "increase",
        "regions_normalized": ["hippocampus"],
        "regions": ["hippocampus"],
        "species": ["mouse"],
        "tasks": ["spatial navigation"],
        "confidence": 0.9,
    }
    for i in range(5)
]

FINDINGS_MIXED = [
    *FINDINGS,
    {
        "finding_id": "f_contra",
        "paper_id": "paper:openalex:W99",
        "finding_text": "Theta oscillations decrease after lesion",
        "result_direction": "decrease",
        "regions_normalized": ["hippocampus"],
        "regions": ["hippocampus"],
        "species": ["mouse"],
        "tasks": ["spatial navigation"],
        "confidence": 0.8,
    },
]


def test_cluster_findings_groups_by_region_direction_species():
    clusters = cluster_findings(FINDINGS, min_size=2)
    assert len(clusters) == 1
    c = clusters[0]
    assert c["direction"] == "increase"
    assert "hippocampus" in c["regions"]
    assert c["n_findings"] == 5


def test_cluster_findings_min_size_filters_small_clusters():
    clusters = cluster_findings(FINDINGS[:2], min_size=3)
    assert len(clusters) == 0


def test_cluster_findings_separate_directions():
    clusters = cluster_findings(FINDINGS_MIXED, min_size=1)
    directions = {c["direction"] for c in clusters}
    assert "increase" in directions
    assert "decrease" in directions


def test_claim_id_from_cluster_is_stable():
    clusters = cluster_findings(FINDINGS, min_size=1)
    id1 = _claim_id_from_cluster(clusters[0])
    id2 = _claim_id_from_cluster(clusters[0])
    assert id1 == id2
    assert id1.startswith("node:claim:")


def test_opposite_directions():
    assert _opposite_directions("increase", "decrease")
    assert _opposite_directions("decrease", "increase")
    assert not _opposite_directions("increase", "correlation")
    assert not _opposite_directions("increase", "increase")


def test_detect_contradictions_marks_contested_claims():
    claims = [
        {
            "claim_id": "node:claim:theta_increase_001",
            "direction": "increase",
            "regions": ["hippocampus"],
            "species": ["mouse"],
            "contradicted_by": [],
            "status": "active",
        },
        {
            "claim_id": "node:claim:theta_decrease_001",
            "direction": "decrease",
            "regions": ["hippocampus"],
            "species": ["mouse"],
            "contradicted_by": [],
            "status": "active",
        },
    ]
    result = detect_contradictions(claims)
    assert "node:claim:theta_decrease_001" in result[0]["contradicted_by"]
    assert result[0]["status"] == "contested"
    assert result[1]["status"] == "contested"


def test_detect_contradictions_no_false_positives():
    claims = [
        {
            "claim_id": "node:claim:theta_increase_001",
            "direction": "increase",
            "regions": ["hippocampus"],
            "species": ["mouse"],
            "contradicted_by": [],
            "status": "active",
        },
        {
            "claim_id": "node:claim:pfc_increase_001",
            "direction": "increase",
            "regions": ["prefrontal cortex"],
            "species": ["mouse"],
            "contradicted_by": [],
            "status": "active",
        },
    ]
    result = detect_contradictions(claims)
    assert result[0]["status"] == "active"
    assert result[1]["status"] == "active"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_claim_synthesizer.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Create `neural_search/literature/claim_synthesizer.py`**

```python
"""Cluster findings into consensus claims and detect contradictions.

Three public functions:
  cluster_findings()       — group findings by (region, direction, species)
  synthesize_claim()       — LLM call to generate consensus claim text
  detect_contradictions()  — mark opposing claims as contested
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_OPPOSITE: dict[str, str] = {
    "increase": "decrease",
    "decrease": "increase",
}

DEFAULT_CONFIG = Path("configs/literature/synthesis_v1.yaml")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _opposite_directions(a: str, b: str) -> bool:
    return _OPPOSITE.get(a) == b


def _cluster_key(finding: dict[str, Any]) -> tuple[str, ...]:
    regions = tuple(sorted(finding.get("regions_normalized") or finding.get("regions") or []))
    direction = finding.get("result_direction", "other")
    species = tuple(sorted(finding.get("species") or []))
    return regions + (direction,) + species


def _claim_id_from_cluster(cluster: dict[str, Any]) -> str:
    key = json.dumps(
        {"regions": sorted(cluster["regions"]), "direction": cluster["direction"], "species": sorted(cluster["species"])},
        sort_keys=True,
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:8]
    slug = "_".join(cluster["regions"][:2] + [cluster["direction"]]).replace(" ", "_")[:40]
    return f"node:claim:{slug}_{digest}"


def cluster_findings(
    findings: list[dict[str, Any]],
    min_size: int = 3,
) -> list[dict[str, Any]]:
    """Group findings by (normalized_regions, direction, species).

    Returns clusters with >= min_size findings.
    Each cluster: {cluster_id, regions, direction, species, n_findings, findings}
    """
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for f in findings:
        key = _cluster_key(f)
        buckets[key].append(f)

    clusters = []
    for key, group in buckets.items():
        if len(group) < min_size:
            continue
        regions = list(dict.fromkeys(
            r for f in group for r in (f.get("regions_normalized") or f.get("regions") or [])
        ))
        species = list(dict.fromkeys(s for f in group for s in (f.get("species") or [])))
        direction = group[0].get("result_direction", "other")
        cluster = {
            "regions": regions,
            "direction": direction,
            "species": species,
            "n_findings": len(group),
            "findings": group,
        }
        cluster["cluster_id"] = _claim_id_from_cluster(cluster)
        clusters.append(cluster)

    return clusters


def _load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def synthesize_claim(
    cluster: dict[str, Any],
    client: Any,  # anthropic.Anthropic
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call Claude to generate a consensus claim for a finding cluster.

    Returns a claim dict ready for KG ingestion.
    Config keys: model, max_tokens, temperature, system_prompt, user_template, prompt_version.
    """
    if config is None:
        config = _load_config()

    sample = cluster["findings"][:10]
    findings_text = "\n".join(
        f"- {f.get('finding_text', '')[:200]}" for f in sample
    )
    user_message = config.get("user_template", "").format(
        regions=", ".join(cluster["regions"]),
        species=", ".join(cluster["species"]) or "unspecified",
        direction=cluster["direction"],
        n_findings=cluster["n_findings"],
        findings_text=findings_text,
    )

    response = client.messages.create(
        model=config.get("model", "claude-haiku-4-5-20251001"),
        max_tokens=config.get("max_tokens", 512),
        temperature=config.get("temperature", 0.0),
        system=config.get("system_prompt", ""),
        messages=[{"role": "user", "content": user_message}],
    )
    raw = response.content[0].text.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse synthesis response: %s", raw[:200])
        parsed = {
            "statement": raw[:500],
            "magnitude_summary": "N/A",
            "timescale": "unknown",
            "evidence_strength": "indirect",
        }

    paper_ids = list(dict.fromkeys(
        f.get("paper_id", "") for f in cluster["findings"] if f.get("paper_id")
    ))
    dataset_ids = list(dict.fromkeys(
        d for f in cluster["findings"] for d in (f.get("linked_datasets") or [])
    ))

    return {
        "claim_id": cluster["cluster_id"],
        "statement": parsed.get("statement", ""),
        "direction": cluster["direction"],
        "regions": cluster["regions"],
        "species": cluster["species"],
        "consensus_confidence": round(
            sum(f.get("confidence", 0.0) for f in cluster["findings"]) / cluster["n_findings"], 3
        ),
        "n_supporting_findings": cluster["n_findings"],
        "n_contradicting_findings": 0,
        "magnitude_summary": parsed.get("magnitude_summary", "N/A"),
        "timescale": parsed.get("timescale", "unknown"),
        "evidence_strength": parsed.get("evidence_strength", "indirect"),
        "status": "active",
        "supporting_papers": paper_ids[:20],
        "supporting_datasets": dataset_ids[:20],
        "contradicted_by": [],
        "synthesis_model": config.get("model", "claude-haiku-4-5-20251001"),
        "synthesis_prompt_version": config.get("prompt_version", "synthesis_v1"),
        "synthesized_at": _now(),
    }


def detect_contradictions(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find pairs of claims with opposing directions that share a brain region.

    Mutates and returns the claims list with `contradicted_by` and `status` updated.
    Status becomes "contested" when any contradiction is found.
    """
    claims = [dict(c) for c in claims]  # shallow copy each claim

    for i, a in enumerate(claims):
        for j, b in enumerate(claims):
            if i >= j:
                continue
            if not _opposite_directions(a["direction"], b["direction"]):
                continue
            shared_regions = set(a["regions"]) & set(b["regions"])
            if not shared_regions:
                continue
            if b["claim_id"] not in a["contradicted_by"]:
                a["contradicted_by"].append(b["claim_id"])
                a["status"] = "contested"
                a["n_contradicting_findings"] = a.get("n_contradicting_findings", 0) + b.get("n_supporting_findings", 1)
            if a["claim_id"] not in b["contradicted_by"]:
                b["contradicted_by"].append(a["claim_id"])
                b["status"] = "contested"
                b["n_contradicting_findings"] = b.get("n_contradicting_findings", 0) + a.get("n_supporting_findings", 1)

    return claims
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_claim_synthesizer.py -v
```
Expected: PASS (7 tests)

- [ ] **Step 6: Commit**

```bash
git add neural_search/literature/claim_synthesizer.py configs/literature/synthesis_v1.yaml tests/test_claim_synthesizer.py
git commit -m "feat(literature): add claim synthesizer — cluster, synthesize, contradiction detection"
```

---

## Task 4: Claim KG Builder — Write Claim Nodes + Edges

**Files:**
- Create: `neural_search/literature/claim_kg_builder.py`
- Test: `tests/test_claim_kg_builder.py`

**Interfaces:**
- Consumes: claim dicts from Task 3 (`synthesize_claim()` output); existing `KnowledgeGraph` object
- Produces: `add_claims_to_graph(graph: KnowledgeGraph, claims_path: Path) -> dict[str, int]` — returns `{"claims_added": N, "edges_added": M}`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_claim_kg_builder.py
from __future__ import annotations

import json
from pathlib import Path

from neural_search.graph.schema import KnowledgeGraph, make_node_id, validate_graph
from neural_search.literature.claim_kg_builder import add_claims_to_graph


def _write_claims(path: Path, claims: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(c) for c in claims), encoding="utf-8")


SAMPLE_CLAIM = {
    "claim_id": "node:claim:hippocampus_increase_abc12345",
    "statement": "Theta oscillations increase during spatial navigation in mouse hippocampus",
    "direction": "increase",
    "regions": ["hippocampus"],
    "species": ["mouse"],
    "consensus_confidence": 0.87,
    "n_supporting_findings": 5,
    "n_contradicting_findings": 0,
    "magnitude_summary": "r=0.7",
    "timescale": "millisecond",
    "evidence_strength": "direct",
    "status": "active",
    "supporting_papers": ["paper:openalex:W123"],
    "supporting_datasets": ["dandi:000026"],
    "contradicted_by": [],
    "synthesis_model": "claude-haiku-4-5-20251001",
    "synthesis_prompt_version": "synthesis_v1",
    "synthesized_at": "2026-06-21T00:00:00+00:00",
}


def test_add_claims_creates_claim_node(tmp_path):
    claims_path = tmp_path / "claims.jsonl"
    _write_claims(claims_path, [SAMPLE_CLAIM])

    graph = KnowledgeGraph()
    stats = add_claims_to_graph(graph, claims_path)

    assert stats["claims_added"] == 1
    assert SAMPLE_CLAIM["claim_id"] in graph.nodes


def test_add_claims_creates_supported_by_dataset_edge(tmp_path):
    claims_path = tmp_path / "claims.jsonl"
    _write_claims(claims_path, [SAMPLE_CLAIM])

    graph = KnowledgeGraph()
    add_claims_to_graph(graph, claims_path)

    edge_ids = list(graph.edges.keys())
    dataset_edges = [e for e in edge_ids if "claim_supported_by_dataset" in e]
    assert len(dataset_edges) == 1


def test_add_claims_creates_supported_by_paper_edge(tmp_path):
    claims_path = tmp_path / "claims.jsonl"
    _write_claims(claims_path, [SAMPLE_CLAIM])

    graph = KnowledgeGraph()
    add_claims_to_graph(graph, claims_path)

    edge_ids = list(graph.edges.keys())
    paper_edges = [e for e in edge_ids if "claim_supported_by_paper" in e]
    assert len(paper_edges) == 1


def test_add_claims_graph_validates(tmp_path):
    claims_path = tmp_path / "claims.jsonl"
    _write_claims(claims_path, [SAMPLE_CLAIM])

    graph = KnowledgeGraph()
    add_claims_to_graph(graph, claims_path)
    validate_graph(graph)  # raises on invalid graph


def test_add_claims_skips_missing_claim_id(tmp_path):
    claims_path = tmp_path / "claims.jsonl"
    bad_claim = {"statement": "no id here"}
    _write_claims(claims_path, [bad_claim])

    graph = KnowledgeGraph()
    stats = add_claims_to_graph(graph, claims_path)
    assert stats["claims_added"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_claim_kg_builder.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create `neural_search/literature/claim_kg_builder.py`**

```python
"""Add synthesized claim nodes and edges to the knowledge graph.

Follows the exact pattern of neural_search.literature.kg_builder.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neural_search.graph.schema import (
    GraphEvidence,
    KnowledgeGraph,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    make_edge_id,
    make_node_id,
)

logger = logging.getLogger(__name__)

BUILDER_NAME = "neural_search.literature.claim_kg_builder"
BUILDER_VERSION = "v0.1.0"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _evidence(*, source_id: str, source_field: str, text: str | None, confidence: float) -> GraphEvidence:
    return GraphEvidence(
        evidence_id=f"evidence:claim:{source_id}:{source_field}",
        source_type="claim_synthesis",
        source_id=source_id,
        source_field=source_field,
        evidence_text=text,
        confidence=confidence,
        extractor_name=BUILDER_NAME,
        extractor_version=BUILDER_VERSION,
    )


def _add_node(graph: KnowledgeGraph, node: KnowledgeGraphNode) -> bool:
    if node.node_id in graph.nodes:
        return False
    graph.nodes[node.node_id] = node
    return True


def _add_edge(graph: KnowledgeGraph, edge: KnowledgeGraphEdge) -> bool:
    if edge.edge_id in graph.edges:
        return False
    graph.edges[edge.edge_id] = edge
    return True


def _claim_node(claim: dict[str, Any]) -> KnowledgeGraphNode:
    claim_id = claim["claim_id"]
    confidence = float(claim.get("consensus_confidence", 0.5))
    ev = _evidence(
        source_id=claim_id,
        source_field="statement",
        text=claim.get("statement"),
        confidence=confidence,
    )
    return KnowledgeGraphNode(
        node_id=claim_id,
        node_type="claim",
        label=(claim.get("statement") or claim_id)[:200],
        aliases=[claim_id],
        source_ids=[claim_id],
        properties={
            "direction": claim.get("direction"),
            "regions": claim.get("regions", []),
            "species": claim.get("species", []),
            "n_supporting_findings": claim.get("n_supporting_findings", 0),
            "n_contradicting_findings": claim.get("n_contradicting_findings", 0),
            "magnitude_summary": claim.get("magnitude_summary"),
            "timescale": claim.get("timescale"),
            "evidence_strength": claim.get("evidence_strength"),
            "status": claim.get("status", "active"),
            "contradicted_by": claim.get("contradicted_by", []),
            "synthesis_model": claim.get("synthesis_model"),
            "synthesis_prompt_version": claim.get("synthesis_prompt_version"),
            "synthesized_at": claim.get("synthesized_at"),
        },
        evidence=[ev],
        confidence=confidence,
        created_at=claim.get("synthesized_at") or _now(),
    )


def add_claims_to_graph(
    graph: KnowledgeGraph,
    claims_path: Path,
) -> dict[str, int]:
    """Read claims JSONL and add claim nodes + edges to graph.

    Edge types created:
      claim_supported_by_dataset  (claim → dataset)
      claim_supported_by_paper    (claim → paper)
    """
    stats = {"claims_added": 0, "edges_added": 0}

    for claim in _iter_jsonl(claims_path):
        claim_id = claim.get("claim_id")
        if not claim_id:
            continue

        node = _claim_node(claim)
        if _add_node(graph, node):
            stats["claims_added"] += 1

        confidence = float(claim.get("consensus_confidence", 0.5))
        ev = node.evidence[0]

        for dataset_id in claim.get("supporting_datasets") or []:
            # Ensure dataset placeholder node exists
            ds_node_id = make_node_id("dataset", dataset_id)
            if ds_node_id not in graph.nodes:
                _add_node(
                    graph,
                    KnowledgeGraphNode(
                        node_id=ds_node_id,
                        node_type="dataset",
                        label=dataset_id,
                        aliases=[dataset_id],
                        source_ids=[dataset_id],
                        properties={"placeholder": True},
                        confidence=0.35,
                        created_at=_now(),
                    ),
                )
            edge = KnowledgeGraphEdge(
                edge_id=make_edge_id(claim_id, "claim_supported_by_dataset", ds_node_id),
                source_node_id=claim_id,
                target_node_id=ds_node_id,
                edge_type="claim_supported_by_dataset",
                evidence=[ev],
                confidence=confidence,
                created_at=_now(),
            )
            if _add_edge(graph, edge):
                stats["edges_added"] += 1

        for paper_id in claim.get("supporting_papers") or []:
            paper_parts = paper_id.split(":")
            p_node_id = make_node_id("paper", *paper_parts[1:]) if len(paper_parts) > 1 else make_node_id("paper", paper_id)
            if p_node_id not in graph.nodes:
                _add_node(
                    graph,
                    KnowledgeGraphNode(
                        node_id=p_node_id,
                        node_type="paper",
                        label=paper_id,
                        aliases=[paper_id],
                        source_ids=[paper_id],
                        properties={"placeholder": True},
                        confidence=0.35,
                        created_at=_now(),
                    ),
                )
            edge = KnowledgeGraphEdge(
                edge_id=make_edge_id(claim_id, "claim_supported_by_paper", p_node_id),
                source_node_id=claim_id,
                target_node_id=p_node_id,
                edge_type="claim_supported_by_paper",
                evidence=[ev],
                confidence=confidence,
                created_at=_now(),
            )
            if _add_edge(graph, edge):
                stats["edges_added"] += 1

    return stats
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_claim_kg_builder.py -v
```
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add neural_search/literature/claim_kg_builder.py tests/test_claim_kg_builder.py
git commit -m "feat(kg): add claim KG builder — claim nodes + supported_by edges"
```

---

## Task 5: Pipeline Scripts — Normalize → Cluster → Synthesize → Detect → Ingest

**Files:**
- Create: `scripts/literature/normalize_regions.py`
- Create: `scripts/literature/cluster_findings.py`
- Create: `scripts/literature/synthesize_claims.py`
- Create: `scripts/literature/detect_contradictions.py`
- Create: `scripts/literature/ingest_claims_to_kg.py`

**Interfaces:**
- Consumes: `artifacts/literature/findings_tier1_ollama.jsonl` (input to first script)
- Produces pipeline:
  - `artifacts/claims/findings_normalized.jsonl`
  - `artifacts/claims/finding_clusters.jsonl`
  - `artifacts/claims/claims_raw.jsonl`
  - `artifacts/claims/claims_validated.jsonl`
  - `data/graph/claims_kg.jsonl`

- [ ] **Step 1: Create `scripts/literature/normalize_regions.py`**

```python
"""Normalize brain region strings in findings using Allen CCF.

Usage: python scripts/literature/normalize_regions.py
Input:  artifacts/literature/findings_tier1_ollama.jsonl
Output: artifacts/claims/findings_normalized.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from neural_search.literature.region_normalizer import (
    build_name_index,
    fetch_allen_ccf,
    normalize_finding,
)

INPUT_PATH = REPO_ROOT / "artifacts/literature/findings_tier1_ollama.jsonl"
OUTPUT_PATH = REPO_ROOT / "artifacts/claims/findings_normalized.jsonl"
CACHE_PATH = REPO_ROOT / "data/ontology/allen_ccf.json"


def main() -> None:
    print("Fetching/loading Allen CCF...")
    structures = fetch_allen_ccf(CACHE_PATH)
    name_index = build_name_index(structures)
    print(f"  {len(structures)} structures loaded")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with INPUT_PATH.open(encoding="utf-8") as fin, OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            finding = json.loads(line)
            normalized = normalize_finding(finding, name_index, structures)
            fout.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            count += 1
            if count % 10000 == 0:
                print(f"  {count} findings normalized...")

    print(f"Done. {count} findings written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `scripts/literature/cluster_findings.py`**

```python
"""Cluster normalized findings by (region, direction, species).

Usage: python scripts/literature/cluster_findings.py [--min-size N]
Input:  artifacts/claims/findings_normalized.jsonl
Output: artifacts/claims/finding_clusters.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from neural_search.literature.claim_synthesizer import cluster_findings

INPUT_PATH = REPO_ROOT / "artifacts/claims/findings_normalized.jsonl"
OUTPUT_PATH = REPO_ROOT / "artifacts/claims/finding_clusters.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-size", type=int, default=3)
    args = parser.parse_args()

    print(f"Loading findings from {INPUT_PATH}...")
    findings = []
    with INPUT_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                findings.append(json.loads(line))
    print(f"  {len(findings)} findings loaded")

    print(f"Clustering (min_size={args.min_size})...")
    clusters = cluster_findings(findings, min_size=args.min_size)
    print(f"  {len(clusters)} clusters formed")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for cluster in clusters:
            # Strip individual findings to reduce file size (keep IDs only)
            slim = {k: v for k, v in cluster.items() if k != "findings"}
            slim["finding_ids"] = [fi.get("finding_id") for fi in cluster["findings"]]
            slim["findings"] = cluster["findings"]  # keep full for synthesis
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")

    print(f"Done. Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create `scripts/literature/synthesize_claims.py`**

```python
"""Synthesize one consensus claim per finding cluster via Claude Haiku.

Usage: python scripts/literature/synthesize_claims.py [--config PATH] [--max-clusters N]
Input:  artifacts/claims/finding_clusters.jsonl
Output: artifacts/claims/claims_raw.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import anthropic
import yaml
from neural_search.literature.claim_synthesizer import synthesize_claim

INPUT_PATH = REPO_ROOT / "artifacts/claims/finding_clusters.jsonl"
OUTPUT_PATH = REPO_ROOT / "artifacts/claims/claims_raw.jsonl"
DEFAULT_CONFIG = REPO_ROOT / "configs/literature/synthesis_v1.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-clusters", type=int, default=None)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    clusters = []
    with INPUT_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                clusters.append(json.loads(line))
    if args.max_clusters:
        clusters = clusters[: args.max_clusters]
    print(f"Synthesizing {len(clusters)} clusters...")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    success = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for i, cluster in enumerate(clusters):
            try:
                claim = synthesize_claim(cluster, client, config)
                fout.write(json.dumps(claim, ensure_ascii=False) + "\n")
                success += 1
            except Exception as e:
                print(f"  ERROR cluster {i}: {e}", file=sys.stderr)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(clusters)} done ({success} succeeded)...")

    print(f"Done. {success}/{len(clusters)} claims written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create `scripts/literature/detect_contradictions.py`**

```python
"""Detect and mark contradicting claim pairs.

Usage: python scripts/literature/detect_contradictions.py
Input:  artifacts/claims/claims_raw.jsonl
Output: artifacts/claims/claims_validated.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from neural_search.literature.claim_synthesizer import detect_contradictions

INPUT_PATH = REPO_ROOT / "artifacts/claims/claims_raw.jsonl"
OUTPUT_PATH = REPO_ROOT / "artifacts/claims/claims_validated.jsonl"


def main() -> None:
    claims = []
    with INPUT_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                claims.append(json.loads(line))
    print(f"Loaded {len(claims)} claims")

    validated = detect_contradictions(claims)
    contested = sum(1 for c in validated if c["status"] == "contested")
    print(f"  {contested} contested claims found")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for claim in validated:
            f.write(json.dumps(claim, ensure_ascii=False) + "\n")
    print(f"Done. {len(validated)} claims written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Create `scripts/literature/ingest_claims_to_kg.py`**

```python
"""Ingest validated claims into the knowledge graph.

Usage: python scripts/literature/ingest_claims_to_kg.py
Input:  artifacts/claims/claims_validated.jsonl
Output: data/graph/claims_kg.jsonl (KG JSONL format)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from neural_search.graph.schema import KnowledgeGraph, write_graph_jsonl
from neural_search.literature.claim_kg_builder import add_claims_to_graph

INPUT_PATH = REPO_ROOT / "artifacts/claims/claims_validated.jsonl"
OUTPUT_PATH = REPO_ROOT / "data/graph/claims_kg.jsonl"


def main() -> None:
    print(f"Ingesting claims from {INPUT_PATH}...")
    graph = KnowledgeGraph()
    stats = add_claims_to_graph(graph, INPUT_PATH)
    print(f"  {stats['claims_added']} claim nodes added")
    print(f"  {stats['edges_added']} edges added")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_graph_jsonl(graph, OUTPUT_PATH)
    print(f"Done. KG written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Verify scripts are importable**

```bash
python -c "import scripts.literature.normalize_regions" 2>/dev/null || python scripts/literature/normalize_regions.py --help 2>&1 | head -3
python scripts/literature/cluster_findings.py --help 2>&1 | head -3
python scripts/literature/detect_contradictions.py --help 2>&1 | head -1
```
Expected: no import errors (scripts have no `--help` but should exit cleanly if inputs missing)

- [ ] **Step 7: Run full test suite to confirm no regressions**

```bash
pytest tests/ -x -q
```
Expected: all tests pass

- [ ] **Step 8: Commit**

```bash
git add scripts/literature/normalize_regions.py scripts/literature/cluster_findings.py scripts/literature/synthesize_claims.py scripts/literature/detect_contradictions.py scripts/literature/ingest_claims_to_kg.py
git commit -m "feat(pipeline): add 5-stage claim synthesis pipeline scripts"
```

---

## Task 6: Vault Templates + Export Scripts

**Files:**
- Modify: `neural_search/obsidian/templates.py`
- Modify: `neural_search/obsidian/io.py` (add vault paths constant)
- Create: `scripts/obsidian/export_literature.py`
- Create: `scripts/obsidian/export_claims.py`
- Test: add to existing `tests/test_obsidian_templates.py` or create if not present

**Interfaces:**
- Consumes: paper metadata JSONL (from `data/corpus/normalized/` shards), claims from `artifacts/claims/claims_validated.jsonl`
- Produces: `obsidian_vault/09_Literature/paper_{id}.md`, `obsidian_vault/10_Claims/cl_{slug}.md`

- [ ] **Step 1: Check if obsidian template test file exists**

```bash
ls tests/test_obsidian*.py 2>/dev/null || echo "none"
```

- [ ] **Step 2: Write failing tests for new template functions**

```python
# If tests/test_obsidian_templates.py exists, append these tests.
# If not, create the file with this content:

# tests/test_obsidian_templates.py
from neural_search.obsidian.templates import (
    paper_card_frontmatter,
    paper_card_body,
    claim_card_frontmatter,
    claim_card_body,
)

SAMPLE_PAPER = {
    "paper_id": "paper:openalex:W123",
    "doi": "10.1234/example",
    "title": "Theta oscillations in CA1",
    "authors": ["Buzsaki G"],
    "year": 2021,
    "n_findings": 3,
    "finding_ids": ["f1", "f2", "f3"],
    "linked_datasets": ["dandi:000026"],
    "modalities": ["neuropixels"],
    "regions": ["hippocampus"],
    "species": ["mouse"],
    "extraction_model": "claude-haiku-4-5-20251001",
    "extraction_prompt_version": "extraction_v2",
}

SAMPLE_CLAIM = {
    "claim_id": "node:claim:hippocampus_increase_abc12345",
    "statement": "Theta oscillations increase during spatial navigation",
    "direction": "increase",
    "regions": ["hippocampus"],
    "species": ["mouse"],
    "consensus_confidence": 0.87,
    "n_supporting_findings": 5,
    "n_contradicting_findings": 0,
    "magnitude_summary": "r=0.7",
    "timescale": "millisecond",
    "evidence_strength": "direct",
    "status": "active",
    "supporting_datasets": ["dandi:000026"],
    "supporting_papers": ["paper:openalex:W123"],
    "contradicted_by": [],
    "synthesis_model": "claude-haiku-4-5-20251001",
    "synthesis_prompt_version": "synthesis_v1",
    "synthesized_at": "2026-06-21T00:00:00+00:00",
    "agent_digest": "5 findings show theta increases during spatial nav.",
}


def test_paper_card_frontmatter_has_required_fields():
    fm = paper_card_frontmatter(SAMPLE_PAPER)
    assert fm["paper_id"] == "paper:openalex:W123"
    assert fm["type"] == "paper"
    assert fm["n_findings"] == 3
    assert "dandi:000026" in fm["linked_datasets"]


def test_claim_card_frontmatter_has_required_fields():
    fm = claim_card_frontmatter(SAMPLE_CLAIM)
    assert fm["claim_id"] == "node:claim:hippocampus_increase_abc12345"
    assert fm["type"] == "claim"
    assert fm["status"] == "active"
    assert fm["consensus_confidence"] == 0.87


def test_claim_card_body_includes_agent_digest():
    body = claim_card_body(SAMPLE_CLAIM)
    assert "Agent Digest" in body
    assert "5 findings show theta increases" in body


def test_paper_card_body_includes_title():
    body = paper_card_body(SAMPLE_PAPER)
    assert "Theta oscillations in CA1" in body
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_obsidian_templates.py -v -k "paper_card or claim_card"
```
Expected: FAIL — `ImportError`

- [ ] **Step 4: Add template functions to `neural_search/obsidian/templates.py`**

Open `neural_search/obsidian/templates.py` and append at the end:

```python
def paper_card_frontmatter(record: dict) -> dict:
    """Build frontmatter dict for a 09_Literature paper card."""
    return {
        "type": "paper",
        "paper_id": record.get("paper_id"),
        "doi": record.get("doi"),
        "title": record.get("title"),
        "authors": record.get("authors") or [],
        "year": record.get("year"),
        "n_findings": record.get("n_findings", 0),
        "finding_ids": record.get("finding_ids") or [],
        "linked_datasets": record.get("linked_datasets") or [],
        "modalities": record.get("modalities") or [],
        "regions": record.get("regions") or [],
        "species": record.get("species") or [],
        "extraction_model": record.get("extraction_model"),
        "extraction_prompt_version": record.get("extraction_prompt_version"),
        "tags": ["paper", "literature"],
    }


def paper_card_body(record: dict) -> str:
    title = record.get("title") or record.get("paper_id") or "Unknown Paper"
    authors = ", ".join(record.get("authors") or []) or "_Unknown_"
    year = record.get("year") or ""
    finding_ids = record.get("finding_ids") or []
    datasets = record.get("linked_datasets") or []

    findings_section = "\n".join(f"- finding_{fid}" for fid in finding_ids) or "_None extracted._"
    datasets_section = "\n".join(f"- [[{d}]]" for d in datasets) or "_None linked._"

    return (
        f"# {title}\n\n"
        f"**Authors:** {authors}  \n"
        f"**Year:** {year}\n\n"
        f"## Findings\n{findings_section}\n\n"
        f"## Linked Datasets\n{datasets_section}\n"
    )


def claim_card_frontmatter(claim: dict) -> dict:
    """Build frontmatter dict for a 10_Claims claim card."""
    return {
        "type": "claim",
        "claim_id": claim.get("claim_id"),
        "statement": claim.get("statement"),
        "direction": claim.get("direction"),
        "regions": claim.get("regions") or [],
        "species": claim.get("species") or [],
        "consensus_confidence": claim.get("consensus_confidence"),
        "n_supporting_findings": claim.get("n_supporting_findings", 0),
        "n_contradicting_findings": claim.get("n_contradicting_findings", 0),
        "magnitude_summary": claim.get("magnitude_summary"),
        "timescale": claim.get("timescale"),
        "evidence_strength": claim.get("evidence_strength"),
        "status": claim.get("status", "active"),
        "supporting_datasets": claim.get("supporting_datasets") or [],
        "supporting_papers": claim.get("supporting_papers") or [],
        "contradicted_by": claim.get("contradicted_by") or [],
        "synthesis_model": claim.get("synthesis_model"),
        "synthesis_prompt_version": claim.get("synthesis_prompt_version"),
        "synthesized_at": claim.get("synthesized_at"),
        "tags": ["claim", claim.get("direction", "other")],
    }


def claim_card_body(claim: dict) -> str:
    statement = claim.get("statement") or "_No statement._"
    agent_digest = claim.get("agent_digest") or "_Not yet generated._"
    supporting = claim.get("supporting_papers") or []
    contradicted = claim.get("contradicted_by") or []
    datasets = claim.get("supporting_datasets") or []

    supporting_section = "\n".join(f"- [[{p}]]" for p in supporting[:20]) or "_None._"
    contradicted_section = "\n".join(f"- [[{c}]]" for c in contradicted) or "_None._"
    datasets_section = "\n".join(f"- [[{d}]]" for d in datasets[:20]) or "_None._"

    return (
        f"# {statement}\n\n"
        f"## Agent Digest\n{agent_digest}\n\n"
        f"## Supporting Datasets\n{datasets_section}\n\n"
        f"## Supporting Papers\n{supporting_section}\n\n"
        f"## Contradicted By\n{contradicted_section}\n"
    )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_obsidian_templates.py -v -k "paper_card or claim_card"
```
Expected: PASS (4 tests)

- [ ] **Step 6: Create `scripts/obsidian/export_literature.py`**

```python
"""Export paper cards to obsidian_vault/09_Literature/.

Usage: python scripts/obsidian/export_literature.py [--vault PATH] [--findings PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from neural_search.obsidian.io import safe_write_note
from neural_search.obsidian.templates import paper_card_body, paper_card_frontmatter

DEFAULT_FINDINGS = REPO_ROOT / "artifacts/literature/findings_tier1_ollama.jsonl"
DEFAULT_LINKS = REPO_ROOT / "artifacts/literature/paper_dataset_links.jsonl"
DEFAULT_VAULT = REPO_ROOT / "obsidian_vault"


def _aggregate_findings(findings_path: Path) -> dict[str, dict]:
    """Return {paper_id: {title, authors, year, n_findings, finding_ids, regions, species, modalities}}."""
    papers: dict[str, dict] = {}
    with findings_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get("paper_id")
            if not pid:
                continue
            if pid not in papers:
                papers[pid] = {
                    "paper_id": pid,
                    "doi": rec.get("paper_doi"),
                    "title": rec.get("paper_title") or pid,
                    "authors": [],
                    "year": rec.get("year"),
                    "finding_ids": [],
                    "regions": [],
                    "species": [],
                    "modalities": [],
                    "extraction_model": rec.get("extraction_model"),
                    "extraction_prompt_version": "extraction_v2",
                    "linked_datasets": [],
                }
            p = papers[pid]
            fid = rec.get("finding_id")
            if fid and fid not in p["finding_ids"]:
                p["finding_ids"].append(fid)
            for r in rec.get("regions") or []:
                if r not in p["regions"]:
                    p["regions"].append(r)
            for s in rec.get("species") or []:
                if s not in p["species"]:
                    p["species"].append(s)
            for m in rec.get("modalities") or []:
                if m not in p["modalities"]:
                    p["modalities"].append(m)

    for p in papers.values():
        p["n_findings"] = len(p["finding_ids"])
    return papers


def _load_links(links_path: Path) -> dict[str, list[str]]:
    """Return {paper_id: [dataset_id, ...]}."""
    links: dict[str, list[str]] = defaultdict(list)
    if not links_path.exists():
        return links
    with links_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get("paper_id")
            did = rec.get("dataset_id")
            if pid and did and did not in links[pid]:
                links[pid].append(did)
    return links


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    parser.add_argument("--findings", type=Path, default=DEFAULT_FINDINGS)
    parser.add_argument("--links", type=Path, default=DEFAULT_LINKS)
    args = parser.parse_args()

    out_dir = args.vault / "09_Literature"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Aggregating findings by paper...")
    papers = _aggregate_findings(args.findings)
    links = _load_links(args.links)

    for pid, paper in papers.items():
        paper["linked_datasets"] = links.get(pid, [])
        safe_id = pid.replace(":", "_").replace("/", "_")
        path = out_dir / f"paper_{safe_id}.md"
        fm = paper_card_frontmatter(paper)
        body = paper_card_body(paper)
        safe_write_note(path, fm, body)

    print(f"Done. {len(papers)} paper cards written to {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Create `scripts/obsidian/export_claims.py`**

```python
"""Export claim cards to obsidian_vault/10_Claims/.

Usage: python scripts/obsidian/export_claims.py [--vault PATH] [--claims PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from neural_search.obsidian.io import safe_write_note
from neural_search.obsidian.templates import claim_card_body, claim_card_frontmatter

DEFAULT_CLAIMS = REPO_ROOT / "artifacts/claims/claims_validated.jsonl"
DEFAULT_VAULT = REPO_ROOT / "obsidian_vault"


def _safe_filename(claim_id: str) -> str:
    return claim_id.replace("node:claim:", "cl_").replace("/", "_").replace(":", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    args = parser.parse_args()

    out_dir = args.vault / "10_Claims"
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.claims.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            claim = json.loads(line)
            claim_id = claim.get("claim_id")
            if not claim_id:
                continue
            filename = _safe_filename(claim_id) + ".md"
            path = out_dir / filename
            fm = claim_card_frontmatter(claim)
            body = claim_card_body(claim)
            safe_write_note(path, fm, body)
            count += 1

    print(f"Done. {count} claim cards written to {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Run full test suite**

```bash
pytest tests/ -x -q
```
Expected: all tests pass

- [ ] **Step 9: Commit**

```bash
git add neural_search/obsidian/templates.py scripts/obsidian/export_literature.py scripts/obsidian/export_claims.py tests/test_obsidian_templates.py
git commit -m "feat(vault): add paper + claim card templates and export scripts"
```

---

## Task 7: Claims API Router — FastAPI Endpoints for Agent Interface

**Files:**
- Create: `apps/api/claims_router.py`
- Modify: `apps/api/main.py` (include router)
- Test: `tests/test_claims_api.py`

**Interfaces:**
- Consumes: `artifacts/claims/claims_validated.jsonl` (loaded at startup, cached in memory)
- Produces 6 endpoints:
  - `GET /api/claims` — filtered list of claims
  - `GET /api/claims/contradictions` — contested pairs
  - `GET /api/claims/gaps` — (region) combos with no claims
  - `GET /api/claims/digest` — compact agent-ready batch
  - `GET /api/claims/{claim_id}` — full single claim
  - `GET /api/claims/{claim_id}/evidence` — supporting/contradicting datasets + papers

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_claims_api.py
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

CLAIMS = [
    {
        "claim_id": "node:claim:hippocampus_increase_abc12345",
        "statement": "Theta oscillations increase during spatial navigation",
        "direction": "increase",
        "regions": ["hippocampus"],
        "species": ["mouse"],
        "consensus_confidence": 0.87,
        "n_supporting_findings": 5,
        "n_contradicting_findings": 0,
        "magnitude_summary": "r=0.7",
        "timescale": "millisecond",
        "evidence_strength": "direct",
        "status": "active",
        "supporting_datasets": ["dandi:000026"],
        "supporting_papers": ["paper:openalex:W123"],
        "contradicted_by": [],
        "synthesis_model": "claude-haiku-4-5-20251001",
        "synthesis_prompt_version": "synthesis_v1",
        "synthesized_at": "2026-06-21T00:00:00+00:00",
    },
    {
        "claim_id": "node:claim:pfc_correlation_def67890",
        "statement": "Prefrontal theta correlates with working memory load",
        "direction": "correlation",
        "regions": ["prefrontal cortex"],
        "species": ["human"],
        "consensus_confidence": 0.75,
        "n_supporting_findings": 8,
        "n_contradicting_findings": 1,
        "magnitude_summary": "r=0.5",
        "timescale": "second",
        "evidence_strength": "direct",
        "status": "contested",
        "supporting_datasets": ["openneuro:ds000120"],
        "supporting_papers": ["paper:openalex:W456"],
        "contradicted_by": ["node:claim:pfc_no_change_ghi11111"],
        "synthesis_model": "claude-haiku-4-5-20251001",
        "synthesis_prompt_version": "synthesis_v1",
        "synthesized_at": "2026-06-21T00:00:00+00:00",
    },
]


@pytest.fixture
def claims_file(tmp_path) -> Path:
    p = tmp_path / "claims_validated.jsonl"
    p.write_text("\n".join(json.dumps(c) for c in CLAIMS), encoding="utf-8")
    return p


@pytest.fixture
def client(claims_file, monkeypatch):
    from apps.api import claims_router
    monkeypatch.setattr(claims_router, "CLAIMS_PATH", claims_file)
    claims_router._claims_cache = None  # reset cache
    from apps.api.claims_router import router
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_get_claims_returns_all(client):
    resp = client.get("/api/claims")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["claims"]) == 2


def test_get_claims_filter_by_direction(client):
    resp = client.get("/api/claims?direction=increase")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["claims"][0]["direction"] == "increase"


def test_get_claims_filter_by_region(client):
    resp = client.get("/api/claims?regions=hippocampus")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1


def test_get_claims_filter_by_min_confidence(client):
    resp = client.get("/api/claims?min_confidence=0.80")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["claims"][0]["consensus_confidence"] >= 0.80


def test_get_claim_by_id(client):
    # Colons are valid URL path chars — no encoding needed
    resp = client.get("/api/claims/node%3Aclaim%3Ahippocampus_increase_abc12345")
    assert resp.status_code == 200
    data = resp.json()
    assert data["claim_id"] == "node:claim:hippocampus_increase_abc12345"


def test_get_claim_by_id_not_found(client):
    resp = client.get("/api/claims/node%3Aclaim%3Anonexistent")
    assert resp.status_code == 404


def test_get_claim_evidence(client):
    resp = client.get("/api/claims/node%3Aclaim%3Ahippocampus_increase_abc12345/evidence")
    assert resp.status_code == 200
    data = resp.json()
    assert "supporting_datasets" in data
    assert "supporting_papers" in data
    assert "dandi:000026" in data["supporting_datasets"]


def test_get_contradictions(client):
    resp = client.get("/api/claims/contradictions")
    assert resp.status_code == 200
    data = resp.json()
    assert "contested_claims" in data


def test_get_digest_returns_compact_objects(client):
    resp = client.get("/api/claims/digest")
    assert resp.status_code == 200
    data = resp.json()
    assert "claims" in data
    assert len(data["claims"]) == 2
    # digest objects must have agent_digest field
    for c in data["claims"]:
        assert "agent_digest" in c
        assert "claim_id" in c
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_claims_api.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create `apps/api/claims_router.py`**

```python
"""Claims API — FastAPI router exposing synthesized claim objects to agents.

Loads claims from artifacts/claims/claims_validated.jsonl at first request
and caches in memory. Reset _claims_cache = None to force reload.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Query

REPO_ROOT = Path(__file__).parent.parent.parent
CLAIMS_PATH = REPO_ROOT / "artifacts/claims/claims_validated.jsonl"

router = APIRouter()

_claims_cache: list[dict[str, Any]] | None = None


def _load_claims() -> list[dict[str, Any]]:
    global _claims_cache
    if _claims_cache is None:
        if not CLAIMS_PATH.exists():
            _claims_cache = []
        else:
            rows = []
            with CLAIMS_PATH.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            _claims_cache = rows
    return _claims_cache


def _agent_digest(claim: dict[str, Any]) -> str:
    """Generate a compact prior-loading sentence for an agent."""
    n = claim.get("n_supporting_findings", 0)
    n_contra = claim.get("n_contradicting_findings", 0)
    regions = ", ".join(claim.get("regions") or []) or "unspecified regions"
    direction = claim.get("direction", "correlates")
    mag = claim.get("magnitude_summary") or "N/A"
    contra_note = f" {n_contra} contradicting findings exist." if n_contra else ""
    return (
        f"{n} findings in {regions} show a {direction} effect "
        f"(magnitude: {mag}).{contra_note}"
    )


def _compact_claim(claim: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim_id": claim["claim_id"],
        "statement": claim.get("statement"),
        "confidence": claim.get("consensus_confidence"),
        "direction": claim.get("direction"),
        "regions": claim.get("regions", []),
        "species": claim.get("species", []),
        "n_evidence": claim.get("n_supporting_findings", 0),
        "status": claim.get("status", "active"),
        "contradicted_by": claim.get("contradicted_by", []),
        "supporting_datasets": claim.get("supporting_datasets", []),
        "agent_digest": claim.get("agent_digest") or _agent_digest(claim),
    }


@router.get("/api/claims")
def list_claims(
    regions: str | None = Query(None, description="Comma-separated region names"),
    species: str | None = Query(None),
    direction: str | None = Query(None),
    status: str | None = Query(None),
    min_confidence: float | None = Query(None, ge=0.0, le=1.0),
) -> dict[str, Any]:
    claims = _load_claims()

    if regions:
        region_list = [r.strip().lower() for r in regions.split(",")]
        claims = [
            c for c in claims
            if any(r in [x.lower() for x in (c.get("regions") or [])] for r in region_list)
        ]
    if species:
        species_list = [s.strip().lower() for s in species.split(",")]
        claims = [
            c for c in claims
            if any(s in [x.lower() for x in (c.get("species") or [])] for s in species_list)
        ]
    if direction:
        claims = [c for c in claims if c.get("direction") == direction]
    if status:
        claims = [c for c in claims if c.get("status") == status]
    if min_confidence is not None:
        claims = [c for c in claims if (c.get("consensus_confidence") or 0) >= min_confidence]

    return {"claims": claims, "total": len(claims)}


@router.get("/api/claims/contradictions")
def list_contradictions() -> dict[str, Any]:
    claims = _load_claims()
    contested = [c for c in claims if c.get("status") == "contested"]
    return {
        "contested_claims": contested,
        "total": len(contested),
    }


@router.get("/api/claims/gaps")
def list_gaps(
    region: str | None = Query(None),
) -> dict[str, Any]:
    """Return (region, direction) combinations that have no claims."""
    claims = _load_claims()
    covered: set[tuple[str, str]] = set()
    for c in claims:
        for r in c.get("regions") or []:
            covered.add((r.lower(), c.get("direction", "other")))

    all_directions = {"increase", "decrease", "correlation", "no_change"}
    all_regions = {r.lower() for c in claims for r in (c.get("regions") or [])}

    if region:
        all_regions = {region.lower()}

    gaps = [
        {"region": r, "direction": d}
        for r in sorted(all_regions)
        for d in sorted(all_directions)
        if (r, d) not in covered
    ]
    return {"gaps": gaps, "total": len(gaps)}


@router.get("/api/claims/digest")
def digest(
    topic: str | None = Query(None, description="Filter by keyword in statement"),
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    """Compact claim objects for agent context loading."""
    claims = _load_claims()
    if topic:
        topic_lower = topic.lower()
        claims = [c for c in claims if topic_lower in (c.get("statement") or "").lower()]
    claims = claims[:limit]
    return {
        "claims": [_compact_claim(c) for c in claims],
        "total": len(claims),
        "generated_at": datetime.now(UTC).isoformat(),
    }


@router.get("/api/claims/{claim_id}/evidence")
def get_claim_evidence(claim_id: str) -> dict[str, Any]:
    """Evidence endpoint must be defined BEFORE the bare /{claim_id} route."""
    claim_id = unquote(claim_id)
    claims = _load_claims()
    for c in claims:
        if c.get("claim_id") == claim_id:
            return {
                "claim_id": claim_id,
                "supporting_datasets": c.get("supporting_datasets", []),
                "supporting_papers": c.get("supporting_papers", []),
                "contradicted_by": c.get("contradicted_by", []),
                "n_supporting_findings": c.get("n_supporting_findings", 0),
            }
    raise HTTPException(status_code=404, detail=f"Claim not found: {claim_id}")


@router.get("/api/claims/{claim_id}")
def get_claim(claim_id: str) -> dict[str, Any]:
    """Claim IDs contain colons (node:claim:*) but no slashes — {claim_id} without
    :path is sufficient and avoids routing conflicts with /evidence."""
    claim_id = unquote(claim_id)
    claims = _load_claims()
    for c in claims:
        if c.get("claim_id") == claim_id:
            return c
    raise HTTPException(status_code=404, detail=f"Claim not found: {claim_id}")
```

- [ ] **Step 4: Wire router into `apps/api/main.py`**

Find the line `from apps.api.graph_router import router as graph_router` in `apps/api/main.py` and add below it:

```python
from apps.api.claims_router import router as claims_router
```

Then find the line `app.include_router(graph_router)` and add below it:

```python
app.include_router(claims_router)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_claims_api.py -v
```
Expected: PASS (9 tests)

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/ -x -q
```
Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add apps/api/claims_router.py apps/api/main.py tests/test_claims_api.py
git commit -m "feat(api): add /api/claims/* router with 6 endpoints for agent interface"
```

---

## Task 8: Prompt Configs + Vault Prompt Directory

**Files:**
- Create: `configs/literature/validation_v1.yaml`
- Create: `obsidian_vault/00_Project/prompts/extraction_v2.yaml` (copy of existing)
- Create: `obsidian_vault/00_Project/prompts/synthesis_v1.yaml` (copy of configs version)
- Create: `obsidian_vault/00_Project/prompts/validation_v1.yaml`

- [ ] **Step 1: Create `configs/literature/validation_v1.yaml`**

```yaml
# configs/literature/validation_v1.yaml
prompt_version: validation_v1
description: >
  Used by detect_contradictions.py to validate claim pairs.
  This config documents the contradiction detection logic.
  Contradiction = opposing directions (increase vs decrease)
  sharing at least one brain region.
opposing_direction_pairs:
  - [increase, decrease]
  - [decrease, increase]
min_shared_regions: 1
contested_threshold: 1   # any contradiction → contested status
```

- [ ] **Step 2: Create vault prompt directory and copy configs**

```bash
mkdir -p obsidian_vault/00_Project/prompts
cp configs/literature/synthesis_v1.yaml obsidian_vault/00_Project/prompts/synthesis_v1.yaml
cp configs/literature/validation_v1.yaml obsidian_vault/00_Project/prompts/validation_v1.yaml
# Copy the existing extraction config if it exists
cp configs/literature/extraction_v2.yaml obsidian_vault/00_Project/prompts/extraction_v2.yaml 2>/dev/null || \
  cp configs/literature/*.yaml obsidian_vault/00_Project/prompts/ 2>/dev/null || true
```

- [ ] **Step 3: Create the Claim Coverage Dataview dashboard**

```bash
cat > obsidian_vault/08_Dashboards/Claim\ Coverage.md << 'EOF'
# Claim Coverage Dashboard

## Active Claims by Direction

```dataview
TABLE direction, regions, consensus_confidence, n_supporting_findings, status
FROM "10_Claims"
WHERE type = "claim" AND status = "active"
SORT consensus_confidence DESC
LIMIT 50
```

## Contested Claims

```dataview
TABLE statement, direction, regions, n_contradicting_findings
FROM "10_Claims"
WHERE status = "contested"
SORT n_contradicting_findings DESC
```

## Paper Coverage

```dataview
TABLE title, n_findings, linked_datasets
FROM "09_Literature"
WHERE type = "paper"
SORT n_findings DESC
LIMIT 30
```
EOF
```

- [ ] **Step 4: Verify vault directory structure**

```bash
ls obsidian_vault/00_Project/prompts/
ls obsidian_vault/08_Dashboards/
ls obsidian_vault/09_Literature/ 2>/dev/null || echo "(empty — run export_literature.py to populate)"
ls obsidian_vault/10_Claims/ 2>/dev/null || echo "(empty — run export_claims.py to populate)"
```
Expected: prompts directory has at least synthesis_v1.yaml and validation_v1.yaml

- [ ] **Step 5: Run full test suite one final time**

```bash
pytest tests/ -x -q
```
Expected: all tests pass

- [ ] **Step 6: Final commit**

```bash
git add configs/literature/validation_v1.yaml obsidian_vault/00_Project/prompts/ "obsidian_vault/08_Dashboards/Claim Coverage.md"
git commit -m "feat(vault): add prompt versioning directory and Claim Coverage dashboard"
```

---

## Running the Full Pipeline

After all tasks are complete, run the end-to-end pipeline:

```bash
# 1. Fetch Allen CCF and normalize regions (~5 min for 122K findings)
python scripts/literature/normalize_regions.py

# 2. Cluster findings (fast, rule-based)
python scripts/literature/cluster_findings.py --min-size 3

# 3. Synthesize claims via Claude Haiku (costs ~$0.50 for 1K clusters)
python scripts/literature/synthesize_claims.py --max-clusters 100  # start small

# 4. Detect contradictions
python scripts/literature/detect_contradictions.py

# 5. Ingest into KG
python scripts/literature/ingest_claims_to_kg.py

# 6. Export to vault
python scripts/obsidian/export_literature.py
python scripts/obsidian/export_claims.py
```

Then start the API and verify:
```bash
# Start API
uvicorn apps.api.main:app --reload

# Check claims endpoint
curl http://localhost:8000/api/claims | python -m json.tool | head -30
curl "http://localhost:8000/api/claims/digest?topic=hippocampus&limit=5" | python -m json.tool
```
