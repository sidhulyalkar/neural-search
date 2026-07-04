# Testing & Expansion Roadmap
# Post-Ingestion: From 255K Papers to Full Neuroscience Coverage

**Date:** 2026-06-17  
**Status:** Active — tier1 extraction running (550/255,940 papers, 262 findings)  
**Goal:** After tier1 extraction completes, systematically validate quality, expand to full OpenAlex coverage, build spatial ontology, and turn the system into a search tool that spans all neuroscience ever done — across space, time, mechanisms, and hidden relationships.

---

## Where We Are

| Layer | Status | Scale |
|---|---|---|
| Dataset corpus | Done | 7,171 datasets, 15+ sources |
| OpenAlex tier1 papers | Ingested | 255,940 papers |
| Paper-dataset links | Done | 7,171 / 7,171 linked |
| Finding extraction (tier1) | Running | 550 done, ~38h remaining |
| Knowledge graph | Current | 7,593 nodes / 31,920 edges |
| Literature search | Implemented | Offline, awaiting findings |
| Spatial ontology | Partial | ~122 canonical regions |
| Evaluation | Infrastructure ready | 0 gold qrels, 13 LFs built |

---

## Sprint 2: Quality Validation & KG Rebuild (immediately after extraction)

**Trigger:** `findings_tier1_ollama.checkpoint.json` reaches 255,940 processed IDs.

### Task 2.1: Finding extraction quality audit

Run a precision spot-check before committing findings to the KG.

```bash
python3 -c "
import json, random
from pathlib import Path
findings = [json.loads(l) for l in Path('artifacts/literature/findings_tier1_ollama.jsonl').open()]
random.seed(42)
sample = random.sample(findings, 100)
# Write audit sheet
with open('artifacts/eval/finding_audit_100.jsonl', 'w') as f:
    for finding in sample:
        f.write(json.dumps(finding) + '\n')
print(f'Total findings: {len(findings)}')
print(f'Yield rate: {len(findings)/255940*100:.1f}%')
"
```

**Audit criteria (manual spot-check, 30 minutes):**
- Is the finding_text a real claim from the abstract? (not hallucinated)
- Is the result_direction correct?
- Are regions/species canonical and accurate?
- Confidence threshold: reject findings with `confidence < 0.6` and `regions == []` and `species == []` (likely ML/methods abstracts mis-classified)

**Pass criterion:** >80% of sampled findings are accurate and traceable to the abstract.

### Task 2.2: KG rebuild with findings

```bash
python scripts/literature/build_literature_kg.py \
  --findings artifacts/literature/findings_tier1_ollama.jsonl \
  --papers data/corpus/normalized/openalex_neuro \
  --links artifacts/literature/paper_dataset_links.jsonl \
  --out artifacts/literature/kg_with_findings.json
```

Expected new node types and counts after rebuild:
- `paper` nodes: ~255,940 (all ingested papers)
- `finding` nodes: ~80,000–95,000 (37–47% yield from tier1)
- `venue` nodes: ~5,000–15,000 (journals and conferences)
- New edges: `paper_reports_finding`, `finding_involves_region/species/modality/task`, `dataset_linked_to_paper`

**Validate KG growth:**
```bash
python3 -c "
import json
from pathlib import Path
manifest = json.loads(Path('artifacts/field_state/current_manifest.json').read_text())
print('Nodes:', manifest.get('node_count'))
print('Edges:', manifest.get('edge_count'))
"
```

Expected: node count increases by ~330K+, edge count by 500K+.

### Task 2.3: Literature search quality evaluation

Build a curated 50-query test set covering:
- Brain region queries: "hippocampal place cells spatial navigation"
- Mechanism queries: "NMDA receptor long-term potentiation"
- Species queries: "mouse barrel cortex whisker stimulation"
- Finding type queries: "dopamine reward prediction error"
- Cross-scale queries: "fMRI BOLD default mode network resting state"

Evaluate BM25 paper search and finding search:
```bash
python scripts/literature/eval_search_quality.py \
  --queries data/eval/literature_search_queries.yaml \
  --findings artifacts/literature/findings_tier1_ollama.jsonl \
  --papers data/corpus/normalized/openalex_neuro \
  --out artifacts/eval/literature_search_eval.md
```

**Accept criterion:** Top-3 results relevant to query for >80% of test queries.

---

## Sprint 3: Spatial Ontology at Scale

**Goal:** Expand from ~122 canonical regions to full hierarchical brain ontologies covering every published neuroscience study's anatomical vocabulary.

### Task 3.1: Allen CCF full hierarchy integration

Allen Common Coordinate Framework v3 has 13,000+ named regions in a tree from whole brain → hemisphere → lobe → area → layer → subregion.

```bash
# Download Allen CCF region tree (JSON)
python scripts/ontology/download_allen_ccf.py \
  --out data/ontology/allen_ccf_v3_full.json

# Integrate into KG
python scripts/ontology/build_spatial_hierarchy.py \
  --ccf data/ontology/allen_ccf_v3_full.json \
  --out data/ontology/spatial_hierarchy.jsonl
```

New KG edges: `region_is_part_of` (parent-child), `region_overlaps` (cross-atlas), `finding_involves_region` (existing, now hits 13K regions instead of 122).

**Impact:** Queries like "what is known about layer 5 pyramidal cells" now resolve through the hierarchy to include CA1 stratum pyramidale, neocortical L5, cerebellum purkinje layer, etc.

### Task 3.2: UBERON cross-species ontology

UBERON maps across species (mouse, rat, macaque, human, zebrafish). Integrate the neuroscience-relevant subset (~3,000 terms).

```bash
python scripts/ontology/download_uberon.py \
  --subset neuroscience \
  --out data/ontology/uberon_neuro.owl

python scripts/ontology/align_uberon_to_ccf.py \
  --uberon data/ontology/uberon_neuro.owl \
  --ccf data/ontology/allen_ccf_v3_full.json \
  --out data/ontology/cross_species_alignment.jsonl
```

**Impact:** A query for "hippocampus" in mouse studies and "cornu ammonis" in human studies both resolve to the same UBERON concept, enabling true cross-species comparison.

### Task 3.3: HCP MMP parcellation (360 cortical areas)

Human Connectome Project Multi-Modal Parcellation covers 360 human cortical areas with functional and structural characterization.

```bash
python scripts/ontology/integrate_hcp_mmp.py \
  --parcellation data/ontology/hcp_mmp_v1.json \
  --align-to-uberon data/ontology/uberon_neuro.owl \
  --out data/ontology/hcp_mmp_nodes.jsonl
```

**Impact:** fMRI studies referencing HCP parcellation areas become searchable and cross-referenced with findings from invasive recording studies.

### Task 3.4: Re-extract findings with spatial normalization

After ontology integration, re-run finding extraction with a spatial normalization step: map free-text regions in extracted findings to canonical UBERON/CCF IDs.

```bash
python scripts/literature/normalize_finding_regions.py \
  --findings artifacts/literature/findings_tier1_ollama.jsonl \
  --ontology data/ontology/spatial_hierarchy.jsonl \
  --uberon data/ontology/uberon_neuro.owl \
  --out artifacts/literature/findings_tier1_normalized.jsonl
```

**Validation:** Check that >70% of non-empty `regions` fields map to a canonical ontology ID.

---

## Sprint 4: Scale to Full OpenAlex Coverage

**Goal:** Expand from 255,940 tier1 papers to 1.39M tier2 (open-access with abstract) and eventually 4.36M total.

### Task 4.1: Tier 2 ingest (1.39M papers)

```bash
python scripts/ingestion/bulk_ingest_openalex.py \
  --tier tier2 \
  --out data/corpus/normalized/openalex_neuro_tier2 \
  --resume
```

Estimated time: ~40h at 0.12s/request rate limit.

### Task 4.2: Tier 2 finding extraction

With 8GB VRAM, extraction rate is ~0.54s/paper. For 1.39M papers: ~210h. Strategies to accelerate:

**Option A: Parallel extraction across model copies**
- Run 2 Ollama instances with different ports if VRAM allows (quantized 7B ≈ 5GB, so two fit marginally)

**Option B: Pre-filter by abstract quality**
- Skip papers where abstract word count < 50 (likely proceedings abstracts or review stubs)
- Skip papers where no neuroscience mesh terms match

**Option C: Upgrade to tier2 GPU**
- RTX 4090 (24GB VRAM) can run 13B-70B quantized models, 5-10x throughput
- Or use a cloud GPU for the bulk extraction sprint (A100 at ~$1.50/hr × 40h = $60)

### Task 4.3: Tier 3 planning (4.36M total)

Tier 3 includes all neuroscience-tagged OpenAlex papers regardless of open access. Abstracts available for ~40%.

Strategy: extract only for papers with DOI-linked corpus datasets (ensures extraction effort maps to searchable datasets), plus highly-cited papers (tier1 already done).

---

## Sprint 5: Cross-Scale Synthesis & Hidden Relationships

**Goal:** Connect the dots across spatial scales (from molecules to circuits), temporal scales (from milliseconds to years), and measurement modalities.

### Task 5.1: Temporal dynamics metadata

Extract from findings:
- Timescale: millisecond (spike), second (LFP), minute (calcium), hour (EEG sleep), year (longitudinal)
- Directionality over time: onset, offset, sustained, oscillatory
- Frequency bands: theta (4-8Hz), gamma (30-100Hz), delta (0.5-4Hz), etc.

Add `timescale` and `frequency_band` to `FindingRecord` schema, update extraction prompt, re-extract a subset.

### Task 5.2: Mechanistic relationship graph

Build a "mechanism graph" layer in the KG connecting:
- Neurotransmitter → receptor → downstream effect
- Gene → protein → circuit function
- Lesion/stimulation → behavior change
- Drug → molecular target → cognitive effect

Populate from extracted findings where `result_direction` is `mechanism`.

### Task 5.3: Hidden relationship discovery

Implement multi-hop graph traversal queries:

```python
# "What brain regions are jointly activated in working memory and spatial navigation?"
results = kg.traverse(
    start_nodes=["working_memory", "spatial_navigation"],  # task nodes
    edge_types=["finding_involves_task", "finding_involves_region"],
    hops=2,
    aggregation="intersection",
)

# "What datasets contain recordings from regions implicated in fear conditioning?"
results = kg.traverse(
    start_concept="fear_conditioning",
    follow=["finding_involves_task", "finding_involves_region", "dataset_linked_to_paper"],
    filter_node_type="dataset",
)
```

These become natural-language search queries surfaced in the UI.

### Task 5.4: Contradiction and consensus detection

For each finding, find "contradictory" findings in the KG:
- Same brain region, same modality, same species
- Opposite `result_direction`

Surface these as "contested findings" in search results. Also compute "consensus strength" as the proportion of findings agreeing on direction.

---

## Sprint 6: Evaluation & Benchmark

**Goal:** Establish credible, reproducible metrics that can be cited in a paper.

### Task 6.1: Finding extraction precision benchmark

Manual evaluation of 300 randomly sampled findings (100 per rater × 3 raters):
- Precision: fraction of extracted findings that are accurate, complete, and traceable to the source abstract
- Recall: estimated by checking 50 papers end-to-end and counting missed findings
- Inter-rater agreement (Cohen's κ)

**Target:** Precision > 0.82, Recall > 0.65 (estimated), κ > 0.70.

### Task 6.2: Literature search relevance benchmark

50-query evaluation set, 3 annotators, pooling depth 10:
- BM25 finding search baseline
- BM25 paper search baseline
- KG-augmented finding search (finding + associated region/species/task)
- Full pipeline: finding → paper → dataset chain

Metrics: NDCG@10, MRR, P@5. Bootstrap 95% CI.

### Task 6.3: Dataset discovery benchmark

End-to-end query-to-dataset benchmark:
- Given a neuroscience research question, does the system surface the correct publicly-known dataset?
- Ground truth: 100 hand-curated (query, relevant-dataset) pairs from published papers

Baselines: PubMed search → linked data, Google Scholar, manual OpenNeuro browse.

### Task 6.4: Frozen snapshot and reproducibility

```bash
# Freeze the final benchmark corpus
python scripts/eval/freeze_corpus_snapshot.py \
  --corpus data/corpus/normalized/combined_corpus.jsonl \
  --findings artifacts/literature/findings_tier1_normalized.jsonl \
  --kg artifacts/literature/kg_with_findings.json \
  --out artifacts/benchmark_v2/snapshot_manifest.json
```

All benchmark results reference the frozen snapshot hash so others can reproduce.

---

## Sprint 7: Retrieval Integration & Search UX

**Goal:** Surface literature findings in the main search path, not as a separate index.

### Task 7.1: Unified search across datasets, papers, and findings

Merge paper/finding results into the main `search()` response:

```python
# Current: only dataset results
results = search(query="hippocampal theta spatial navigation")

# Target: unified results
results = unified_search(query="hippocampal theta spatial navigation")
# Returns:
#   - 5 datasets matching the query
#   - 3 key findings (theta increases during spatial navigation, r=0.7)
#   - 2 papers (linked to those findings)
#   - "Related regions" panel from KG traversal
```

### Task 7.2: Finding cards in search UI

Render `FindingRecord` objects as cards in search results:
- Finding text (headline)
- Result direction badge (increase ↑ / decrease ↓ / correlation ~)
- Regions, species, modality chips
- Confidence meter
- "View paper" and "View dataset" links
- "Related findings" (from KG traversal)

### Task 7.3: Graph-driven "hidden insights" panel

For any search result, show:
- "3 datasets studying this phenomenon across different species"
- "2 contradictory findings (effect direction disputed)"
- "Most-studied co-occurring region: prefrontal cortex (47 datasets)"
- "Common experimental modality: electrophysiology (72%)"

These come directly from multi-hop KG traversal and are automatically updated as findings are added.

---

## Testing Strategy Summary

| Test Type | When | Target | Method |
|---|---|---|---|
| Finding extraction precision | After tier1 complete | >80% | Manual audit of 100 random findings |
| Finding extraction recall | After tier1 complete | >60% estimated | End-to-end check on 20 papers |
| KG connectivity | After KG rebuild | 100% dataset-paper links traversable | Automated path check |
| Literature search relevance | After KG rebuild | NDCG@10 > 0.35 | 50-query eval set |
| Spatial normalization coverage | After Sprint 3 | >70% regions map to ontology ID | Automated mapping check |
| Cross-species alignment | After Sprint 3 | >80% UBERON terms have CCF mapping | Alignment report |
| Tier2 extraction quality | After Sprint 4 | Similar to tier1 precision | Random sample audit |
| Full pipeline benchmark | Sprint 6 | Dataset discovery recall > 0.75 | 100-query curated benchmark |
| Inter-rater agreement | Sprint 6 | κ > 0.70 | 3 annotators × 100 findings |

---

## Priority Order

**Now (while extraction runs):**
1. Update whitepaper ✓
2. Keep extraction running — it's the foundation for everything else
3. Plan spatial ontology scripts

**After extraction completes (~38h):**
1. Precision audit (30-min manual check, unblocks publishing claims)
2. KG rebuild with findings
3. Literature search quality eval (50 queries)

**Next sprint (1-2 weeks):**
1. Allen CCF full hierarchy integration (13K regions)
2. UBERON cross-species alignment
3. Tier2 ingest (1.39M papers)
4. Unified search UI with finding cards

**Following sprint (2-4 weeks):**
1. Full benchmark suite (300-finding precision eval, 50-query search eval)
2. Hidden relationship discovery (multi-hop traversal)
3. Contradiction/consensus detection
4. Frozen snapshot and reproducibility package

---

## Success Definition

The system has "spanned all of neuroscience ever done" when:

1. **Coverage**: >90% of published neuroscience papers (OpenAlex tier3, 4.36M) have abstracts extracted and findings indexed.
2. **Spatial**: All brain regions in Allen CCF v3 (13K+), UBERON, and HCP MMP are traversable in the KG.
3. **Temporal**: Findings are tagged with timescale (ms→years) and frequency band where applicable.
4. **Cross-scale**: A query like "what is known about hippocampal memory consolidation" returns findings from single-unit ephys, LFP, fMRI, lesion, pharmacology, and genetics studies — all linked to available datasets.
5. **Mechanistic**: Molecular → circuit → behavior chains are traversable in the KG.
6. **Validated**: Precision > 0.80, dataset discovery recall > 0.75, with frozen reproducible benchmarks.
