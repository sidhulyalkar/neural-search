# Optimization Plan: Latent Future Usefulness Retrieval

**Core Research Question**: How do we retrieve objects based on latent future usefulness rather than surface similarity?

**Date**: 2026-05-31  
**Status**: Planning Phase

---

## 1. Problem Definition

### Surface Similarity vs Latent Usefulness

Traditional retrieval systems optimize for **surface similarity**:
- Query: "decision-making task"
- Retrieved: Datasets with "decision-making" in title/description
- Metric: BM25/semantic similarity to query text

**Latent future usefulness** asks a different question:
- Query: "decision-making task"
- Goal: Datasets that will be *scientifically useful* for the user's actual research
- Includes datasets the user didn't know to ask for

### Why This Matters

| Scenario | Surface Match | Latent Usefulness |
|----------|--------------|-------------------|
| Same task, different modality | High | Variable (depends on analysis goal) |
| Different task, same neural signature | Low | High (novel cross-task comparison) |
| Same modality, different species | High | High for methods, Low for biology |
| Complementary datasets for meta-analysis | Low | Very High |

---

## 2. Current System Capabilities

### Implemented Components

1. **Multi-dimensional relatedness scoring** (9 dimensions)
   - Modality alignment, task compatibility, species match
   - Affordance compatibility, graph proximity
   - Reusability vs Comparability composites

2. **Graph-based implicit signals**
   - Metapath counts (same-task, same-modality paths)
   - Shared analysis affordances
   - Common linked papers

3. **Boolean constraint parsing**
   - NOT/AND/OR operators
   - Implicit constraints from domain ontology

4. **Neural signature schema** (NeuralSignatureV1)
   - Content-derived features from NWB files
   - Modality-specific statistics

### Current Limitations

- **Static weights**: Dimension weights are fixed, not adapted to user intent
- **No feedback loop**: No learning from user interactions
- **Surface-level embeddings**: Semantic embeddings capture topical similarity, not usefulness
- **No usefulness labels**: Ground truth is relevance, not actual downstream utility

---

## 3. Optimization Directions

### Direction A: Intent-Aware Weight Adaptation

**Goal**: Dynamically adjust relatedness dimension weights based on inferred user intent.

**Approach**:
```
Intent → Weight Profile
─────────────────────────
replication    → high(task, species, region), low(modality flexibility)
meta-analysis  → high(task, comparability), medium(species, modality)  
pipeline-reuse → high(modality, affordance), low(task specificity)
exploration    → uniform weights, high(graph_proximity)
```

**Implementation**:
1. Extend `QueryIntent` enum with usefulness-oriented intents
2. Create `IntentWeightProfile` mapping intent → dimension weights
3. Add intent classifier that examines query structure:
   - "datasets like X" → pipeline-reuse intent
   - "replicate finding Y" → replication intent
   - "compare across species" → meta-analysis intent

**Evaluation**: A/B test with user satisfaction surveys

---

### Direction B: Graph-Derived Usefulness Signals

**Goal**: Extract implicit usefulness signals from knowledge graph structure.

**Signals to Compute**:

1. **Co-citation strength**: Datasets cited together in papers
   ```
   CoRef(d1, d2) = |papers citing both d1 and d2| / |papers citing d1 or d2|
   ```

2. **Analysis pipeline overlap**: Shared supported analyses
   ```
   PipelineOverlap(d1, d2) = |affordances(d1) ∩ affordances(d2)| / |affordances(d1) ∪ affordances(d2)|
   ```

3. **Downstream impact**: Papers using dataset for novel analyses
   ```
   NoveltyScore(d) = count(papers using d for analysis not in original paper)
   ```

4. **Complementarity score**: Datasets that fill each other's gaps
   ```
   Complement(d1, d2) = |missing_affordances(d1) ∩ has_affordances(d2)|
   ```

**Implementation**:
1. Add paper-dataset citation edges to graph (from OpenAlex)
2. Compute co-citation matrix offline
3. Add `graph_usefulness_signals` to retrieval scoring

---

### Direction C: Neural Signature Similarity

**Goal**: Retrieve datasets with similar neural population dynamics, not just similar metadata.

**Approach**:
1. Extract signatures from NWB files via streaming
2. Build signature embedding space
3. Enable "datasets with similar dynamics to [reference]" queries

**Signature Features** (from NeuralSignatureV1):
- Firing rate statistics (mean, std, percentiles)
- ISI distribution parameters
- Population dimensionality (PCA explained variance)
- Event-aligned modulation depth
- Temporal correlation structure

**Implementation**:
1. Run signature extraction on 100+ NWB files via DANDI streaming
2. Train signature encoder (contrastive learning on same-experiment pairs)
3. Add `signature_similarity` signal to hybrid scoring

---

### Direction D: Feedback-Driven Learning

**Goal**: Learn usefulness weights from user interaction patterns.

**Data Sources**:
1. **Click-through patterns**: Which results users explore
2. **Session depth**: How deep users go into result details
3. **Download/citation tracking**: Long-term utility signals
4. **Explicit feedback**: Thumbs up/down on results

**Learning Approach**:
```
Observed: User clicked datasets D1, D3, D5 for query Q
Not clicked: D2, D4 in top-5
Signal: (Q, D1) is more useful than (Q, D2)

Update: Increase weight of dimensions where D1 > D2
        (e.g., D1 has better affordance match)
```

**Implementation**:
1. Add telemetry to search interface (privacy-preserving)
2. Build preference dataset: (query, preferred, non-preferred) triples
3. Train pairwise ranking model on dimension scores
4. Deploy as online weight adapter

---

## 4. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Implement `IntentWeightProfile` with 4 intent types
- [ ] Add intent classifier based on query patterns
- [ ] Validate on existing benchmark (expect modest gains)

### Phase 2: Graph Signals (Weeks 3-4)
- [ ] Ingest paper-dataset citations from OpenAlex
- [ ] Compute co-citation matrix for corpus
- [ ] Add `complementarity_score` to relatedness scoring
- [ ] Create test cases for complementary dataset retrieval

### Phase 3: Neural Signatures (Weeks 5-8)
- [ ] Scale signature extraction to 200+ NWB files
- [ ] Train signature encoder (or use simple Euclidean distance)
- [ ] Add "similar to [dataset]" query type
- [ ] Evaluate on manually curated similarity pairs

### Phase 4: Feedback Loop (Weeks 9-12)
- [ ] Deploy search interface with telemetry
- [ ] Collect 1000+ search sessions
- [ ] Train preference model on click data
- [ ] A/B test learned vs static weights

---

## 5. Evaluation Framework

### Metrics for Latent Usefulness

| Metric | Definition | Target |
|--------|------------|--------|
| Reusability Precision | Fraction of top-5 results that share analysis pipeline | > 0.7 |
| Complementarity Recall | Fraction of useful complementary datasets retrieved | > 0.5 |
| Novel Discovery Rate | User finds datasets they didn't know to search for | Survey |
| Downstream Utility | Retrieved datasets actually used in user's research | Long-term |

### Benchmark Extensions

1. **Complementary Pair Benchmark**
   - 100 pairs of datasets that are scientifically complementary
   - Test: Given one, does system retrieve the other?

2. **Replication Benchmark**
   - Known replication studies (paper A replicates paper B)
   - Test: Given paper A's dataset, retrieve paper B's dataset

3. **Pipeline Transfer Benchmark**
   - Analysis code that works on multiple datasets
   - Test: Given dataset where code works, retrieve other compatible datasets

---

## 6. Quick Wins

### Immediate Improvements (No New Data Required)

1. **Affordance-weighted scoring** (1 day)
   - Boost datasets sharing specific affordances mentioned in query
   - "Q-learning" query → boost datasets with Q-learning affordance

2. **Negative modality penalty** (1 day)
   - If query mentions modality, penalize mismatched modalities
   - "calcium imaging" query → strong penalty for ephys-only datasets

3. **Graph proximity normalization** (2 days)
   - Current: Raw metapath count
   - Better: Normalized by node degree (popular nodes shouldn't dominate)

4. **Intent-based RRF weights** (2 days)
   - Lookup query → high BM25 weight
   - Exploration query → high graph weight
   - Affordance query → high affordance weight

---

## 7. Success Criteria

### Short-term (3 months)
- [ ] Intent-aware retrieval deployed
- [ ] Graph usefulness signals integrated
- [ ] 10% improvement in user-reported satisfaction (survey)

### Medium-term (6 months)
- [ ] Neural signature search operational
- [ ] Feedback loop collecting data
- [ ] Complementary dataset retrieval functional

### Long-term (12 months)
- [ ] Learned weights outperform static weights
- [ ] Documented cases of novel scientific discoveries via system
- [ ] Published evaluation comparing surface vs latent retrieval

---

## 8. Research Questions

1. **Dimension orthogonality**: Are the 9 relatedness dimensions truly orthogonal, or do they collapse to fewer factors?

2. **Intent stability**: Do users have consistent intent across sessions, or does it shift query-by-query?

3. **Feedback sparsity**: Can we learn from sparse feedback (most users don't click most results)?

4. **Cross-domain transfer**: Do learned weights transfer across neuroscience subfields?

5. **Complementarity definition**: How do we formally define "complementary" datasets?

---

## Next Steps

1. **Implement quick wins** (Direction A partial, 1 week)
2. **Design complementary pair benchmark** (10 expert-annotated pairs)
3. **Prototype signature extraction pipeline** (Direction C foundation)
4. **Instrument search interface for telemetry** (Direction D foundation)
