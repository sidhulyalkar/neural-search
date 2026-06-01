# Latent Search v0.3 Roadmap

**Status:** Design Document
**Created:** 2026-05-23
**Target:** Neural Search v0.3 Alpha

---

## Executive Summary

The latent search system provides similarity-based retrieval using neural and behavioral feature fingerprints extracted from datasets and sessions. In v0.3, we focus on **deterministic, metadata-derived fingerprints** rather than learned embeddings, preparing interfaces for future model integration while avoiding expensive cloud training.

---

## Current State (v0.2)

### Implemented Components

| Component | Location | Status |
|-----------|----------|--------|
| Feature schema | `neural_search/latent/embedding_schema.py` | Working |
| Feature extraction | `neural_search/latent/summary_features.py` | Placeholder |
| Similarity search | `neural_search/latent/search.py` | Prototype |
| Hybrid search API | `neural_search/search/core.py:hybrid_search_with_latent` | Scaffold |

### Feature Types Defined

**Neural Features:**
- `NEURAL_SUMMARY_STATISTICS` - Aggregate neural statistics
- `FIRING_RATE_HISTOGRAM` - Firing rate distributions
- `SPIKE_TRAIN_STATISTICS` - Spike train properties
- `LFP_POWER_SPECTRUM` - LFP frequency content
- `CALCIUM_TRACE_SUMMARY` - Calcium imaging summaries
- `NEURAL_EMBEDDING` - Placeholder for learned embeddings

**Behavioral Features:**
- `EVENT_HISTOGRAM` - Event type distributions
- `BEHAVIOR_TRANSITION_SUMMARY` - State transition patterns
- `TASK_STATE_LABELS` - Task structure encoding
- `BEHAVIOR_TRANSITION_MATRIX` - Transition probabilities
- `TASK_STATE_SEQUENCE` - State sequences
- `TRIAL_OUTCOME_DISTRIBUTION` - Trial outcomes
- `BEHAVIOR_EMBEDDING` - Placeholder for learned embeddings

**Quality:**
- `SESSION_QC_VECTOR` - Data quality indicators

### Current Limitations

1. **Feature extraction is metadata-only**: `summary_features.py` derives features from dataset metadata, not actual NWB/BIDS file inspection
2. **Embeddings are hash-based**: Deterministic SHA256 hashes, not semantic embeddings
3. **No file-level inspection**: Cannot extract unit counts, trial structures, or electrode layouts from actual data files
4. **No learned models**: No trained encoder for neural activity patterns
5. **No indexed retrieval**: Linear scan over all sessions for similarity search

---

## v0.3 Scope: Deterministic Fingerprints

### Goals

1. Define deterministic session/dataset fingerprints derivable from NWB/BIDS metadata without file parsing
2. Implement fingerprint extraction for normalized records
3. Prepare clean interfaces for v0.4+ learned embeddings
4. Do NOT train any large models
5. Do NOT require cloud compute

### Fingerprint Features (v0.3)

These features can be derived from dataset metadata and lightweight file inspection:

| Feature | Source | Description |
|---------|--------|-------------|
| `number_of_units` | NWB metadata / BIDS participants | Approximate unit/channel count |
| `number_of_trials` | NWB intervals / BIDS events | Trial count from trial tables |
| `event_type_names` | NWB intervals / BIDS events.tsv | Event type vocabulary |
| `sampling_rates` | NWB acquisition / BIDS sidecar | Recording sample rates |
| `brain_regions` | Normalized metadata | Brain region coverage |
| `electrode_groups` | NWB electrode table / BIDS electrodes | Electrode group names |
| `behavioral_timeseries_present` | NWB / BIDS | Whether continuous behavior exists |
| `trial_interval_tables_present` | NWB / BIDS | Whether trial structure exists |
| `modality_composition` | Normalized metadata | Modality distribution vector |
| `file_format_composition` | Normalized metadata | File format distribution |
| `events_tsv_columns` | BIDS events.tsv | Column vocabulary |
| `subject_count` | NWB / BIDS | Number of subjects |
| `run_count` | BIDS | Number of runs |
| `session_duration_estimate` | NWB / BIDS | Approximate recording duration |
| `task_vocabulary` | Normalized metadata | Task labels present |

### Fingerprint Schema

```python
@dataclass
class DatasetFingerprint:
    """Deterministic fingerprint for dataset similarity."""

    dataset_id: str
    fingerprint_version: str = "v0.3.0"

    # Structural features
    subject_count: int | None = None
    session_count: int | None = None
    run_count: int | None = None
    trial_count: int | None = None
    unit_count: int | None = None

    # Duration features
    total_duration_seconds: float | None = None
    mean_session_duration_seconds: float | None = None

    # Content features (normalized histograms)
    modality_histogram: dict[str, float] = field(default_factory=dict)
    brain_region_histogram: dict[str, float] = field(default_factory=dict)
    task_histogram: dict[str, float] = field(default_factory=dict)
    event_type_histogram: dict[str, float] = field(default_factory=dict)
    file_format_histogram: dict[str, float] = field(default_factory=dict)

    # Capability flags
    has_trials: bool = False
    has_continuous_behavior: bool = False
    has_spike_data: bool = False
    has_lfp_data: bool = False
    has_calcium_data: bool = False
    has_imaging_data: bool = False

    # Sampling info
    primary_sampling_rate: float | None = None
    sampling_rates: list[float] = field(default_factory=list)

    # Data standard
    data_standard: str | None = None

    # Extraction metadata
    extracted_at: str = ""
    extraction_method: str = "metadata_derived"
```

### Implementation Plan

#### 1. Add fingerprint module

```
neural_search/latent/fingerprint.py
├── DatasetFingerprint  # Schema class
├── extract_fingerprint_from_normalized()  # From NormalizedDatasetRecord
├── fingerprint_similarity()  # Cosine/weighted similarity
└── fingerprint_to_vector()  # Convert to fixed-length vector
```

#### 2. Integrate with normalized records

```python
# In NormalizedDatasetRecord
class NormalizedDatasetRecord(BaseModel):
    # ... existing fields ...
    fingerprint: DatasetFingerprint | None = None
```

#### 3. Add fingerprint-based search

```python
def search_by_fingerprint(
    query_fingerprint: DatasetFingerprint,
    corpus: list[NormalizedDatasetRecord],
    top_k: int = 10,
) -> list[FingerprintSearchResult]:
    """Find datasets with similar structural fingerprints."""
```

#### 4. Hybrid integration

```python
def hybrid_search_with_fingerprint(
    query: str,
    query_dataset_id: str | None,
    corpus: list[NormalizedDatasetRecord],
    fingerprint_weight: float = 0.2,
    limit: int = 10,
) -> SearchResponse:
    """Combine ontology + fingerprint similarity."""
```

---

## v0.4 Preview: Learned Embeddings

### Planned Components

| Component | Description |
|-----------|-------------|
| `SentenceTransformerProvider` | Field-specific text embeddings |
| `SPECTEREmbedding` | Scientific paper embeddings |
| `NeuralActivityEncoder` | Learned neural pattern encoder |
| `BehaviorPatternEncoder` | Learned behavioral sequence encoder |
| `MultiModalFusion` | Cross-modal embedding fusion |

### Interface Preparation

v0.3 fingerprint interfaces are designed to be extended:

```python
# v0.3: Deterministic fingerprint
class FingerprintProvider(Protocol):
    def extract(self, record: NormalizedDatasetRecord) -> DatasetFingerprint: ...
    def similarity(self, a: DatasetFingerprint, b: DatasetFingerprint) -> float: ...

# v0.4: Can swap in learned provider
class LearnedEmbeddingProvider(Protocol):
    def embed(self, record: NormalizedDatasetRecord) -> np.ndarray: ...
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float: ...
```

---

## v0.5 Preview: File-Level Inspection

### NWB Inspection

```python
def inspect_nwb_file(path: str) -> NWBInspectionResult:
    """Extract structural features from NWB file."""
    return NWBInspectionResult(
        num_units=len(nwb.units) if nwb.units else 0,
        num_electrodes=len(nwb.electrodes) if nwb.electrodes else 0,
        trial_count=len(nwb.trials) if nwb.trials else 0,
        trial_columns=list(nwb.trials.colnames) if nwb.trials else [],
        timeseries_names=[ts.name for ts in nwb.acquisition.values()],
        processing_modules=list(nwb.processing.keys()),
        # ...
    )
```

### BIDS Inspection

```python
def inspect_bids_dataset(path: str) -> BIDSInspectionResult:
    """Extract structural features from BIDS dataset."""
    return BIDSInspectionResult(
        subject_count=len(layout.get_subjects()),
        session_count=len(layout.get_sessions()),
        run_count=len(layout.get_runs()),
        task_names=layout.get_tasks(),
        modalities=layout.get_modalities(),
        events_columns=extract_events_columns(layout),
        # ...
    )
```

---

## Non-Goals for v0.3

1. **No model training**: Do not train any encoder models
2. **No cloud compute**: All extraction runs locally
3. **No file downloading**: Use metadata only, no large file fetches
4. **No vector database**: Use in-memory similarity for now
5. **No embeddings API**: No external embedding service calls

---

## Success Criteria

### v0.3 Fingerprint Milestones

- [ ] `DatasetFingerprint` schema implemented
- [ ] `extract_fingerprint_from_normalized()` working
- [ ] `fingerprint_similarity()` computes meaningful similarity
- [ ] Tests cover fingerprint extraction determinism
- [ ] At least 100 normalized records have fingerprints
- [ ] Fingerprint search returns reasonable results
- [ ] Hybrid search with fingerprint weight configurable
- [ ] No learned models introduced

### Interface Readiness

- [ ] `FingerprintProvider` protocol defined
- [ ] Clear extension point for `LearnedEmbeddingProvider`
- [ ] Fingerprint vectors compatible with future vector indices
- [ ] Documentation explains v0.4 upgrade path

---

## Risks and Mitigations

### Risk: Fingerprints too coarse

**Mitigation:** Start with high-dimensional histograms; add feature engineering iteratively.

### Risk: Metadata incompleteness

**Mitigation:** Fingerprints handle missing fields gracefully; use `None` for unknown values.

### Risk: Scope creep into file parsing

**Mitigation:** Strictly defer file inspection to v0.5. v0.3 uses only normalized metadata.

### Risk: Similarity not meaningful

**Mitigation:** Validate fingerprint similarity against human judgment for 20+ dataset pairs.

---

## Open Questions

1. Should fingerprints be stored in the database or as derived artifacts?
2. What weighting should be used for different fingerprint components?
3. How should fingerprint versions be tracked as the schema evolves?
4. Should fingerprint vectors be normalized before similarity computation?

---

## Dependencies

| Dependency | Version | Required For |
|------------|---------|--------------|
| numpy | 1.24+ | Vector operations |
| pydantic | 2.5+ | Schema validation |
| None (new) | - | v0.3 has no new dependencies |

---

## Timeline

| Phase | Deliverable | Dependencies |
|-------|-------------|--------------|
| Phase 5a | Fingerprint schema | WS3 (normalized schema) |
| Phase 5b | Extraction from normalized | Phase 5a |
| Phase 5c | Similarity function | Phase 5b |
| Phase 5d | Hybrid integration | Phase 5c, WS7 (scoring) |
| Phase 5e | Tests and validation | All above |

---

## References

- [instruction_set:WS10](../neural_search_v0_3_claude_codex_instructionset.xml)
- [existing:latent/search.py](../neural_search/latent/search.py)
- [existing:latent/embedding_schema.py](../neural_search/latent/embedding_schema.py)
- [existing:latent/summary_features.py](../neural_search/latent/summary_features.py)
