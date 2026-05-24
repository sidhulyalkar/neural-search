# Embedding Model Strategy

This document defines the embedding model strategy for Neural Search v0.4, including model selection, field-specific embedding approaches, and evaluation criteria.

## Overview

Neural Search uses embeddings to enable semantic similarity search beyond exact keyword and ontology matching. The embedding strategy must:

1. Support scientific terminology and neuroscience-specific concepts
2. Enable field-specific embeddings (not one merged text blob)
3. Work offline without requiring API calls
4. Preserve interpretability alongside semantic scores
5. Allow provider swapping without breaking existing functionality

## Recommended Defaults for v0.4

### Primary Provider: HashingEmbeddingProvider

**Use Case:** CI/CD, deterministic tests, offline development

```python
from neural_search.embeddings import HashingEmbeddingProvider

provider = HashingEmbeddingProvider(dimensions=64)
```

**Advantages:**
- Zero dependencies
- Deterministic (same input → same output)
- Fast
- No model downloads required

**Limitations:**
- No semantic understanding
- Token overlap drives similarity, not meaning
- Cannot capture synonyms or paraphrases

---

### Secondary Provider: SentenceTransformerEmbeddingProvider

**Use Case:** Production semantic search, similarity ranking

```python
from neural_search.embeddings import SentenceTransformerEmbeddingProvider

provider = SentenceTransformerEmbeddingProvider(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

**Recommended Models:**

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `all-MiniLM-L6-v2` | 384 | 80MB | Fast, general-purpose, good baseline |
| `all-mpnet-base-v2` | 768 | 420MB | Higher quality, slower |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | 80MB | Optimized for Q&A, good for queries |

**Installation:**
```bash
pip install neural-search[embeddings]
# or
pip install sentence-transformers
```

---

### Future Providers (v0.5+)

#### SPECTER for Paper Abstracts

```python
# Not yet implemented
provider = SpecterEmbeddingProvider(
    model_name="allenai/specter2"
)
```

**Use Case:** Paper-to-paper similarity, citation prediction

**Why:** SPECTER is trained on scientific papers and captures research concepts better than general-purpose models.

**Limitation:** Requires separate model, higher memory, slower inference.

---

#### SciBERT for Scientific Text

```python
# Not yet implemented
provider = SciBERTEmbeddingProvider(
    model_name="allenai/scibert_scivocab_uncased"
)
```

**Use Case:** Scientific term understanding, biomedical concepts

**Why:** Pretrained on scientific literature, better vocabulary coverage for neuroscience terms.

---

#### BiomedBERT for Clinical/Medical

```python
# Not yet implemented
provider = BiomedBERTEmbeddingProvider(
    model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
)
```

**Use Case:** Clinical datasets, disease-related search

---

## Field-Specific Embedding Strategy

### Why Field-Specific?

Different fields carry different semantic meaning:

- **Title similarity:** "Similar datasets by name"
- **Task similarity:** "Datasets studying similar cognitive processes"
- **Modality similarity:** "Datasets using similar recording techniques"
- **Analysis goal similarity:** "Datasets supporting similar analyses"

Merging all fields into one text blob loses this distinction.

### Dataset Embedding Fields

```python
DATASET_EMBEDDING_FIELDS = [
    "title",                      # Direct lookup, naming similarity
    "description",                # Context, methodology
    "tasks",                      # Task-based search
    "behavioral_events",          # Event-aligned analysis
    "modalities",                 # Recording technique similarity
    "brain_regions",              # Anatomical targeting
    "analysis_goals",             # Affordance matching
    "data_standards",             # Format compatibility
    "combined_scientific_summary" # General semantic search
]
```

### Paper Embedding Fields

```python
PAPER_EMBEDDING_FIELDS = [
    "title",                      # Direct lookup
    "abstract",                   # Methodology, findings
    "extracted_labels",           # Structured concepts
    "combined_scientific_summary" # General semantic search
]
```

### Field Embedding Generation

```python
from neural_search.embeddings import (
    build_field_embedding_records,
    field_texts_for_record,
)

# Generate field texts
field_texts = field_texts_for_record(dataset)
# {'title': '...', 'tasks': '...', 'modalities': '...', ...}

# Build embeddings for all fields
embeddings = build_field_embedding_records([dataset], provider)
```

---

## When NOT to Trust Semantic Embeddings

### 1. Negative Constraints

Embeddings cannot reliably encode "NOT X" semantics:
- "Mouse electrophysiology, NOT calcium imaging" may still match calcium imaging datasets
- Always apply hard exclusion filters **after** semantic ranking

### 2. Exact ID Lookups

For dataset IDs, DOIs, source IDs:
- Use exact string matching, not embeddings
- Embeddings may rank similar-looking IDs higher than exact matches

### 3. Species/Modality Confusion

Embeddings may conflate:
- "Mouse" and "rat" (both rodents)
- "EEG" and "ECoG" (both electrical recordings)
- "Calcium imaging" and "fiber photometry" (both optical)

**Mitigation:** Use ontology matching for strict constraints, embeddings for soft ranking.

### 4. Novel Scientific Terms

New techniques, tasks, or methods not in training data:
- "Neuropixels" may not be well-represented in older models
- Custom neuroscience terminology may be underrepresented

**Mitigation:** Combine with lexical/keyword matching.

### 5. Low-Quality Metadata

If dataset descriptions are sparse or inconsistent:
- Embeddings from "Neuropixels recording" vs "Neural activity" may differ significantly
- Garbage in → garbage out

---

## Embedding Cache Format

### Cache File Structure

```
data/indexes/embeddings/
├── hashing_64/
│   └── field_embeddings.jsonl
├── sentence-transformer_all-MiniLM-L6-v2/
│   └── field_embeddings.jsonl
└── metadata.json
```

### Cache Record Schema

```python
class FieldEmbeddingRecord(BaseModel):
    record_id: str                # dataset:dandi:000026
    record_type: Literal["dataset", "paper"]
    field_name: str               # title, description, tasks, etc.
    text: str                     # Original text that was embedded
    embedding: list[float]        # The embedding vector
    provider_name: str            # hashing, sentence-transformer
    model_name: str               # signed-token-hashing-64, all-MiniLM-L6-v2
    dimension: int                # 64, 384, 768
    normalize: bool               # Whether L2-normalized
    created_at: str               # ISO timestamp
```

### Cache Validation

Before using cached embeddings:

```python
from neural_search.embeddings import validate_field_embedding_cache

validate_field_embedding_cache(
    records,
    expected_provider_name="sentence-transformer",
    expected_model_name="all-MiniLM-L6-v2",
    expected_dimension=384,
)
```

This prevents mixing incompatible embeddings.

---

## Evaluation Criteria for Future Providers

### Retrieval Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Precision@5 | ≥ 70% | Benchmark suite |
| Label Recall@10 | ≥ 80% | Expected label coverage |
| MRR | ≥ 0.6 | Mean Reciprocal Rank |
| Exclusion Correctness | 100% | Hard negatives filtered |

### Computational Requirements

| Requirement | Threshold |
|-------------|-----------|
| Embedding latency (batch 100) | < 5 seconds |
| Model size | < 500MB |
| Memory usage | < 2GB |
| Offline capability | Required |

### Scientific Domain Coverage

Test on domain-specific queries:
- Neuropixels-specific terminology
- Task names (go/no-go, reversal learning)
- Brain region abbreviations (V1, PFC, OFC)
- Analysis methods (PSTH, GLM, RSA)

---

## CLI Commands

### Build Embedding Index with Hashing Provider

```bash
python -m neural_search.embeddings.build_index \
  --input data/corpus/normalized \
  --out data/indexes/embeddings/hashing_64 \
  --provider hashing \
  --dimensions 64
```

### Build Embedding Index with Sentence Transformer

```bash
python -m neural_search.embeddings.build_index \
  --input data/corpus/normalized \
  --out data/indexes/embeddings/sentence-transformer \
  --provider sentence-transformer \
  --model sentence-transformers/all-MiniLM-L6-v2
```

---

## Provider Comparison Summary

| Provider | Quality | Speed | Offline | Dependencies | Use Case |
|----------|---------|-------|---------|--------------|----------|
| Hashing | Low | Fast | Yes | None | CI, testing |
| all-MiniLM-L6-v2 | Medium | Medium | Yes | sentence-transformers | Production |
| all-mpnet-base-v2 | High | Slow | Yes | sentence-transformers | High-quality search |
| SPECTER2 | High (papers) | Slow | Yes | transformers | Paper similarity |
| OpenAI | High | Fast | No | openai, API key | Cloud deployment |

---

## Migration Path

### v0.4 (Current)

1. HashingEmbeddingProvider for tests
2. SentenceTransformerEmbeddingProvider for optional semantic search
3. Field-specific embeddings with cache

### v0.5 (Planned)

1. Add SPECTER provider for paper abstracts
2. Add SciBERT provider for scientific text
3. Evaluate domain-specific fine-tuning

### v0.6+ (Future)

1. Contrastive learning on dataset metadata
2. Task-specific embeddings trained on neuroscience corpora
3. Multi-modal embeddings (text + structure)
