# Human Relevance Labeling Protocol

This document defines a lightweight protocol for manually reviewing search results so retrieval can be evaluated against human scientific judgment.

## Purpose

Automated metrics (P@K, MRR, label recall) measure system behavior but cannot capture:
- Whether a result is scientifically useful for a specific research question
- Subtle distinctions between "relevant" and "exactly what I needed"
- Why a result is wrong (wrong modality vs. wrong task vs. missing data)

Human relevance judgments provide ground truth for evaluation.

---

## Relevance Labels

### Label Definitions

| Label | Code | Description |
|-------|------|-------------|
| **Exact** | `exact` | This is exactly the dataset/paper the query was looking for |
| **Highly Relevant** | `relevant` | Strong match for the scientific intent; would use this |
| **Partially Relevant** | `partial` | Some overlap but missing key requirements |
| **Wrong Modality** | `wrong_modality` | Correct task/species but wrong recording technique |
| **Wrong Task** | `wrong_task` | Correct modality/species but wrong experimental paradigm |
| **Wrong Species** | `wrong_species` | Correct task/modality but wrong animal/human |
| **Missing Data** | `missing_data` | Metadata suggests relevant but data is incomplete |
| **Unclear** | `unclear` | Cannot determine relevance from metadata alone |
| **Not Relevant** | `not_relevant` | No scientific connection to query intent |

### Label Hierarchy

```
exact (1.0)
  └── relevant (0.8)
        └── partial (0.5)
              ├── wrong_modality (0.2)
              ├── wrong_task (0.2)
              ├── wrong_species (0.2)
              └── missing_data (0.3)
        └── unclear (0.3)
              └── not_relevant (0.0)
```

---

## Review Format

### Review File Schema

Reviews are stored as JSONL for easy append-only collection:

```jsonl
{"query_id": "q001", "result_id": "dataset:dandi:000026", "rank": 1, "relevance_label": "exact", "scientific_rationale": "This is the Steinmetz Neuropixels dataset requested", "reviewer": "reviewer_a", "reviewed_at": "2026-05-24T10:00:00Z"}
{"query_id": "q001", "result_id": "dataset:dandi:000034", "rank": 2, "relevance_label": "relevant", "scientific_rationale": "Also Neuropixels visual coding, similar paradigm", "reviewer": "reviewer_a", "reviewed_at": "2026-05-24T10:01:00Z"}
```

### Review Record Fields

```python
@dataclass
class RelevanceJudgment:
    # Required fields
    query_id: str           # Benchmark query ID (e.g., "q001")
    result_id: str          # Dataset or paper ID
    rank: int               # Position in result list (1-indexed)
    relevance_label: str    # One of the defined labels
    scientific_rationale: str  # Brief explanation of judgment

    # Metadata
    reviewer: str           # Reviewer identifier
    reviewed_at: str        # ISO timestamp

    # Optional fields
    confidence: float = 1.0      # Reviewer's confidence in judgment (0-1)
    alternative_labels: list[str] = field(default_factory=list)
    notes: str = ""
```

---

## Review Workflow

### Step 1: Generate Review Batch

```bash
python -m neural_search.evaluation.generate_review_batch \
  --suite demo_v02 \
  --top-k 5 \
  --output reviews/batch_2026_05_24.jsonl
```

This generates a template with queries and top-k results to review.

### Step 2: Review Results

For each query-result pair:

1. Read the query
2. Examine the result metadata (title, description, labels)
3. Assign a relevance label
4. Write a brief scientific rationale (1-2 sentences)

### Step 3: Submit Reviews

```bash
python -m neural_search.evaluation.validate_reviews \
  --input reviews/batch_2026_05_24_completed.jsonl

python -m neural_search.evaluation.merge_reviews \
  --input reviews/batch_2026_05_24_completed.jsonl \
  --output data/eval/relevance_judgments/demo_v02.jsonl
```

---

## Review Guidelines

### Be Consistent

- Use the same criteria across all queries
- If unsure between two labels, choose the more conservative (lower relevance)
- Document edge cases in notes

### Focus on Scientific Utility

Ask: "Would a neuroscientist searching for [query intent] find this useful?"

- A dataset with perfect metadata but no downloadable data → `missing_data`
- A dataset that's close but uses different analysis methods → `partial`
- A dataset from a different species but same task → `wrong_species`

### Rationale Requirements

Good rationales:
- "Exact match: Neuropixels visual coding in mouse V1, matches query exactly"
- "Wrong modality: Query asks for calcium imaging but this is electrophysiology"
- "Partial: Contains reward learning data but no behavioral events labeled"

Avoid vague rationales:
- "Looks good"
- "Not quite"
- "Maybe relevant"

---

## Evaluation Metrics with Judgments

### Metrics Using Human Labels

```python
def human_precision_at_k(results: list, judgments: dict, k: int) -> float:
    """Precision using human relevance labels instead of expected IDs."""
    relevant_count = 0
    for i, result in enumerate(results[:k]):
        judgment = judgments.get((query_id, result.id))
        if judgment and judgment.relevance_label in {"exact", "relevant"}:
            relevant_count += 1
    return relevant_count / k


def human_ndcg_at_k(results: list, judgments: dict, k: int) -> float:
    """NDCG using graded relevance from human labels."""
    GRADE_MAP = {
        "exact": 3,
        "relevant": 2,
        "partial": 1,
        "wrong_modality": 0,
        "wrong_task": 0,
        "wrong_species": 0,
        "missing_data": 0.5,
        "unclear": 0,
        "not_relevant": 0,
    }
    # ... compute DCG / ideal DCG
```

### Comparison Report

```markdown
## Retrieval Quality: Automated vs Human

| Metric | Automated | Human-Judged | Delta |
|--------|-----------|--------------|-------|
| P@5 | 76.7% | 72.3% | -4.4% |
| NDCG@10 | 0.812 | 0.756 | -0.056 |
| Wrong Modality Rate | - | 12.3% | - |
| Wrong Species Rate | - | 5.1% | - |
```

---

## Storage Format

### Directory Structure

```
data/eval/relevance_judgments/
├── demo_v02.jsonl           # Judgments for demo benchmark
├── real_corpus.jsonl        # Judgments for real corpus
├── adversarial.jsonl        # Judgments for adversarial queries
└── metadata.json            # Reviewer info, collection dates
```

### Metadata File

```json
{
  "reviewers": {
    "reviewer_a": {
      "expertise": "systems neuroscience",
      "affiliation": "anonymous"
    }
  },
  "collection_periods": {
    "demo_v02": {
      "started_at": "2026-05-24",
      "completed_at": "2026-05-25",
      "queries_reviewed": 30,
      "results_reviewed": 150
    }
  }
}
```

---

## Inter-Rater Agreement

For critical benchmarks, collect judgments from multiple reviewers:

### Agreement Metrics

```python
def cohen_kappa(judgments_a: list, judgments_b: list) -> float:
    """Measure agreement between two reviewers."""
    ...

def fleiss_kappa(all_judgments: list[list]) -> float:
    """Measure agreement among multiple reviewers."""
    ...
```

### Handling Disagreements

1. For labels within one level (e.g., `relevant` vs `partial`): Use majority vote
2. For labels across levels (e.g., `relevant` vs `not_relevant`): Adjudicate manually
3. Document disagreements in notes

---

## Example Reviews

### Example 1: Exact Match

```json
{
  "query_id": "q_neuropixels_v1",
  "query": "Neuropixels recordings in mouse V1 visual cortex",
  "result_id": "dataset:dandi:000026",
  "rank": 1,
  "relevance_label": "exact",
  "scientific_rationale": "Steinmetz Neuropixels dataset with V1 recordings in mouse, exactly matches query intent",
  "reviewer": "reviewer_a",
  "reviewed_at": "2026-05-24T10:00:00Z"
}
```

### Example 2: Wrong Modality

```json
{
  "query_id": "q_calcium_ofc",
  "query": "Calcium imaging in orbitofrontal cortex during reversal learning",
  "result_id": "dataset:dandi:000012",
  "rank": 3,
  "relevance_label": "wrong_modality",
  "scientific_rationale": "This is electrophysiology not calcium imaging, even though task and region match",
  "reviewer": "reviewer_a",
  "reviewed_at": "2026-05-24T10:05:00Z"
}
```

### Example 3: Partial Relevance

```json
{
  "query_id": "q_choice_decoding",
  "query": "Datasets for choice decoding with trial structure",
  "result_id": "dataset:openneuro:ds003505",
  "rank": 2,
  "relevance_label": "partial",
  "scientific_rationale": "Has decision task but behavioral events not clearly labeled, would need preprocessing",
  "reviewer": "reviewer_a",
  "reviewed_at": "2026-05-24T10:10:00Z",
  "notes": "Could be upgraded to 'relevant' if events file is present"
}
```

---

## CLI Commands

### Generate Review Template

```bash
python -m neural_search.evaluation.generate_review_batch \
  --suite demo_v02 \
  --top-k 5 \
  --output reviews/demo_v02_batch.jsonl
```

### Validate Completed Reviews

```bash
python -m neural_search.evaluation.validate_reviews \
  --input reviews/demo_v02_completed.jsonl
```

### Compute Human-Judged Metrics

```bash
python -m neural_search.evaluation.human_metrics \
  --judgments data/eval/relevance_judgments/demo_v02.jsonl \
  --benchmark data/eval/benchmark_queries_demo_v02.yaml \
  --output data/reports/human_eval_report.md
```
