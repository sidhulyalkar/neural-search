# API Contract Examples v0.7

## Search Response

```json
{
  "schema_version": "v1",
  "query": "mouse Neuropixels visual decision-making without EEG",
  "parsed_query": {
    "modalities": ["neuropixels"],
    "excluded_modalities": ["eeg"]
  },
  "filtered_constraints": [
    {
      "dataset_id": "dataset:openneuro:ds003505",
      "violations": ["eeg"]
    }
  ],
  "results": [
    {
      "dataset_id": "dataset:dandi:000026",
      "score": 81.2,
      "why_matched": ["Modality matched: neuropixels"],
      "score_breakdown": {
        "graph_score": 0.4,
        "field_semantic_score": 0.3
      },
      "graph_context": {
        "linked_papers": [{"paper_id": "paper:openalex:W2963345511"}],
        "analysis_affordances": ["event_aligned_analysis"]
      },
      "linked_papers": [{"paper_id": "paper:openalex:W2963345511"}],
      "missing_metadata": ["license"]
    }
  ]
}
```

## Search Trace

```python
from neural_search.search.trace import capture_search_trace

trace = capture_search_trace(
    "mouse visual decision making without EEG",
    retrieval_config={"hard_negative_filters": {"enabled": True}},
)
trace.model_dump(mode="json")
```

The trace contains parsed query fields, filtered constraints, score breakdowns,
warnings, and parse/search timings.
