# Task 14: Awareness-Integrated Retrieval

Status: started

## Goal

Make the Task 13 neuroscience awareness layer actionable in retrieval while keeping the current public search API stable and avoiding churn in the actively edited search core.

## Implementation Slice

- Add `search_datasets_with_awareness` as a wrapper around the existing `search_datasets`.
- Annotate responses with `parsed_query.query_awareness`.
- Add `awareness_score` into each result `score_breakdown`.
- Add full awareness evidence under `dataset_card_preview.data_form_awareness`.
- Add awareness reasons and warnings to existing `why_matched` and `warnings` fields.
- Support opt-in reranking with an `awareness_weight` while preserving default ordering unless `rerank=True`.

## Next Steps

- Move awareness scoring into the main retrieval path after concurrent Task 12 semantic-search work settles.
- Add benchmark cases for fMRI, MEG, connectomics, molecular, clinical, computational models, and cross-modal searches.
- Tune awareness weights by query intent once benchmark labels cover broader neuroscience data forms.
