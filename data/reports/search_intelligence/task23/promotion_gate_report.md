# Search Intelligence Promotion Gates

- Default enabled: false
- Promotion ready: false
- Human judgments: 9

## Global Blockers

- total query_count 6 < required 10
- default promotion disabled in manifest

## Intent Decisions

| Intent | Ready | Enabled | Query Count | MRR Delta | Hard Neg Delta | Blockers |
|---|---|---|---:|---:|---:|---|
| analysis_affordance | false | false | 1 | 0.6667 | 0 | query_count 1 < required 5; intent disabled in manifest |
| cross_modal | false | false | 1 | 0.0 | 0 | query_count 1 < required 5; intent disabled in manifest |
| data_form_search | false | false | 1 | 0.0 | 0 | query_count 1 < required 5; intent disabled in manifest |
| dataset_lookup | false | false | 0 | 0.0 | 0 | missing evaluation summary; query_count 0 < required 5; intent disabled in manifest; human judgment_count 0 < required 1 |
| graph_similarity | false | false | 0 | 0.0 | 0 | missing evaluation summary; query_count 0 < required 5; intent disabled in manifest; human judgment_count 0 < required 1 |
| hard_negative | false | false | 3 | 0.0 | 0 | query_count 3 < required 5; intent disabled in manifest |
