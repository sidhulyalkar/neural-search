# Search Intelligence Promotion Gates

- Default enabled: false
- Promotion ready: false

## Global Blockers

- total query_count 3 < required 10
- default promotion disabled in manifest

## Intent Decisions

| Intent | Ready | Enabled | Query Count | MRR Delta | Hard Neg Delta | Blockers |
|---|---|---|---:|---:|---:|---|
| analysis_affordance | false | false | 1 | 0.0 | 0 | query_count 1 < required 5; intent disabled in manifest |
| cross_modal | false | false | 0 | 0.0 | 0 | missing evaluation summary; query_count 0 < required 5; intent disabled in manifest |
| dataset_lookup | false | false | 1 | 0.0 | 0 | query_count 1 < required 5; intent disabled in manifest |
| graph_similarity | false | false | 0 | 0.0 | 0 | missing evaluation summary; query_count 0 < required 5; intent disabled in manifest |
| hard_negative | false | false | 1 | 0.0 | 0 | query_count 1 < required 5; intent disabled in manifest |
