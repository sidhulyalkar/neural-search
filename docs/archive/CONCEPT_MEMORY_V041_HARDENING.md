# Concept Memory v0.4.1 Hardening

Concept Memory v0.4.1 hardens the v0.4 graph artifact pipeline for reproducibility, evidence polarity, provenance, scoring transparency, and claim hygiene.

It remains structural/provenance infrastructure. It does not establish that Concept Memory improves retrieval. Retrieval-improvement claims still require qrels-backed ablation results.

## Deterministic Artifact Mode

Use deterministic mode when producing CI or paper artifacts:

```bash
python -m neural_search.field_state.concept_memory.cli --root . concept-build --deterministic
python -m neural_search.field_state.concept_memory.cli --root . concept-report --deterministic
python -m neural_search.field_state.concept_memory.cli --root . concept-validate --deterministic
```

Alternatively set:

```bash
NEURAL_SEARCH_DETERMINISTIC_ARTIFACTS=1
```

In deterministic mode, generated timestamps use:

```text
1970-01-01T00:00:00+00:00
```

Normal operational mode still uses wall-clock UTC timestamps.

## Semantic Hash Manifest

Builds now write:

```text
artifacts/field_state/concept_memory/manifest.json
```

The manifest records artifact paths, record counts, byte hashes, semantic hashes, build mode, optional source corpus hash, field-state input hashes, and GraphML export status.

Semantic hashes ignore approved volatile fields such as `created_at`, `generated_at`, and `validated_at`. This allows normal builds to be compared semantically even when operational timestamps differ.

## Evidence Polarity

`ConceptBasis` now separates:

- `supporting_count`
- `contradicting_count`
- `neutral_or_metadata_count`
- `missing_count`
- reviewed counts for each polarity
- contradictory and metadata evidence link IDs

Evidence strength is computed from supporting evidence, not raw reviewed-link count. Metadata-only concepts remain weak/metadata-derived. Contradictory evidence is visible and does not inflate positive support.

## Corpus Provenance

Corpus-derived evidence links now include available structured provenance:

- source repository
- source record/dataset ID
- source field, such as `modalities`, `tasks`, `brain_regions`, or `species`
- extractor name/version
- evidence text snippet

Unavailable provenance is left missing rather than invented.

## Scoring Decomposition

Concept search keeps the zero-lexical-match guard from the v0.4 audit. Graph boost cannot retrieve a concept by itself.

Search results now expose:

- `lexical_score`
- `graph_boost_raw`
- `graph_boost_degree_normalized`
- `missingness_penalty`
- `final_score`
- `matched_terms`
- `matched_concepts`
- `warnings`

Graph boost is capped and degree-normalized so high-degree concepts have less opportunity to dominate purely by popularity. Sparse metadata is reflected as a bounded missingness penalty.

## Report Wording

Reports now include a Claim Safety Notice and avoid treating unreviewed metadata links as paper-grade evidence.

Preferred language includes:

- metadata/evidence links
- most connected metadata concepts
- reviewed supporting evidence
- reviewed contradictory evidence
- missing reviewed evidence

Reports distinguish metadata-derived links, reviewed support, contradiction, and missing support where basis data is available.

## Validation

Run:

```bash
python -m neural_search.field_state.concept_memory.cli --root . concept-validate
```

Validation now reports:

- deterministic mode status
- manifest presence
- GraphML written status or optional-export warning
- reviewed/unreviewed link counts
- supporting/contradicting/metadata link counts
- orphan concept count
- source provenance completeness
- whether qrels-backed retrieval evidence exists

Absence of qrels-backed retrieval evidence is a scientific-readiness warning, not a structural validation error.

## Retrieval Ablation Harness v0.4.2

Run qrels-backed retrieval ablation with:

```bash
python -m neural_search.field_state.concept_memory.cli --root . concept-ablate-retrieval \
  --queries artifacts/benchmark_queries.jsonl \
  --qrels path/to/qrels.jsonl \
  --corpus data/corpus/normalized/combined_corpus.jsonl \
  --out reports/eval/concept_memory_ablation.json \
  --deterministic
```

The command writes:

```text
reports/eval/concept_memory_ablation.json
reports/eval/concept_memory_ablation.md
```

It compares four variants:

- baseline retrieval without Concept Memory graph, evidence, or hard-negative signals
- Concept Memory enabled
- Concept Memory lexical-only matching
- Concept Memory graph boost enabled with capped, degree-normalized scoring

When qrels are present, the report includes NDCG@10, MRR@10, Recall@10, Precision@5, hard-negative violation rate, evaluated/skipped query counts, per-intent breakdowns when query intents are available, reliability bins/ECE for scored overlaps, and bootstrap confidence intervals for deltas against the baseline.

Missing, empty, or malformed qrels produce placeholder reports with warnings and no fabricated metrics. Small qrels sets are flagged because uncertainty estimates are unstable.

The ablation report is claim-safe by construction: it can describe qrels-backed associations on the evaluated snapshot, but it must not claim that Concept Memory improves retrieval without adequate qrels-backed evidence and broader validation.

## Supported Claims

Currently supported:

- Concept Memory can generate local graph artifacts from local inputs.
- Deterministic mode supports byte-stable timestamps for generated Concept Memory artifacts.
- Semantic manifests support reproducibility checks that ignore approved volatile fields.
- Basis records explicitly separate support, contradiction, metadata, and missing evidence.
- Concept search exposes score decomposition and guards graph boost behind lexical evidence.
- The v0.4.2 ablation harness can compare Concept Memory retrieval variants against qrels and report snapshot-local metrics with uncertainty.

Still unsupported:

- Concept Memory improves retrieval quality.
- Graph boost improves qrels-backed metrics.
- Metadata-derived links are reviewed scientific evidence.
- Concept evidence strength is paper-grade without human review and source audits.
