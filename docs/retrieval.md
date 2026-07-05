# Retrieval and Ranking

Neural Search uses a hybrid, explainable retrieval pipeline, not a single learned model:

1. Parse the query into scientific intent (tasks, behaviors, modalities, species, brain regions, analysis goals, negative constraints).
2. Generate candidates from BM25 sparse scoring and BGE-large-en-v1.5 dense field embeddings, fused via reciprocal rank fusion (RRF).
3. Add graph and concept-memory context: shared region/task/species edges, method-supports-analysis links, evidence-tiered reanalysis/reinterpretation signals.
4. Return explanations that expose both positive evidence and reuse risks — including each linked paper's evidence tier and retraction/correction status.

Retrieval weights and parser aliases live in config; the ranking pipeline itself is inspectable end to end, and every score decomposes into named components rather than a single opaque similarity number.

## Query Parsing

`search/core.py` detects:

- Task intent from the behavioral task ontology, including labels and synonyms.
- Behavior intent from ontology behavior labels.
- Modality intent from explicit modality terms plus broad phrases such as "neural recordings."
- Species and brain-region intent from configured ontology aliases.
- Analysis intent from configured phrases such as "decode choice," "event alignment," or "latent state modeling."
- Negative constraints ("NOT EEG," "without fMRI") as hard exclusions, not soft preferences.

## Ranking Signals

| Signal | Purpose |
|--------|---------|
| BM25 (sparse) | Keyword/field-level lexical match |
| Dense (BGE-large) | Semantic similarity in a 1,024-dim embedding space |
| RRF fusion | Combines sparse + dense rankings without hand-tuned score blending |
| Graph context | Shared region/task/species edges, method-supports-analysis links |
| Concept memory | Ontology-normalized concept overlap |
| Readiness | Boosts datasets that are analysis-ready per their card |
| Evidence tier | Distinguishes a metadata-inferred claim from a file-validated or human-validated one |
| Coverage-gap boost | A small, capped preference for underrepresented regions/modalities |
| Source diversity | Prevents a single high-volume repository from dominating result lists |

Every KG-affecting change to this pipeline is checked against the NDCG@10 ablation ladder (`scripts/eval/run_ablation_ladder.py`) before being considered done — see [Evaluation](evaluation.md).

## Explanations

Each result includes:

- `why_matched`: human-readable reasons such as task, behavior, modality, readiness, and provenance.
- `matched_terms` / `inferred_concepts`: normalized labels and concepts that matched the query.
- `evidence_snippets`: short source snippets from dataset text or card evidence.
- `missing_metadata_warnings`: required metadata gaps.
- `score_breakdown`: normalized component scores and penalties.
- `linked_papers[].evidence_tier` / `retraction_status`: per-link trust signal, surfaced directly rather than only living in the graph.

## Running the Benchmark

```bash
python scripts/eval/run_ablation_ladder.py --skip-rungs bm25 bm25_structured dense_bge
```

This runs the graph-aware rungs (`hybrid_graph`, `typed_kg`, `typed_kg_qualified`, `full`) against the 317-query canonical benchmark and reports NDCG@10 per rung, writing `reports/eval/ablation_ladder_report.partial.{json,md}`. Drop `--skip-rungs` for the full ladder including the sparse/dense-only baselines.

Some benchmark queries describe concepts that are only weakly represented in the corpus today — these are useful parser-coverage checks even when they can't be fully satisfied by ranking alone.
