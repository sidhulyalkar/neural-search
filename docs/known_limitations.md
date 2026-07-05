# Known Limitations

Neural Search runs against a real corpus (7,171 records) and a real knowledge graph (~12,750 nodes / ~150,000 edges), with hybrid retrieval measured at NDCG@10 = 0.8594 on a 317-query canonical benchmark. It is not yet a fully validated production data platform. The honest gaps are below — this list is deliberately specific rather than generic, and is kept current as gaps close.

## The single biggest gap: gold qrels

Every ranking number in this project is measured against silver/LLM-judged relevance labels, not human-adjudicated ones. Gold qrels are currently **0 rows**. Until a human-labeled benchmark exists, treat NDCG/MRR numbers as internal regression-detection signals, not published retrieval-quality claims.

## Evidence tiers are real, but coverage is uneven

- A 6-tier evidence framework (`heuristic_candidate` → `computed`) is enforced in code, and reanalysis suggestions, paper links, and retraction status are tagged. But the two most-populated tiers today are the two weakest (`heuristic_candidate`, `evidence_backed_bridge`) — `file_validated` coverage comes from a live validator run against only the top-300 highest-confidence reanalysis suggestions (not the full corpus), and `human_validated`/`computed` have zero occupants.
- Live file validation only exists for DANDI (NWB header streaming) and OpenNeuro (BIDS). Figshare, GIN, CRCNS, Zenodo, and other sources have no live validator and are recorded as `validator: "none"` rather than silently skipped.

## Literature linking coverage varies sharply by source

Combined real paper-dataset link coverage across 5 sources is ~35% (2,510/7,171). Crossref and PubMed/bioRxiv processed the full corpus cleanly; DataCite is limited to the ~2,400 records with a dataset DOI (DANDI/OpenNeuro records mostly have none); Semantic Scholar's unauthenticated tier is blocked by a rate limit after a single request and needs an API key to use at scale.

## Known code-health items, not yet resolved

- `neural_search/graph/semantic_edges.py` appears to be dead code (zero real callers found in an audit) — flagged, not yet deleted, pending a review pass.
- `neural_search/graph/paper_linking.py`'s reachability from the production search path (vs. a separate evaluation-only retrieval implementation) is unresolved — an open question from the last connectivity audit, not yet traced to a conclusion.

## ExperimentGlancer's evidence tiers are not yet unified with the KG's

ExperimentGlancer uses its own 4-tier scheme (`available`/`probable`/`placeholder`/`unsupported`) at the layer level. It's conceptually aligned with the KG's 6-tier scheme but not the same enum — a deliberate scope decision to ship the bridge without a cross-cutting evidence-tier migration, tracked as future work rather than silently glossed over.

## Retrieval

- Ranking is a hybrid of BM25, dense embeddings, and graph/concept signals — interpretable and tunable, not a trained learned-to-rank model (that requires the gold qrels above).
- Query understanding depends on ontology coverage and synonyms; some analysis-goal phrasing may be represented only indirectly.

## Dataset Cards

- Cards are generated and should be reviewed before being treated as fully trusted, especially for sources without a live file validator.
- Retraction/evidence-tier badges now surface automatically where the graph has the data, but absence of a badge means "nothing to show," not "verified clean."

## Notebook Generation

- Starter notebooks are scaffolds for inspection and first analysis, not complete publication-ready pipelines.

## Frontend and API

- QA updates and the demo corpus toggle (`NEURAL_SEARCH_DEMO_MODE`) are still local/in-process state, not a durable multi-user store.
- Authentication, user roles, and audit logs are not implemented.

## Future Work

- Human-adjudicated gold qrels (highest priority).
- File validation at corpus scale, not just the top-N reanalysis suggestions.
- Resolve `paper_linking.py`'s reachability and either delete or reconnect `semantic_edges.py`.
- Unify the KG's and ExperimentGlancer's evidence-tier vocabularies.
- Multi-modal retrieval over learned neural/behavioral representations, once a real signature extractor exists (currently a placeholder).
