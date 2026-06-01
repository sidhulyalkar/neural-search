# Known Limitations

Neural Search is currently a public technical demo, not a production data platform.

## Demo Data

- The default app uses curated in-memory seed data.
- Live archive ingestion exists only as early connector work.
- Benchmark coverage is small and demo-oriented.
- Some example queries intentionally expose corpus gaps.

## Retrieval

- Ranking is interpretable and rule-weighted, not a fully trained retrieval model.
- Embedding-style semantic matching is represented in the architecture, but the local demo path is optimized for reliability and inspection.
- Query understanding depends heavily on ontology coverage and synonyms.
- Some scientific concepts, especially analysis goals, may be represented indirectly.

## Dataset Cards

- Cards are generated and need human review before being treated as trusted.
- Missing metadata warnings depend on fixture completeness.
- Linked literature confidence is heuristic in the demo.
- Provenance is surfaced, but not yet backed by a full claim-level citation graph.

## Notebook Generation

- Starter notebooks are scaffolds.
- They are intended to inspect files and suggest first analyses, not to complete a publication-ready workflow.
- Template availability depends on detected data standard, modality, and metadata.

## Frontend and API

- The API uses in-memory demo records at startup.
- QA updates are demo-local state.
- Error messages are designed for local demo recovery, not full production observability.
- Authentication, user roles, and audit logs are not implemented.

## Future Work

- Durable indexing and persistent QA state.
- Larger benchmark sets with human relevance judgments.
- Production source ingestion from DANDI, OpenNeuro, OpenAlex, and related archives.
- Better embedding infrastructure and vector index persistence.
- Claim-level provenance for generated cards.
- Latent neural-state search over learned neural and behavioral representations.
