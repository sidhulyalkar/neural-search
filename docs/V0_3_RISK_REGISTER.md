# Neural Search v0.3 Risk Register

**Created:** 2026-05-23
**Status:** Active

---

## Risk Categories

| Category | Description |
|----------|-------------|
| **TECHNICAL** | Implementation complexity, dependencies, architecture |
| **DATA** | Corpus quality, ingestion reliability, API availability |
| **EVALUATION** | Benchmark validity, metric interpretation |
| **SCOPE** | Feature creep, timeline pressure |
| **INTEGRATION** | Breaking changes, backward compatibility |

---

## High Priority Risks

### RISK-001: v0.2 Benchmark Score Inflation
**Category:** EVALUATION
**Severity:** High
**Probability:** Confirmed (current state)

**Description:**
The 29/30 benchmark pass rate may reflect co-evolution between curated demo corpus, ontology labels, and benchmark queries rather than genuine retrieval quality.

**Mitigation:**
- Freeze v0.2 benchmark as regression-only suite
- Create adversarial benchmark with hard negatives
- Create real_corpus benchmark with external data
- Do not tune system against v0.2 queries

**Status:** Actively mitigated by v0.3 design

**2026-05-23 update:** Mitigated in code by adding the `demo_v02` suite and writing results to `data/eval/results/demo_v02/`. The latest run preserved the v0.2 regression profile at 30 queries with 29/30 pass shape.

---

### RISK-002: Label Hallucination Without Evidence
**Category:** DATA
**Severity:** High
**Probability:** Medium

**Description:**
Current extraction may infer labels (e.g., tasks from brain regions, analyses from modalities) without sufficient evidence, creating false positives in search.

**Mitigation:**
- Require evidence_text for all EvidenceLabel instances
- Implement confidence tiers with strict cutoffs
- Add false positive tests for common traps
- Do not infer behavioral tasks from brain region alone

**Owner:** WS5 (Label Extraction Rules)

---

### RISK-003: External API Instability
**Category:** DATA
**Severity:** Medium
**Probability:** Medium

**Description:**
DANDI, OpenNeuro, and OpenAlex APIs may change, rate-limit, or become unavailable, breaking ingestion pipelines.

**Mitigation:**
- Create fixture payloads for representative records
- Implement `--dry-run` mode for all ingestion CLIs
- Add retry/backoff logic to API clients
- Run tests against fixtures, not live APIs

**Owner:** WS4 (Ingestion Hardening)

**2026-05-23 update:** Partially mitigated by fixture-backed v0.3 normalizer tests for DANDI, OpenNeuro, and OpenAlex. Live API resilience and larger saved raw payload coverage remain open.

---

### RISK-004: Schema Migration Complexity
**Category:** INTEGRATION
**Severity:** Medium
**Probability:** Medium

**Description:**
Adding EvidenceLabel provenance to existing records may require database migrations and break existing cards/notebooks.

**Mitigation:**
- Create new schema models alongside existing ones
- Add migration path from simple labels to EvidenceLabel
- Preserve backward compatibility with existing Dataset/Paper ORM
- Test roundtrip serialization before deploying

**Owner:** WS3 (Provenance Schema)

**2026-05-23 update:** Partially mitigated by additive Pydantic models and JSONL roundtrip tests. No ORM migration has been introduced yet, preserving existing DB behavior.

---

### RISK-005: Scoring Head Weight Tuning
**Category:** TECHNICAL
**Severity:** Medium
**Probability:** Medium

**Description:**
Adding 7+ scoring heads creates a large hyperparameter space. Incorrect weights may degrade search quality on one dimension while improving another.

**Mitigation:**
- Start with equal weights for new heads
- Document default weights in retrieval.yaml
- Add ablation tests for individual heads
- Do not tune weights against v0.2 benchmark
- Use adversarial benchmark to validate head contributions

**Owner:** WS7 (Multi-Head Scoring)

**2026-05-23 update:** Search results now expose v0.3 score heads without major rank-formula churn. Weight calibration and ablation reporting remain open.

---

## Medium Priority Risks

### RISK-006: Adversarial Benchmark Design Bias
**Category:** EVALUATION
**Severity:** Medium
**Probability:** Medium

**Description:**
Adversarial queries may reflect author assumptions about "hard" cases rather than real user confusion patterns.

**Mitigation:**
- Document scientific rationale for each adversarial query
- Include diverse query categories (hard negatives, ambiguity, missing metadata)
- Accept that some adversarial queries will fail initially
- Iterate on benchmark based on user feedback (future)

**Owner:** WS8 (Adversarial Benchmark)

---

### RISK-007: Analysis Affordance False Positives
**Category:** DATA
**Severity:** Medium
**Probability:** Medium

**Description:**
Rule-based affordance detection may claim a dataset supports an analysis (e.g., Q-learning) when required fields are present but data quality is insufficient.

**Mitigation:**
- Use conservative support_level assignments
- Expose missing_fields explicitly
- Add confidence to affordance estimates
- Do not claim "high" support without strong evidence

**Owner:** WS9 (Analysis Affordance)

---

### RISK-008: Corpus Report Scalability
**Category:** TECHNICAL
**Severity:** Low
**Probability:** Medium

**Description:**
Corpus reports that iterate over all records may become slow as corpus grows beyond 1000+ records.

**Mitigation:**
- Use streaming/generator patterns for large corpora
- Cache intermediate aggregations
- Add progress indicators for CLI
- Design reports to run in reasonable time (<5 min for 1000 records)

**Owner:** WS6 (Corpus Reports)

**2026-05-23 update:** Initial report generator runs against normalized JSON/JSONL and tolerates small fixture corpora. Streaming/scaling optimization remains deferred.

---

### RISK-009: Latent Search Scope Creep
**Category:** SCOPE
**Severity:** Medium
**Probability:** Low

**Description:**
Temptation to add learned embeddings or NWB file parsing in v0.3, exceeding planned scope.

**Mitigation:**
- Explicitly defer learned models to v0.4
- Explicitly defer NWB/BIDS file inspection to v0.5
- Document what latent search WILL and WILL NOT do in v0.3
- Focus on deterministic fingerprints only

**Owner:** WS10 (Latent Roadmap)

---

### RISK-010: Test Coverage Gaps
**Category:** TECHNICAL
**Severity:** Medium
**Probability:** Medium

**Description:**
New modules may have insufficient test coverage, leading to regressions or silent failures.

**Mitigation:**
- Require tests for every new module
- Add fixture-backed tests for ingestion normalization
- Add tests for score decomposition correctness
- Add tests for adversarial benchmark metric calculation
- Run pytest as quality gate

**Owner:** All workstreams

---

## Low Priority Risks

### RISK-011: Fixture Payload Staleness
**Category:** DATA
**Severity:** Low
**Probability:** Low

**Description:**
Fixture payloads from DANDI/OpenNeuro/OpenAlex may become outdated as APIs evolve.

**Mitigation:**
- Document fixture creation dates
- Periodically refresh fixtures (quarterly)
- Design normalization to handle missing/extra fields gracefully

---

### RISK-012: Frontend Pressure
**Category:** SCOPE
**Severity:** Low
**Probability:** Medium

**Description:**
Stakeholders may request frontend polish or demo improvements during v0.3.

**Mitigation:**
- Explicitly state "no frontend work" in v0.3 scope
- Document frontend as v0.4+ work
- Keep API stable for future frontend integration

---

### RISK-013: Breaking Changes to CLI
**Category:** INTEGRATION
**Severity:** Low
**Probability:** Low

**Description:**
Adding new CLI flags may break existing scripts or workflows.

**Mitigation:**
- Make new flags optional with sensible defaults
- Preserve existing command signatures
- Document CLI changes in release notes

---

## Risk Summary Matrix

| Risk ID | Severity | Probability | Category | Owner |
|---------|----------|-------------|----------|-------|
| RISK-001 | High | Confirmed | EVALUATION | WS2, WS8 |
| RISK-002 | High | Medium | DATA | WS5 |
| RISK-003 | Medium | Medium | DATA | WS4 |
| RISK-004 | Medium | Medium | INTEGRATION | WS3 |
| RISK-005 | Medium | Medium | TECHNICAL | WS7 |
| RISK-006 | Medium | Medium | EVALUATION | WS8 |
| RISK-007 | Medium | Medium | DATA | WS9 |
| RISK-008 | Low | Medium | TECHNICAL | WS6 |
| RISK-009 | Medium | Low | SCOPE | WS10 |
| RISK-010 | Medium | Medium | TECHNICAL | All |
| RISK-011 | Low | Low | DATA | WS4 |
| RISK-012 | Low | Medium | SCOPE | - |
| RISK-013 | Low | Low | INTEGRATION | All |

---

## Monitoring

Risks will be reviewed at each phase completion:
1. After Phase 1 (Freeze and Plan): Re-assess RISK-001, RISK-009
2. After Phase 2 (Schema and Ingestion): Re-assess RISK-002, RISK-003, RISK-004
3. After Phase 3 (Reports and Scoring): Re-assess RISK-005, RISK-008
4. After Phase 4 (Scientific Evaluation): Re-assess RISK-006, RISK-007
5. After Phase 5 (Latent Foundation): Re-assess RISK-009, RISK-010

---

## Contingency Plans

### If API ingestion fails completely:
- Use fixture payloads for all normalization testing
- Document which tests require fixtures vs. live APIs
- Add manual data entry path as fallback

### If v0.2 benchmark regresses:
- Investigate which changes caused regression
- Roll back specific changes if necessary
- Do not "fix" by tuning against v0.2 queries

### If scoring weights cause quality degradation:
- Reset to baseline equal weights
- Add per-head ablation tests
- Use adversarial benchmark to identify problematic heads

### If schema migration breaks existing data:
- Keep old schema models for backward compatibility
- Add migration script from old to new format
- Test migration on copy of production data first
