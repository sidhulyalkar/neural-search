# Concept Memory v0.4 Audit

Audit date: 2026-06-10 America/Los_Angeles

Auditor: Codex

Scope: independent scientific and engineering audit of `neural_search/field_state/concept_memory/` after Graph-Indexed Concept Memory v0.4 landed. This audit intentionally avoids broad rewrites and only adds focused safeguards/tests for confirmed risks.

## Executive Summary

Concept Memory v0.4 is credible as a deterministic, locally rebuildable graph artifact generator. A clean temporary rebuild produced the expected parsed counts: 13,281 concepts, 29,684 evidence links, and 13,281 basis records. A second rebuild produced the same counts, same concept/evidence/basis ID sets, and the same ID order.

It is not yet scientifically safe to claim that concept memory improves retrieval. The current artifacts validate structurally, but validation reported 13,279 orphan concepts and zero reviewed concepts/links in the clean rebuild. Most concept links are metadata-derived relations, not adjudicated scientific evidence. This is useful infrastructure, but not yet qrels-backed retrieval evidence.

One retrieval risk was patched: graph boost could previously return concepts with zero lexical match. The retrieval layer now requires a positive lexical score before graph boost can rerank a concept.

## Reproducibility Findings

Command audited from a temporary clean root:

```bash
python -m neural_search.field_state.concept_memory.cli --root /tmp/concept_memory_audit_root concept-build
```

Input setup used copied `artifacts/field_state/` files and a symlinked `data/corpus/normalized/combined_corpus.jsonl`, so generated concept-memory artifacts were written only under `/tmp/concept_memory_audit_root`.

Observed clean-build counts:

| Artifact | Parsed records |
|---|---:|
| `concepts.jsonl` | 13,281 |
| `evidence_links.jsonl` | 29,684 |
| `concept_basis.jsonl` | 13,281 |

Observed type counts:

| Type | Count |
|---|---:|
| dataset | 10,402 |
| task | 2,742 |
| modality | 62 |
| brain_region | 39 |
| species | 16 |
| opportunity | 8 |
| claim | 6 |
| benchmark_gap | 6 |

Two consecutive clean builds had identical counts, identical ID sets, and identical ID order for concepts, evidence links, and bases. Stable IDs are generated deterministically from normalized names and SHA-256-derived evidence IDs.

Byte reproducibility is mixed:

| Artifact | Byte-stable across two builds? | Finding |
|---|---:|---|
| `concepts.jsonl` | yes | Hash unchanged. |
| `concept_basis.jsonl` | yes | Hash unchanged. |
| `concept_index.json` | yes | Hash unchanged. |
| `concept_graph.json` | yes | Hash unchanged. |
| `evidence_links.jsonl` | no | Only observed difference was `created_at`, which defaults to wall-clock time in `EvidenceLink`. |

Representative evidence-link diff:

```text
created_at: 2026-06-11T01:56:43.974630+00:00
created_at: 2026-06-11T02:01:09.675716+00:00
```

Validation on the clean rebuild:

```text
Concept memory validation: VALID
errors: 0
warnings: 0
total_concepts: 13281
total_evidence_links: 29684
total_bases: 13281
orphan_concepts: 13279
reviewed_concepts: 0
```

Reproducibility risks:

- `EvidenceLink.created_at` and `ConceptEmbeddingRecord.created_at` use wall-clock defaults, so JSONL artifacts with those records are not byte-stable unless timestamps are injected deterministically.
- Markdown reports embed wall-clock `Generated:` timestamps, so they are not byte-stable enough for strict CI artifact hashing.
- `concept_graph.graphml` was not emitted in the temp clean build. The writer silently suppresses GraphML export exceptions, so missing GraphML is not visible as a validation warning.
- Full `concept-report` generation completed, but only after a long wait on the full artifact set. The code path includes repeated retrieval scoring over all concepts/links and likely needs indexing before it is comfortable as a CI step.

## Schema Findings

The core models are useful and typed enough for a v0.4 artifact layer:

- `ConceptNode` validates non-empty IDs/names, concept type, confidence range, and carries source IDs, note paths, artifacts, review status, counts, tags, and metadata.
- `EvidenceLink` validates IDs, relation type, confidence, source/target IDs, review status, and optional evidence text/provenance fields.
- `ConceptBasis` validates evidence strength and keeps linked evidence IDs plus uncertainty notes and next validation actions.
- `ConceptSearchResult` decomposes final score into lexical score and graph boost.

Necessary schema concerns:

- `evidence_type` is a free string while `relation_type` is enumerated. This allows ambiguous evidence classes even when graph semantics are constrained.
- `review_status` is a free string across nodes/links. For paper-grade artifacts, it should become an enum such as `unreviewed`, `machine_derived`, `human_reviewed`, `rejected`.
- `created_at` defaults are useful operationally but harm reproducibility.
- `ConceptBasis` does not distinguish supporting from contradicting evidence in separate fields; contradiction is only represented in the linked `EvidenceLink`.
- `ConceptEmbeddingRecord` has model/source hash fields, but the schema does not enforce embedding dimensionality, vector normalization, or source text immutability.

Recommendation: do not broadly redesign the schema now. First add deterministic timestamp support, enumerated evidence/review status fields, and contradiction-aware basis fields.

## Graph Semantics Findings

Current relation types are scientifically meaningful for metadata topology: `has_modality`, `has_task`, `has_brain_region`, `has_species`, `linked_to_benchmark_gap`, and `linked_to_opportunity`.

However, current artifacts are mostly metadata links, not evaluated scientific support:

| Relation type | Count |
|---|---:|
| `has_modality` | 10,001 |
| `has_task` | 8,061 |
| `has_species` | 5,930 |
| `has_brain_region` | 5,665 |
| `linked_to_benchmark_gap` | 16 |
| `linked_to_opportunity` | 11 |

All 29,684 clean-build evidence links were `derived_from_artifact` and `unreviewed`. That is acceptable for an index, but not enough to support claims about scientific validity or retrieval improvement.

GraphML export is optional and silently ignored on failure. If GraphML is a promised artifact, failed export should become at least a validation warning.

## Normalization Risks

Alias normalization is conservative and exact-match based. The audit added regression coverage for dangerous aliases:

- `AP` does not merge with `action potential`.
- `AP` does not merge with `anterior-posterior`.
- `M1` does not merge with `primary motor cortex`.
- `VISp` does not merge with `visual cortex`.
- `DLC coating` does not merge with `DeepLabCut`.
- A dataset title/description mentioning a computer mouse does not create a species `mouse` concept unless `species` contains `mouse`.

Remaining risk: exact short labels such as `M1` still normalize to `m1` if supplied in the same concept type. That is acceptable for now only because the loader scopes IDs by concept type and brain regions are not alias-normalized in corpus loading.

## Scoring Risks

Before this audit, graph boost was additive even when lexical score was zero. That could make a connected concept appear for an unrelated query. This was patched in `search_concepts`: graph boost now only applies after positive lexical evidence.

Remaining scoring concerns:

- Graph boost is capped at 0.3, which limits but does not calibrate the effect.
- Boost is based on connected reviewed links, claims, and datasets. It does not normalize by total degree beyond caps, so hubs can still win ties.
- Missingness is not explicitly modeled in `ConceptSearchResult`; sparse metadata may reduce lexical opportunity without a visible confidence penalty/decomposition.
- All clean-build links were unreviewed, so graph boost currently has limited scientific meaning on full artifacts.
- There is no qrels-backed evidence that concept retrieval improves dataset retrieval or scientific task success.

Recommended next safeguard: add degree-normalized graph features and an explicit `match_explanation`/`missingness_penalty` field before using concept search as part of production retrieval.

## Evidence-Grounding Findings

`basis.py` is mostly grounded: summaries are constructed from `canonical_name`, `concept_type`, `description`, and the first `source_artifact`. No LLM calls or unsupported natural-language synthesis were found.

Weak evidence remains weak under the count-based strength function:

- no links -> `none`
- links but no reviewed links -> `weak`
- one reviewed link with one total link -> `weak`
- reviewed links can become `moderate` or `strong` based only on counts

The audit added tests for no-evidence and single contradictory-evidence cases. Both remain weak or missing, not inflated.

Concern: contradiction is not surfaced strongly enough in basis output. A reviewed `contradicts` link contributes to `reviewed_count`, so enough contradictory links could increase evidence strength even though they are evidence against, not for, a concept claim. Basis generation should split positive, negative, and neutral evidence counts before it is used in reports or papers.

## Report and Obsidian Export Findings

Markdown reports are readable and distinguish missing evidence in several places. `unsupported_concepts.md` explicitly reports unsupported concepts, and `claim_basis_map.md` includes a "Missing evidence" line.

Reproducibility and wording issues:

- Report files include wall-clock timestamps, so generated Markdown is not byte-stable.
- Full report generation on the full clean artifact set completed but was slow, likely due retrieval scoring and full dataset map rendering.
- Full-corpus report sizes were moderate but large enough to matter for review ergonomics: `dataset_method_concept_map.md` was 3.2 MB, `concept_basis_summary.md` was 3.0 MB, and `unsupported_concepts.md` was 1.8 MB in the temp run.
- Report wording such as "Top Concepts" and "High-evidence concepts" may overstate support because most links are unreviewed metadata links.
- Reports should label metadata-derived links as metadata-derived, not scientific evidence.

Obsidian export preserves human blocks and safe human frontmatter fields. Existing tests cover human block preservation, and the code explicitly extracts the human block before rewriting generated sections.

## Whitepaper Claim-Readiness

| Claim | Evidence currently available | Evidence missing | Risk level | Recommended next validation |
|---|---|---|---|---|
| Concept Memory v0.4 can rebuild deterministic concept and graph artifacts from local inputs. | Two clean temp builds matched counts, ID sets, and ID order. Most non-timestamp artifacts were byte-identical. | Deterministic timestamp mode and GraphML failure visibility. | Low | Add CI job that rebuilds in temp root and compares semantic hashes ignoring approved volatile fields. |
| Stable IDs are stable across rebuilds. | Concept, evidence, and basis ID sets/order matched across two rebuilds. | Cross-platform and changed-input regression snapshots. | Low | Store ID manifest snapshots for a small fixture and full corpus. |
| Generated artifacts are paper/CI reproducible. | Concepts, basis, index, and graph JSON were byte-stable in the temp run. | Evidence links have wall-clock timestamps; reports have wall-clock timestamps. | Medium | Add deterministic build mode or strip volatile fields for paper artifacts. |
| The concept graph is scientifically meaningful. | Relation schema is reasonable for metadata topology; counts validate structurally. | Human review, provenance quality, contradiction handling, and relation-specific confidence. | High | Audit samples by source/type and adjudicate link correctness. |
| Alias normalization is conservative. | Exact-match alias map; added dangerous-alias regression tests. | Broader abbreviation inventory and context-aware disambiguation. | Medium | Build an ambiguity blacklist and fixture corpus for AP/M1/VISp/DLC/mouse edge cases. |
| Graph-boosted concept retrieval is safe to connect to retrieval. | Added guard preventing graph-only zero-lexical results; score decomposition exists. | Qrels evaluation, degree normalization, sparse-metadata fairness tests, source-skew analysis. | High | Run qrels-backed ablations: lexical only vs lexical+graph, with per-source/per-metadata missingness slices. |
| Concept memory improves retrieval. | No qrels-backed improvement evidence found in this audit. | Evaluation showing statistically meaningful gains on held-out qrels/tasks. | High | Do not claim yet. Run retrieval benchmarks and report confidence intervals. |
| Basis generation is grounded and non-hallucinatory. | Basis text uses local fields/links only; no LLM calls. Added no-evidence and contradiction tests. | Positive vs contradictory evidence separation. | Medium | Add contradiction-aware basis schema and report sections. |
| Obsidian export is safe for human edits. | Existing and focused tests show human block preservation. | Larger round-trip test with modified safe frontmatter and malformed notes. | Low | Add fixture vault re-export tests with frontmatter preservation checks. |

## Recommended Next Work

1. Add deterministic artifact mode: fixed or omitted `created_at`, stable report timestamps, and semantic hash manifests.
2. Make GraphML export failure visible in validation output.
3. Split evidence polarity in `ConceptBasis`: supporting, contradicting, neutral/metadata.
4. Add source-level provenance fields for corpus-derived links: source field, extractor name/version, source record ID, and evidence text where available.
5. Add degree-normalized graph boost and missingness-aware scoring fields.
6. Run qrels-backed retrieval ablations before connecting concept memory to production retrieval or making whitepaper claims.
7. Add report wording that distinguishes metadata facts, inferred links, reviewed evidence, and missing evidence.

## Patches and Tests Added

Code patch:

- `neural_search/field_state/concept_memory/retrieval.py`: require positive lexical evidence before graph boost can affect retrieval results.

Focused tests added:

- `test_dangerous_aliases_do_not_false_merge`
- `test_loader_does_not_infer_mouse_species_from_free_text`
- `test_basis_no_evidence_stays_missing_not_inferred`
- `test_basis_contradictory_single_link_remains_weak`
- `test_graph_boost_does_not_retrieve_zero_lexical_matches`

Verification completed:

```text
pytest tests/test_field_state_concept_memory.py
25 passed

ruff check neural_search/field_state/concept_memory/retrieval.py tests/test_field_state_concept_memory.py
All checks passed

mypy neural_search/field_state/concept_memory/retrieval.py
Success: no issues found
```

Broader gate status in the current dirty worktree:

- Full `pytest tests/ -q` was started and was still running during report finalization; it had reached `tests/test_api_smoke.py` at approximately 14 percent completion with no failure output yet.
- Full `concept-report` generation on the temp clean full artifact set completed after a long wait; generated reports were readable, but large.
- Full `ruff check .` failed with 415 repo-wide issues, mostly in unrelated app, generated notebook, migration, script, and test files outside this audit patch.
- Full `mypy neural_search/` failed with 321 errors in 108 files, mostly pre-existing typing/stub issues outside this audit patch.
