# Next-Generation Knowledge Graph: Development Plan

**Date:** 2026-06-22
**Status:** Reconciles `docs/deep-research-report.md` against current code and against `reports/strategy/brainknow_comparison_publishable_plan.md` (2026-06-20). Supersedes the deep-research report as the active plan.

---

## 0. Why this document exists

`docs/deep-research-report.md` was generated externally and audits the repo as of an earlier snapshot. Two of its central claims do not hold against the current codebase:

| Report claim | Verified current reality |
|---|---|
| "No explicit claim nodes" | `claim` is a first-class node type ([`neural_search/graph/schema.py:36`](../../neural_search/graph/schema.py)) |
| "No contradiction model" | `claim_contradicts_claim` edge type exists ([`schema.py:101`](../../neural_search/graph/schema.py)); `detect_contradictions()` in [`neural_search/literature/claim_synthesizer.py`](../../neural_search/literature/claim_synthesizer.py) implements it; claims are served live via [`apps/api/claims_router.py`](../../apps/api/claims_router.py) (wired into `main.py` as of commit `9f3a7dc`) |
| "No claim-level provenance" | Each claim node carries a `GraphEvidence` bundle with statement text, confidence, extractor name/version ([`claim_kg_builder.py:89-122`](../../neural_search/literature/claim_kg_builder.py)) |

The report's deeper recommendations (evidence span offsets, latent-edge lifecycle, ontology grounding, durable persistence) are still directionally correct, but a more current and more specific strategic plan already exists and is being actively executed: `reports/strategy/brainknow_comparison_publishable_plan.md`. That plan positions Neural Search against BrainKnow (a large-scale, untyped, literature co-occurrence KG) on exactly the axis the deep-research report was reaching for — typed, mechanistic, provenance-backed relations instead of generic claim/contradiction plumbing for its own sake.

This document does three things:
1. Identifies a concrete, verified wiring gap blocking the BrainKnow-differentiation plan from working end-to-end (Phase 0).
2. Folds in the still-valid parts of the deep-research report's recommendations, scoped down to what's actually load-bearing right now (Phases 1-3).
3. Explicitly lists what NOT to build yet, per the caution already raised in `reports/strategy/neural_search_project_roadmap.md` (2026-06-11) about overbuilding graph plumbing ahead of validation.

---

## 1. Phase 0 — Wire the relationship builder into the graph (1-2 days)

**This is the highest-leverage fix and should happen first.**

`neural_search/literature/relationship_builder.py` already implements exactly the cross-paper "relate findings/papers" logic this whole initiative is about:
- `build_cross_finding_edges()` → `supports` / `contradicts` edges between findings from different papers, based on shared region+task and matching/opposing `result_direction`
- `build_region_cooccurrence_edges()` → `region_co_occurs_with` edges
- `build_consensus_summaries()` → per-(region, direction, task) consensus strength

But none of these edge type strings — `supports`, `contradicts`, `co_occurs_in`, `region_co_occurs_with` — appear in `SUPPORTED_EDGE_TYPES` ([`schema.py:59-135`](../../neural_search/graph/schema.py)). `write_edges_jsonl()` writes plain dataclasses to a sidecar JSONL file; nothing reads that file back into a `KnowledgeGraph`. The relationship layer is fully built and tested (`tests/test_finding_relationships.py`) but disconnected from the queryable graph.

**Tasks:**
- Add edge types to `SUPPORTED_EDGE_TYPES`: `finding_supports_finding`, `finding_contradicts_finding`, `finding_co_occurs_with_finding`, `region_co_occurs_with_region`
- Add `neural_search/literature/relationship_kg_builder.py` following the exact pattern of `claim_kg_builder.py` (`_evidence()`, `_add_node()`, `_add_edge()`): read `FindingEdge`/`RegionCooccurrenceEdge` JSONL, emit `KnowledgeGraphEdge` objects with `GraphEvidence` carrying `paper_id_a`/`paper_id_b`, shared regions/tasks, and confidence
- Wire it into `scripts/literature/build_finding_relationships.py` and the main graph build path so relationship edges land in the same graph as claims and dataset/paper nodes
- Add `tests/test_relationship_kg_builder.py` mirroring the existing `test_finding_relationships.py` + `test_graph_builder.py` patterns

**Exit criteria:** A finding-to-finding `contradicts` edge produced by `relationship_builder.py` is traversable from `KnowledgeGraph.edges` and shows up in `python -m neural_search.graph.quality --report`.

---

## 2. Phase 1 — Evidence precision (1 week)

The deep-research report's evidence-span gap is real, just narrower than stated: `GraphEvidence` ([`schema.py:215-236`](../../neural_search/graph/schema.py)) has `evidence_text` and `source_field` but no character offsets, so a reviewer can't jump from a graph edge to the exact sentence in a paper.

**Tasks:**
- Add optional `char_start`, `char_end`, `sentence_id` fields to `GraphEvidence` (additive, non-breaking)
- Capture these in `typed_finding_extractor.py` / `finding_extractor.py` when a finding is extracted, so offsets flow through `relationship_kg_builder.py` (Phase 0) and `claim_kg_builder.py` for free
- This is exactly the input the "Evidence Graph" dataset-card view (BrainKnow plan, Milestone 5) needs — don't build it twice

---

**Status: done (2026-06-22).** Phases 0-2 implemented and tested (~250 new/updated tests passing). See repo history for `relationship_kg_builder.py`, `evidence_span.py`, the typed-field graph wiring, and the negation-aware relationship builder.

## 3. Phase 2 — Finish the typed finding schema (1-2 weeks)

This is BrainKnow plan Milestone 2, already in progress: `typed_finding_extractor.py` (untracked, new) implements 27 rule-based typed fields — frequency band, temporal pattern, negation, spatial frame, effect scale, etc. — purely via regex, no model calls.

**Tasks:**
- Wire the 27 typed fields into normalized `FindingRecord` and into graph edges (`finding_has_frequency_band`, `finding_has_temporal_pattern`, `finding_has_spatial_frame`, per BrainKnow plan §6)
- Make `relationship_builder.py`'s `supports`/`contradicts` logic typed-field-aware: a finding with `negation=True` should not count as supporting evidence at face value — right now `detect_negation()` exists but isn't consulted by `build_cross_finding_edges()`
- Build the 30-50 gold-reviewed finding examples the BrainKnow plan calls for (Milestone 2) — needed to know if the regex extractor is precise enough to trust before more graph structure is built on top of it

---

## 4. Phase 3 — Scoped latent-edge lifecycle and ontology grounding (2-3 weeks)

**Status: done (2026-06-22).**

The deep-research report's "latent edge lifecycle" and "ontology crosswalk" recommendations were sound but assumed a blanker slate than actually existed. Before implementing, an audit of `data/ontology/` turned up that the Allen CCF and Cognitive Atlas *data* layers were already built (likely by a concurrent session) but never wired into the graph:

- `data/ontology/brain_regions.yaml` already carries `atlas_refs` (UBERON + Allen CCF mouse IDs) for 106+ regions, and `neural_search/ontology/loader.py` already exposed `get_region_atlas_refs()` / `get_region_allen_ccf_id()` / `get_region_uberon_id()` — but nothing called them when building graph nodes.
- `data/ontology/task_atlas.yaml` already mapped 87 Neural Search task IDs to Cognitive Atlas term IDs — but **65 of those 87 (75%) point to a single empty/placeholder Cognitive Atlas concept** (`trm_4f244f46ebf58`, `name=""`, `definition_text="None"`) left over from a substring-match fallback. The file's own `_meta` block claims "matched: 87/87," which is not true once you check what's actually behind the IDs. Real, non-empty matches: 22/87 (~25%).

**What was built:**
- Added `get_region_id_by_alias()` and `get_task_id_by_alias()` to `neural_search/ontology/loader.py` — cached, separator-tolerant exact-match indexes (deliberately not the slower fuzzy `match_brain_regions()` matcher, which rebuilds its lookup table on every call and would not scale to 127K+ literature findings).
- New `neural_search/ontology/cognitive_atlas.py` loads `task_atlas.yaml` and **filters out the 65 placeholder matches**, exposing only the 22 validated ones via `get_cogat_match()` / `get_cogat_coverage()`.
- Wired both crosswalks into brain_region and task node creation in both graph builders (`neural_search/graph/builder.py` for the dataset/paper graph, `neural_search/literature/kg_builder.py` for the literature graph) — nodes now carry `atlas_refs` / `cogat_id` / `cogat_label` properties when a match exists, and carry nothing extra when it doesn't (no silent garbage).
- `dataset_similar_to_dataset` edges ([`similarity.py`](../../neural_search/graph/similarity.py)) now carry `derivation_method`, `review_status="unreviewed"`, `calibration_bin` (low/medium/high), and `refresh_due` (+90 days) — this is the one latent-edge type the system actually produces today.
- openMINDS/BIDS/NWB/ModelDB remain deferred — no concrete ingestion path consumes them yet, unchanged from the original assessment.

44 new tests added across `test_atlas_refs.py`, `test_cognitive_atlas.py`, `test_similarity.py`, `test_graph_builder.py`, `test_literature_kg.py`. All pass.

---

## 5. Phase 4 — Graph persistence migration (gated, not scheduled)

**Status: re-checked 2026-06-22, gate still closed.**

Keep the file-backed JSON/JSONL graph as canonical. Do not migrate to Postgres-backed graph tables or a dedicated graph database speculatively — `reports/strategy/neural_search_project_roadmap.md` (2026-06-11) already flagged this exact risk ("build the rocket before the runway"). Revisit only if graph size or multi-writer concurrency becomes an actual bottleneck; the project already has SQLAlchemy + optional pgvector to extend if/when that happens.

Current scale check: `data/graph/neural_search_graph.real_corpus.json` is 80MB; `data/graph/relationships_kg.jsonl` (new, from Phase 0) is 206MB. Both load and validate in seconds in this session's testing — single-writer, single-reader, file-backed is still adequate. Not migrating.

---

## 6. Phase 5 — Prove it matters (gated on real qrels)

**Status: re-checked 2026-06-22, gate still closed — and this is the actual bottleneck, not anything code-shaped.**

This is BrainKnow plan Milestone 4: a benchmark showing typed contradiction/frequency/region edges retrieve datasets that a BM25 + same-sentence co-occurrence baseline miss, on real human-labeled queries.

Current state per `reports/eval/qrels_progress_report.md` (generated 2026-06-20): **0 gold qrels**, 13 total human-labelled pairs against a target of 1,500, 4 unique queries covered against a target of 100, 1 annotator against a target of ≥2. Unchanged since the 2026-06-11 roadmap flagged it. This phase cannot be advanced by writing more code — it requires an actual annotation session. If/when that happens, the existing eval infrastructure (already flagged as "overbuilt relative to 0 labels" in the 2026-06-11 roadmap) should be enough to run the comparison without new tooling.

---

## 6b. Phase 6 — Multi-field consensus & contradiction (done 2026-06-23)

**Status: done.** Chosen from a brainstorm of four options (multi-field consensus, value-to-value typed-field relationships, dataset-can-test-finding bridge, cross-species crosswalk) as the cheapest high-value next step that builds directly on Phases 2-3.

**What actually shipped, vs. plan:**
- `_consensus_records_for_key()` is the shared implementation behind both `build_consensus_summaries()` (unchanged output — still writes `consensus_summaries.jsonl`) and the new `build_qualified_consensus_summaries()` (writes a **separate** `consensus_summaries_qualified.jsonl`). The plan originally sketched one combined `build_consensus_summaries_multi_field()` — split into two output files instead, after checking and finding that `apps/api/graph_router.py` and `scripts/literature/build_cluster_graph.py` both read `consensus_summaries.jsonl` live and assume exactly one row per `(region, direction, task)`. Folding the qualified tier into the same file would have multiplied `n_findings` counts in the existing graph visualization.
- `FindingEdge.contradiction_subtype` (`opposite_direction` | `direct_refutation`) and the frequency_band/injury_model context-tightening landed in `build_cross_finding_edges` as planned.
- Claims-level tightening required one addition beyond the original plan: claims didn't carry `frequency_band`/`injury_model` at all, so `detect_contradictions` had nothing to check. `cluster_findings()` now aggregates `frequency_bands`/`injury_models` (union across each cluster's findings) and `synthesize_claim()` propagates them onto the claim dict — additive fields only, the clustering *key* itself is unchanged, consistent with the "explicitly out of scope" note below.
- Test file ended up as additions to the existing `tests/test_claim_synthesizer.py` rather than a new `test_claim_contradiction_context.py` — matched existing convention instead of fragmenting it.
- Verified end-to-end on a real (if small, due to system load from the concurrent extraction job) 200-finding sample via `scripts/literature/build_finding_relationships.py`: produced real `contradiction_subtype` values and a genuinely meaningful qualified consensus record (`region=hippocampus, direction=increase, species=human, n_papers=2`) — not synthetic test data. Also fixed a pre-existing Windows console crash in that script (`≥` character, same class of bug as the typed-field audit sampler in Phase 2).
- 60 new tests across `test_finding_relationships.py` and `test_claim_synthesizer.py`. All pass; full suite collection holds at 2,827 tests.

**Context at planning time:** the literature corpus is growing live (144,401 findings / 202,850 papers processed as of this check, up from 127,232 findings a few hours earlier — extraction is running concurrently on a separate machine) and qrels are being LLM-labeled on another machine (silver-tier, not gold — Phase 5's gate stays closed until human-adjudicated). Anything built here should tolerate re-runs on a growing corpus and findings that predate the Phase 2 typed-field enrichment (missing fields, not malformed ones).

**The gap:** `build_consensus_summaries` and `detect_contradictions` ([`relationship_builder.py:297`](../../neural_search/literature/relationship_builder.py), [`claim_synthesizer.py:172`](../../neural_search/literature/claim_synthesizer.py)) key only on `(region, direction, task)`. Findings have carried 27 typed fields since Phase 2 (`frequency_band`, `injury_model`, `molecular_marker`, `species`, etc.) but consensus/contradiction never looks at them. The result: "12 papers say X increases in hippocampus" when the corpus actually supports the far sharper "12 papers say theta-band hippocampal activity increases specifically in Alzheimer mouse models during reward learning, 3 say it decreases" — the typed fields exist but the meaning they'd unlock isn't being surfaced.

**Design — tiered facets, not a full cross-product:** Adding all qualifier fields to one combined key (`region × direction × task × frequency_band × injury_model × molecular_marker × species`) would fragment findings into mostly-singleton buckets and lose the statistical power (`min_papers >= 2`) that makes a consensus record meaningful. Instead, compute consensus at the existing base tier (unchanged, backward compatible) plus one additional qualifier field at a time:

- **Base tier** (existing): `(region, direction, task)` — unchanged.
- **Qualified tier** (new): base key + exactly one of `frequency_band`, `injury_model`, `molecular_marker`, `species` — four extra passes over the findings, not a 4-dimensional cross product.

Each `ConsensusRecord` gains `facet_fields: list[str]` (`[]` for base, e.g. `["frequency_band"]` for qualified) and `specificity_tier: "base" | "qualified"`, so consumers can choose granularity rather than the function silently picking one.

**Contradiction refinement:**
- New `contradiction_subtype` on `FindingEdge`: `"opposite_direction"` (today's logic, unchanged) vs `"direct_refutation"` — a finding with `negation=True` directly disputing a same-direction finding in matching context is a different (and arguably stronger) kind of evidence than two papers landing on opposite directions.
- Context tightening, additive only: when **both** findings in a candidate contradiction pair have a populated `frequency_band` (or `injury_model`), require it to match before flagging a contradiction. Prevents false contradictions like "theta increases" vs "gamma decreases" in the same region being flagged as opposing evidence about the same phenomenon — they're not opposing, they're talking about different signals. Falls back to today's region-only behavior when either finding lacks the field, so older un-enriched findings keep working as-is.
- Same tightening applied to `detect_contradictions` in `claim_synthesizer.py` (claims-level), so mouse and human claims (or different injury models) don't get cross-contradicted just for sharing a region and an opposite direction.

**Explicitly out of scope for this phase:** extending `cluster_findings`' claim-clustering key (currently `(regions, direction, species)`) with the same qualifier fields. Claims are meant to be the broad, human-readable consensus layer; the new tiered `ConsensusRecord`s are where the fine-grained mechanistic detail should live. Doing both would create near-duplicate narrow claims for marginal benefit.

**Files to add/touch:** `relationship_builder.py` (new `build_consensus_summaries_multi_field`, `contradiction_subtype` on `FindingEdge`, context-tightening in `build_cross_finding_edges`), `claim_synthesizer.py` (context-tightening in `detect_contradictions`), `tests/test_finding_relationships.py` (tiered-consensus and contradiction-subtype cases), new `tests/test_claim_contradiction_context.py` for the claims-level tightening.

---

## 7. Explicitly not doing right now

- **`experiment_plan` / `hypothesis` / `gap` planning nodes** (deep-research report's top recommendation) — no evidence this is needed before the dataset-paper-finding bridge (BrainKnow Milestone 3) and a real benchmark (Milestone 4) exist. Speculative ahead of validation.
- **Migrating off file-backed graph storage** — not a bottleneck at current scale; see Phase 4.
- **Full openMINDS/BIDS/NWB/ModelDB crosswalk registries** — scoped down to Allen + Cognitive Atlas (Phase 3) until a concrete need surfaces.
- **GraphRAG-style subgraph retrieval as a new product surface** — the existing `apps/api` retrieval stack should consume typed edges as ranking/explanation features first; a separate path-retrieval system is premature.

---

## 8. Immediate next action

Start with **Phase 0**. It is a 1-2 day fix that converts literature-mining work that is already built and tested (`relationship_builder.py`, `claim_synthesizer.py`, `typed_finding_extractor.py`) from an orphaned JSONL sidecar into traversable, queryable graph edges — which is the concrete mechanism behind "better relate neuroscience data and papers."
