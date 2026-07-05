# ExperimentGlancer Integration Spec for Claude Code

**Date:** 2026-07-03  
**Repository:** `neural-search`  
**Working title:** ExperimentGlancer  
**Alternative names:** NeuroCinema, SynapseScope, ExperimentGlancer  
**Goal:** Design and implement the bridge between Neural Search as the intelligence/retrieval layer and ExperimentGlancer as a multimodal neural-behavior browser.

## Claude Code Prompt

You are working in the `neural-search` repository. Your task is to add a new ExperimentGlancer integration layer that turns Neural Search results into portable, shareable multimodal visualization scenes.

Neural Search should remain the intelligence layer:

- Parse research intent.
- Retrieve datasets.
- Explain why each dataset matched.
- Surface evidence, affordance support, missing metadata, source links, and provenance.
- Inspect datasets enough to know what visual layers can be opened.

ExperimentGlancer should be treated as the browser/visualization layer:

- It receives scene JSON.
- It renders synchronized video, pose, events, neural traces, trials, model outputs, and metadata when source data or derived assets are available.
- It supports shareable URLs that encode scientific coordinates.

The bridge layer should convert search results, affordance matches, dataset cards, and dataset introspection outputs into shareable scene JSON.

Do not build a generic dashboard or another static dataset page. The new feature is a timeline-first scientific browser: a user searches by research intent, inspects why a dataset was retrieved, and opens an automatically generated ExperimentGlancer scene centered on the relevant trial/event/neural/behavior layers.

## Executive Product Concept

ExperimentGlancer is "Neuroglancer for experiment time."

Instead of slicing through a 3D EM volume, the scientist slices through a synchronized experiment timeline:

- video frames
- pose keypoints
- pupil diameter
- lick events
- wheel movement
- stimulus identity
- reward delivery
- trial labels
- spike rasters
- calcium traces
- population manifolds
- model predictions
- latent state embeddings
- metadata and provenance

The core interaction is not "show me neuron 542." It is:

- "Show me the moment the animal switched strategy."
- "Show trials where the model was confident but the animal failed."
- "Jump to pupil dilation events preceding hippocampal replay."
- "Open session X, trial 128, 2.3 seconds before lick onset, neurons sorted by ramping activity."

The secret sauce is portable scientific coordinates: search result -> scene JSON -> URL -> exact dataset/session/trial/time/layer state.

## Current Repository Assessment

The repo already has most of the intelligence-side ingredients:

- Search stack: `neural_search/search/core.py`
  - Parses queries into tasks, modalities, species, brain regions, affordances, exclusions, and analysis intent.
  - Produces `SearchResult` with score breakdowns, `why_matched`, warnings, evidence snippets, missing metadata, graph context, usefulness score, linked papers, and explanation.
- API search endpoint: `apps/api/main.py`
  - Converts backend search results into `FrontendSearchResult`.
  - Adds neuro-judge snapshots, evidence packets, prior feedback, memory graph evidence, linked papers, and rank.
- Frontend result contract: `apps/web/src/types/index.ts`
  - `SearchResultItem` already carries dataset, score, evidence, score breakdown, neuro-judge, evidence packet, memory graph evidence, and feedback.
- Dataset cards: `neural_search/cards/generator.py`
  - Generates experimental structure, neural data, analysis plan, linked literature, reuse instructions, missing metadata, and provenance.
- Affordance detection:
  - `neural_search/analysis_affordances.py`
  - `neural_search/affordances/registry.py`
  - `apps/api/main.py` endpoint `/api/datasets/{dataset_id}/affordances`
- Dataset context/usefulness:
  - `neural_search/retrieval/dataset_context_bridge.py`
  - `neural_search/retrieval/usefulness_scorer.py`
- DANDI/NWB streaming:
  - `neural_search/data/dandi_streaming.py`
  - Can list DANDI assets and inspect NWB metadata via byte-range streaming.
- Graph and provenance:
  - `neural_search/graph/*`
  - Dataset-paper, typed KG, temporal, coverage, and reanalysis bridge modules exist.
- Existing frontend pattern:
  - `apps/web/src/pages/ResultsPage.tsx`
  - `apps/web/src/pages/DatasetPage.tsx`
  - `apps/web/src/components/DatasetCard.tsx`

The missing category is not another retriever. It is a scene compiler and browser bridge.

## Architectural Decision

Add ExperimentGlancer as a bounded integration layer with three responsibilities:

1. Scene planning
   - Decide which visualization layers should be included for a result and why.
   - Use query intent, matched terms, affordances, dataset card fields, evidence packet fields, and source-specific file introspection.

2. Scene JSON generation
   - Produce a stable `ExperimentGlancerSceneV1` schema.
   - Include data sources, layers, synchronized clocks, anchors, selected trials/events, display settings, provenance, and warnings.

3. Scene delivery
   - Provide API endpoints and frontend actions that let a user open a scene from a search result or dataset page.
   - Support shareable URLs by either storing scene JSON server-side or encoding a compact scene reference in the URL.

Do not put rendering logic in Neural Search. Neural Search should output a scene contract. ExperimentGlancer owns rendering.

## Proposed Package Layout

Add:

```text
neural_search/experimentglancer/
  __init__.py
  schemas.py
  layer_planner.py
  scene_builder.py
  source_resolvers.py
  anchors.py
  serialization.py

apps/api/experimentglancer_router.py

tests/test_experimentglancer_schema.py
tests/test_experimentglancer_scene_builder.py
tests/test_experimentglancer_api.py
```

Optional later:

```text
apps/web/src/pages/ExperimentGlancerPage.tsx
apps/web/src/components/experimentglancer/SceneLaunchButton.tsx
apps/web/src/components/experimentglancer/LayerSummary.tsx
apps/web/src/components/experimentglancer/ScenePreview.tsx
```

## Scene Schema

Create `ExperimentGlancerSceneV1` in `neural_search/experimentglancer/schemas.py`.

The schema should be explicit, versioned, and conservative. It must never imply a layer exists when it is only inferred from metadata.

```python
class ExperimentGlancerSceneV1(BaseModel):
    schema_version: Literal["experimentglancer.scene.v1"] = "experimentglancer.scene.v1"
    scene_id: str
    created_at: str
    source: SceneSource
    query_context: QueryContext | None = None
    dataset: SceneDatasetRef
    coordinate_space: CoordinateSpace
    anchors: list[SceneAnchor] = []
    layers: list[SceneLayer] = []
    layout: SceneLayout = Field(default_factory=SceneLayout)
    provenance: SceneProvenance
    warnings: list[str] = []
```

Minimum nested objects:

- `SceneSource`
  - `kind`: `search_result`, `dataset_card`, `manual`, `reanalysis_candidate`
  - `query`, `rank`, `retrieval_method`, `score`, `score_breakdown`
- `SceneDatasetRef`
  - `dataset_id`, `source`, `source_id`, `title`, `url`, `doi`, `data_standard`
  - `species`, `modalities`, `brain_regions`, `tasks`
- `CoordinateSpace`
  - `clock`: `experiment_time_seconds`, `nwb_time_seconds`, `bids_events_seconds`, or `metadata_only`
  - `time_unit`: usually `s`
  - `default_window`: `{center_time, pre, post}`
  - `session_id`, `subject_id`, `trial_id`
- `SceneAnchor`
  - `anchor_id`, `kind`, `label`
  - `time`, `trial_id`, `event_type`
  - `reason`: why Neural Search selected this anchor
  - `evidence`: evidence strings or references
- `SceneLayer`
  - `layer_id`, `kind`, `label`
  - `status`: `available`, `probable`, `placeholder`, `unsupported`
  - `data_ref`: source URL/path/asset ID/column/table reference
  - `alignment`: clock and offset metadata
  - `display`: compact display config
  - `provenance`: evidence tier and detector
  - `warnings`
- `SceneProvenance`
  - `generated_by`, `generator_version`
  - `inputs`: search result hash, dataset card hash, introspection hash if available
  - `evidence_tier`: highest available tier for layer planning
  - `missing_requirements`

Layer kinds should start with:

- `timeline.events`
- `timeline.trials`
- `video.frames`
- `behavior.pose`
- `behavior.pupil`
- `behavior.licks`
- `behavior.wheel`
- `stimulus.identity`
- `reward.delivery`
- `neural.spikes`
- `neural.calcium`
- `neural.lfp`
- `neural.population_heatmap`
- `model.predictions`
- `model.latent_state`
- `metadata.labels`
- `provenance.evidence`

## Scene Planning Rules

Implement `plan_layers_for_result(...)` in `layer_planner.py`.

Inputs:

- dataset record
- optional dataset card
- `SearchResult` or frontend result-like dict
- parsed query
- optional affordance validation results
- optional file introspection metadata

Rules:

- Always include `metadata.labels` and `provenance.evidence`.
- Include `timeline.trials` if card experimental structure or dataset usability says trials exist.
- Include `timeline.events` if behavioral events, event timestamps, BIDS events, or NWB trials are present.
- Include `neural.spikes` for Neuropixels/electrophysiology/NWB units when file-derived or metadata-derived support exists.
- Include `neural.calcium` for two-photon/calcium imaging/ophys/ROI traces.
- Include `behavior.pose` only if behavior tracking, pose, DeepLabCut, keypoints, kinematics, or video-derived pose is detected.
- Include `video.frames` only when video assets or source metadata explicitly expose video. Otherwise create a `placeholder` warning, not an available layer.
- Include `behavior.pupil`, `behavior.licks`, `behavior.wheel`, `stimulus.identity`, and `reward.delivery` when events/trial columns suggest those signals.
- Include `model.predictions` and `model.latent_state` only if generated derivatives exist or the query asks for model overlays and the scene can create placeholder layers explaining what derivative is required.
- Mark all file-unverified layers as `probable` with evidence tier `metadata_inferred`.
- Mark file-derived NWB/BIDS layers as `available` with evidence tier `file_derived`.

The planner should produce warnings such as:

- `video layer requested by query but no video asset detected`
- `event timestamps inferred from behavioral labels; file validation pending`
- `model latent state layer requires derived artifact generation`

## Anchor Selection

Implement `select_scene_anchors(...)` in `anchors.py`.

Anchor priorities:

1. Query-specified event or trial:
   - lick onset
   - reward omission
   - stimulus onset
   - trial outcome
   - movement onset
   - replay event
   - strategy switch
2. Affordance-related anchor:
   - `event_aligned_activity` -> first relevant event timestamp if available.
   - `choice_decoding` -> choice/response trial.
   - `q_learning_modeling` -> reward/outcome trial.
   - `brain_behavior_alignment` -> behavior event + neural layer.
   - `latent_dynamics_modeling` -> trial or session midpoint if no event.
3. Dataset-card structure:
   - first trial, representative trial, or metadata-only overview.
4. Fallback:
   - `anchor_id="overview"`, `kind="dataset_overview"`, no concrete time.

Do not invent actual timestamps. If introspection cannot find concrete times, create semantic anchors with `time=None` and a clear warning.

## Source Resolvers

Implement `source_resolvers.py` as adapters that translate dataset identity into resolvable source references.

Start with:

- DANDI/NWB resolver:
  - Use `source == "dandi"` or DANDI-like `source_id`.
  - Reuse `neural_search/data/dandi_streaming.py` where possible.
  - Return asset refs and lightweight metadata, not downloaded data.
- OpenNeuro/BIDS resolver:
  - Use local fixture paths when available.
  - Detect `events.tsv`, EEG/iEEG/fMRI JSON sidecars, channels files, and derivatives if present.
- Metadata-only resolver:
  - Works for all datasets.
  - Produces labels, provenance, and placeholders.

Return a normalized `DatasetIntrospectionV1`:

- assets
- clocks
- sessions
- subjects
- trial/event columns
- available modalities
- detected layers
- missing layer requirements
- source warnings

## API Endpoints

Add `apps/api/experimentglancer_router.py` and include it in `apps/api/main.py`.

Endpoints:

```text
POST /api/experimentglancer/scenes/from-search-result
GET  /api/experimentglancer/scenes/{scene_id}
GET  /api/experimentglancer/datasets/{dataset_id}/scene
GET  /api/experimentglancer/datasets/{dataset_id}/introspection
```

Suggested request for `from-search-result`:

```json
{
  "query": "visual decision-making with Neuropixels and failed confident model trials",
  "dataset_id": "dandi:000XXX",
  "rank": 1,
  "include_probable_layers": true,
  "requested_layers": ["timeline.events", "neural.spikes", "behavior.pupil", "model.predictions"],
  "anchor_hint": {
    "kind": "event",
    "event_type": "lick_onset",
    "relative_time": -2.3
  }
}
```

Response:

```json
{
  "scene": { "...": "ExperimentGlancerSceneV1" },
  "scene_url": "/experimentglancer?scene_id=...",
  "external_url": null,
  "warnings": []
}
```

For MVP, store generated scenes in an in-memory TTL cache, similar to `_SEARCH_CACHE` in `apps/api/main.py`. Later, persist by hash under `artifacts/experimentglancer/scenes/`.

## Frontend Integration

MVP UI:

- Add `experimentglancer_scene?: { scene_id, scene_url, layer_summary, warnings }` to `SearchResultItem`.
- In `DatasetCard`, add a compact action:
  - label: `Open Scene`
  - disabled or warning state if only metadata/placeholder layers exist.
- On click, call `POST /api/experimentglancer/scenes/from-search-result`.
- Open `/experimentglancer?scene_id=...` if a local page exists, or show the JSON/URL in a modal during MVP.

Better first product experience:

- In each result card, show `Scene layers: trials, events, spikes, calcium, behavior, metadata`.
- Make unavailable layers visible as warnings:
  - `No video detected`
  - `Pose requires derived DeepLabCut output`
  - `Model overlays not generated yet`
- On dataset page, add an `ExperimentGlancer` panel below "What can I do with this dataset?"

Do not create a marketing landing page. The scene launch belongs directly in search results and dataset pages.

## Scene JSON Example

```json
{
  "schema_version": "experimentglancer.scene.v1",
  "scene_id": "eg_dandi_000000_queryhash",
  "created_at": "2026-07-03T00:00:00Z",
  "source": {
    "kind": "search_result",
    "query": "visual behavior calcium imaging with lick events",
    "rank": 1,
    "retrieval_method": "hybrid_search",
    "score": 0.87,
    "score_breakdown": {
      "ontology_score": 0.8,
      "analysis_fit_score": 0.72,
      "readiness": 0.65
    }
  },
  "dataset": {
    "dataset_id": "dandi:000000",
    "source": "dandi",
    "source_id": "000000",
    "title": "Example visual behavior dataset",
    "url": "https://dandiarchive.org/dandiset/000000",
    "data_standard": "NWB",
    "species": ["mouse"],
    "modalities": ["calcium_imaging", "behavior_video"],
    "brain_regions": ["visual_cortex"],
    "tasks": ["visual_change_detection"]
  },
  "coordinate_space": {
    "clock": "nwb_time_seconds",
    "time_unit": "s",
    "default_window": {"center_time": null, "pre": 2.0, "post": 5.0},
    "subject_id": null,
    "session_id": null,
    "trial_id": null
  },
  "anchors": [
    {
      "anchor_id": "overview",
      "kind": "dataset_overview",
      "label": "Overview from search match",
      "time": null,
      "trial_id": null,
      "event_type": null,
      "reason": "No concrete event timestamp was file-validated yet.",
      "evidence": ["matched calcium imaging", "matched lick events"]
    }
  ],
  "layers": [
    {
      "layer_id": "trials",
      "kind": "timeline.trials",
      "label": "Trials",
      "status": "probable",
      "data_ref": {"kind": "nwb_trials_table", "asset_id": null},
      "alignment": {"clock": "nwb_time_seconds", "offset": 0},
      "display": {"track": "timeline", "color": "#7dd3fc"},
      "provenance": {"evidence_tier": "metadata_inferred", "detector": "dataset_card"},
      "warnings": ["Trial structure inferred from metadata; file validation pending."]
    }
  ],
  "layout": {
    "primary_view": "timeline",
    "tracks": ["timeline", "behavior", "neural", "model", "metadata"]
  },
  "provenance": {
    "generated_by": "neural_search.experimentglancer.scene_builder",
    "generator_version": "v0.1.0",
    "inputs": {},
    "evidence_tier": "metadata_inferred",
    "missing_requirements": ["file validated event timestamps"]
  },
  "warnings": []
}
```

## Development Phases

### Phase 0: Design Contract

Deliver:

- `schemas.py`
- JSON schema test
- example scene fixture under `tests/fixtures/experimentglancer_scene_v1.json`

Acceptance:

- Scene model validates.
- Scene serializes to stable JSON.
- Missing layers are represented as warnings, not false availability.

### Phase 1: Metadata-Only Scene Builder

Deliver:

- `scene_builder.py`
- `layer_planner.py`
- `anchors.py`
- `source_resolvers.py` metadata-only resolver

Acceptance:

- Any search result/dataset can produce a scene with metadata/provenance layers.
- Datasets with trials/events/neural modalities produce probable layer summaries.
- Query terms influence anchors and requested layers.

### Phase 2: API Bridge

Deliver:

- `apps/api/experimentglancer_router.py`
- Router included in `apps/api/main.py`
- API tests

Acceptance:

- `GET /api/experimentglancer/datasets/{dataset_id}/scene` works for a demo dataset.
- `POST /api/experimentglancer/scenes/from-search-result` works with query + dataset ID.
- Response includes `scene_url`, layer summary, and warnings.

### Phase 3: Search Result UI

Deliver:

- Add `Open Scene` to result cards.
- Add type definitions.
- Add API client helper.

Acceptance:

- A user can search, inspect why a result matched, and launch a scene.
- The UI distinguishes available/probable/placeholder layers.

### Phase 4: Source-Specific Introspection

Deliver:

- DANDI/NWB lightweight introspection using streaming metadata.
- OpenNeuro/BIDS local fixture introspection.

Acceptance:

- DANDI NWB datasets expose file-derived trial/unit/ophys layers when metadata can be streamed.
- OpenNeuro fixtures expose events/channel layers when files exist locally.
- Network failures degrade to metadata-only scenes with warnings.

### Phase 5: ExperimentGlancer Viewer Contract

Deliver:

- A documented external viewer URL format:
  - `https://experimentglancer.example/viewer?scene_url=...`
  - or local `/experimentglancer?scene_id=...`
- Stable scene JSON docs.

Acceptance:

- The scene contract can be implemented by an independent viewer without importing Neural Search internals.

## Testing Strategy

Add focused tests first:

- `test_experimentglancer_schema.py`
  - validates required fields and enum values
  - rejects unsupported layer statuses/kinds
- `test_experimentglancer_scene_builder.py`
  - metadata-only dataset creates metadata/provenance layers
  - calcium dataset creates calcium layer
  - Neuropixels/ephys dataset creates spike layer
  - behavior tracking dataset creates pose layer
  - query "lick onset" creates lick/event anchor if labels exist
  - missing video becomes warning, not available layer
- `test_experimentglancer_api.py`
  - dataset scene endpoint
  - from-search-result endpoint
  - unknown dataset returns 404

Suggested lightweight verification:

```bash
python -m pytest tests/test_experimentglancer_schema.py tests/test_experimentglancer_scene_builder.py tests/test_experimentglancer_api.py -q
python -m pytest tests/test_api_smoke.py -q
npm --prefix apps/web run build
```

Do not run full corpus rebuilds, full embedding rebuilds, or full extraction unless explicitly asked.

## Risk Controls

- Do not overclaim layer availability.
  - `available` requires source/file evidence.
  - `probable` means metadata suggests it.
  - `placeholder` means desired but absent until derivatives are generated.
- Do not download large datasets in tests.
- Do not require network for core tests.
- Keep scene generation deterministic.
- Keep scene JSON independent of React or FastAPI internals.
- Preserve current search API behavior unless adding optional fields.
- Respect the dirty worktree. Do not revert unrelated changes.

## Strategic Fit

This expansion turns Neural Search from "find me a dataset" into "take me to the scientifically relevant moment in the dataset."

That is the right next leap because the repo already differentiates itself on:

- experiment-aware retrieval
- analysis affordances
- provenance
- dataset cards
- notebook generation
- graph-backed explanations
- DANDI/NWB and OpenNeuro/BIDS awareness

ExperimentGlancer should not compete with those systems. It should make their output visible, shareable, and actionable.

The north-star workflow is:

1. Researcher searches by intent.
2. Neural Search returns datasets with evidence and affordances.
3. Researcher sees why a dataset matched.
4. Researcher clicks `Open Scene`.
5. ExperimentGlancer opens a synchronized, shareable timeline centered on the relevant neural, behavioral, trial, model, and metadata layers.

That is the missing category.
