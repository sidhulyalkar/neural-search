# ExperimentGlancer Scene Contract (`experimentglancer.scene.v1`)

ExperimentGlancer is a multimodal experiment-timeline browser: instead of
slicing through a 3D EM volume like Neuroglancer, it slices through a
synchronized experiment timeline (video, pose, events, neural traces,
trials, model outputs). Neural Search is the intelligence/retrieval layer;
it compiles search results and dataset introspection into a portable scene
that an independent viewer renders. Neural Search never renders anything,
and a viewer implementing this contract never needs to import Neural
Search internals.

Implementation: `neural_search/experimentglancer/`. Full design rationale:
`reports/strategy/2026-07-03_experimentglancer_integration_spec_for_claude.md`.

## Stability

`schema_version` is always `"experimentglancer.scene.v1"`. Additive,
backward-compatible fields may be added to this version; breaking changes
require a new `v2` schema literal. Treat unknown top-level fields as
forward-compatible additions a viewer can ignore.

## Fetching a scene

```text
POST /api/experimentglancer/scenes/from-search-result
GET  /api/experimentglancer/scenes/{scene_id}
GET  /api/experimentglancer/datasets/{dataset_id}/scene
GET  /api/experimentglancer/datasets/{dataset_id}/introspection
```

Every scene-producing endpoint returns:

```json
{
  "scene": { "...": "ExperimentGlancerSceneV1" },
  "scene_url": "/experimentglancer?scene_id=eg_...",
  "external_url": null,
  "warnings": []
}
```

`scene_url` is the shareable, local-viewer URL (`/experimentglancer?scene_id=...`).
`external_url` is reserved for a future standalone ExperimentGlancer
deployment (`https://experimentglancer.example/viewer?scene_url=...`); it is
`null` until that deployment exists.

Scenes are generated deterministically: the same dataset + query + evidence
always produce the same `scene_id` (see `neural_search/experimentglancer/serialization.py`),
so a shared URL keeps pointing at the same scene content.

## Evidence tiers (the core trust rule)

A layer's `status` must never imply availability beyond what was actually
verified:

| `status`      | `provenance.evidence_tier` | Meaning |
|---------------|----------------------------|---------|
| `available`   | `file_derived`             | A resolver read real file bytes (streamed NWB headers, local BIDS events/channels files) and confirmed this layer. |
| `probable`    | `metadata_inferred`        | Inferred from dataset-card metadata (modalities, experimental structure) or a query keyword match. No file was read. |
| `placeholder` | `query_requested`          | Explicitly asked for (by query or caller) but no evidence exists yet — surfaced as a warning, not a layer. |
| `unsupported` | `unknown`                  | Reserved for layers a viewer determines it cannot render at all. |

A viewer should render `probable` layers with a visibly different treatment
(e.g. dashed outline, "unverified" badge) than `available` layers, and
should surface every entry in `scene.warnings` and per-layer `warnings`
rather than hiding them.

## Layer kinds

```text
timeline.events   timeline.trials     video.frames
behavior.pose     behavior.pupil      behavior.licks      behavior.wheel
stimulus.identity reward.delivery
neural.spikes     neural.calcium      neural.lfp          neural.population_heatmap
model.predictions model.latent_state
metadata.labels   provenance.evidence
```

`metadata.labels` and `provenance.evidence` are present on every scene.

## Coordinate space and clocks

`coordinate_space.clock` is one of `experiment_time_seconds`,
`nwb_time_seconds`, `bids_events_seconds`, or `metadata_only`. Every layer's
`alignment.clock` should match (or be explicitly offset from) the scene's
coordinate space. When no resolver could confirm a real clock, the scene
uses `metadata_only` and every anchor's `time` stays `null` — see below.

## Anchors: never invent a timestamp

`anchors[].time` is `null` unless a resolver file-validated a concrete
time. Anchor selection priority (see `neural_search/experimentglancer/anchors.py`):

1. An explicit `anchor_hint` from the caller (e.g. frontend request body).
2. A query keyword match (`"lick onset"`, `"reward omission"`, `"replay"`, ...).
3. An affordance-implied anchor (`event_aligned_psth` -> event,
   `choice_decoding` / `q_learning` / `latent_dynamics_modeling` -> trial,
   `pose_neural_correlation` -> event).
4. Dataset-card trial structure -> a representative "first trial" anchor.
5. Fallback: `anchor_id="overview"`, `kind="dataset_overview"`.

`anchors[0].reason` always explains *why* that anchor was chosen, and
`evidence` lists the matched query term / affordance id / structure field
that justified it.

## Source resolvers

| Resolver               | Network? | Trigger |
|-------------------------|----------|---------|
| `metadata_only`          | No       | Default for any dataset without a more specific resolver. |
| `openneuro_bids_local`   | No       | `dataset.source == "openneuro"`; reads local BIDS fixtures under `data/corpus/fixtures/real_v07/bids/<accession>/`. Always attempted (fast, local-only). |
| `dandi_nwb_streaming`    | Yes      | `dataset.source == "dandi"` **and** the caller passes `deep_introspection: true` (API) or `deep=True` (Python). Streams NWB headers via `neural_search/data/dandi_streaming.py` without downloading full files. |

Any resolver failure (missing fixture, network error, missing optional
dependency) degrades to `metadata_only` with a warning explaining why —
never an exception, never a false claim of availability.

## Example scene

See `tests/fixtures/experimentglancer_scene_v1.json` for a complete,
schema-valid example, and `neural_search/experimentglancer/schemas.py` for
the full Pydantic model (`ExperimentGlancerSceneV1` and nested types).
