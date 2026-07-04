const API_BASE = '/api'

export type ClockKind =
  | 'experiment_time_seconds'
  | 'nwb_time_seconds'
  | 'bids_events_seconds'
  | 'metadata_only'

export type LayerStatus = 'available' | 'probable' | 'placeholder' | 'unsupported'

export type EvidenceTier = 'file_derived' | 'metadata_inferred' | 'query_requested' | 'unknown'

export interface QueryContext {
  query: string
  parsed_query: Record<string, unknown>
  requested_layers: string[]
  affordance_ids: string[]
}

export interface SceneSource {
  kind: 'search_result' | 'dataset_card' | 'manual' | 'reanalysis_candidate'
  query?: string | null
  rank?: number | null
  retrieval_method?: string | null
  score?: number | null
  score_breakdown: Record<string, number>
}

export interface SceneDatasetRef {
  dataset_id: string
  source?: string | null
  source_id?: string | null
  title?: string | null
  url?: string | null
  doi?: string | null
  data_standard?: string | null
  species: string[]
  modalities: string[]
  brain_regions: string[]
  tasks: string[]
}

export interface DefaultWindow {
  center_time: number | null
  pre: number
  post: number
}

export interface CoordinateSpace {
  clock: ClockKind
  time_unit: string
  default_window: DefaultWindow
  session_id?: string | null
  subject_id?: string | null
  trial_id?: string | null
}

export interface ExperimentGlancerAnchor {
  anchor_id: string
  kind: string
  label: string
  time: number | null
  trial_id?: string | null
  event_type?: string | null
  reason: string
  evidence: string[]
}

export interface LayerDataRef {
  kind: string
  asset_id?: string | null
  path?: string | null
  column?: string | null
  table?: string | null
}

export interface LayerAlignment {
  clock: ClockKind
  offset: number
}

export interface LayerDisplay {
  track: string
  color?: string | null
}

export interface LayerProvenance {
  evidence_tier: EvidenceTier
  detector: string
}

export interface ExperimentGlancerLayer {
  layer_id: string
  kind: string
  label: string
  status: LayerStatus
  data_ref: LayerDataRef
  alignment: LayerAlignment
  display: LayerDisplay
  provenance: LayerProvenance
  warnings: string[]
}

export interface SceneLayout {
  primary_view: string
  tracks: string[]
}

export interface SceneProvenance {
  generated_by: string
  generator_version: string
  inputs: Record<string, string>
  evidence_tier: EvidenceTier
  missing_requirements: string[]
}

export interface ExperimentGlancerScene {
  schema_version: string
  scene_id: string
  created_at: string
  source: SceneSource
  query_context: QueryContext | null
  dataset: SceneDatasetRef
  coordinate_space: CoordinateSpace
  anchors: ExperimentGlancerAnchor[]
  layers: ExperimentGlancerLayer[]
  layout: SceneLayout
  provenance: SceneProvenance
  warnings: string[]
}

export interface ExperimentGlancerSceneResponse {
  scene: ExperimentGlancerScene
  scene_url: string
  external_url: string | null
  warnings: string[]
}

export interface AnchorHint {
  kind?: string
  event_type?: string
  relative_time?: number
}

export interface CreateSceneFromSearchResultPayload {
  query: string
  dataset_id: string
  rank?: number | null
  retrieval_method?: string
  score?: number
  score_breakdown?: Record<string, number>
  include_probable_layers?: boolean
  requested_layers?: string[]
  affordance_ids?: string[]
  anchor_hint?: AnchorHint
  deep_introspection?: boolean
}

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })

  if (!response.ok) {
    let detail = `API request failed with status ${response.status}.`
    try {
      const payload = await response.json()
      if (typeof payload?.detail === 'string') detail = payload.detail
    } catch {
      // Response body wasn't JSON; fall back to the generic message above.
    }
    throw new Error(detail)
  }

  return response.json()
}

export async function createSceneFromSearchResult(
  payload: CreateSceneFromSearchResultPayload,
): Promise<ExperimentGlancerSceneResponse> {
  return fetchJSON<ExperimentGlancerSceneResponse>(
    `${API_BASE}/experimentglancer/scenes/from-search-result`,
    {
      method: 'POST',
      body: JSON.stringify(payload),
    },
  )
}

export async function getDatasetScene(datasetId: string): Promise<ExperimentGlancerSceneResponse> {
  return fetchJSON<ExperimentGlancerSceneResponse>(
    `${API_BASE}/experimentglancer/datasets/${encodeURIComponent(datasetId)}/scene`,
  )
}

export async function getScene(sceneId: string): Promise<ExperimentGlancerSceneResponse> {
  return fetchJSON<ExperimentGlancerSceneResponse>(
    `${API_BASE}/experimentglancer/scenes/${encodeURIComponent(sceneId)}`,
  )
}
