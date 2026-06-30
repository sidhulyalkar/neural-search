const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export type CoverageSummary = {
  total_datasets: number
  total_entries: number
  dimension_coverage: Record<string, { datasets: number; pct: number }>
}

export type SourceRate = {
  source: string
  n_total: number
  regions_covered: number
  regions_pct: number
  modalities_covered: number
  modalities_pct: number
  species_covered: number
  species_pct: number
  tasks_covered: number
  tasks_pct: number
}

export type UncoveredRegion = {
  id: string
  label: string
  uberon_id: string | null
  allen_ccf_mouse_id: string | null
  parents: string[]
}

export type GapMatrixRow = {
  row: string
  col: string
  n_datasets: number
  row_dim: string
  col_dim: string
}

export type DarkPair = {
  a_value: string
  b_value: string
  n_observed: number
  a_marginal: number
  b_marginal: number
  opportunity_score: number
  dim_a: string
  dim_b: string
}

export type RegionCount = {
  region_id: string
  region_label: string
  n_datasets: number
}

export type RegionDataset = {
  dataset_id: string
  source: string
  title: string
  access_tier: string | null
  confidence: number
}

export type RegionDatasetsResponse = {
  region_id: string
  datasets: RegionDataset[]
  count: number
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json() as Promise<T>
}

export const coverageApi = {
  summary: () => get<CoverageSummary>('/api/coverage/summary'),
  sourceRates: () => get<SourceRate[]>('/api/coverage/source-rates'),
  uncoveredRegions: () => get<UncoveredRegion[]>('/api/coverage/uncovered-regions'),
  gapMatrix: (params?: { rowDim?: string; colDim?: string; species?: string; limit?: number }) => {
    const q = new URLSearchParams()
    if (params?.rowDim) q.set('row_dim', params.rowDim)
    if (params?.colDim) q.set('col_dim', params.colDim)
    if (params?.species) q.set('species', params.species)
    if (params?.limit) q.set('limit', String(params.limit))
    return get<GapMatrixRow[]>(`/api/coverage/gap-matrix?${q}`)
  },
  darkPairs: (params?: { dimA?: string; dimB?: string; limit?: number }) => {
    const q = new URLSearchParams()
    if (params?.dimA) q.set('dim_a', params.dimA)
    if (params?.dimB) q.set('dim_b', params.dimB)
    if (params?.limit) q.set('limit', String(params.limit))
    return get<DarkPair[]>(`/api/coverage/dark-pairs?${q}`)
  },
  regionCounts: () => get<RegionCount[]>('/api/coverage/region-counts'),
  regionDatasets: (regionId: string, params?: { limit?: number; offset?: number }) => {
    const q = new URLSearchParams()
    if (params?.limit != null) q.set('limit', String(params.limit))
    if (params?.offset != null) q.set('offset', String(params.offset))
    const qs = q.toString() ? `?${q}` : ''
    return get<RegionDatasetsResponse>(
      `/api/coverage/region/${encodeURIComponent(regionId)}/datasets${qs}`
    )
  },
}

// ── Allen Brain Atlas types ───────────────────────────────────────────────────

export type AllenStructure = {
  allen_id: number
  acronym: string
  name: string
  parent_id: number | null
  color_hex: string
  st_level: number
  children_ids: number[]
  atlas_id: number
}

export type AllenStructuresResponse = {
  species: string
  total: number
  structures: AllenStructure[]
}

export type AllenMappingResponse = {
  total_mapped: number
  mapping: Record<string, number>
}

export type AtlasCoverageResponse = {
  total_mouse_structures: number
  total_ontology_mapped: number
  by_level: Record<number, { total: number; mapped: number }>
}

// ── Allen Atlas API client ────────────────────────────────────────────────────

export type RegionDetailTopic = {
  id: string
  label: string
  description: string
  color: string
  companion_topics: string[]
}

export type RegionDetail = {
  id: string
  label: string
  aliases: string[]
  is_strict: boolean
  parents: Array<{ id: string; label: string }>
  children: Array<{ id: string; label: string }>
  siblings: Array<{ id: string; label: string }>
  atlas_refs: {
    allen_ccf_mouse: string | null
    allen_human: string | null
    uberon: string | null
    waxholm_rat: string | null
  }
  allen_structure: {
    allen_id: number
    acronym: string
    color_hex: string
    st_level: number | null
  } | null
  connected_topics: RegionDetailTopic[]
  functional_systems: string[]
}

export type CircuitRegion = {
  id: string
  label: string
  role: string
}

export type Circuit = {
  id: string
  label: string
  description: string
  color: string
  regions: CircuitRegion[]
  topics: string[]
}

export const atlasApi = {
  structures: (species = 'mouse', level?: number, limit = 200) => {
    const q = new URLSearchParams({ species, limit: String(limit) })
    if (level != null) q.set('level', String(level))
    return get<AllenStructuresResponse>(`/api/atlas/structures?${q}`)
  },

  structure: (allenId: number) =>
    get<AllenStructure>(`/api/atlas/structures/${allenId}`),

  children: (allenId: number, recursive = false) => {
    const q = new URLSearchParams({ recursive: String(recursive) })
    return get<AllenStructure[]>(`/api/atlas/structures/${allenId}/children?${q}`)
  },

  mapping: () => get<AllenMappingResponse>('/api/atlas/regions/mapping'),

  coverage: () => get<AtlasCoverageResponse>('/api/atlas/coverage'),

  regionDetail: (regionId: string) =>
    get<RegionDetail>(`/api/atlas/regions/${encodeURIComponent(regionId)}/detail`),

  circuits: () => get<Circuit[]>('/api/atlas/circuits'),
}
