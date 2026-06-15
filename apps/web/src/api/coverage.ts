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
}
