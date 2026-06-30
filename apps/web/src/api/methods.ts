const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const get = async <T>(path: string, params?: Record<string, string | number | boolean | undefined>): Promise<T> => {
  const url = new URL(`${API_BASE}${path}`)
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined) url.searchParams.set(k, String(v))
    }
  }
  const res = await fetch(url.toString())
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json() as Promise<T>
}

export type MethodCategory = {
  id: string
  label: string
  description: string
  count: number
}

export type MethodSummary = {
  id: string
  label: string
  category: string
  category_label: string
  formula: string
  computes: string[]
  topics: string[]
  aliases: string[]
}

export type MethodDetail = MethodSummary & {
  assumptions?: Record<string, string>
  limitations?: string[]
  mathematical_basis?: string[]
  key_papers?: string[]
  related_methods?: string[]
  principle?: string
  neural_relevance?: string | Record<string, string>
  advantages?: string[]
  software?: string[]
  species_note?: string
  why_it_matters?: string[]
}

export type HomologyGroup = {
  group_id: string
  confidence: 'high' | 'medium' | 'low'
  basis: string[]
  notes: string
  divergence: string
  members: Array<{
    region_id: string
    species: string
    allen_acronym?: string
    notes?: string
  }>
}

export type OscillationSignature = {
  region_id: string
  frequency_band: string
  species: string[]
  functional_role: string
  condition: string
  generator?: string
  key_papers?: string[]
  topics: string[]
  pac_coupling?: string
  coherent_with?: string[]
  clinical_relevance?: string
  translational_significance?: string
}

export type FrequencyBand = {
  id: string
  label: string
  range_hz: [number, number]
}

export type Paradigm = {
  id: string
  label: string
  cognitive_construct: string
  description: string
  topics: string[]
  circuits_engaged: string[]
  species_available: string[]
  key_neural_signal: string
  key_finding: string
}

export type StructuralConnection = {
  source: string
  target: string
  pathway: string
  fa_estimate: number
  streamline_density?: string
  functional_relevance?: string
  topics?: string[]
  circuits?: string[]
  human_specific?: boolean
}

export const fetchMethodCategories = () =>
  get<MethodCategory[]>('/methods/categories')

export const fetchMethods = (params?: { category?: string; topic?: string }) =>
  get<MethodSummary[]>('/methods/list', params)

export const fetchMethodDetail = (id: string) =>
  get<MethodDetail>(`/methods/detail/${id}`)

export const fetchOscillations = (params?: {
  region?: string
  band?: string
  species?: string
  topic?: string
}) =>
  get<{ frequency_bands: FrequencyBand[]; signatures: OscillationSignature[]; total: number }>(
    '/methods/oscillations',
    params,
  )

export const fetchRegionOscillations = (regionId: string) =>
  get<OscillationSignature[]>(`/methods/oscillations/region/${regionId}`)

export const fetchHomologyGroups = (params?: { confidence?: string; species?: string }) =>
  get<HomologyGroup[]>('/methods/homology/groups', params)

export const fetchParadigms = (params?: { species?: string; topic?: string }) =>
  get<Paradigm[]>('/methods/paradigms', params)

export const fetchStructuralConnections = (params?: { region?: string; min_fa?: number }) =>
  get<StructuralConnection[]>('/methods/structural', params)
