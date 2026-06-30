const API = import.meta.env.VITE_API_URL ?? ''

export type DisorderSummary = {
  id: string
  label: string
  icd11?: string
  type?: string
  disrupted_circuits: string[]
  n_biomarkers: number
  topics: string[]
}

export type OscillationBiomarker = {
  band: string
  direction?: string
  region?: string
  note?: string
}

export type SpeciesModel = {
  species: string
  model: string
  face_validity?: string
}

export type DisorderDetail = {
  id: string
  label: string
  icd11?: string
  dsm5?: string
  type?: string
  disrupted_circuits: string[]
  oscillation_biomarkers: OscillationBiomarker[]
  species_models: SpeciesModel[]
  diagnostic_methods: string[]
  topics: string[]
  key_papers: string[]
}

export type DisorderListResponse = {
  total: number
  disorders: DisorderSummary[]
  by_type: Record<string, DisorderSummary[]>
}

export type CircuitRow = {
  circuit: string
  disorders: { id: string; label: string; type?: string }[]
  n_disorders: number
}

export type CircuitMatrixResponse = {
  circuits: CircuitRow[]
  disorders: { id: string; label: string; type?: string }[]
  n_circuits: number
  n_disorders: number
}

export type ConceptSummary = {
  id: string
  label: string
  aliases: string[]
  concept_type?: string
  broader_concept?: string
  narrower_concepts: string[]
  definition: string
  formula?: string
  related_methods: string[]
  related_regions: string[]
  testable_predictions: string[]
  topics: string[]
  scholarpedia_url?: string
}

export type ConceptListResponse = {
  total: number
  concepts: ConceptSummary[]
}

export type HierarchyNode = {
  id: string
  label: string
  concept_type?: string
  formula?: string
  definition: string
  children: HierarchyNode[]
}

export type ConceptHierarchyResponse = {
  roots: HierarchyNode[]
  total: number
}

async function get<T>(path: string, params?: Record<string, string | number | undefined>): Promise<T> {
  const url = new URL(`${API}${path}`)
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined) url.searchParams.set(k, String(v))
    }
  }
  const res = await fetch(url.toString())
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export const fetchDisorders = (params?: { disorder_type?: string; circuit?: string }) =>
  get<DisorderListResponse>('/kg/disorders', params)

export const fetchDisorder = (id: string) =>
  get<DisorderDetail>(`/kg/disorders/${id}`)

export const fetchCircuitMatrix = () =>
  get<CircuitMatrixResponse>('/kg/disorders/matrix/circuits')

export const fetchConcepts = (params?: { concept_type?: string; topic?: string }) =>
  get<ConceptListResponse>('/kg/concepts', params)

export const fetchConcept = (id: string) =>
  get<ConceptSummary & { narrower_resolved: { id: string; label: string }[] }>(`/kg/concepts/${id}`)

export const fetchConceptHierarchy = () =>
  get<ConceptHierarchyResponse>('/kg/concepts/hierarchy/tree')
