import type {
  ConsensusRow,
  DatasetNeighborhood,
  FindingRow,
  GalaxyLayout,
  SubgraphResponse,
  SuggestedView,
  TopicGraphResponse,
} from '../types/graph'

const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

async function get<T>(path: string, params?: Record<string, string | number>): Promise<T> {
  const url = new URL(`${BASE}${path}`)
  if (params) {
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, String(v)))
  }
  const res = await fetch(url.toString())
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} — ${path}`)
  return res.json() as Promise<T>
}

export function fetchGalaxyLayout(): Promise<GalaxyLayout> {
  // Static file served by vite dev server or built assets
  return fetch('/graph/galaxy_points.json').then((r) => {
    if (!r.ok) throw new Error('galaxy_points.json not found — run compute_layout.py first')
    return r.json() as Promise<GalaxyLayout>
  })
}

export function fetchGraphOverview(limit = 400): Promise<SubgraphResponse> {
  return get('/api/graph/overview', { limit })
}

export function fetchSubgraph(params: {
  regions?: string
  species?: string
  tasks?: string
  limit?: number
}): Promise<SubgraphResponse> {
  return get('/api/graph/subgraph', {
    regions: params.regions ?? '',
    species: params.species ?? '',
    tasks: params.tasks ?? '',
    limit: params.limit ?? 400,
  })
}

export function fetchTopicGraph(slug: string): Promise<TopicGraphResponse> {
  return get(`/api/graph/topic/${slug}`)
}

export function fetchSuggestedViews(): Promise<SuggestedView[]> {
  return get('/api/graph/suggested-views')
}

export function fetchConsensus(region?: string): Promise<ConsensusRow[]> {
  return get('/api/literature/consensus', region ? { region } : {})
}

export function fetchFindings(params: {
  region?: string
  direction?: string
  limit?: number
}): Promise<FindingRow[]> {
  return get('/api/literature/findings', {
    region: params.region ?? '',
    direction: params.direction ?? '',
    limit: params.limit ?? 20,
  })
}

export function fetchDatasetNeighborhood(datasetId: string): Promise<DatasetNeighborhood> {
  return get(`/api/datasets/${encodeURIComponent(datasetId)}/neighborhood`)
}
