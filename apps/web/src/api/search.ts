import type { SearchResult, DatasetCard, Ontology } from '../types'

const API_BASE = '/api'

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
    },
    ...options,
  })

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`)
  }

  return response.json()
}

export async function searchDatasets(query: string): Promise<SearchResult> {
  return fetchJSON<SearchResult>(`${API_BASE}/search`, {
    method: 'POST',
    body: JSON.stringify({ query }),
  })
}

export async function getDataset(id: string): Promise<DatasetCard> {
  return fetchJSON<DatasetCard>(`${API_BASE}/datasets/${id}/card`)
}

export async function generateNotebook(datasetId: string): Promise<Blob> {
  const response = await fetch(`${API_BASE}/datasets/${datasetId}/notebook`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
  })

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`)
  }

  return response.blob()
}

export async function getOntology(): Promise<Ontology> {
  return fetchJSON<Ontology>(`${API_BASE}/ontology/tasks`)
}

export async function healthCheck(): Promise<{ status: string }> {
  return fetchJSON<{ status: string }>('/healthz')
}
