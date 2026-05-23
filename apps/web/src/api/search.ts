import type {
  SearchResult,
  DatasetCard,
  Ontology,
  EvaluationReport,
  CompilationReport,
  DatasetQAUpdate,
  ExperimentQuery,
} from '../types'

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

export async function searchDatasets(
  query: string,
  filters: Record<string, unknown> = {},
  structuredQuery?: ExperimentQuery,
): Promise<SearchResult> {
  return fetchJSON<SearchResult>(`${API_BASE}/search`, {
    method: 'POST',
    body: JSON.stringify({ query, filters, structured_query: structuredQuery }),
  })
}

export async function getDataset(id: string): Promise<DatasetCard> {
  return fetchJSON<DatasetCard>(`${API_BASE}/datasets/${id}/card`)
}

export async function exportDatasetCard(id: string, format: 'json' | 'markdown'): Promise<Blob> {
  const response = await fetch(`${API_BASE}/datasets/${id}/card/export/${format}`)

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`)
  }

  return response.blob()
}

export async function generateNotebook(datasetId: string, templateId?: string): Promise<Blob> {
  const response = await fetch(`${API_BASE}/datasets/${datasetId}/notebook`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ template_id: templateId || 'generic_nwb_inspection' }),
  })

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`)
  }

  return response.blob()
}

export async function updateDatasetQA(
  datasetId: string,
  updates: DatasetQAUpdate,
): Promise<DatasetQAUpdate> {
  return fetchJSON<DatasetQAUpdate>(`${API_BASE}/datasets/${datasetId}/qa`, {
    method: 'PATCH',
    body: JSON.stringify(updates),
  })
}

export async function getOntology(): Promise<Ontology> {
  return fetchJSON<Ontology>(`${API_BASE}/ontology/tasks`)
}

export async function healthCheck(): Promise<{ status: string }> {
  return fetchJSON<{ status: string }>('/healthz')
}

export async function getEvaluationReport(): Promise<EvaluationReport> {
  return fetchJSON<EvaluationReport>(`${API_BASE}/evaluation/report`)
}

export async function runEvaluation(): Promise<EvaluationReport> {
  return fetchJSON<EvaluationReport>(`${API_BASE}/evaluation/run`, {
    method: 'POST',
  })
}

export async function getCompilationReport(): Promise<CompilationReport> {
  return fetchJSON<CompilationReport>(`${API_BASE}/reports/compilation`)
}
