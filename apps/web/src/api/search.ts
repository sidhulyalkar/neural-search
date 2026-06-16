import type {
  SearchResult,
  DatasetCard,
  Ontology,
  EvaluationReport,
  CompilationReport,
  DatasetQAUpdate,
  ExperimentQuery,
  ComparisonResult,
  RetrievalFeedbackEvent,
} from '../types'

const API_BASE = '/api'

export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

function getErrorMessage(error: unknown): string {
  if (typeof error === 'string') return error
  if (error && typeof error === 'object') {
    const detail = 'detail' in error ? (error as { detail?: unknown }).detail : undefined
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
      return detail
        .map((item) => {
          if (item && typeof item === 'object' && 'msg' in item) {
            return String((item as { msg: unknown }).msg)
          }
          return String(item)
        })
        .join('; ')
    }
    const message = 'message' in error ? (error as { message?: unknown }).message : undefined
    if (typeof message === 'string') return message
  }
  return 'The API returned an unexpected error.'
}

async function errorFromResponse(response: Response, fallback: string): Promise<ApiError> {
  let payload: unknown = null
  try {
    payload = await response.json()
  } catch {
    try {
      payload = await response.text()
    } catch {
      payload = fallback
    }
  }
  return new ApiError(getErrorMessage(payload) || fallback, response.status)
}

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
    },
    ...options,
  })

  if (!response.ok) {
    throw await errorFromResponse(response, `API request failed with status ${response.status}.`)
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

export async function createSearchSession(payload: {
  query_text: string
  query_id?: string | null
  retrieval_method?: string
  filters?: Record<string, unknown>
  structured_query?: ExperimentQuery
}): Promise<{ session_id: string; timestamp: string; provenance: string }> {
  return fetchJSON<{ session_id: string; timestamp: string; provenance: string }>(`${API_BASE}/frontend/search-sessions`, {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export async function logRetrievalFeedback(
  event: RetrievalFeedbackEvent,
): Promise<RetrievalFeedbackEvent> {
  return fetchJSON<RetrievalFeedbackEvent>(`${API_BASE}/frontend/feedback`, {
    method: 'POST',
    body: JSON.stringify(event),
  })
}

export async function saveFrontendDataset(payload: {
  session_id?: string | null
  query_id?: string | null
  query_text: string
  dataset_id: string
  dataset_title: string
  rank?: number | null
  retrieval_method?: string
  exported?: boolean
  judge_snapshot?: unknown
}): Promise<Record<string, unknown>> {
  return fetchJSON<Record<string, unknown>>(`${API_BASE}/frontend/saved-datasets`, {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export async function getDataset(id: string): Promise<DatasetCard> {
  return fetchJSON<DatasetCard>(`${API_BASE}/datasets/${id}/card`)
}

export async function exportDatasetCard(id: string, format: 'json' | 'markdown'): Promise<Blob> {
  const response = await fetch(`${API_BASE}/datasets/${id}/card/export/${format}`)

  if (!response.ok) {
    throw await errorFromResponse(
      response,
      `Could not export this dataset card. The API returned ${response.status}.`,
    )
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
    throw await errorFromResponse(
      response,
      `Could not generate this notebook. The API returned ${response.status}.`,
    )
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

// Dataset Comparison API
export async function compareDatasets(datasetIds: string[]): Promise<ComparisonResult> {
  return fetchJSON<ComparisonResult>(`${API_BASE}/datasets/compare`, {
    method: 'POST',
    body: JSON.stringify({ dataset_ids: datasetIds }),
  })
}

export async function exportComparisonMarkdown(datasetIds: string[]): Promise<Blob> {
  const response = await fetch(`${API_BASE}/datasets/compare/export/markdown`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ dataset_ids: datasetIds }),
  })

  if (!response.ok) {
    throw await errorFromResponse(
      response,
      `Could not export comparison. The API returned ${response.status}.`,
    )
  }

  return response.blob()
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function getDemoReport(): Promise<any> {
  return fetchJSON<any>(`${API_BASE}/demo/report`)
}

export async function exportComparisonJson(datasetIds: string[]): Promise<Blob> {
  const response = await fetch(`${API_BASE}/datasets/compare/export/json`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ dataset_ids: datasetIds }),
  })

  if (!response.ok) {
    throw await errorFromResponse(
      response,
      `Could not export comparison. The API returned ${response.status}.`,
    )
  }

  return response.blob()
}

export type AffordanceSupportLevel = 'high' | 'medium' | 'low' | 'unsupported' | 'unknown'

export type AffordanceResult = {
  affordance_id: string
  label: string
  support_level: AffordanceSupportLevel
  confidence: number
  found_required_features: string[]
  missing_required_features: string[]
  found_optional_features: string[]
}

export type DatasetAffordancesResponse = {
  dataset_id: string
  affordances: AffordanceResult[]
  detection_method: string
}

export async function getDatasetAffordances(datasetId: string): Promise<DatasetAffordancesResponse> {
  return fetchJSON<DatasetAffordancesResponse>(
    `${API_BASE}/datasets/${encodeURIComponent(datasetId)}/affordances`
  )
}
