const API_BASE = '/api'

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!response.ok) {
    let detail = `API request failed with status ${response.status}`
    try {
      const payload = await response.json() as { detail?: unknown }
      if (typeof payload.detail === 'string') detail = payload.detail
    } catch {
      // Keep the status-based fallback.
    }
    throw new Error(detail)
  }
  return response.json() as Promise<T>
}

export type RuntimeCapability = {
  capability: string
  artifact_id: string
  available: boolean
  state: string
  version: string | null
  lineage_id: string | null
  resolution_source: string
  description: string
}

export type RuntimeCapabilities = {
  profile: string | null
  available: number
  total: number
  capabilities: RuntimeCapability[]
}

export type CompatibilityIssue = {
  artifact_id: string
  lineage_id?: string
  reason?: string
  issues?: string[]
  expected_parents?: string[]
}

export type RuntimeStatus = {
  profile: {
    name: string
    description: string
  }
  ready: boolean
  health: 'ready' | 'degraded' | 'unhealthy'
  missing_modules: string[]
  recommended_missing: string[]
  compatibility: {
    state: 'compatible' | 'unknown' | 'incompatible'
    compatible: boolean
    unknown: CompatibilityIssue[]
    incompatible: CompatibilityIssue[]
  }
  capabilities: RuntimeCapabilities
}

export type BundleState = {
  index_path: string
  available_bundles: Array<{
    name: string
    version: string
    ref: string
    manifest_url: string
    compatibility_group: string | null
    deprecated: boolean
  }>
  lock: {
    lock_path: string
    bundles: Record<string, unknown>
    artifacts: Record<string, unknown>
  }
  verification: {
    valid: boolean
    artifacts: Array<Record<string, unknown>>
  }
}

export type ReanalysisEvidence = {
  kind: string
  source_id: string
  summary: string
  confidence: number | null
  evidence_tier: string
  metadata: Record<string, unknown>
}

export type ReanalysisCandidate = {
  method_id: string
  method_label: string
  analysis_family: string
  data_form: string
  priority_score: number
  feasibility_score: number
  feasibility_status: string
  novelty_status: string
  rationale: string
  required_signals: string[]
  present_required_signals: string[]
  missing_required_signals: string[]
  assumptions: string[]
  limitations: string[]
  computes: string[]
  precedent_datasets: Array<Record<string, unknown>>
  evidence: ReanalysisEvidence[]
  requires_human_review: boolean
}

export type ReanalysisPlan = {
  dataset_id: string
  dataset_title: string
  generated_at: string
  matched_data_forms: string[]
  candidates: ReanalysisCandidate[]
  uncovered_analysis_families: string[]
  warnings: string[]
  evidence_policy: string
  corpus_source: string
  evidence_capabilities: Record<string, boolean>
}

export type AdoptionEvent = {
  session_id: string
  timestamp: string
  event_type: string
  success?: boolean
  dataset_id?: string
  usefulness?: 'useful' | 'partially_useful' | 'not_useful' | 'unsure'
  would_use_for_analysis?: 'yes' | 'maybe' | 'no'
  known_before?: boolean
  metadata?: Record<string, unknown>
}

export function getRuntimeStatus(profile?: string): Promise<RuntimeStatus> {
  const query = profile ? `?profile=${encodeURIComponent(profile)}` : ''
  return fetchJSON<RuntimeStatus>(`${API_BASE}/runtime/status${query}`)
}

export function getRuntimeCapabilities(profile?: string): Promise<RuntimeCapabilities> {
  const query = profile ? `?profile=${encodeURIComponent(profile)}` : ''
  return fetchJSON<RuntimeCapabilities>(`${API_BASE}/runtime/capabilities${query}`)
}

export function getBundleState(): Promise<BundleState> {
  return fetchJSON<BundleState>(`${API_BASE}/runtime/bundles`)
}

export function getReanalysisPlan(datasetId: string, limit = 12): Promise<ReanalysisPlan> {
  return fetchJSON<ReanalysisPlan>(
    `${API_BASE}/reanalysis/${encodeURIComponent(datasetId)}?limit=${limit}`,
  )
}

export function recordAdoptionEvent(event: AdoptionEvent): Promise<{ recorded: boolean }> {
  return fetchJSON<{ recorded: boolean }>(`${API_BASE}/adoption/events`, {
    method: 'POST',
    body: JSON.stringify(event),
  })
}
