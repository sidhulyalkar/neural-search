export type DatasetSource = 'dandi' | 'openneuro' | 'other'

export interface DatasetRecord {
  id: string
  source: DatasetSource
  source_id: string
  title: string
  description?: string
  contributors: string[]
  species: string[]
  license?: string
  doi?: string
  url?: string
  version?: string
  created_at?: string
  updated_at?: string
  data_standard?: 'nwb' | 'bids' | 'other'
  modalities: string[]
  brain_regions: string[]
  tasks: string[]
  nwb_count: number
  total_size_bytes?: number
}

export interface SearchResultItem {
  dataset: DatasetRecord
  score: number
  why_matched: string[]
  warnings: string[]
  suggested_next_actions: string[]
}

export interface SearchResult {
  query: string
  total_count: number
  results: SearchResultItem[]
  facets?: Record<string, Record<string, number>>
  search_time_ms: number
}

export interface ExtractionLabel {
  label: string
  confidence: number
  evidence: string
  source_span?: string
  extractor: string
}

export interface AnalysisReadiness {
  score: number
  strengths: string[]
  limitations: string[]
  missing_metadata: string[]
  suggested_analyses: string[]
}

export interface DatasetCard {
  dataset_id: string
  title: string
  summary: string
  source: DatasetSource
  data_standard?: 'nwb' | 'bids' | 'other'
  species: string[]
  modalities: string[]
  brain_regions: string[]
  tasks: string[]
  readiness?: AnalysisReadiness
  url?: string
  doi?: string
  related_papers: string[]
  markdown?: string
  generated_at: string
}

export interface Task {
  id: string
  label: string
  category: string
  definition: string
  synonyms: string[]
  common_events: string[]
  relevant_modalities: string[]
  relevant_regions: string[]
  suggested_analyses: string[]
}

export interface Ontology {
  tasks: Task[]
}
