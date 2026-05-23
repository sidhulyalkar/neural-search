export type DatasetSource = 'dandi' | 'openneuro' | 'demo' | 'other'
export type DatasetQAStatus =
  | 'unreviewed'
  | 'auto_generated'
  | 'needs_review'
  | 'reviewed'
  | 'trusted'
  | 'rejected'

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
  data_standard?: string
  data_standards?: string[]
  modalities: string[]
  brain_regions: string[]
  tasks: string[]
  behaviors: string[]
  nwb_count: number
  total_size_bytes?: number
  qa_status: DatasetQAStatus
  task_labels_verified?: boolean
  modality_labels_verified?: boolean
  behavior_labels_verified?: boolean
  brain_regions_verified?: boolean
  linked_papers_verified?: boolean
  notebook_tested?: boolean
  reviewer_notes?: string
}

export interface Asset {
  id: string
  path: string
  asset_type?: string
  file_format?: string
  size_bytes?: number
  content_type?: string
  modality?: string
}

export interface LinkedPaper {
  id: string
  title: string
  authors: string[]
  year?: number
  doi?: string
  openalex_id?: string
  url?: string
  confidence?: number
  link_evidence?: string[]
}

export interface SearchResultItem {
  dataset: DatasetRecord
  score: number
  why_matched: string[]
  warnings: string[]
  matched_terms?: string[]
  inferred_concepts?: string[]
  evidence_snippets?: string[]
  missing_metadata_warnings?: string[]
  reusable_reason?: string
  suggested_next_actions: string[]
  readiness_score?: number
  linked_papers?: LinkedPaper[]
}

export interface SearchResult {
  query: string
  total_count: number
  results: SearchResultItem[]
  facets?: Record<string, Record<string, number>>
  search_time_ms: number
}

export interface ExperimentQuery {
  task: string[]
  behavior: string[]
  modality: string[]
  species: string[]
  brain_region: string[]
  data_standard: string[]
  source_archive: string[]
  analysis_goal: string[]
  min_analysis_readiness_score?: number
  reviewed_trusted_only: boolean
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
  summary_details?: {
    one_sentence?: string
    scientific_use_case?: string
    why_this_dataset_matters?: string
  }
  source: DatasetSource
  source_id?: string
  data_standard?: string
  license?: string
  species: string[]
  modalities: string[]
  brain_regions: string[]
  tasks: string[]
  behaviors: string[]
  experimental_structure?: Record<string, unknown>
  neural_data?: Record<string, unknown>
  analysis_plan?: {
    readiness_score?: number
    strengths?: string[]
    limitations?: string[]
    missing_metadata?: string[]
    suggested_first_analysis?: string
    suggested_advanced_analysis?: string
    starter_recipes?: Array<{
      id: string
      title: string
      summary?: string
      level?: string
      analyses?: string[]
      required_fields?: string[]
      matched_tasks?: string[]
      match_score?: number
    }>
    [key: string]: unknown
  }
  linked_literature?: {
    candidate_papers?: LinkedPaper[]
    link_confidence_summary?: string
  }
  reuse_instructions?: {
    source_link?: string
    how_to_load?: string
    notebook_generation_status?: string
    notebook_templates?: NotebookTemplateAvailability[]
    known_caveats?: string[]
    recommended_first_steps?: string[]
    [key: string]: unknown
  }
  readiness?: AnalysisReadiness
  url?: string
  doi?: string
  related_papers: LinkedPaper[]
  assets?: Asset[]
  missing_metadata?: string[]
  provenance?: {
    dataset_source?: string
    dataset_source_id?: string
    linked_paper_count?: number
    claim_policy?: string
  }
  markdown?: string
  generated_at: string
  qa_status: DatasetQAStatus
  task_labels_verified: boolean
  modality_labels_verified: boolean
  behavior_labels_verified: boolean
  brain_regions_verified: boolean
  linked_papers_verified: boolean
  notebook_tested: boolean
  reviewer_notes: string
}

export interface NotebookTemplateAvailability {
  id: string
  title: string
  description?: string
  available: boolean
  missing_requirements: string[]
  recipes?: string[]
}

export interface DatasetQAUpdate {
  qa_status?: DatasetQAStatus
  task_labels_verified?: boolean
  modality_labels_verified?: boolean
  behavior_labels_verified?: boolean
  brain_regions_verified?: boolean
  linked_papers_verified?: boolean
  notebook_tested?: boolean
  reviewer_notes?: string
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
  behavior_labels?: Array<{
    id: string
    label: string
    synonyms: string[]
  }>
  modalities?: string[]
  brain_regions?: string[]
  analysis_goals?: string[]
}

export interface QueryEvaluation {
  query_id: string
  query: string
  expected_tasks: string[]
  expected_modalities: string[]
  found_tasks: string[]
  found_modalities: string[]
  precision_at_5: number
  label_recall: number
  passed: boolean
  top_results: Array<{
    dataset_id: string
    title: string
    score: number
  }>
}

export interface EvaluationReport {
  timestamp: string
  total_queries: number
  passed_queries: number
  avg_precision_at_5: number
  avg_label_recall_at_10: number
  query_evaluations: QueryEvaluation[]
}

export interface CompilationReport {
  generated_at: string
  total_datasets: number
  qa_review_counts?: Record<DatasetQAStatus, number>
  common_missing_metadata?: Record<string, number>
  datasets_by_source: Record<string, number>
  datasets_by_task: Record<string, number>
  datasets_by_modality: Record<string, number>
  datasets_by_species: Record<string, number>
  datasets_by_brain_region: Record<string, number>
  datasets_by_data_standard: Record<string, number>
  top_analysis_ready: Array<{
    dataset_id: string
    title: string
    score: number
    source: string
  }>
  top_demo_ready?: Array<{
    dataset_id: string
    source_id?: string
    title: string
    score: number
    source: string
    qa_status: DatasetQAStatus
    analysis_readiness_score?: number
    notebook_tested?: boolean
  }>
}
