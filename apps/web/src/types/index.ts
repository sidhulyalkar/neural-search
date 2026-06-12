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
  abstract?: string
  abstract_snippet?: string
}

export interface NeuroJudgeSnapshot {
  label?: number
  confidence?: number
  rationale_short?: string
  evidence_for?: string[]
  evidence_against?: string[]
  missing_information?: string[]
  matched_dimensions?: string[]
  failure_modes?: string[]
  hard_negative_detected?: boolean
  judge_model?: string
  judge_models?: string[]
  prompt_version?: string
  evidence_packet_hash?: string
  label_provenance?: string
  evidence_completeness?: number
  required_dimensions_present?: string[]
  required_dimensions_missing?: string[]
  abstain_recommended?: boolean
  abstain_reason?: string | null
  watermark?: string
}

export interface EvidencePacketSummary {
  query_id?: string
  query_text?: string
  query_intent?: string
  hard_negatives?: string[]
  expected_species?: string[]
  expected_modalities?: string[]
  expected_brain_regions?: string[]
  expected_tasks?: string[]
  expected_analysis_affordances?: string[]
  dataset_id?: string
  title?: string
  source_archive?: string
  source_url?: string
  description?: string
  affordance_matches?: Array<{
    affordance?: string
    matched?: boolean
    confidence?: number
    missing_requirements?: string[]
    rationale?: string
  }>
  concept_explanation_summary?: string
  matched_concept_names?: string[]
  concept_missing_evidence?: string[]
  concept_hard_negative_conflicts?: string[]
  has_raw_data?: boolean | null
  has_processed_data?: boolean | null
  file_format_evidence?: string[]
  known_failure_warnings?: string[]
  linked_papers?: LinkedPaper[]
  raw_json?: Record<string, unknown>
}

export type FeedbackUsefulness = 'useful' | 'partially_useful' | 'not_useful' | 'unsure'
export type WouldUseForAnalysis = 'yes' | 'maybe' | 'no'

export interface RetrievalFeedbackEvent {
  feedback_id?: string
  timestamp?: string
  session_id?: string | null
  query_id?: string | null
  query_text: string
  retrieval_method: string
  rank?: number | null
  dataset_id: string
  dataset_title: string
  usefulness: FeedbackUsefulness
  would_use_for_analysis?: WouldUseForAnalysis | null
  clicked?: boolean
  opened_evidence?: boolean
  saved?: boolean
  exported?: boolean
  reason_tags: string[]
  free_text_note?: string
  judge_snapshot?: NeuroJudgeSnapshot
  provenance?: 'user_feedback_downstream_signal'
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
  rank?: number
  retrieval_method?: string
  score_breakdown?: Record<string, number>
  neuro_judge?: NeuroJudgeSnapshot | null
  evidence_packet?: EvidencePacketSummary | null
  prior_feedback?: RetrievalFeedbackEvent[]
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
  warnings?: string[]
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
  queries_with_results?: number
  avg_precision_at_5: number
  avg_label_recall_at_10: number
  avg_task_match_rate?: number
  avg_modality_match_rate?: number
  avg_behavior_match_rate?: number
  summary_warnings?: string[]
  recommendations?: string[]
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

// Dataset Comparison Types
export interface DatasetComparisonItem {
  dataset_id: string
  title: string
  source: string
  source_id: string
  url?: string
  doi?: string
  license?: string

  task_labels: string[]
  modalities: string[]
  species: string[]
  brain_regions: string[]
  behavior_labels: string[]
  data_standards: string[]

  has_trials: boolean
  has_events: boolean
  has_behavior: boolean
  subject_count?: number
  session_count?: number

  linked_paper_count: number
  linked_papers: Array<{
    title: string
    doi?: string
    year?: number
  }>

  analysis_readiness_score: number
  strengths: string[]
  limitations: string[]
  missing_metadata: string[]

  available_notebook_templates: string[]
  suggested_analyses: string[]
  matched_recipes: Array<{
    id: string
    title: string
    match_score: number
  }>

  qa_status: string
}

export interface FieldComparison {
  field_name: string
  field_label: string
  values: Record<string, unknown>
  all_same: boolean
  union_values: unknown[]
  intersection_values: unknown[]
}

export interface ComparisonSummary {
  dataset_count: number
  common_fields: string[]
  different_fields: string[]
  readiness_ranking: Array<{
    dataset_id: string
    title: string
    score: number
  }>
  highest_readiness?: {
    dataset_id: string
    title: string
    score: number
  }
  most_notebook_templates?: {
    dataset_id: string
    title: string
    count: number
  }
  shared_tasks: string[]
  shared_modalities: string[]
  all_tasks: string[]
  all_modalities: string[]
}

export interface ComparisonResult {
  dataset_ids: string[]
  datasets: DatasetComparisonItem[]
  field_comparisons: FieldComparison[]
  summary: ComparisonSummary
  generated_at: string
}
