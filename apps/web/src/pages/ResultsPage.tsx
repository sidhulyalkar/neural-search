import { useState, useCallback, useEffect, useMemo } from 'react'
import { useSearchParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { createSearchSession, getOntology, searchDatasets } from '../api/search'
import { DatasetCard } from '../components/DatasetCard'
import { ComparisonDrawer, ComparisonBar } from '../components/ComparisonDrawer'
import type { DatasetQAStatus, ExperimentQuery, SearchResultItem } from '../types'

const MAX_COMPARISON_DATASETS = 5
type QAFilter = 'all' | 'reviewed' | 'trusted'
type SortKey =
  | 'retrieval_score'
  | 'neuro_label'
  | 'confidence'
  | 'evidence_completeness'
  | 'source'
  | 'modality'
  | 'recording_scale'
  | 'species'
  | 'abstain'
  | 'feedback'

interface ResultFilters {
  source: string
  species: string
  modality: string
  recordingScale: string
  brainRegion: string
  neuroLabel: string
  minEvidenceCompleteness: number
  hideAbstain: boolean
  rawDataSuitable: boolean
  needsAudit: boolean
}

const emptyExperimentQuery: ExperimentQuery = {
  task: [], behavior: [], modality: [], recording_scale: [], species: [],
  brain_region: [], data_standard: [], source_archive: [],
  analysis_goal: [], reviewed_trusted_only: false,
}

const speciesOptions = ['mouse', 'rat', 'human', 'macaque']
const dataStandardOptions = ['NWB', 'BIDS']
const sourceArchiveOptions = ['demo', 'dandi', 'openneuro']
const defaultResultFilters: ResultFilters = {
  source: '',
  species: '',
  modality: '',
  recordingScale: '',
  brainRegion: '',
  neuroLabel: '',
  minEvidenceCompleteness: 0,
  hideAbstain: false,
  rawDataSuitable: false,
  needsAudit: false,
}

const recoveryQueries = [
  'Find reversal learning datasets with reward omission and trial outcomes',
  'Go/NoGo task with calcium imaging in mPFC and lick events',
  'Visual decision-making with Neuropixels recordings',
]

export function ResultsPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const query = searchParams.get('q') || ''
  const qaFilter = (searchParams.get('qa') as QAFilter | null) || 'all'
  const [inputValue, setInputValue] = useState(query)
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [experimentQuery, setExperimentQuery] = useState<ExperimentQuery>(emptyExperimentQuery)
  const [submittedExperimentQuery, setSubmittedExperimentQuery] = useState<ExperimentQuery>(emptyExperimentQuery)
  const [selectedForComparison, setSelectedForComparison] = useState<string[]>([])
  const [selectedTitles, setSelectedTitles] = useState<Record<string, string>>({})
  const [comparisonDrawerOpen, setComparisonDrawerOpen] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [sortBy, setSortBy] = useState<SortKey>('retrieval_score')
  const [resultFilters, setResultFilters] = useState<ResultFilters>(defaultResultFilters)

  const toggleDatasetSelection = useCallback((datasetId: string, title?: string) => {
    setSelectedForComparison((prev) => {
      if (prev.includes(datasetId)) return prev.filter((id) => id !== datasetId)
      if (prev.length >= MAX_COMPARISON_DATASETS) return prev
      return [...prev, datasetId]
    })
    if (title) setSelectedTitles((prev) => ({ ...prev, [datasetId]: title }))
  }, [])

  const removeFromComparison = useCallback((datasetId: string) => {
    setSelectedForComparison((prev) => prev.filter((id) => id !== datasetId))
  }, [])

  const clearComparison = useCallback(() => {
    setSelectedForComparison([])
    setSelectedTitles({})
  }, [])

  const { data: ontology } = useQuery({ queryKey: ['ontology'], queryFn: getOntology })

  const { data, isLoading, error } = useQuery({
    queryKey: ['search', query, qaFilter, submittedExperimentQuery],
    queryFn: () => searchDatasets(query, qaFilterToFilters(qaFilter), cleanStructuredQuery(submittedExperimentQuery)),
    enabled: Boolean(query || hasStructuredQuery(submittedExperimentQuery)),
  })

  useEffect(() => {
    let cancelled = false
    if (!query && !hasStructuredQuery(submittedExperimentQuery)) return undefined

    createSearchSession({
      query_text: query,
      retrieval_method: 'hybrid_search',
      filters: qaFilterToFilters(qaFilter),
      structured_query: cleanStructuredQuery(submittedExperimentQuery),
    })
      .then((session) => {
        if (!cancelled) setSessionId(session.session_id)
      })
      .catch(() => {
        if (!cancelled) setSessionId(null)
      })

    return () => { cancelled = true }
  }, [query, qaFilter, submittedExperimentQuery])

  const filteredResults = useMemo(
    () => sortResults(filterResults(data?.results || [], resultFilters), sortBy),
    [data?.results, resultFilters, sortBy],
  )

  const filterOptions = useMemo(
    () => buildResultFilterOptions(data?.results || []),
    [data?.results],
  )

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (inputValue.trim() || hasStructuredQuery(experimentQuery)) {
      setSubmittedExperimentQuery(cleanStructuredQuery(experimentQuery))
      setSearchParams(qaFilter === 'all' ? { q: inputValue.trim() } : { q: inputValue.trim(), qa: qaFilter })
    }
  }

  const setExperimentField = <K extends keyof ExperimentQuery>(field: K, value: ExperimentQuery[K]) => {
    setExperimentQuery((current) => ({ ...current, [field]: value }))
  }

  return (
    <div className="max-w-5xl mx-auto px-6 lg:px-8 py-10">
      {/* Search bar */}
      <form onSubmit={handleSearch} className="mb-8">
        <div className="flex gap-3 mb-3">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Search for neural datasets…"
            className="flex-1 bg-neural-900 border border-neural-700 rounded-lg px-5 py-3 text-neural-100 placeholder-neural-600 focus:outline-none focus:border-neural-500 focus:ring-1 focus:ring-neural-500 transition-colors"
          />
          <button type="submit" className="bg-accent-cyan text-neural-950 font-medium px-5 py-3 rounded-lg hover:bg-accent-cyan/90 transition-colors text-sm whitespace-nowrap">
            Search
          </button>
        </div>

        {/* Advanced toggle */}
        <div className="border border-neural-800/50 rounded-lg overflow-hidden">
          <button
            type="button"
            onClick={() => setAdvancedOpen((v) => !v)}
            className="w-full flex items-center justify-between px-4 py-2.5 text-left hover:bg-neural-900/30 transition-colors"
          >
            <span className="text-xs text-neural-500 uppercase tracking-widest">Filters</span>
            <span className="text-neural-600 text-xs">{advancedOpen ? '−' : '+'}</span>
          </button>

          {advancedOpen && (
            <div className="border-t border-neural-800/50 p-4">
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                <FilterSelect label="Task" values={experimentQuery.task}
                  options={(ontology?.tasks || []).map((t) => ({ value: t.id, label: t.label }))}
                  onChange={(v) => setExperimentField('task', v)} />
                <FilterSelect label="Modality" values={experimentQuery.modality}
                  options={(ontology?.modalities || []).map((v) => ({ value: v, label: v.replace(/_/g, ' ') }))}
                  onChange={(v) => setExperimentField('modality', v)} />
                <FilterSelect label="Recording Scale" values={experimentQuery.recording_scale}
                  options={(ontology?.recording_scales || []).map((v) => ({ value: v.id, label: v.label }))}
                  onChange={(v) => setExperimentField('recording_scale', v)} />
                <FilterSelect label="Species" values={experimentQuery.species}
                  options={speciesOptions.map((v) => ({ value: v, label: v }))}
                  onChange={(v) => setExperimentField('species', v)} />
                <FilterSelect label="Brain Region" values={experimentQuery.brain_region}
                  options={(ontology?.brain_regions || []).map((v) => ({ value: v, label: v.replace(/_/g, ' ') }))}
                  onChange={(v) => setExperimentField('brain_region', v)} />
                <FilterSelect label="Data Standard" values={experimentQuery.data_standard}
                  options={dataStandardOptions.map((v) => ({ value: v, label: v }))}
                  onChange={(v) => setExperimentField('data_standard', v)} />
                <FilterSelect label="Archive" values={experimentQuery.source_archive}
                  options={sourceArchiveOptions.map((v) => ({ value: v, label: v.toUpperCase() }))}
                  onChange={(v) => setExperimentField('source_archive', v)} />
                <FilterSelect label="Behavior" values={experimentQuery.behavior}
                  options={(ontology?.behavior_labels || []).map((b) => ({ value: b.id, label: b.label }))}
                  onChange={(v) => setExperimentField('behavior', v)} />
                <FilterSelect label="Analysis Goal" values={experimentQuery.analysis_goal}
                  options={(ontology?.analysis_goals || []).map((v) => ({ value: v, label: v.replace(/_/g, ' ') }))}
                  onChange={(v) => setExperimentField('analysis_goal', v)} />
              </div>

              <div className="mt-3 flex items-center gap-4">
                <label className="flex items-center gap-2 text-xs text-neural-500 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={experimentQuery.reviewed_trusted_only}
                    onChange={(e) => setExperimentField('reviewed_trusted_only', e.target.checked)}
                    className="rounded border-neural-700 bg-neural-900 text-accent-cyan focus:ring-accent-cyan/30"
                  />
                  Reviewed/trusted only
                </label>

                <div className="flex items-center gap-2">
                  <span className="text-xs text-neural-500">Min readiness</span>
                  <input
                    type="range" min="0" max="100" step="5"
                    value={experimentQuery.min_analysis_readiness_score ?? 0}
                    onChange={(e) => setExperimentField('min_analysis_readiness_score', Number(e.target.value))}
                    className="w-24"
                  />
                  <span className="text-xs text-neural-400 tabular-nums w-6">
                    {experimentQuery.min_analysis_readiness_score ?? 0}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </form>

      {/* Loading */}
      {isLoading && (
        <div className="py-12 text-center">
          <div className="inline-flex items-center gap-3 text-neural-500 text-sm">
            <span className="w-4 h-4 border-2 border-neural-700 border-t-accent-cyan rounded-full animate-spin" />
            Searching ontology, metadata, and embeddings…
          </div>
          <div className="mt-8 space-y-px">
            {[0, 1, 2].map((i) => (
              <div key={i} className="border-b border-neural-800/40 py-6 animate-pulse">
                <div className="flex gap-6">
                  <div className="flex-1 space-y-3">
                    <div className="h-4 w-24 bg-neural-800 rounded" />
                    <div className="h-5 w-2/3 bg-neural-800 rounded" />
                    <div className="h-4 w-full bg-neural-800 rounded" />
                    <div className="flex gap-2">
                      <div className="h-5 w-16 bg-neural-800 rounded" />
                      <div className="h-5 w-20 bg-neural-800 rounded" />
                      <div className="h-5 w-14 bg-neural-800 rounded" />
                    </div>
                  </div>
                  <div className="flex flex-col gap-3">
                    <div className="h-8 w-12 bg-neural-800 rounded" />
                    <div className="h-8 w-12 bg-neural-800 rounded" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="py-10 border border-red-500/30 rounded-lg px-6">
          <p className="text-red-300 text-sm font-medium mb-1">Search could not complete</p>
          <p className="text-neural-500 text-sm">
            {error instanceof Error ? error.message : 'Unexpected error from API.'}
          </p>
          <p className="text-neural-600 text-xs mt-3">
            Confirm API is running: <code className="font-mono text-neural-400">make api</code>
          </p>
        </div>
      )}

      {/* Results */}
      {data && !isLoading && !error && (
        <>
          <div className="flex items-center justify-between mb-6">
            <p className="text-sm text-neural-500">
              <span className="text-neural-200">{data.total_count}</span> datasets
              {data.search_time_ms && (
                <span className="text-neural-700"> · {data.search_time_ms.toFixed(0)}ms</span>
              )}
            </p>

            <div className="flex gap-1">
              {(['all', 'reviewed', 'trusted'] as QAFilter[]).map((f) => (
                <button
                  key={f}
                  type="button"
                  onClick={() => setSearchParams(f === 'all' ? { q: query } : { q: query, qa: f })}
                  className={`px-3 py-1 text-xs rounded transition-colors capitalize ${
                    qaFilter === f
                      ? 'bg-neural-800 text-neural-200'
                      : 'text-neural-600 hover:text-neural-300'
                  }`}
                >
                  {f}
                </button>
              ))}
            </div>
          </div>

          {data.results.length > 0 ? (
            <div>
              <ResultControls
                sortBy={sortBy}
                onSortChange={setSortBy}
                filters={resultFilters}
                onFiltersChange={setResultFilters}
                options={filterOptions}
                resultCount={filteredResults.length}
                totalCount={data.results.length}
              />

              {filteredResults.map((result) => (
                <DatasetCard
                  key={result.dataset.id}
                  result={result}
                  queryText={query}
                  sessionId={sessionId}
                  isSelected={selectedForComparison.includes(result.dataset.id)}
                  onToggleSelect={(id) => toggleDatasetSelection(id, result.dataset.title)}
                  selectionDisabled={
                    selectedForComparison.length >= MAX_COMPARISON_DATASETS &&
                    !selectedForComparison.includes(result.dataset.id)
                  }
                />
              ))}
            </div>
          ) : (
            <div className="py-16 text-center">
              <p className="text-neural-300 mb-2">No datasets matched</p>
              <p className="text-sm text-neural-600 mb-8">
                Try loosening filters or using a query from the demo corpus.
              </p>
              <div className="space-y-1 max-w-lg mx-auto">
                {recoveryQueries.map((q) => (
                  <button
                    key={q}
                    type="button"
                    onClick={() => { setInputValue(q); setSubmittedExperimentQuery(emptyExperimentQuery); setSearchParams({ q }) }}
                    className="block w-full text-left text-sm text-neural-500 hover:text-neural-200 py-2 border-b border-neural-800/40 last:border-0 transition-colors"
                  >
                    → {q}
                  </button>
                ))}
              </div>
              <p className="mt-6 text-xs text-neural-600">
                Browse the{' '}
                <Link to="/ontology" className="text-accent-cyan hover:underline">ontology</Link>
                {' '}for available task labels.
              </p>
            </div>
          )}
        </>
      )}

      <ComparisonBar
        selectedIds={selectedForComparison}
        selectedTitles={selectedTitles}
        onRemove={removeFromComparison}
        onClear={clearComparison}
        onCompare={() => setComparisonDrawerOpen(true)}
      />

      <ComparisonDrawer
        selectedIds={selectedForComparison}
        onRemove={removeFromComparison}
        onClear={clearComparison}
        isOpen={comparisonDrawerOpen}
        onClose={() => setComparisonDrawerOpen(false)}
      />
    </div>
  )
}

function qaFilterToFilters(filter: QAFilter): Record<string, DatasetQAStatus[]> {
  if (filter === 'trusted') return { qa_status: ['trusted'] }
  if (filter === 'reviewed') return { qa_status: ['reviewed', 'trusted'] }
  return {}
}

function FilterSelect({
  label, values, options, onChange,
}: {
  label: string
  values: string[]
  options: Array<{ value: string; label: string }>
  onChange: (values: string[]) => void
}) {
  return (
    <label className="block">
      <span className="block text-xs text-neural-500 mb-1.5 uppercase tracking-wide">{label}</span>
      <select
        multiple
        value={values}
        onChange={(e) => onChange(Array.from(e.target.selectedOptions).map((o) => o.value))}
        className="w-full bg-neural-900 border border-neural-700 rounded px-3 py-1.5 text-sm text-neural-200 focus:outline-none focus:border-neural-500 h-24"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </label>
  )
}

function ResultControls({
  sortBy,
  onSortChange,
  filters,
  onFiltersChange,
  options,
  resultCount,
  totalCount,
}: {
  sortBy: SortKey
  onSortChange: (value: SortKey) => void
  filters: ResultFilters
  onFiltersChange: (filters: ResultFilters) => void
  options: ReturnType<typeof buildResultFilterOptions>
  resultCount: number
  totalCount: number
}) {
  const setFilter = <K extends keyof ResultFilters>(key: K, value: ResultFilters[K]) => {
    onFiltersChange({ ...filters, [key]: value })
  }

  return (
    <div className="border border-neural-800/50 rounded-lg p-4 mb-3">
      <div className="flex flex-wrap items-end gap-3">
        <label className="block">
          <span className="block text-xs text-neural-500 mb-1 uppercase tracking-wide">Sort</span>
          <select
            value={sortBy}
            onChange={(e) => onSortChange(e.target.value as SortKey)}
            className="bg-neural-900 border border-neural-700 rounded px-3 py-1.5 text-sm text-neural-200"
          >
            <option value="retrieval_score">Retrieval score</option>
            <option value="neuro_label">Neuro-judge label</option>
            <option value="confidence">Judge confidence</option>
            <option value="evidence_completeness">Evidence completeness</option>
            <option value="source">Source/archive</option>
            <option value="modality">Modality</option>
            <option value="recording_scale">Recording scale</option>
            <option value="species">Species</option>
            <option value="abstain">Abstain recommended</option>
            <option value="feedback">Feedback score</option>
          </select>
        </label>

        <CompactSelect label="Archive" value={filters.source} options={options.sources}
          onChange={(value) => setFilter('source', value)} />
        <CompactSelect label="Species" value={filters.species} options={options.species}
          onChange={(value) => setFilter('species', value)} />
        <CompactSelect label="Modality" value={filters.modality} options={options.modalities}
          onChange={(value) => setFilter('modality', value)} />
        <CompactSelect label="Scale" value={filters.recordingScale} options={options.recordingScales}
          onChange={(value) => setFilter('recordingScale', value)} />
        <CompactSelect label="Region" value={filters.brainRegion} options={options.brainRegions}
          onChange={(value) => setFilter('brainRegion', value)} />
        <CompactSelect label="Judge" value={filters.neuroLabel} options={['0', '1', '2', '3']}
          onChange={(value) => setFilter('neuroLabel', value)} />

        <label className="block min-w-40">
          <span className="block text-xs text-neural-500 mb-1 uppercase tracking-wide">
            Min evidence {filters.minEvidenceCompleteness.toFixed(1)}
          </span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={filters.minEvidenceCompleteness}
            onChange={(e) => setFilter('minEvidenceCompleteness', Number(e.target.value))}
            className="w-full"
          />
        </label>

        <label className="flex items-center gap-2 text-xs text-neural-500">
          <input type="checkbox" checked={filters.hideAbstain}
            onChange={(e) => setFilter('hideAbstain', e.target.checked)}
            className="rounded border-neural-700 bg-neural-900 text-accent-cyan" />
          Hide abstain
        </label>
        <label className="flex items-center gap-2 text-xs text-neural-500">
          <input type="checkbox" checked={filters.rawDataSuitable}
            onChange={(e) => setFilter('rawDataSuitable', e.target.checked)}
            className="rounded border-neural-700 bg-neural-900 text-accent-cyan" />
          Raw data suitable
        </label>
        <label className="flex items-center gap-2 text-xs text-neural-500">
          <input type="checkbox" checked={filters.needsAudit}
            onChange={(e) => setFilter('needsAudit', e.target.checked)}
            className="rounded border-neural-700 bg-neural-900 text-accent-cyan" />
          Needs audit
        </label>

        <button
          type="button"
          onClick={() => onFiltersChange(defaultResultFilters)}
          className="text-xs text-neural-500 hover:text-neural-200"
        >
          Clear
        </button>
      </div>
      <p className="mt-3 text-xs text-neural-600">
        Showing {resultCount} of {totalCount}. Neuro-judge labels are preliminary downstream inspection signals, not human gold.
      </p>
    </div>
  )
}

function CompactSelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: string
  options: string[]
  onChange: (value: string) => void
}) {
  return (
    <label className="block">
      <span className="block text-xs text-neural-500 mb-1 uppercase tracking-wide">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="bg-neural-900 border border-neural-700 rounded px-3 py-1.5 text-sm text-neural-200 max-w-36"
      >
        <option value="">Any</option>
        {options.map((option) => (
          <option key={option} value={option}>{option}</option>
        ))}
      </select>
    </label>
  )
}

function buildResultFilterOptions(results: SearchResultItem[]) {
  const unique = (values: string[]) => Array.from(new Set(values.filter(Boolean))).sort()
  return {
    sources: unique(results.map((r) => r.dataset.source)),
    species: unique(results.flatMap((r) => r.dataset.species || [])),
    modalities: unique(results.flatMap((r) => r.dataset.modalities || [])),
    recordingScales: unique(results.flatMap((r) => r.dataset.recording_scales || [])),
    brainRegions: unique(results.flatMap((r) => r.dataset.brain_regions || [])),
  }
}

function filterResults(results: SearchResultItem[], filters: ResultFilters): SearchResultItem[] {
  return results.filter((result) => {
    const dataset = result.dataset
    const judge = result.neuro_judge
    const packet = result.evidence_packet
    if (filters.source && dataset.source !== filters.source) return false
    if (filters.species && !(dataset.species || []).includes(filters.species)) return false
    if (filters.modality && !(dataset.modalities || []).includes(filters.modality)) return false
    if (filters.recordingScale && !(dataset.recording_scales || []).includes(filters.recordingScale)) return false
    if (filters.brainRegion && !(dataset.brain_regions || []).includes(filters.brainRegion)) return false
    if (filters.neuroLabel && String(judge?.label ?? '') !== filters.neuroLabel) return false
    if ((judge?.evidence_completeness ?? 0) < filters.minEvidenceCompleteness) return false
    if (filters.hideAbstain && judge?.abstain_recommended) return false
    if (filters.rawDataSuitable && packet?.has_raw_data !== true) return false
    if (filters.needsAudit && !needsAudit(result)) return false
    return true
  })
}

function sortResults(results: SearchResultItem[], sortBy: SortKey): SearchResultItem[] {
  const sorted = [...results]
  const first = (values?: string[]) => (values && values[0] ? values[0] : '')
  sorted.sort((a, b) => {
    if (sortBy === 'source') return first([a.dataset.source]).localeCompare(first([b.dataset.source]))
    if (sortBy === 'modality') return first(a.dataset.modalities).localeCompare(first(b.dataset.modalities))
    if (sortBy === 'recording_scale') return first(a.dataset.recording_scales).localeCompare(first(b.dataset.recording_scales))
    if (sortBy === 'species') return first(a.dataset.species).localeCompare(first(b.dataset.species))
    if (sortBy === 'abstain') return Number(b.neuro_judge?.abstain_recommended ?? false) - Number(a.neuro_judge?.abstain_recommended ?? false)
    if (sortBy === 'feedback') return feedbackScore(b) - feedbackScore(a)
    if (sortBy === 'neuro_label') return (b.neuro_judge?.label ?? -1) - (a.neuro_judge?.label ?? -1)
    if (sortBy === 'confidence') return (b.neuro_judge?.confidence ?? -1) - (a.neuro_judge?.confidence ?? -1)
    if (sortBy === 'evidence_completeness') {
      return (b.neuro_judge?.evidence_completeness ?? -1) - (a.neuro_judge?.evidence_completeness ?? -1)
    }
    return b.score - a.score
  })
  return sorted
}

function feedbackScore(result: SearchResultItem): number {
  const weights: Record<string, number> = { useful: 3, partially_useful: 2, unsure: 1, not_useful: 0 }
  const feedback = result.prior_feedback || []
  if (!feedback.length) return -1
  return feedback.reduce((sum, item) => sum + (weights[item.usefulness] ?? 0), 0) / feedback.length
}

function needsAudit(result: SearchResultItem): boolean {
  const judge = result.neuro_judge
  return Boolean(
    judge?.abstain_recommended ||
    judge?.hard_negative_detected ||
    (judge?.required_dimensions_missing && judge.required_dimensions_missing.length > 0) ||
    (judge?.label ?? 0) >= 2 && (judge?.evidence_completeness ?? 1) < 0.75,
  )
}

function hasStructuredQuery(query: ExperimentQuery): boolean {
  return Boolean(
    query.task.length || query.behavior.length || query.modality.length ||
    query.recording_scale.length || query.species.length || query.brain_region.length || query.data_standard.length ||
    query.source_archive.length || query.analysis_goal.length ||
    (query.min_analysis_readiness_score ?? 0) > 0 || query.reviewed_trusted_only,
  )
}

function cleanStructuredQuery(query: ExperimentQuery): ExperimentQuery {
  return {
    ...query,
    min_analysis_readiness_score:
      (query.min_analysis_readiness_score ?? 0) > 0
        ? query.min_analysis_readiness_score
        : undefined,
  }
}
