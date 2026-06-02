import { useState, useCallback } from 'react'
import { useSearchParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getOntology, searchDatasets } from '../api/search'
import { DatasetCard } from '../components/DatasetCard'
import { ComparisonDrawer, ComparisonBar } from '../components/ComparisonDrawer'
import type { DatasetQAStatus, ExperimentQuery } from '../types'

const MAX_COMPARISON_DATASETS = 5
type QAFilter = 'all' | 'reviewed' | 'trusted'

const emptyExperimentQuery: ExperimentQuery = {
  task: [], behavior: [], modality: [], species: [],
  brain_region: [], data_standard: [], source_archive: [],
  analysis_goal: [], reviewed_trusted_only: false,
}

const speciesOptions = ['mouse', 'rat', 'human', 'macaque']
const dataStandardOptions = ['NWB', 'BIDS']
const sourceArchiveOptions = ['demo', 'dandi', 'openneuro']

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
              {data.results.map((result) => (
                <DatasetCard
                  key={result.dataset.id}
                  result={result}
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

function hasStructuredQuery(query: ExperimentQuery): boolean {
  return Boolean(
    query.task.length || query.behavior.length || query.modality.length ||
    query.species.length || query.brain_region.length || query.data_standard.length ||
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
