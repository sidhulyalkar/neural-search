import { useState } from 'react'
import { useSearchParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { SearchIcon, SpinnerIcon, ChevronRightIcon } from '../components/Icons'
import { getOntology, searchDatasets } from '../api/search'
import { DatasetCard } from '../components/DatasetCard'
import type { DatasetQAStatus, ExperimentQuery } from '../types'

type QAFilter = 'all' | 'reviewed' | 'trusted'

const emptyExperimentQuery: ExperimentQuery = {
  task: [],
  behavior: [],
  modality: [],
  species: [],
  brain_region: [],
  data_standard: [],
  source_archive: [],
  analysis_goal: [],
  reviewed_trusted_only: false,
}

const speciesOptions = ['mouse', 'rat', 'human', 'macaque']
const dataStandardOptions = ['NWB', 'BIDS']
const sourceArchiveOptions = ['demo', 'dandi', 'openneuro']

export function ResultsPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const query = searchParams.get('q') || ''
  const qaFilter = (searchParams.get('qa') as QAFilter | null) || 'all'
  const [inputValue, setInputValue] = useState(query)
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [experimentQuery, setExperimentQuery] = useState<ExperimentQuery>(emptyExperimentQuery)
  const [submittedExperimentQuery, setSubmittedExperimentQuery] = useState<ExperimentQuery>(emptyExperimentQuery)

  const { data: ontology } = useQuery({
    queryKey: ['ontology'],
    queryFn: getOntology,
  })

  const { data, isLoading, error } = useQuery({
    queryKey: ['search', query, qaFilter, submittedExperimentQuery],
    queryFn: () => searchDatasets(
      query,
      qaFilterToFilters(qaFilter),
      cleanStructuredQuery(submittedExperimentQuery),
    ),
    enabled: Boolean(query || hasStructuredQuery(submittedExperimentQuery)),
  })

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (inputValue.trim() || hasStructuredQuery(experimentQuery)) {
      setSubmittedExperimentQuery(cleanStructuredQuery(experimentQuery))
      setSearchParams(qaFilter === 'all' ? { q: inputValue.trim() } : { q: inputValue.trim(), qa: qaFilter })
    }
  }

  const handleQAFilter = (nextFilter: QAFilter) => {
    setSearchParams(nextFilter === 'all' ? { q: query } : { q: query, qa: nextFilter })
  }

  const setExperimentField = <K extends keyof ExperimentQuery>(
    field: K,
    value: ExperimentQuery[K],
  ) => {
    setExperimentQuery((current) => ({ ...current, [field]: value }))
  }

  const structuredPreview = cleanStructuredQuery(experimentQuery)
  const naturalPreview = buildNaturalPreview(inputValue, structuredPreview)

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Search bar */}
      <form onSubmit={handleSearch} className="mb-8 space-y-4">
        <div className="relative max-w-3xl">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Search for neural datasets..."
            className="input py-3 pl-12 pr-4"
          />
          <SearchIcon className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-neural-400" />
          <button type="submit" className="absolute right-2 top-1/2 -translate-y-1/2 btn-primary">
            Search
          </button>
        </div>

        <div className="border border-neural-800 rounded bg-neural-900/60">
          <button
            type="button"
            onClick={() => setAdvancedOpen((value) => !value)}
            className="w-full flex items-center justify-between px-4 py-3 text-left"
          >
            <span className="font-medium text-neural-200">Advanced Search</span>
            <ChevronRightIcon
              className={`w-5 h-5 text-neural-500 transition-transform ${advancedOpen ? 'rotate-90' : ''}`}
            />
          </button>
          {advancedOpen && (
            <div className="border-t border-neural-800 p-4 space-y-5">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <MultiSelect
                  label="Task"
                  values={experimentQuery.task}
                  options={(ontology?.tasks || []).map((task) => ({ value: task.id, label: task.label }))}
                  onChange={(values) => setExperimentField('task', values)}
                />
                <MultiSelect
                  label="Behavior"
                  values={experimentQuery.behavior}
                  options={(ontology?.behavior_labels || []).map((behavior) => ({ value: behavior.id, label: behavior.label }))}
                  onChange={(values) => setExperimentField('behavior', values)}
                />
                <MultiSelect
                  label="Modality"
                  values={experimentQuery.modality}
                  options={(ontology?.modalities || []).map((value) => ({ value, label: value.replace(/_/g, ' ') }))}
                  onChange={(values) => setExperimentField('modality', values)}
                />
                <MultiSelect
                  label="Species"
                  values={experimentQuery.species}
                  options={speciesOptions.map((value) => ({ value, label: value }))}
                  onChange={(values) => setExperimentField('species', values)}
                />
                <MultiSelect
                  label="Brain Region"
                  values={experimentQuery.brain_region}
                  options={(ontology?.brain_regions || []).map((value) => ({ value, label: value.replace(/_/g, ' ') }))}
                  onChange={(values) => setExperimentField('brain_region', values)}
                />
                <MultiSelect
                  label="Data Standard"
                  values={experimentQuery.data_standard}
                  options={dataStandardOptions.map((value) => ({ value, label: value }))}
                  onChange={(values) => setExperimentField('data_standard', values)}
                />
                <MultiSelect
                  label="Source Archive"
                  values={experimentQuery.source_archive}
                  options={sourceArchiveOptions.map((value) => ({ value, label: value.toUpperCase() }))}
                  onChange={(values) => setExperimentField('source_archive', values)}
                />
                <MultiSelect
                  label="Analysis Goal"
                  values={experimentQuery.analysis_goal}
                  options={(ontology?.analysis_goals || []).map((value) => ({ value, label: value.replace(/_/g, ' ') }))}
                  onChange={(values) => setExperimentField('analysis_goal', values)}
                />
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-sm text-neural-300">Minimum Readiness</label>
                    <span className="text-sm text-accent-cyan">
                      {experimentQuery.min_analysis_readiness_score ?? 0}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    step="5"
                    value={experimentQuery.min_analysis_readiness_score ?? 0}
                    onChange={(event) => setExperimentField(
                      'min_analysis_readiness_score',
                      Number(event.target.value),
                    )}
                    className="w-full"
                  />
                </div>
              </div>

              <label className="inline-flex items-center gap-2 text-sm text-neural-300">
                <input
                  type="checkbox"
                  checked={experimentQuery.reviewed_trusted_only}
                  onChange={(event) => setExperimentField('reviewed_trusted_only', event.target.checked)}
                  className="h-4 w-4 rounded border-neural-700 bg-neural-900 text-accent-cyan"
                />
                Reviewed/trusted only
              </label>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div>
                  <h3 className="text-sm font-medium text-neural-300 mb-2">Structured Query JSON</h3>
                  <pre className="text-xs text-neural-300 bg-neural-950 border border-neural-800 rounded p-3 overflow-x-auto">
                    {JSON.stringify(structuredPreview, null, 2)}
                  </pre>
                </div>
                <div>
                  <h3 className="text-sm font-medium text-neural-300 mb-2">Natural Language Preview</h3>
                  <div className="text-sm text-neural-300 bg-neural-950 border border-neural-800 rounded p-3 min-h-24">
                    {naturalPreview || 'No query terms selected yet.'}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </form>

      {/* Results */}
      {isLoading && (
        <div className="flex items-center justify-center py-20">
          <SpinnerIcon className="w-8 h-8 text-accent-cyan" />
          <span className="ml-3 text-neural-400">Searching...</span>
        </div>
      )}

      {error && (
        <div className="card border-red-500/50 text-center py-10">
          <p className="text-red-400">Error loading results. Please try again.</p>
        </div>
      )}

      {data && (
        <>
          {/* Results header */}
          <div className="flex items-center justify-between mb-6">
            <p className="text-neural-400">
              Found <span className="text-neural-100 font-medium">{data.total_count}</span> datasets
              {data.search_time_ms && (
                <span className="text-neural-500"> in {data.search_time_ms.toFixed(0)}ms</span>
              )}
            </p>

            <div className="inline-flex rounded border border-neural-700 overflow-hidden">
              {(['all', 'reviewed', 'trusted'] as QAFilter[]).map((filter) => (
                <button
                  key={filter}
                  type="button"
                  onClick={() => handleQAFilter(filter)}
                  className={`px-3 py-1.5 text-xs capitalize transition-colors ${
                    qaFilter === filter
                      ? 'bg-accent-cyan text-neural-950'
                      : 'bg-neural-900 text-neural-400 hover:text-neural-100'
                  }`}
                >
                  {filter}
                </button>
              ))}
            </div>
          </div>

          {/* Results list */}
          {data.results.length > 0 ? (
            <div className="space-y-4">
              {data.results.map((result) => (
                <DatasetCard key={result.dataset.id} result={result} />
              ))}
            </div>
          ) : (
            <div className="card text-center py-16">
              <p className="text-neural-400 mb-4">No datasets found for "{query}"</p>
              <p className="text-sm text-neural-500">
                Try using different keywords or browse the{' '}
                <Link to="/ontology" className="text-accent-cyan hover:underline">
                  ontology
                </Link>
              </p>
            </div>
          )}

          {/* Facets sidebar could go here */}
        </>
      )}
    </div>
  )
}

function qaFilterToFilters(filter: QAFilter): Record<string, DatasetQAStatus[]> {
  if (filter === 'trusted') {
    return { qa_status: ['trusted'] }
  }
  if (filter === 'reviewed') {
    return { qa_status: ['reviewed', 'trusted'] }
  }
  return {}
}

function MultiSelect({
  label,
  values,
  options,
  onChange,
}: {
  label: string
  values: string[]
  options: Array<{ value: string; label: string }>
  onChange: (values: string[]) => void
}) {
  return (
    <label className="block">
      <span className="block text-sm text-neural-300 mb-2">{label}</span>
      <select
        multiple
        value={values}
        onChange={(event) => {
          onChange(Array.from(event.target.selectedOptions).map((option) => option.value))
        }}
        className="input h-28 text-sm"
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  )
}

function hasStructuredQuery(query: ExperimentQuery): boolean {
  return Boolean(
    query.task.length ||
      query.behavior.length ||
      query.modality.length ||
      query.species.length ||
      query.brain_region.length ||
      query.data_standard.length ||
      query.source_archive.length ||
      query.analysis_goal.length ||
      (query.min_analysis_readiness_score ?? 0) > 0 ||
      query.reviewed_trusted_only,
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

function buildNaturalPreview(freeText: string, query: ExperimentQuery): string {
  const parts = [freeText.trim()].filter(Boolean)
  appendPreview(parts, 'task', query.task)
  appendPreview(parts, 'behavior', query.behavior)
  appendPreview(parts, 'modality', query.modality)
  appendPreview(parts, 'species', query.species)
  appendPreview(parts, 'brain region', query.brain_region)
  appendPreview(parts, 'data standard', query.data_standard)
  appendPreview(parts, 'source archive', query.source_archive)
  appendPreview(parts, 'analysis goal', query.analysis_goal)
  if (query.min_analysis_readiness_score !== undefined) {
    parts.push(`minimum analysis readiness ${query.min_analysis_readiness_score}`)
  }
  if (query.reviewed_trusted_only) {
    parts.push('reviewed or trusted dataset card')
  }
  return parts.join('; ')
}

function appendPreview(parts: string[], label: string, values: string[]) {
  if (values.length) {
    parts.push(`${label}: ${values.join(', ')}`)
  }
}
