import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  getDataset,
  generateNotebook,
  exportDatasetCard,
  updateDatasetQA,
  getDatasetAffordances,
  getSimilarDatasets,
  type AffordanceResult,
  type AffordanceSupportLevel,
  type SimilarDataset,
} from '../api/search'
import {
  SpinnerIcon,
  ExternalLinkIcon,
  CodeIcon,
  DocumentIcon,
  CheckCircleIcon,
  WarningIcon,
  FolderIcon,
  BookOpenIcon,
  InformationCircleIcon,
  DownloadIcon,
} from '../components/Icons'
import type { DatasetQAStatus, DatasetQAUpdate } from '../types'

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

function formatValue(value: unknown): string {
  if (value === undefined || value === null || value === '') return 'Not available'
  if (typeof value === 'boolean') return value ? 'yes' : 'no'
  if (Array.isArray(value)) {
    return value.length > 0 ? value.map((item) => String(item).replace(/_/g, ' ')).join(', ') : 'Not available'
  }
  return String(value).replace(/_/g, ' ')
}

function asList(value: unknown): string[] {
  if (!value) return []
  if (Array.isArray(value)) return value.map((item) => String(item))
  return [String(value)]
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

const qaStatusStyles: Record<DatasetQAStatus, string> = {
  unreviewed: 'bg-neural-700 text-neural-300',
  auto_generated: 'bg-amber-500/10 text-amber-300 border border-amber-500/30',
  needs_review: 'bg-red-500/10 text-red-300 border border-red-500/30',
  reviewed: 'bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/30',
  trusted: 'bg-accent-emerald/10 text-accent-emerald border border-accent-emerald/30',
  rejected: 'bg-red-500/10 text-red-300 border border-red-500/30',
}

function formatQAStatus(status: DatasetQAStatus) {
  return status.replace(/_/g, ' ')
}

const RELATION_COLORS: Record<string, string> = {
  same_region_same_task: 'text-accent-cyan',
  same_region_cross_modality: 'text-yellow-400',
  same_task_cross_species: 'text-emerald-400',
}

function SimilarDatasetsPanel({ similar }: { similar: SimilarDataset[] }) {
  return (
    <section className="card">
      <h2 className="text-lg font-semibold mb-3">Similar Datasets</h2>
      <p className="text-xs text-neural-500 mb-3">Related via the knowledge graph.</p>
      <div className="space-y-2">
        {similar.map((ds) => (
          <Link
            key={ds.dataset_id}
            to={`/datasets/${encodeURIComponent(ds.dataset_id)}`}
            className="block py-2 px-2 rounded hover:bg-neural-800/60 transition-colors"
          >
            <div className="text-sm text-neural-200 truncate">{ds.title || ds.dataset_id}</div>
            <div className={`text-xs mt-0.5 ${RELATION_COLORS[ds.relation] ?? 'text-neural-500'}`}>
              {ds.relation_label}
            </div>
          </Link>
        ))}
      </div>
    </section>
  )
}

const SUPPORT_STYLES: Record<AffordanceSupportLevel, string> = {
  high: 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/30',
  medium: 'bg-yellow-500/10 text-yellow-300 border border-yellow-500/30',
  low: 'bg-orange-500/10 text-orange-300 border border-orange-500/30',
  unsupported: 'bg-neural-800 text-neural-500',
  unknown: 'bg-neural-800 text-neural-600',
}

const SUPPORT_LABEL: Record<AffordanceSupportLevel, string> = {
  high: 'HIGH',
  medium: 'MED',
  low: 'LOW',
  unsupported: '—',
  unknown: '?',
}

function AffordancePanel({ affordances }: { affordances: AffordanceResult[] }) {
  const [expanded, setExpanded] = useState<string | null>(null)

  const sorted = [...affordances].sort((a, b) => {
    const order: AffordanceSupportLevel[] = ['high', 'medium', 'low', 'unsupported', 'unknown']
    return order.indexOf(a.support_level) - order.indexOf(b.support_level)
  })

  const supported = sorted.filter((a) => a.support_level === 'high' || a.support_level === 'medium')
  const missing = sorted.filter((a) => a.support_level === 'low' || a.support_level === 'unsupported')

  return (
    <section className="card">
      <h2 className="text-lg font-semibold mb-1">What can I test with this?</h2>
      <p className="text-xs text-neural-500 mb-1">
        Analyses this dataset structurally supports, based on metadata signals.
        <span className="ml-2 text-neural-600">(audit pending · metadata-only)</span>
      </p>
      {supported.length > 0 && (
        <p className="text-xs text-emerald-400 mb-3">
          {supported.length} supported · {missing.length} not supported
        </p>
      )}

      <div className="space-y-1.5">
        {sorted.map((a) => (
          <div key={a.affordance_id}>
            <button
              type="button"
              onClick={() => setExpanded((prev) => (prev === a.affordance_id ? null : a.affordance_id))}
              className="w-full flex items-center justify-between gap-2 py-1.5 px-2 rounded hover:bg-neural-800/60 transition-colors text-left"
            >
              <span className="text-sm text-neural-300 truncate">{a.label}</span>
              <span className={`text-xs font-mono px-1.5 py-0.5 rounded flex-shrink-0 ${SUPPORT_STYLES[a.support_level]}`}>
                {SUPPORT_LABEL[a.support_level]}
              </span>
            </button>
            {expanded === a.affordance_id && (
              <div className="ml-2 mb-1 px-2 py-2 bg-neural-900 rounded text-xs space-y-1.5 border border-neural-800">
                {a.found_required_features.length > 0 && (
                  <div>
                    <span className="text-emerald-400">✓ </span>
                    <span className="text-neural-400">{a.found_required_features.join(', ')}</span>
                  </div>
                )}
                {a.missing_required_features.length > 0 && (
                  <div>
                    <span className="text-red-400">✗ missing: </span>
                    <span className="text-neural-500">{a.missing_required_features.join(', ')}</span>
                  </div>
                )}
                <div className="text-neural-600">
                  conf: {(a.confidence * 100).toFixed(0)}%
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  )
}

export function DatasetPage() {
  const { id } = useParams<{ id: string }>()
  const queryClient = useQueryClient()
  const [qaForm, setQaForm] = useState<DatasetQAUpdate>({})
  const [selectedTemplateId, setSelectedTemplateId] = useState('generic_nwb_inspection')

  const { data: card, isLoading, error } = useQuery({
    queryKey: ['dataset', id],
    queryFn: () => getDataset(id!),
    enabled: !!id,
  })

  const { data: affordancesData } = useQuery({
    queryKey: ['dataset-affordances', id],
    queryFn: () => getDatasetAffordances(id!),
    enabled: !!id,
  })

  const { data: similarData } = useQuery({
    queryKey: ['dataset-similar', id],
    queryFn: () => getSimilarDatasets(id!),
    enabled: !!id,
  })

  const notebookMutation = useMutation({
    mutationFn: () => generateNotebook(id!, selectedTemplateId),
    onSuccess: (blob) => {
      downloadBlob(blob, `${id}_${selectedTemplateId}.ipynb`)
    },
  })

  const markdownExportMutation = useMutation({
    mutationFn: () => exportDatasetCard(id!, 'markdown'),
    onSuccess: (blob) => downloadBlob(blob, `${id}_reuse_card.md`),
  })

  const jsonExportMutation = useMutation({
    mutationFn: () => exportDatasetCard(id!, 'json'),
    onSuccess: (blob) => downloadBlob(blob, `${id}_reuse_card.json`),
  })

  useEffect(() => {
    if (card) {
      setQaForm({
        qa_status: card.qa_status,
        task_labels_verified: card.task_labels_verified,
        modality_labels_verified: card.modality_labels_verified,
        behavior_labels_verified: card.behavior_labels_verified,
        brain_regions_verified: card.brain_regions_verified,
        linked_papers_verified: card.linked_papers_verified,
        notebook_tested: card.notebook_tested,
        reviewer_notes: card.reviewer_notes || '',
      })
    }
  }, [card])

  const qaMutation = useMutation({
    mutationFn: (updates: DatasetQAUpdate) => updateDatasetQA(id!, updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['dataset', id] })
    },
  })

  const updateQAField = (
    key: keyof DatasetQAUpdate,
    value: DatasetQAUpdate[keyof DatasetQAUpdate],
  ) => {
    setQaForm((current) => ({ ...current, [key]: value }))
  }

  const saveQA = (updates: DatasetQAUpdate = qaForm) => {
    qaMutation.mutate(updates)
  }

  if (isLoading) {
    return (
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8" aria-live="polite" aria-busy="true">
        <div className="flex items-center gap-3 text-neural-400 mb-6">
          <SpinnerIcon className="w-5 h-5 text-accent-cyan" />
          <span>Loading dataset card, provenance, and reuse instructions...</span>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 animate-pulse">
          <div className="lg:col-span-2 space-y-4">
            <div className="h-10 w-3/4 rounded bg-neural-800" />
            <div className="card h-44" />
            <div className="card h-56" />
          </div>
          <div className="space-y-4">
            <div className="card h-40" />
            <div className="card h-56" />
          </div>
        </div>
      </div>
    )
  }

  if (error || !card) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-16 text-center">
        <p className="text-red-300 font-medium mb-2">Failed to load dataset card</p>
        <p className="text-sm text-neural-400 mb-4">
          {error instanceof Error
            ? error.message
            : 'The API could not return this generated dataset card.'}
        </p>
        <Link to="/" className="text-accent-cyan hover:underline">
          Return to search
        </Link>
      </div>
    )
  }

  const sourceLabel = card.source || 'other'
  const linkedPapers = card.linked_literature?.candidate_papers?.length
    ? card.linked_literature.candidate_papers
    : card.related_papers
  const notebookTemplates = card.reuse_instructions?.notebook_templates || []
  const selectedTemplate = notebookTemplates.find((template) => template.id === selectedTemplateId)
  const selectedTemplateUnavailable = selectedTemplate ? !selectedTemplate.available : false

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Breadcrumb */}
      <nav className="text-sm text-neural-500 mb-6">
        <Link to="/" className="hover:text-neural-300">Search</Link>
        <span className="mx-2">/</span>
        <span className="text-neural-300">{card.title}</span>
      </nav>

      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-3">
          <span className={`badge ${sourceLabel === 'dandi' ? 'badge-cyan' : 'badge-violet'}`}>
            {sourceLabel.toUpperCase()}
          </span>
          {card.data_standard && (
            <span className="badge badge-emerald">{card.data_standard.toUpperCase()}</span>
          )}
          <span className={`badge ${qaStatusStyles[card.qa_status]}`}>
            {formatQAStatus(card.qa_status)}
          </span>
        </div>
        <h1 className="text-3xl font-bold text-neural-100 mb-4">{card.title}</h1>
        <p className="text-neural-400 text-lg">{card.summary}</p>
      </div>

      {/* Actions */}
      <div className="flex flex-wrap gap-4 mb-8">
        {notebookTemplates.length > 0 && (
          <select
            value={selectedTemplateId}
            onChange={(event) => setSelectedTemplateId(event.target.value)}
            className="bg-neural-900 border border-neural-700 rounded px-3 py-2 text-sm text-neural-200"
          >
            {notebookTemplates.map((template) => (
              <option key={template.id} value={template.id}>
                {template.available ? '' : '[Unavailable] '}{template.title}
              </option>
            ))}
          </select>
        )}
        <button
          onClick={() => notebookMutation.mutate()}
          disabled={notebookMutation.isPending || selectedTemplateUnavailable}
          className="btn-primary flex items-center gap-2"
        >
          <CodeIcon className="w-4 h-4" />
          {notebookMutation.isPending ? 'Generating...' : 'Generate Notebook'}
        </button>
        {card.url && (
          <a
            href={card.url}
            target="_blank"
            rel="noopener noreferrer"
            className="btn-secondary flex items-center gap-2"
          >
            <ExternalLinkIcon className="w-4 h-4" />
            View Source
          </a>
        )}
        {card.doi && (
          <a
            href={`https://doi.org/${card.doi}`}
            target="_blank"
            rel="noopener noreferrer"
            className="btn-secondary flex items-center gap-2"
          >
            <BookOpenIcon className="w-4 h-4" />
            DOI
          </a>
        )}
        <button
          onClick={() => markdownExportMutation.mutate()}
          disabled={markdownExportMutation.isPending}
          className="btn-secondary flex items-center gap-2"
        >
          <DownloadIcon className="w-4 h-4" />
          Markdown
        </button>
        <button
          onClick={() => jsonExportMutation.mutate()}
          disabled={jsonExportMutation.isPending}
          className="btn-secondary flex items-center gap-2"
        >
          <DownloadIcon className="w-4 h-4" />
          JSON
        </button>
      </div>
      {selectedTemplateUnavailable && selectedTemplate && (
        <div className="mb-8 text-sm text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded px-3 py-2">
          Template unavailable: {selectedTemplate.missing_requirements.join('; ')}
        </div>
      )}
      {notebookMutation.error && (
        <div className="mb-8 text-sm text-red-300 bg-red-500/10 border border-red-500/30 rounded px-3 py-2">
          {notebookMutation.error instanceof Error
            ? notebookMutation.error.message
            : 'Notebook generation failed for this dataset.'}
        </div>
      )}
      {(markdownExportMutation.error || jsonExportMutation.error) && (
        <div className="mb-8 text-sm text-red-300 bg-red-500/10 border border-red-500/30 rounded px-3 py-2">
          {(markdownExportMutation.error || jsonExportMutation.error) instanceof Error
            ? (markdownExportMutation.error || jsonExportMutation.error)?.message
            : 'Dataset card export failed.'}
        </div>
      )}

      {/* Content grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main content */}
        <div className="lg:col-span-2 space-y-6">
          {/* Reuse Summary */}
          <section className="card">
            <h2 className="text-lg font-semibold mb-4">Scientific Reuse Summary</h2>
            <div className="space-y-4 text-sm">
              <div>
                <div className="text-neural-500 mb-1">Scientific use case</div>
                <p className="text-neural-300">{card.summary_details?.scientific_use_case || 'Not available'}</p>
              </div>
              <div>
                <div className="text-neural-500 mb-1">Why this dataset matters</div>
                <p className="text-neural-300">{card.summary_details?.why_this_dataset_matters || 'Not available'}</p>
              </div>
            </div>
          </section>

          {/* Experimental Structure */}
          {card.experimental_structure && (
            <section className="card">
              <h2 className="text-lg font-semibold mb-4">Experimental Structure</h2>
              <dl className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
                {[
                  ['Task labels', card.experimental_structure.task_labels],
                  ['Behavior labels', card.experimental_structure.behavior_labels],
                  ['Trial/event structure', card.experimental_structure.trial_event_structure],
                  ['Stimuli', card.experimental_structure.stimuli],
                  ['Rewards/outcomes', card.experimental_structure.rewards_outcomes],
                  ['Species', card.experimental_structure.species],
                  ['Subjects', card.experimental_structure.subjects],
                  ['Sessions', card.experimental_structure.sessions],
                ].map(([label, value]) => (
                  <div key={String(label)}>
                    <dt className="text-neural-500 mb-1">{String(label)}</dt>
                    <dd className="text-neural-300">{formatValue(value)}</dd>
                  </div>
                ))}
              </dl>
            </section>
          )}

          {/* Neural Data */}
          {card.neural_data && (
            <section className="card">
              <h2 className="text-lg font-semibold mb-4">Neural Data</h2>
              <dl className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
                {[
                  ['Modalities', card.neural_data.modalities],
                  ['Brain regions', card.neural_data.brain_regions],
                  ['File formats', card.neural_data.file_formats],
                  ['Raw data', card.neural_data.has_raw_data],
                  ['Processed data', card.neural_data.has_processed_data],
                ].map(([label, value]) => (
                  <div key={String(label)}>
                    <dt className="text-neural-500 mb-1">{String(label)}</dt>
                    <dd className="text-neural-300">{formatValue(value)}</dd>
                  </div>
                ))}
              </dl>
            </section>
          )}

          {/* Scientific Labels */}
          <section className="card">
            <h2 className="text-lg font-semibold mb-4">Scientific Labels</h2>
            {card.tasks.length === 0 &&
              card.modalities.length === 0 &&
              (!card.behaviors || card.behaviors.length === 0) &&
              card.brain_regions.length === 0 &&
              card.species.length === 0 && (
                <div className="text-sm text-neural-500 bg-neural-950 border border-neural-800 rounded p-3">
                  No scientific labels were extracted yet. Review the source metadata or linked papers to improve ontology coverage.
                </div>
              )}
            <dl className="space-y-4">
              {card.tasks.length > 0 && (
                <div>
                  <dt className="text-sm text-neural-500 mb-1.5">Tasks</dt>
                  <dd className="flex flex-wrap gap-2">
                    {card.tasks.map((t) => (
                      <span key={t} className="badge bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/30">
                        {t.replace(/_/g, ' ')}
                      </span>
                    ))}
                  </dd>
                </div>
              )}
              {card.modalities.length > 0 && (
                <div>
                  <dt className="text-sm text-neural-500 mb-1.5">Modalities</dt>
                  <dd className="flex flex-wrap gap-2">
                    {card.modalities.map((m) => (
                      <span key={m} className="badge bg-accent-violet/10 text-accent-violet border border-accent-violet/30">
                        {m.replace(/_/g, ' ')}
                      </span>
                    ))}
                  </dd>
                </div>
              )}
              {card.behaviors && card.behaviors.length > 0 && (
                <div>
                  <dt className="text-sm text-neural-500 mb-1.5">Behaviors</dt>
                  <dd className="flex flex-wrap gap-2">
                    {card.behaviors.map((b) => (
                      <span key={b} className="badge bg-accent-emerald/10 text-accent-emerald border border-accent-emerald/30">
                        {b.replace(/_/g, ' ')}
                      </span>
                    ))}
                  </dd>
                </div>
              )}
              {card.brain_regions.length > 0 && (
                <div>
                  <dt className="text-sm text-neural-500 mb-1.5">Brain Regions</dt>
                  <dd className="flex flex-wrap gap-2">
                    {card.brain_regions.map((r) => (
                      <span key={r} className="badge bg-neural-700 text-neural-300">
                        {r.replace(/_/g, ' ')}
                      </span>
                    ))}
                  </dd>
                </div>
              )}
              {card.species.length > 0 && (
                <div>
                  <dt className="text-sm text-neural-500 mb-1.5">Species</dt>
                  <dd className="flex flex-wrap gap-2">
                    {card.species.map((s) => (
                      <span key={s} className="badge bg-neural-700 text-neural-300">{s}</span>
                    ))}
                  </dd>
                </div>
              )}
            </dl>
          </section>

          {/* Available Assets */}
          {card.assets && card.assets.length > 0 && (
            <section className="card">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <FolderIcon className="w-5 h-5 text-accent-cyan" />
                Available Assets
              </h2>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {card.assets.slice(0, 10).map((asset) => (
                  <div
                    key={asset.id}
                    className="flex items-center justify-between py-2 px-3 bg-neural-800/50 rounded text-sm"
                  >
                    <span className="text-neural-300 font-mono truncate flex-1 mr-4">
                      {asset.path}
                    </span>
                    {asset.size_bytes && (
                      <span className="text-neural-500 flex-shrink-0">
                        {formatBytes(asset.size_bytes)}
                      </span>
                    )}
                  </div>
                ))}
                {card.assets.length > 10 && (
                  <p className="text-sm text-neural-500 pt-2">
                    +{card.assets.length - 10} more files
                  </p>
                )}
              </div>
            </section>
          )}

          {/* Suggested Analyses */}
          {card.readiness?.suggested_analyses && card.readiness.suggested_analyses.length > 0 && (
            <section className="card">
              <h2 className="text-lg font-semibold mb-4">Suggested Analyses</h2>
              <ul className="space-y-2">
                {card.readiness.suggested_analyses.map((analysis) => (
                  <li key={analysis} className="flex items-center gap-2 text-neural-300">
                    <DocumentIcon className="w-4 h-4 text-accent-cyan flex-shrink-0" />
                    {analysis.replace(/_/g, ' ')}
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* Starter Recipes */}
          {card.analysis_plan?.starter_recipes && card.analysis_plan.starter_recipes.length > 0 && (
            <section className="card">
              <h2 className="text-lg font-semibold mb-4">Starter Analysis Recipes</h2>
              <div className="space-y-4">
                {card.analysis_plan.starter_recipes.map((recipe) => (
                  <div key={recipe.id} className="border-b border-neural-800 last:border-0 pb-4 last:pb-0">
                    <div className="flex items-center justify-between gap-3 mb-1">
                      <h3 className="font-medium text-neural-200">{recipe.title}</h3>
                      <span className="badge bg-neural-800 text-neural-300">{recipe.id}</span>
                    </div>
                    {recipe.summary && (
                      <p className="text-sm text-neural-500 mb-2">{recipe.summary}</p>
                    )}
                    {recipe.analyses && recipe.analyses.length > 0 && (
                      <div className="flex flex-wrap gap-1.5">
                        {recipe.analyses.slice(0, 5).map((analysis) => (
                          <span key={analysis} className="badge bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/30">
                            {analysis.replace(/_/g, ' ')}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Linked Papers */}
          {linkedPapers.length > 0 && (
            <section className="card">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <BookOpenIcon className="w-5 h-5 text-accent-violet" />
                Linked Literature
              </h2>
              {card.linked_literature?.link_confidence_summary && (
                <p className="text-sm text-neural-500 mb-4">
                  {card.linked_literature.link_confidence_summary}
                </p>
              )}
              <ul className="space-y-3">
                {linkedPapers.map((paper) => (
                  <li key={paper.id} className="border-b border-neural-800 last:border-0 pb-3 last:pb-0">
                    <div className="text-neural-200 font-medium mb-1">
                      {paper.url ? (
                        <a
                          href={paper.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="hover:text-accent-cyan transition-colors"
                        >
                          {paper.title}
                        </a>
                      ) : (
                        paper.title
                      )}
                    </div>
                    <div className="text-sm text-neural-500">
                      {paper.authors?.slice(0, 3).join(', ')}
                      {paper.authors && paper.authors.length > 3 && ' et al.'}
                      {paper.year && ` (${paper.year})`}
                    </div>
                    {paper.doi && (
                      <div className="text-xs text-neural-600 font-mono mt-1">
                        DOI: {paper.doi}
                      </div>
                    )}
                    {paper.openalex_id && (
                      <div className="text-xs text-neural-600 font-mono mt-1">
                        OpenAlex: {paper.openalex_id}
                      </div>
                    )}
                    {paper.confidence !== undefined && (
                      <div className="text-xs text-neural-500 mt-1">
                        Link confidence: {Math.round(paper.confidence * 100)}%
                        {paper.link_evidence && paper.link_evidence.length > 0 && ` · ${paper.link_evidence.join(', ')}`}
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            </section>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* QA Review */}
          <section className="card">
            <h2 className="text-lg font-semibold mb-4">Corpus QA</h2>
            <div className="flex flex-wrap gap-2 mb-4">
              {(['needs_review', 'reviewed', 'trusted', 'rejected'] as DatasetQAStatus[]).map((status) => (
                <button
                  key={status}
                  type="button"
                  onClick={() => {
                    const updates = { ...qaForm, qa_status: status }
                    setQaForm(updates)
                    saveQA(updates)
                  }}
                  disabled={qaMutation.isPending}
                  className={`badge capitalize cursor-pointer ${
                    qaForm.qa_status === status
                      ? qaStatusStyles[status]
                      : 'bg-neural-800 text-neural-400 hover:text-neural-100'
                  }`}
                >
                  {formatQAStatus(status)}
                </button>
              ))}
            </div>
            <div className="space-y-2 mb-4">
              {[
                ['task_labels_verified', 'Task labels'],
                ['modality_labels_verified', 'Modality labels'],
                ['behavior_labels_verified', 'Behavior labels'],
                ['brain_regions_verified', 'Brain regions'],
                ['linked_papers_verified', 'Linked papers'],
                ['notebook_tested', 'Notebook tested'],
              ].map(([key, label]) => (
                <label key={key} className="flex items-center gap-2 text-sm text-neural-300">
                  <input
                    type="checkbox"
                    checked={Boolean(qaForm[key as keyof DatasetQAUpdate])}
                    onChange={(event) => updateQAField(key as keyof DatasetQAUpdate, event.target.checked)}
                    className="h-4 w-4 rounded border-neural-700 bg-neural-900 text-accent-cyan"
                  />
                  {label}
                </label>
              ))}
            </div>
            <textarea
              value={qaForm.reviewer_notes || ''}
              onChange={(event) => updateQAField('reviewer_notes', event.target.value)}
              rows={4}
              className="input text-sm resize-none mb-3"
              placeholder="Reviewer notes"
            />
            <button
              type="button"
              onClick={() => saveQA()}
              disabled={qaMutation.isPending}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              <CheckCircleIcon className="w-4 h-4" />
              {qaMutation.isPending ? 'Saving...' : 'Save QA Review'}
            </button>
          </section>

          {/* Readiness Score */}
          {card.readiness && (
            <section className="card">
              <h2 className="text-lg font-semibold mb-4">Analysis Readiness</h2>
              <div className="flex items-center gap-4 mb-4">
                <div className="text-4xl font-bold text-accent-cyan">
                  {Math.round(card.readiness.score)}
                </div>
                <div className="text-sm text-neural-400">/ 100</div>
              </div>

              {card.readiness.strengths.length > 0 && (
                <div className="mb-4">
                  <h3 className="text-sm font-medium text-neural-300 mb-2">Strengths</h3>
                  <ul className="space-y-1.5">
                    {card.readiness.strengths.map((s) => (
                      <li key={s} className="flex items-start gap-2 text-sm text-neural-400">
                        <CheckCircleIcon className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
                        {s}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {card.readiness.limitations.length > 0 && (
                <div>
                  <h3 className="text-sm font-medium text-neural-300 mb-2">Limitations</h3>
                  <ul className="space-y-1.5">
                    {card.readiness.limitations.map((l) => (
                      <li key={l} className="flex items-start gap-2 text-sm text-neural-400">
                        <WarningIcon className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                        {l}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {card.analysis_plan && (
                <div className="mt-4 pt-4 border-t border-neural-800 space-y-3">
                  <div>
                    <h3 className="text-sm font-medium text-neural-300 mb-1">First analysis</h3>
                    <p className="text-sm text-neural-400">
                      {formatValue(card.analysis_plan.suggested_first_analysis)}
                    </p>
                  </div>
                  <div>
                    <h3 className="text-sm font-medium text-neural-300 mb-1">Advanced analysis</h3>
                    <p className="text-sm text-neural-400">
                      {formatValue(card.analysis_plan.suggested_advanced_analysis)}
                    </p>
                  </div>
                </div>
              )}
            </section>
          )}

          {/* Analysis Affordances */}
          {affordancesData && affordancesData.affordances.length > 0 && (
            <AffordancePanel affordances={affordancesData.affordances} />
          )}

          {/* Similar Datasets */}
          {similarData && similarData.similar.length > 0 && (
            <SimilarDatasetsPanel similar={similarData.similar} />
          )}

          {/* Reuse Instructions */}
          {card.reuse_instructions && (
            <section className="card">
              <h2 className="text-lg font-semibold mb-4">Reuse Instructions</h2>
              <dl className="space-y-3 text-sm">
                <div>
                  <dt className="text-neural-500">How to load</dt>
                  <dd className="text-neural-300">{formatValue(card.reuse_instructions.how_to_load)}</dd>
                </div>
                <div>
                  <dt className="text-neural-500">Notebook generation</dt>
                  <dd className="text-neural-300">{formatValue(card.reuse_instructions.notebook_generation_status)}</dd>
                </div>
                <div>
                  <dt className="text-neural-500">First steps</dt>
                  <dd className="text-neural-300">
                    <ul className="space-y-1">
                      {asList(card.reuse_instructions.recommended_first_steps).map((step) => (
                        <li key={step}>• {step}</li>
                      ))}
                    </ul>
                  </dd>
                </div>
              </dl>
            </section>
          )}

          {notebookTemplates.length > 0 && (
            <section className="card">
              <h2 className="text-lg font-semibold mb-4">Notebook Templates</h2>
              <div className="space-y-3">
                {notebookTemplates.map((template) => (
                  <div key={template.id} className="border-b border-neural-800 last:border-0 pb-3 last:pb-0">
                    <div className="flex items-center justify-between gap-2">
                      <div className="font-medium text-neural-200">{template.title}</div>
                      <span className={`badge ${template.available ? 'bg-accent-emerald/10 text-accent-emerald border border-accent-emerald/30' : 'bg-neural-800 text-neural-500'}`}>
                        {template.available ? 'available' : 'unavailable'}
                      </span>
                    </div>
                    {template.description && (
                      <p className="text-sm text-neural-500 mt-1">{template.description}</p>
                    )}
                    {!template.available && template.missing_requirements.length > 0 && (
                      <p className="text-xs text-amber-300 mt-2">
                        {template.missing_requirements.join('; ')}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Missing Metadata */}
          {card.missing_metadata && card.missing_metadata.length > 0 && (
            <section className="card border-amber-500/30">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2 text-amber-400">
                <WarningIcon className="w-5 h-5" />
                Missing Metadata
              </h2>
              <ul className="space-y-1.5">
                {card.missing_metadata.map((field) => (
                  <li key={field} className="text-sm text-neural-400">
                    • {field}
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* Provenance */}
          {card.provenance && (
            <section className="card bg-neural-800/50">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <InformationCircleIcon className="w-5 h-5 text-neural-400" />
                Provenance
              </h2>
              <dl className="space-y-2 text-sm">
                {card.provenance.dataset_source && (
                  <div>
                    <dt className="text-neural-500">Source</dt>
                    <dd className="text-neural-300">{card.provenance.dataset_source}</dd>
                  </div>
                )}
                {card.provenance.dataset_source_id && (
                  <div>
                    <dt className="text-neural-500">Source ID</dt>
                    <dd className="text-neural-300 font-mono text-xs">{card.provenance.dataset_source_id}</dd>
                  </div>
                )}
                {card.provenance.linked_paper_count !== undefined && (
                  <div>
                    <dt className="text-neural-500">Linked Papers</dt>
                    <dd className="text-neural-300">{card.provenance.linked_paper_count}</dd>
                  </div>
                )}
                {card.provenance.claim_policy && (
                  <div className="pt-2 border-t border-neural-700">
                    <dt className="text-neural-500 mb-1">Policy</dt>
                    <dd className="text-neural-400 text-xs italic">{card.provenance.claim_policy}</dd>
                  </div>
                )}
              </dl>
            </section>
          )}

          {/* Generated timestamp */}
          {card.generated_at && (
            <div className="text-xs text-neural-600 text-center">
              Card generated: {new Date(card.generated_at).toLocaleDateString()}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
