import { useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import {
  exportDatasetCard,
  generateNotebook,
  logRetrievalFeedback,
  saveFrontendDataset,
} from '../api/search'
import type { DimensionMatch, ExplanationGroups, FeedbackUsefulness, MemoryGraphEvidence, SearchResultItem, WouldUseForAnalysis } from '../types'

interface DatasetCardProps {
  result: SearchResultItem
  queryText?: string
  sessionId?: string | null
  isSelected?: boolean
  onToggleSelect?: (datasetId: string) => void
  selectionDisabled?: boolean
}

const sourceLabel: Record<string, string> = {
  dandi: 'DANDI',
  openneuro: 'OpenNeuro',
  demo: 'Demo',
  other: 'Other',
}

const roleColors: Record<string, string> = {
  dandi: 'text-accent-cyan',
  openneuro: 'text-accent-violet',
  demo: 'text-accent-emerald',
  other: 'text-neural-400',
}

function Score({ value, label }: { value: number; label: string }) {
  return (
    <div className="text-right">
      <div className="text-xl font-light tabular-nums text-neural-200">{value}</div>
      <div className="text-xs text-neural-600">{label}</div>
    </div>
  )
}

function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'neutral' | 'cyan' | 'amber' | 'red' | 'green' | 'violet' }) {
  const tones = {
    neutral: 'text-neural-500 border-neural-800 bg-neural-900',
    cyan: 'text-accent-cyan border-accent-cyan/30 bg-accent-cyan/5',
    amber: 'text-amber-300 border-amber-500/30 bg-amber-500/5',
    red: 'text-red-300 border-red-500/30 bg-red-500/5',
    green: 'text-accent-emerald border-accent-emerald/30 bg-accent-emerald/5',
    violet: 'text-accent-violet border-accent-violet/30 bg-accent-violet/5',
  }
  return (
    <span className={`text-xs border rounded px-2 py-0.5 ${tones[tone]}`}>
      {children}
    </span>
  )
}

function MetaLine({ label, values }: { label: string; values?: string[] }) {
  const visible = (values || []).filter(Boolean)
  return (
    <p className="text-xs text-neural-600 truncate">
      <span className="text-neural-500">{label}:</span>{' '}
      <span className="text-neural-400">{visible.length ? visible.slice(0, 4).join(', ') : 'unknown'}</span>
    </p>
  )
}

function ListBlock({ title, items }: { title: string; items?: string[] }) {
  const visible = (items || []).filter(Boolean)
  if (!visible.length) return null
  return (
    <div>
      <p className="text-xs uppercase tracking-wide text-neural-600 mb-1">{title}</p>
      <ul className="space-y-1">
        {visible.slice(0, 6).map((item) => (
          <li key={item} className="text-xs text-neural-400 leading-relaxed">{item}</li>
        ))}
      </ul>
    </div>
  )
}

const DIMENSION_STYLES: Record<string, { tone: 'cyan' | 'violet' | 'amber' | 'green' | 'neutral'; label: string }> = {
  brain_region: { tone: 'violet', label: 'region' },
  modality:     { tone: 'cyan',   label: 'modality' },
  task:         { tone: 'amber',  label: 'task' },
  species:      { tone: 'green',  label: 'species' },
  recording_scale: { tone: 'neutral', label: 'scale' },
}

function DimensionMatchBadges({ groups }: { groups: ExplanationGroups }) {
  const entries = Object.entries(groups) as [keyof ExplanationGroups, DimensionMatch][]
  const matched = entries.filter(([, d]) => d.matched.length > 0)
  const queried_unmatched = entries.filter(([, d]) => d.queried.length > 0 && d.matched.length === 0)
  if (matched.length === 0 && queried_unmatched.length === 0) return null
  return (
    <div className="mt-3 flex flex-wrap gap-1.5">
      {matched.map(([dim, d]) => {
        const style = DIMENSION_STYLES[dim] ?? { tone: 'neutral', label: dim }
        return d.matched.slice(0, 2).map((term) => (
          <Badge key={`${dim}-${term}`} tone={style.tone}>
            {style.label}: {term.replace(/_/g, ' ')}
          </Badge>
        ))
      })}
      {queried_unmatched.map(([dim, d]) => {
        const style = DIMENSION_STYLES[dim] ?? { tone: 'neutral', label: dim }
        return (
          <span
            key={dim}
            className="text-xs border rounded px-2 py-0.5 text-neural-600 border-neural-800 bg-neural-900 line-through"
            title={`Queried ${style.label} not found`}
          >
            {style.label}: {d.queried[0]?.replace(/_/g, ' ')}
          </span>
        )
      })}
    </div>
  )
}

function MemoryGraphEvidencePanel({ evidence }: { evidence: MemoryGraphEvidence }) {
  const hasAny =
    evidence.modality_matches.length > 0 ||
    (evidence.recording_scale_matches || []).length > 0 ||
    evidence.species_matches.length > 0 ||
    evidence.region_matches.length > 0 ||
    evidence.affordance_matches.length > 0 ||
    evidence.has_raw_signal ||
    evidence.contraindicated.length > 0 ||
    evidence.lacks_evidence_count > 0

  if (!hasAny) return null

  return (
    <div className="mt-4 border-t border-neural-800/60 pt-4">
      <p className="text-xs uppercase tracking-wide text-neural-600 mb-2">Field-State Graph Evidence</p>
      <div className="flex flex-wrap gap-1.5">
        {evidence.modality_matches.map((m) => (
          <span key={m} className="text-xs border rounded px-2 py-0.5 text-accent-cyan border-accent-cyan/30 bg-accent-cyan/5">
            modality: {m.replace(/_/g, ' ')}
          </span>
        ))}
        {(evidence.recording_scale_matches || []).map((s) => (
          <span key={s} className="text-xs border rounded px-2 py-0.5 text-accent-cyan border-accent-cyan/30 bg-accent-cyan/5">
            scale: {s.replace(/_/g, ' ')}
          </span>
        ))}
        {evidence.species_matches.map((s) => (
          <span key={s} className="text-xs border rounded px-2 py-0.5 text-accent-emerald border-accent-emerald/30 bg-accent-emerald/5">
            species: {s.replace(/_/g, ' ')}
          </span>
        ))}
        {evidence.region_matches.map((r) => (
          <span key={r} className="text-xs border rounded px-2 py-0.5 text-accent-violet border-accent-violet/30 bg-accent-violet/5">
            region: {r.replace(/_/g, ' ')}
          </span>
        ))}
        {evidence.affordance_matches.map((a) => (
          <span key={a} className="text-xs border rounded px-2 py-0.5 text-neural-300 border-neural-600/30 bg-neural-800/40">
            affordance: {a.replace(/_/g, ' ')}
          </span>
        ))}
        {evidence.has_raw_signal && (
          <span className="text-xs border rounded px-2 py-0.5 text-accent-emerald border-accent-emerald/30 bg-accent-emerald/5">
            raw signal confirmed
          </span>
        )}
        {evidence.lacks_evidence_count > 0 && (
          <span className="text-xs border rounded px-2 py-0.5 text-amber-300 border-amber-500/30 bg-amber-500/5">
            ⚠ {evidence.lacks_evidence_count} evidence gap{evidence.lacks_evidence_count > 1 ? 's' : ''}
          </span>
        )}
        {evidence.contraindicated.map((c) => (
          <span key={c} className="text-xs border rounded px-2 py-0.5 text-red-300 border-red-500/30 bg-red-500/5">
            ✗ contraindicated: {c.replace(/_/g, ' ')}
          </span>
        ))}
      </div>
    </div>
  )
}

function EvidencePanel({
  result,
  rawJsonOpen,
  onRawJsonToggle,
}: {
  result: SearchResultItem
  rawJsonOpen: boolean
  onRawJsonToggle: () => void
}) {
  const { evidence_packet, neuro_judge, score_breakdown, linked_papers } = result
  const packetPapers = evidence_packet?.linked_papers || linked_papers || []
  const scoreEntries = Object.entries(score_breakdown || {})
    .filter(([, value]) => Number.isFinite(value))
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)

  return (
    <div className="mt-5 border border-neural-800/60 rounded-lg p-4 bg-neural-950/70">
      <div className="flex flex-wrap items-center gap-2 mb-4">
        <Badge tone="amber">preliminary neuro-judge, not human gold</Badge>
        {neuro_judge?.judge_model && <Badge>{neuro_judge.judge_model}</Badge>}
        {neuro_judge?.evidence_packet_hash && <Badge>{neuro_judge.evidence_packet_hash}</Badge>}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div className="space-y-4">
          <div>
            <p className="text-xs uppercase tracking-wide text-neural-600 mb-1">Evidence packet</p>
            <p className="text-xs text-neural-400 leading-relaxed">
              {evidence_packet?.query_text || 'No query-specific evidence packet was attached.'}
            </p>
            {evidence_packet?.description && (
              <p className="mt-2 text-xs text-neural-500 line-clamp-3 leading-relaxed">
                {evidence_packet.description}
              </p>
            )}
          </div>

          <ListBlock title="Concept matches" items={evidence_packet?.matched_concept_names} />
          <ListBlock title="Concept missing evidence" items={evidence_packet?.concept_missing_evidence} />
          <ListBlock title="Evidence for" items={neuro_judge?.evidence_for} />
          <ListBlock title="Evidence against" items={neuro_judge?.evidence_against} />
          <ListBlock title="Missing information" items={neuro_judge?.missing_information} />
        </div>

        <div className="space-y-4">
          <ListBlock title="Required dimensions present" items={neuro_judge?.required_dimensions_present} />
          <ListBlock title="Required dimensions missing" items={neuro_judge?.required_dimensions_missing} />
          <ListBlock title="Failure modes" items={neuro_judge?.failure_modes} />
          <ListBlock title="Hard-negative checks" items={[
            ...(evidence_packet?.hard_negatives || []),
            ...(evidence_packet?.known_failure_warnings || []),
            ...(evidence_packet?.concept_hard_negative_conflicts || []),
          ]} />

          {evidence_packet?.affordance_matches && evidence_packet.affordance_matches.length > 0 && (
            <div>
              <p className="text-xs uppercase tracking-wide text-neural-600 mb-1">Affordance matches</p>
              <div className="space-y-1">
                {evidence_packet.affordance_matches.slice(0, 6).map((match) => (
                  <p key={`${match.affordance}-${match.rationale}`} className="text-xs text-neural-400">
                    {match.affordance}: {match.matched ? 'matched' : 'not matched'}
                    {typeof match.confidence === 'number' ? ` · ${Math.round(match.confidence * 100)}%` : ''}
                    {match.rationale ? ` · ${match.rationale}` : ''}
                  </p>
                ))}
              </div>
            </div>
          )}

          <div>
            <p className="text-xs uppercase tracking-wide text-neural-600 mb-1">Score breakdown</p>
            <div className="grid grid-cols-2 gap-1">
              {scoreEntries.length ? scoreEntries.map(([key, value]) => (
                <p key={key} className="text-xs text-neural-500">
                  {key.replace(/_/g, ' ')}: <span className="text-neural-300">{value.toFixed(3)}</span>
                </p>
              )) : (
                <p className="text-xs text-neural-600">No component scores available.</p>
              )}
            </div>
          </div>
        </div>
      </div>

      {packetPapers.length > 0 && (
        <div className="mt-4 border-t border-neural-800/60 pt-4">
          <p className="text-xs uppercase tracking-wide text-neural-600 mb-2">Linked paper</p>
          {packetPapers.slice(0, 1).map((paper) => (
            <div key={paper.id}>
              <p className="text-sm text-neural-300">{paper.title}</p>
              {(paper.abstract_snippet || paper.abstract) && (
                <p className="mt-1 text-xs text-neural-500 line-clamp-3 leading-relaxed">
                  {paper.abstract_snippet || paper.abstract}
                </p>
              )}
            </div>
          ))}
        </div>
      )}

      {result.memory_graph_evidence && (
        <MemoryGraphEvidencePanel evidence={result.memory_graph_evidence} />
      )}

      <div className="mt-4 border-t border-neural-800/60 pt-4">
        <button
          type="button"
          onClick={onRawJsonToggle}
          className="text-xs text-neural-400 hover:text-neural-200"
        >
          {rawJsonOpen ? 'Hide raw evidence JSON' : 'View raw evidence JSON'}
        </button>
        {rawJsonOpen && (
          <pre className="mt-3 max-h-72 overflow-auto rounded border border-neural-800 bg-neural-900 p-3 text-xs text-neural-400">
            {JSON.stringify(evidence_packet?.raw_json || evidence_packet || neuro_judge || {}, null, 2)}
          </pre>
        )}
      </div>
    </div>
  )
}

const reasonTagOptions = [
  'wrong_modality',
  'wrong_species',
  'wrong_region',
  'missing_raw_data',
  'insufficient_metadata',
  'good_match',
  'interesting_reuse_candidate',
  'needs_manual_review',
  'wrong_task',
  'processed_only',
  'low_evidence',
]

export function DatasetCard({
  result,
  queryText = '',
  sessionId = null,
  isSelected = false,
  onToggleSelect,
  selectionDisabled = false,
}: DatasetCardProps) {
  const {
    dataset,
    score,
    why_matched,
    warnings,
    readiness_score,
    linked_papers,
    neuro_judge,
    evidence_packet,
    score_breakdown,
  } = result
  const [detailsOpen, setDetailsOpen] = useState(false)
  const [rawJsonOpen, setRawJsonOpen] = useState(false)
  const [usefulness, setUsefulness] = useState<FeedbackUsefulness>('unsure')
  const [wouldUse, setWouldUse] = useState<WouldUseForAnalysis>('maybe')
  const [reasonTags, setReasonTags] = useState<string[]>([])
  const [note, setNote] = useState('')

  const notebookMutation = useMutation({
    mutationFn: () => generateNotebook(dataset.id),
    onSuccess: (blob) => {
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${dataset.id}_starter.ipynb`
      a.click()
      URL.revokeObjectURL(url)
    },
  })

  const feedbackMutation = useMutation({
    mutationFn: (overrides: Partial<Parameters<typeof logRetrievalFeedback>[0]> = {}) => logRetrievalFeedback({
      session_id: sessionId,
      query_id: evidence_packet?.query_id || null,
      query_text: queryText || evidence_packet?.query_text || '',
      retrieval_method: result.retrieval_method || 'hybrid_search',
      rank: result.rank ?? null,
      dataset_id: dataset.id,
      dataset_title: dataset.title,
      usefulness,
      would_use_for_analysis: wouldUse,
      clicked: false,
      opened_evidence: detailsOpen,
      saved: false,
      exported: false,
      reason_tags: reasonTags,
      free_text_note: note,
      judge_snapshot: neuro_judge || {},
      ...overrides,
    }),
  })

  const saveMutation = useMutation({
    mutationFn: (exported = false) => saveFrontendDataset({
      session_id: sessionId,
      query_id: evidence_packet?.query_id || null,
      query_text: queryText || evidence_packet?.query_text || '',
      dataset_id: dataset.id,
      dataset_title: dataset.title,
      rank: result.rank ?? null,
      retrieval_method: result.retrieval_method || 'hybrid_search',
      exported,
      judge_snapshot: neuro_judge || {},
    }),
  })

  const exportMutation = useMutation({
    mutationFn: () => exportDatasetCard(dataset.id, 'json'),
    onSuccess: (blob) => {
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${dataset.id}_reuse_card.json`
      a.click()
      URL.revokeObjectURL(url)
      saveMutation.mutate(true)
      feedbackMutation.mutate({ exported: true })
    },
  })

  const tags = [
    ...(dataset.species || []).slice(0, 2),
    ...(dataset.modalities || []).slice(0, 2),
    ...(dataset.recording_scales || []).slice(0, 2),
    ...(dataset.brain_regions || []).slice(0, 2),
    ...(dataset.tasks || []).slice(0, 2),
  ]
    .map((t) => t.replace(/_/g, ' '))
    .filter(Boolean)
    .slice(0, 7)
  const dataStandards = [
    dataset.data_standard,
    ...(dataset.data_standards || []),
    ...(evidence_packet?.file_format_evidence || []),
  ].filter((value): value is string => Boolean(value))
  const hardNegativeWarnings = [
    ...(warnings || []),
    ...(evidence_packet?.known_failure_warnings || []),
    ...(evidence_packet?.concept_hard_negative_conflicts || []),
  ]
  const judgeLabelTone = neuro_judge?.label_provenance === 'expert_audited_consensus'
    ? 'green'
    : neuro_judge?.abstain_recommended
    ? 'amber'
    : 'violet'
  const rawSummary = evidence_packet
    ? `raw: ${evidence_packet.has_raw_data === true ? 'yes' : evidence_packet.has_raw_data === false ? 'no' : 'unknown'} · processed: ${evidence_packet.has_processed_data === true ? 'yes' : evidence_packet.has_processed_data === false ? 'no' : 'unknown'}`
    : 'raw/processed evidence unavailable'

  const toggleReasonTag = (tag: string) => {
    setReasonTags((current) => current.includes(tag)
      ? current.filter((item) => item !== tag)
      : [...current, tag])
  }

  return (
    <article
      className={`border-b border-neural-800/50 py-6 transition-colors group ${
        isSelected ? 'bg-accent-cyan/5' : ''
      }`}
    >
      <div className="flex items-start justify-between gap-6">
        {/* Left: content */}
        <div className="flex-1 min-w-0">
          {/* Source + title row */}
          <div className="flex items-baseline gap-3 mb-2 flex-wrap">
            <span className={`font-mono text-xs font-medium ${roleColors[dataset.source] || 'text-neural-500'}`}>
              {sourceLabel[dataset.source] || dataset.source.toUpperCase()}
            </span>

            <span className="font-mono text-xs text-neural-600">{dataset.source_id || dataset.id}</span>

            {dataStandards.slice(0, 2).map((standard) => (
              <span key={standard} className="font-mono text-xs text-neural-600">{standard}</span>
            ))}

            {/* Comparison checkbox */}
            {onToggleSelect && (
              <button
                onClick={(e) => { e.preventDefault(); onToggleSelect(dataset.id) }}
                disabled={selectionDisabled && !isSelected}
                className={`font-mono text-xs transition-colors ${
                  isSelected
                    ? 'text-accent-cyan'
                    : selectionDisabled
                    ? 'text-neural-700 cursor-not-allowed'
                    : 'text-neural-600 hover:text-neural-300'
                }`}
                title={isSelected ? 'Remove from comparison' : 'Add to comparison'}
              >
                {isSelected ? '− unselect' : '+ compare'}
              </button>
            )}
          </div>

          <Link
            to={`/datasets/${dataset.id}`}
            className="block text-lg font-medium text-neural-100 hover:text-white mb-2 transition-colors leading-snug"
          >
            {dataset.title}
          </Link>

          {dataset.description && (
            <p className="text-sm text-neural-500 line-clamp-2 mb-3 leading-relaxed">
              {dataset.description}
            </p>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1 mb-3">
            <MetaLine label="Species" values={dataset.species} />
            <MetaLine label="Modality" values={dataset.modalities} />
            <MetaLine label="Scale" values={dataset.recording_scales} />
            <MetaLine label="Region" values={dataset.brain_regions} />
            <MetaLine label="Tasks" values={dataset.tasks} />
            <MetaLine label="License" values={dataset.license ? [dataset.license] : []} />
            <MetaLine label="Raw data" values={[rawSummary]} />
          </div>

          {/* Tags */}
          {tags.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mb-3">
              {tags.map((tag) => (
                <span
                  key={tag}
                  className="text-xs text-neural-600 bg-neural-900 border border-neural-800 rounded px-2 py-0.5"
                >
                  {tag}
                </span>
              ))}
            </div>
          )}

          {/* Why matched */}
          {why_matched.length > 0 && (
            <p className="text-xs text-neural-600 mb-1 line-clamp-1">
              {why_matched.slice(0, 3).join(' · ')}
            </p>
          )}

          {/* Dimension match badges — region / modality / task / species */}
          {result.dataset_card_preview?.explanation?.match_summary?.dimension_matches && (
            <DimensionMatchBadges
              groups={result.dataset_card_preview.explanation.match_summary.dimension_matches as ExplanationGroups}
            />
          )}

          <div className="flex flex-wrap gap-2 mb-4">
            <Badge tone="cyan">retrieval {Math.round(score * 100)}</Badge>
            {neuro_judge ? (
              <>
                <Badge tone={judgeLabelTone}>neuro-judge {neuro_judge.label}</Badge>
                <Badge>conf {Math.round((neuro_judge.confidence || 0) * 100)}%</Badge>
                <Badge>evidence {Math.round((neuro_judge.evidence_completeness || 0) * 100)}%</Badge>
                {neuro_judge.abstain_recommended && <Badge tone="amber">abstain recommended</Badge>}
                <Badge tone={neuro_judge.label_provenance === 'expert_audited_consensus' ? 'green' : 'neutral'}>
                  {neuro_judge.label_provenance || 'neuro_judge'}
                </Badge>
              </>
            ) : (
              <Badge>no neuro-judge snapshot</Badge>
            )}
            {result.prior_feedback && result.prior_feedback.length > 0 && (
              <Badge tone="green">{result.prior_feedback.length} feedback event{result.prior_feedback.length > 1 ? 's' : ''}</Badge>
            )}
            {typeof score_breakdown?.memory_graph_score === 'number' && score_breakdown.memory_graph_score !== 0 && (
              <Badge tone={score_breakdown.memory_graph_score > 0 ? 'cyan' : 'amber'}>
                graph {score_breakdown.memory_graph_score > 0 ? '+' : ''}{score_breakdown.memory_graph_score.toFixed(2)}
              </Badge>
            )}
          </div>

          {/* Actions */}
          <div className="flex items-center gap-4">
            <Link
              to={`/datasets/${dataset.id}`}
              className="text-xs text-neural-400 hover:text-accent-cyan transition-colors"
            >
              View card →
            </Link>

            {dataset.url && (
              <a
                href={dataset.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-neural-400 hover:text-accent-cyan transition-colors"
              >
                Source ↗
              </a>
            )}

            {dataset.doi && !dataset.url && (
              <a
                href={`https://doi.org/${dataset.doi}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-neural-400 hover:text-accent-cyan transition-colors"
              >
                DOI ↗
              </a>
            )}

            <button
              onClick={(e) => { e.preventDefault(); notebookMutation.mutate() }}
              disabled={notebookMutation.isPending}
              className="text-xs text-neural-400 hover:text-neural-200 transition-colors"
            >
              {notebookMutation.isPending ? 'Generating…' : 'Notebook'}
            </button>

            <button
              onClick={(e) => {
                e.preventDefault()
                setDetailsOpen((value) => !value)
                feedbackMutation.mutate({ opened_evidence: true })
              }}
              className="text-xs text-neural-400 hover:text-neural-200 transition-colors"
            >
              {detailsOpen ? 'Close evidence' : 'Open evidence'}
            </button>

            <button
              onClick={(e) => {
                e.preventDefault()
                saveMutation.mutate(false)
                feedbackMutation.mutate({ saved: true })
              }}
              disabled={saveMutation.isPending}
              className="text-xs text-neural-400 hover:text-neural-200 transition-colors"
            >
              {saveMutation.isPending ? 'Saving…' : 'Save'}
            </button>

            <button
              onClick={(e) => { e.preventDefault(); exportMutation.mutate() }}
              disabled={exportMutation.isPending}
              className="text-xs text-neural-400 hover:text-neural-200 transition-colors"
            >
              {exportMutation.isPending ? 'Exporting…' : 'Export JSON'}
            </button>

            {linked_papers && linked_papers.length > 0 && (
              <span className="text-xs text-neural-600">
                {linked_papers.length} paper{linked_papers.length > 1 ? 's' : ''}
              </span>
            )}

            {warnings && warnings.length > 0 && (
              <span className="text-xs text-amber-500/70" title={warnings.join('; ')}>
                ⚠ {warnings.length}
              </span>
            )}
          </div>

          {neuro_judge?.rationale_short && (
            <p className="mt-3 text-xs text-neural-500 leading-relaxed">
              Judge rationale: {neuro_judge.rationale_short}
            </p>
          )}

          {hardNegativeWarnings.length > 0 && (
            <p className="mt-2 text-xs text-amber-300/80">
              Hard-negative warning: {hardNegativeWarnings.slice(0, 2).join(' · ')}
            </p>
          )}

          {notebookMutation.error && (
            <p className="mt-2 text-xs text-red-400">
              {notebookMutation.error instanceof Error
                ? notebookMutation.error.message
                : 'Notebook generation failed.'}
            </p>
          )}

          {detailsOpen && (
            <EvidencePanel
              result={result}
              rawJsonOpen={rawJsonOpen}
              onRawJsonToggle={() => setRawJsonOpen((value) => !value)}
            />
          )}

          <div className="mt-5 border border-neural-800/60 rounded-lg p-3">
            <div className="flex flex-wrap gap-2 mb-3">
              {(['useful', 'partially_useful', 'not_useful', 'unsure'] as FeedbackUsefulness[]).map((value) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setUsefulness(value)}
                  className={`text-xs rounded px-2 py-1 border transition-colors ${
                    usefulness === value
                      ? 'border-accent-cyan text-accent-cyan bg-accent-cyan/5'
                      : 'border-neural-800 text-neural-500 hover:text-neural-200'
                  }`}
                >
                  {value.replace(/_/g, ' ')}
                </button>
              ))}
              <select
                value={wouldUse}
                onChange={(e) => setWouldUse(e.target.value as WouldUseForAnalysis)}
                className="bg-neural-900 border border-neural-800 rounded px-2 py-1 text-xs text-neural-300"
              >
                <option value="yes">would use: yes</option>
                <option value="maybe">would use: maybe</option>
                <option value="no">would use: no</option>
              </select>
            </div>

            <div className="flex flex-wrap gap-1.5 mb-3">
              {reasonTagOptions.map((tag) => (
                <button
                  key={tag}
                  type="button"
                  onClick={() => toggleReasonTag(tag)}
                  className={`text-xs rounded px-2 py-0.5 border ${
                    reasonTags.includes(tag)
                      ? 'border-accent-violet text-accent-violet bg-accent-violet/5'
                      : 'border-neural-800 text-neural-600 hover:text-neural-300'
                  }`}
                >
                  {tag}
                </button>
              ))}
            </div>

            <div className="flex gap-2">
              <input
                value={note}
                onChange={(e) => setNote(e.target.value)}
                placeholder="Feedback note"
                className="flex-1 bg-neural-950 border border-neural-800 rounded px-3 py-1.5 text-xs text-neural-300 placeholder-neural-700"
              />
              <button
                type="button"
                onClick={() => feedbackMutation.mutate()}
                disabled={feedbackMutation.isPending}
                className="text-xs bg-neural-800 text-neural-200 rounded px-3 py-1.5 hover:bg-neural-700 disabled:opacity-50"
              >
                {feedbackMutation.isPending ? 'Logging…' : 'Log feedback'}
              </button>
            </div>
            <p className="mt-2 text-xs text-neural-700">
              Stored as user_feedback_downstream_signal, not a gold relevance label.
            </p>
            {feedbackMutation.isSuccess && (
              <p className="mt-2 text-xs text-accent-emerald">Feedback logged.</p>
            )}
            {(feedbackMutation.error || saveMutation.error || exportMutation.error) && (
              <p className="mt-2 text-xs text-red-400">Action could not be logged or exported.</p>
            )}
          </div>
        </div>

        {/* Right: scores */}
        <div className="flex flex-col items-end gap-3 flex-shrink-0">
          <Score value={Math.round(score * 100)} label="match" />
          {readiness_score !== undefined && (
            <Score value={Math.round(readiness_score)} label="readiness" />
          )}
          {dataset.nwb_count > 0 && (
            <div className="text-right">
              <div className="font-mono text-xs text-neural-600">{dataset.nwb_count} NWB</div>
            </div>
          )}
        </div>
      </div>
    </article>
  )
}
