import { Link } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { ChevronRightIcon, WarningIcon, ExternalLinkIcon, CodeIcon, BookOpenIcon } from './Icons'
import { generateNotebook } from '../api/search'
import type { SearchResultItem } from '../types'

const qaStatusStyles: Record<string, string> = {
  unreviewed: 'bg-neural-700 text-neural-300',
  auto_generated: 'bg-amber-500/10 text-amber-300 border border-amber-500/30',
  needs_review: 'bg-red-500/10 text-red-300 border border-red-500/30',
  reviewed: 'bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/30',
  trusted: 'bg-accent-emerald/10 text-accent-emerald border border-accent-emerald/30',
  rejected: 'bg-red-500/10 text-red-300 border border-red-500/30',
}

function formatQAStatus(status: string | undefined) {
  return (status || 'auto_generated').replace(/_/g, ' ')
}

interface DatasetCardProps {
  result: SearchResultItem
}

export function DatasetCard({ result }: DatasetCardProps) {
  const {
    dataset,
    score,
    why_matched,
    warnings,
    readiness_score,
    linked_papers,
    reusable_reason,
    evidence_snippets,
    matched_terms,
  } = result

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

  const sourceColors: Record<string, string> = {
    dandi: 'badge-cyan',
    openneuro: 'badge-violet',
    other: 'badge-emerald',
  }

  const handleNotebookClick = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    notebookMutation.mutate()
  }

  const handleSourceClick = (e: React.MouseEvent) => {
    e.stopPropagation()
  }

  return (
    <div className="card-hover group">
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          {/* Header: source badge and title */}
          <div className="flex items-center gap-2 mb-2">
            <span className={`badge ${sourceColors[dataset.source] || 'badge-emerald'}`}>
              {dataset.source.toUpperCase()}
            </span>
            {dataset.data_standard && (
              <span className="badge badge-emerald">
                {dataset.data_standard.toUpperCase()}
              </span>
            )}
            <span className={`badge ${qaStatusStyles[dataset.qa_status] || qaStatusStyles.auto_generated}`}>
              {formatQAStatus(dataset.qa_status)}
            </span>
            <Link
              to={`/datasets/${dataset.id}`}
              className="text-lg font-medium text-neural-100 truncate hover:text-accent-cyan transition-colors"
            >
              {dataset.title}
            </Link>
          </div>

          {/* Description */}
          {dataset.description && (
            <p className="text-sm text-neural-400 line-clamp-2 mb-3">
              {dataset.description}
            </p>
          )}

          {/* Scientific labels grid */}
          <div className="space-y-2 mb-3">
            {/* Tasks */}
            {dataset.tasks && dataset.tasks.length > 0 && (
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-xs text-neural-500 w-16">Tasks:</span>
                {dataset.tasks.slice(0, 3).map((t) => (
                  <span key={t} className="badge bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/30">
                    {t.replace(/_/g, ' ')}
                  </span>
                ))}
                {dataset.tasks.length > 3 && (
                  <span className="text-xs text-neural-500">+{dataset.tasks.length - 3}</span>
                )}
              </div>
            )}

            {/* Modalities */}
            {dataset.modalities && dataset.modalities.length > 0 && (
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-xs text-neural-500 w-16">Modality:</span>
                {dataset.modalities.slice(0, 3).map((m) => (
                  <span key={m} className="badge bg-accent-violet/10 text-accent-violet border border-accent-violet/30">
                    {m.replace(/_/g, ' ')}
                  </span>
                ))}
                {dataset.modalities.length > 3 && (
                  <span className="text-xs text-neural-500">+{dataset.modalities.length - 3}</span>
                )}
              </div>
            )}

            {/* Behaviors */}
            {dataset.behaviors && dataset.behaviors.length > 0 && (
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-xs text-neural-500 w-16">Behavior:</span>
                {dataset.behaviors.slice(0, 3).map((b) => (
                  <span key={b} className="badge bg-accent-emerald/10 text-accent-emerald border border-accent-emerald/30">
                    {b.replace(/_/g, ' ')}
                  </span>
                ))}
                {dataset.behaviors.length > 3 && (
                  <span className="text-xs text-neural-500">+{dataset.behaviors.length - 3}</span>
                )}
              </div>
            )}

            {/* Species */}
            {dataset.species && dataset.species.length > 0 && (
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-xs text-neural-500 w-16">Species:</span>
                {dataset.species.slice(0, 2).map((s) => (
                  <span key={s} className="badge bg-neural-700 text-neural-300">
                    {s}
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Match reasons */}
          {why_matched.length > 0 && (
            <div className="text-xs text-neural-500 mb-2 bg-neural-800/50 rounded px-2 py-1.5">
              <span className="text-neural-400 font-medium">Why matched: </span>
              {why_matched.slice(0, 2).join(' · ')}
              {why_matched.length > 2 && <span className="text-neural-600"> +{why_matched.length - 2} more</span>}
            </div>
          )}

          {reusable_reason && (
            <div className="text-xs text-neural-500 mb-2">
              <span className="text-neural-400 font-medium">Reusable: </span>
              {reusable_reason}
            </div>
          )}

          {matched_terms && matched_terms.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5 mb-2">
              <span className="text-xs text-neural-500">Matched:</span>
              {matched_terms.slice(0, 4).map((term) => (
                <span key={term} className="badge bg-neural-800 text-neural-300">
                  {term.replace(/_/g, ' ')}
                </span>
              ))}
            </div>
          )}

          {evidence_snippets && evidence_snippets.length > 0 && (
            <div className="text-xs text-neural-500 mb-2 line-clamp-1">
              <span className="text-neural-400 font-medium">Evidence: </span>
              {evidence_snippets[0]}
            </div>
          )}

          {/* Warnings */}
          {warnings.length > 0 && (
            <div className="flex items-center gap-1.5 text-xs text-amber-400 mb-3">
              <WarningIcon className="w-3.5 h-3.5 flex-shrink-0" />
              <span>{warnings[0]}</span>
              {warnings.length > 1 && <span className="text-neural-500">+{warnings.length - 1} more</span>}
            </div>
          )}

          {/* Linked papers */}
          {linked_papers && linked_papers.length > 0 && (
            <div className="flex items-center gap-1.5 text-xs text-neural-400 mb-3">
              <BookOpenIcon className="w-3.5 h-3.5 flex-shrink-0" />
              <span>{linked_papers.length} linked paper{linked_papers.length > 1 ? 's' : ''}</span>
            </div>
          )}

          {/* Action buttons */}
          <div className="flex items-center gap-2 mt-3 pt-3 border-t border-neural-800">
            <Link
              to={`/datasets/${dataset.id}`}
              className="btn-primary text-xs py-1.5 px-3 flex items-center gap-1.5"
              onClick={handleSourceClick}
            >
              View Card
              <ChevronRightIcon className="w-3.5 h-3.5" />
            </Link>
            <button
              onClick={handleNotebookClick}
              disabled={notebookMutation.isPending}
              className="btn-secondary text-xs py-1.5 px-3 flex items-center gap-1.5"
            >
              <CodeIcon className="w-3.5 h-3.5" />
              {notebookMutation.isPending ? 'Generating...' : 'Notebook'}
            </button>
            {dataset.url && (
              <a
                href={dataset.url}
                target="_blank"
                rel="noopener noreferrer"
                onClick={handleSourceClick}
                className="btn-secondary text-xs py-1.5 px-3 flex items-center gap-1.5"
              >
                <ExternalLinkIcon className="w-3.5 h-3.5" />
                Source
              </a>
            )}
          </div>
        </div>

        {/* Score column */}
        <div className="flex flex-col items-end gap-2 ml-6 flex-shrink-0">
          {/* Match score */}
          <div className="text-right">
            <div className="text-2xl font-bold text-accent-cyan">
              {Math.round(score * 100)}
            </div>
            <div className="text-xs text-neural-500">match score</div>
          </div>

          {/* Readiness score */}
          {readiness_score !== undefined && (
            <div className="text-right mt-2">
              <div className="text-lg font-semibold text-accent-emerald">
                {Math.round(readiness_score)}
              </div>
              <div className="text-xs text-neural-500">readiness</div>
            </div>
          )}

          {/* NWB count */}
          {dataset.nwb_count > 0 && (
            <div className="text-right mt-2">
              <span className="badge-cyan text-xs">
                {dataset.nwb_count} NWB
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
