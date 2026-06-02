import { Link } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { generateNotebook } from '../api/search'
import type { SearchResultItem } from '../types'

interface DatasetCardProps {
  result: SearchResultItem
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

export function DatasetCard({
  result,
  isSelected = false,
  onToggleSelect,
  selectionDisabled = false,
}: DatasetCardProps) {
  const { dataset, score, why_matched, warnings, readiness_score, linked_papers } = result

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

  const tags = [
    ...(dataset.species || []).slice(0, 2),
    ...(dataset.modalities || []).slice(0, 2),
    ...(dataset.brain_regions || []).slice(0, 2),
    ...(dataset.tasks || []).slice(0, 2),
  ]
    .map((t) => t.replace(/_/g, ' '))
    .filter(Boolean)
    .slice(0, 7)

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

            {dataset.data_standard && (
              <span className="font-mono text-xs text-neural-600">{dataset.data_standard}</span>
            )}

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
            <p className="text-xs text-neural-600 mb-3 line-clamp-1">
              {why_matched.slice(0, 3).join(' · ')}
            </p>
          )}

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

          {notebookMutation.error && (
            <p className="mt-2 text-xs text-red-400">
              {notebookMutation.error instanceof Error
                ? notebookMutation.error.message
                : 'Notebook generation failed.'}
            </p>
          )}
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
