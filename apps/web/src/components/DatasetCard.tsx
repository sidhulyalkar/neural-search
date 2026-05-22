import { Link } from 'react-router-dom'
import { ChevronRightIcon, WarningIcon } from './Icons'
import type { SearchResultItem } from '../types'

interface DatasetCardProps {
  result: SearchResultItem
}

export function DatasetCard({ result }: DatasetCardProps) {
  const { dataset, score, why_matched, warnings } = result

  const sourceColors: Record<string, string> = {
    dandi: 'badge-cyan',
    openneuro: 'badge-violet',
    other: 'badge-emerald',
  }

  return (
    <Link
      to={`/datasets/${dataset.id}`}
      className="card-hover block group"
    >
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          {/* Title and source */}
          <div className="flex items-center gap-2 mb-2">
            <span className={sourceColors[dataset.source] || 'badge-emerald'}>
              {dataset.source.toUpperCase()}
            </span>
            <h3 className="text-lg font-medium text-neural-100 truncate group-hover:text-accent-cyan transition-colors">
              {dataset.title}
            </h3>
          </div>

          {/* Description */}
          {dataset.description && (
            <p className="text-sm text-neural-400 line-clamp-2 mb-3">
              {dataset.description}
            </p>
          )}

          {/* Metadata badges */}
          <div className="flex flex-wrap gap-2 mb-3">
            {dataset.species?.slice(0, 2).map((s) => (
              <span key={s} className="badge bg-neural-700 text-neural-300">
                {s}
              </span>
            ))}
            {dataset.tasks?.slice(0, 2).map((t) => (
              <span key={t} className="badge bg-neural-700 text-neural-300">
                {t}
              </span>
            ))}
            {dataset.modalities?.slice(0, 2).map((m) => (
              <span key={m} className="badge bg-neural-700 text-neural-300">
                {m}
              </span>
            ))}
            {dataset.nwb_count > 0 && (
              <span className="badge-cyan">{dataset.nwb_count} NWB files</span>
            )}
          </div>

          {/* Match reasons */}
          {why_matched.length > 0 && (
            <div className="text-xs text-neural-500 mb-2">
              <span className="text-neural-400">Matched:</span>{' '}
              {why_matched.slice(0, 2).join(' | ')}
            </div>
          )}

          {/* Warnings */}
          {warnings.length > 0 && (
            <div className="flex items-center gap-1 text-xs text-amber-400">
              <WarningIcon className="w-3 h-3" />
              {warnings[0]}
            </div>
          )}
        </div>

        {/* Score and arrow */}
        <div className="flex items-center gap-4 ml-4">
          <div className="text-right">
            <div className="text-2xl font-bold text-accent-cyan">
              {Math.round(score * 100)}
            </div>
            <div className="text-xs text-neural-500">match</div>
          </div>
          <ChevronRightIcon className="w-5 h-5 text-neural-500 group-hover:text-accent-cyan transition-colors" />
        </div>
      </div>
    </Link>
  )
}
