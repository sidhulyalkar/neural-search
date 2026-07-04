import { Link } from 'react-router-dom'
import type { QueryContext, SceneDatasetRef, SceneSource } from '../../api/experimentglancer'
import { GlassPanel } from './GlassPanel'

interface SceneHeroHeaderProps {
  dataset: SceneDatasetRef
  source: SceneSource
  queryContext: QueryContext | null
}

export function SceneHeroHeader({ dataset, source, queryContext }: SceneHeroHeaderProps) {
  const tags = [...dataset.species, ...dataset.modalities, ...dataset.brain_regions, ...dataset.tasks]
    .map((tag) => tag.replace(/_/g, ' '))
    .filter(Boolean)
    .slice(0, 8)

  return (
    <GlassPanel tone="cyan" className="p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-widest text-accent-cyan/80 mb-2">
            {(dataset.source ?? 'dataset').toUpperCase()} · {dataset.source_id ?? dataset.dataset_id}
            {dataset.data_standard ? ` · ${dataset.data_standard}` : ''}
          </p>
          <h1 className="text-2xl font-medium text-neural-50 mb-2">
            {dataset.title ?? dataset.dataset_id}
          </h1>
          {queryContext?.query && (
            <p className="text-sm text-neural-400 italic">&ldquo;{queryContext.query}&rdquo;</p>
          )}
        </div>

        <div className="text-right flex-shrink-0">
          {typeof source.score === 'number' && (
            <div>
              <div className="text-2xl font-light tabular-nums text-neural-100">
                {Math.round(source.score * 100)}
              </div>
              <div className="text-xs text-neural-600">match score</div>
            </div>
          )}
          {source.retrieval_method && (
            <p className="mt-1 text-xs text-neural-600">{source.retrieval_method}</p>
          )}
        </div>
      </div>

      {tags.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-1.5">
          {tags.map((tag) => (
            <span
              key={tag}
              className="text-xs rounded px-2 py-0.5 border border-white/10 bg-white/[0.03] text-neural-400"
            >
              {tag}
            </span>
          ))}
        </div>
      )}

      <div className="mt-4 flex items-center gap-4">
        <Link
          to={`/datasets/${encodeURIComponent(dataset.dataset_id)}`}
          className="text-xs text-accent-cyan hover:text-white transition-colors"
        >
          View dataset card →
        </Link>
        {dataset.url && (
          <a
            href={dataset.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-neural-500 hover:text-neural-300 transition-colors"
          >
            Source ↗
          </a>
        )}
      </div>
    </GlassPanel>
  )
}
