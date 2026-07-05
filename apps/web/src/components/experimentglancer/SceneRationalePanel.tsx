import type { ExperimentGlancerScene } from '../../api/experimentglancer'

interface SceneRationalePanelProps {
  scene: ExperimentGlancerScene
}

/** Bridges retrieval intelligence and the visualization: explains, in one
 * place, why this specific scene (this anchor, these layers) was produced
 * for this dataset -- rather than leaving the user to infer it from the
 * layer chips and warnings. */
export function SceneRationalePanel({ scene }: SceneRationalePanelProps) {
  const { source, query_context, anchors, provenance } = scene
  const primaryAnchor = anchors[0]

  const matchClauses: string[] = []
  if (source.query) matchClauses.push(`the query "${source.query}"`)
  if (query_context?.affordance_ids?.length) {
    matchClauses.push(`matched affordances ${query_context.affordance_ids.join(', ')}`)
  }
  if (query_context?.requested_layers?.length) {
    matchClauses.push(`requested layers ${query_context.requested_layers.join(', ')}`)
  }

  const retrievalClause =
    source.kind === 'search_result'
      ? `a ${source.retrieval_method || 'search'} match${
          typeof source.rank === 'number' ? ` (rank ${source.rank})` : ''
        }${typeof source.score === 'number' ? ` scoring ${source.score.toFixed(2)}` : ''}`
      : source.kind === 'dataset_card'
      ? 'this dataset’s own card'
      : 'a manual request'

  return (
    <div className="text-xs text-neural-400 leading-relaxed space-y-1.5">
      <p>
        Created from {retrievalClause}
        {matchClauses.length > 0 ? ` that ${matchClauses.join(' and ')}.` : '.'}
      </p>
      {primaryAnchor && (
        <p>
          Opens on <span className="text-neural-200">{primaryAnchor.label}</span> — {primaryAnchor.reason}
        </p>
      )}
      <p className="text-neural-600">
        Highest evidence tier available: <span className="text-neural-400">{provenance.evidence_tier}</span>
        {provenance.missing_requirements.length > 0 &&
          ` · still missing: ${provenance.missing_requirements.join(', ')}`}
      </p>
    </div>
  )
}
