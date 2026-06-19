import { useNavigate } from 'react-router-dom'
import type { ConsensusRow, FindingRow, GraphNode } from '../../types/graph'

interface NodeDetailPanelProps {
  node: GraphNode | null
  findings: FindingRow[]
  consensus: ConsensusRow[]
  onClose: () => void
}

const DIRECTION_LABELS: Record<string, string> = {
  increase: '↑ increase',
  decrease: '↓ decrease',
  correlation: '↔ correlation',
  no_change: '— no change',
}
const DIRECTION_COLORS: Record<string, string> = {
  increase: 'text-accent-emerald',
  decrease: 'text-red-400',
  correlation: 'text-accent-violet',
  no_change: 'text-neural-500',
}

function ConsensusMiniBar({ strength }: { strength: number }) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1 bg-neural-800 rounded overflow-hidden">
        <div
          className="h-full bg-accent-emerald rounded transition-all"
          style={{ width: `${Math.round(strength * 100)}%` }}
        />
      </div>
      <span className="text-xs text-neural-500 tabular-nums w-8 text-right">
        {(strength * 100).toFixed(0)}%
      </span>
    </div>
  )
}

export function NodeDetailPanel({ node, findings, consensus, onClose }: NodeDetailPanelProps) {
  const navigate = useNavigate()

  if (!node) return null

  const searchQuery = node.type === 'finding_cluster'
    ? `${(node.meta.region as string) ?? ''} ${(node.meta.direction as string) ?? ''}`.trim()
    : node.type === 'region'
    ? (node.label ?? '')
    : node.type === 'dataset'
    ? node.label
    : ''

  return (
    <div className="absolute bottom-0 left-0 right-0 z-20 bg-neural-950/95 backdrop-blur border-t border-neural-800/50 p-4">
      <div className="max-w-5xl mx-auto">
        <div className="flex items-start justify-between gap-6">
          {/* Left: node info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs uppercase tracking-widest text-neural-600">{node.type.replace('_', ' ')}</span>
              <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: node.color }} />
            </div>
            <h3 className="text-base font-medium text-neural-100 mb-2 truncate">{node.label}</h3>

            {/* Finding cluster details */}
            {node.type === 'finding_cluster' && (
              <div className="flex flex-wrap gap-4 text-xs text-neural-400">
                <span>{node.meta.n_findings as number} findings</span>
                <span>{node.meta.n_papers as number} papers</span>
                <span className={DIRECTION_COLORS[node.meta.direction as string] ?? ''}>
                  {DIRECTION_LABELS[node.meta.direction as string] ?? node.meta.direction as string}
                </span>
                <span>consensus strength</span>
                <div className="w-32">
                  <ConsensusMiniBar strength={node.meta.consensus_strength as number ?? 0} />
                </div>
              </div>
            )}

            {/* Region details */}
            {node.type === 'region' && consensus.length > 0 && (
              <div className="flex flex-wrap gap-3 mt-1">
                {consensus.slice(0, 4).map((c) => (
                  <div key={`${c.region}-${c.direction}`} className="flex items-center gap-1.5">
                    <span className={`text-xs ${DIRECTION_COLORS[c.direction] ?? 'text-neural-400'}`}>
                      {DIRECTION_LABELS[c.direction]}
                    </span>
                    <span className="text-xs text-neural-600">{c.n_findings} findings</span>
                  </div>
                ))}
              </div>
            )}

            {/* Dataset details */}
            {node.type === 'dataset' && (
              <div className="text-xs text-neural-500">
                Source: {(node.meta.source as string) ?? '—'} · Readiness: {(node.meta.readiness as number) ?? 0}
              </div>
            )}

            {/* Paper details */}
            {node.type === 'paper' && (
              <div className="text-xs text-neural-500">
                {(node.meta.year as number) ? `${node.meta.year as number} · ` : ''}
                {(node.meta.doi as string) ?? ''}
              </div>
            )}

            {/* Top findings */}
            {findings.length > 0 && (
              <div className="mt-2 space-y-1">
                {findings.slice(0, 2).map((f) => (
                  <p key={f.finding_id} className="text-xs text-neural-500 line-clamp-1 italic">
                    "{f.finding_text}"
                  </p>
                ))}
              </div>
            )}
          </div>

          {/* Right: actions */}
          <div className="flex flex-col gap-2 flex-shrink-0">
            {searchQuery && (
              <button
                type="button"
                onClick={() => navigate(`/search?q=${encodeURIComponent(searchQuery)}`)}
                className="text-xs text-accent-cyan hover:text-white border border-accent-cyan/30 hover:border-accent-cyan rounded px-3 py-1.5 transition-colors whitespace-nowrap"
              >
                Search datasets →
              </button>
            )}
            {node.type === 'dataset' && (
              <button
                type="button"
                onClick={() => navigate(`/datasets/${node.id.replace('dataset:', '')}`)}
                className="text-xs text-neural-400 hover:text-neural-200 border border-neural-700 rounded px-3 py-1.5 transition-colors"
              >
                View card →
              </button>
            )}
            <button
              type="button"
              onClick={onClose}
              className="text-xs text-neural-600 hover:text-neural-400"
            >
              Close ✕
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
