import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { ConsensusRow, FindingRow, GraphNode } from '../../types/graph'
import { TimelinePanel } from './TimelinePanel'
import { FoundationalPapersPanel } from './FoundationalPapersPanel'

interface NodeDetailPanelProps {
  node: GraphNode | null
  findings: FindingRow[]
  consensus: ConsensusRow[]
  onClose: () => void
}

type TopicTab = 'overview' | 'timeline' | 'ancestors'

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

const TOPIC_TABS: { id: TopicTab; label: string }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'timeline', label: 'Timeline' },
  { id: 'ancestors', label: 'Ancestors' },
]

function NodeOverview({
  node,
  findings,
  consensus,
}: {
  node: GraphNode
  findings: FindingRow[]
  consensus: ConsensusRow[]
}) {
  return (
    <div className="min-w-0 flex-1">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-xs uppercase tracking-widest text-neural-600">
          {node.type.replace('_', ' ')}
        </span>
        <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: node.color }} />
      </div>
      <h3 className="text-base font-medium text-neural-100 mb-2 truncate">{node.label}</h3>

      {node.type === 'finding_cluster' && (
        <div className="flex flex-wrap gap-4 text-xs text-neural-400">
          <span>{(node.meta?.n_findings as number) ?? 0} findings</span>
          <span>{(node.meta?.n_papers as number) ?? 0} papers</span>
          <span className={DIRECTION_COLORS[node.meta?.direction as string] ?? ''}>
            {DIRECTION_LABELS[node.meta?.direction as string] ?? (node.meta?.direction as string)}
          </span>
          <span>consensus strength</span>
          <div className="w-32">
            <ConsensusMiniBar strength={(node.meta?.consensus_strength as number) ?? 0} />
          </div>
        </div>
      )}

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

      {node.type === 'dataset' && (
        <div className="text-xs text-neural-500">
          Source: {(node.meta?.source as string) ?? '—'} · Readiness:{' '}
          {(node.meta?.readiness as number) ?? 0}
        </div>
      )}

      {node.type === 'paper' && (
        <div className="text-xs text-neural-500">
          {(node.meta?.year as number) ? `${node.meta?.year as number} · ` : ''}
          {(node.meta?.doi as string) ?? ''}
        </div>
      )}

      {node.type === 'system' && (
        <div className="text-xs text-neural-500">
          {(node.meta?.description as string) ?? 'Topic node'}
        </div>
      )}

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
  )
}

export function NodeDetailPanel({ node, findings, consensus, onClose }: NodeDetailPanelProps) {
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState<TopicTab>('overview')

  if (!node) return null

  const isTopic = node.type === 'system'
  // Derive slug from id (e.g. "topic:working-memory" -> "working-memory", or use label)
  const topicSlug = node.id.includes(':') ? node.id.split(':').slice(1).join(':') : node.id

  const searchQuery =
    node.type === 'finding_cluster'
      ? `${(node.meta?.region as string) ?? ''} ${(node.meta?.direction as string) ?? ''}`.trim()
      : node.type === 'region'
      ? (node.label ?? '')
      : node.type === 'dataset'
      ? node.label
      : ''

  return (
    <div className="absolute bottom-0 left-0 right-0 z-20 bg-neural-950/95 backdrop-blur border-t border-neural-800/50">
      <div className="max-w-5xl mx-auto">
        {/* Header row */}
        <div className="flex items-center gap-3 px-4 pt-3 pb-2 border-b border-neural-800/40">
          {/* Tab pills for topic nodes */}
          {isTopic ? (
            <div className="flex items-center gap-1">
              {TOPIC_TABS.map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setActiveTab(tab.id)}
                  className={`text-xs px-3 py-1 rounded-full border transition-colors ${
                    activeTab === tab.id
                      ? 'bg-neural-800 border-neural-700 text-white'
                      : 'bg-neural-900 border-neural-800 text-neural-500 hover:text-neural-300 hover:border-neural-700'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <span className="text-xs uppercase tracking-widest text-neural-600">
                {node.type.replace('_', ' ')}
              </span>
              <div
                className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ backgroundColor: node.color }}
              />
              <span className="text-sm font-medium text-neural-100 truncate">{node.label}</span>
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center gap-2 ml-auto flex-shrink-0">
            {searchQuery && (
              <button
                type="button"
                onClick={() => navigate(`/search?q=${encodeURIComponent(searchQuery)}`)}
                className="text-xs text-accent-cyan hover:text-white border border-accent-cyan/30 hover:border-accent-cyan rounded px-3 py-1 transition-colors whitespace-nowrap"
              >
                Search datasets →
              </button>
            )}
            {node.type === 'dataset' && (
              <button
                type="button"
                onClick={() => navigate(`/datasets/${node.id.replace('dataset:', '')}`)}
                className="text-xs text-neural-400 hover:text-neural-200 border border-neural-700 rounded px-3 py-1 transition-colors"
              >
                View card →
              </button>
            )}
            <button
              type="button"
              onClick={() => {
                setActiveTab('overview')
                onClose()
              }}
              className="text-xs text-neural-600 hover:text-neural-400"
            >
              Close ✕
            </button>
          </div>
        </div>

        {/* Body */}
        {isTopic ? (
          <div>
            {activeTab === 'overview' && (
              <div className="px-4 py-3">
                <NodeOverview node={node} findings={findings} consensus={consensus} />
              </div>
            )}
            {activeTab === 'timeline' && (
              <TimelinePanel topicId={topicSlug} topicLabel={node.label} />
            )}
            {activeTab === 'ancestors' && (
              <div className="px-4 max-h-[320px] overflow-y-auto">
                <FoundationalPapersPanel topicId={topicSlug} />
              </div>
            )}
          </div>
        ) : (
          <div className="px-4 py-3 flex items-start justify-between gap-6">
            <NodeOverview node={node} findings={findings} consensus={consensus} />
          </div>
        )}
      </div>
    </div>
  )
}
