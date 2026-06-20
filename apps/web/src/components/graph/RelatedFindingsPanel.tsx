import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { fetchDatasetNeighborhood } from '../../api/graph'
import type { ConsensusRow } from '../../types/graph'
import type { LinkedPaper } from '../../types'

interface RelatedFindingsPanelProps {
  datasetId: string
  brainRegions: string[]
  linkedPapers: LinkedPaper[]
}

const DIRECTION_LABELS: Record<string, string> = {
  increase: '↑',
  decrease: '↓',
  correlation: '↔',
  no_change: '—',
}
const DIRECTION_COLORS: Record<string, string> = {
  increase: 'text-accent-emerald border-accent-emerald/30 bg-accent-emerald/5',
  decrease: 'text-red-400 border-red-500/30 bg-red-500/5',
  correlation: 'text-accent-violet border-accent-violet/30 bg-accent-violet/5',
  no_change: 'text-neural-500 border-neural-700 bg-neural-900',
}

function ConsensusBadge({ row }: { row: ConsensusRow }) {
  const cls = DIRECTION_COLORS[row.direction] ?? 'text-neural-500 border-neural-700 bg-neural-900'
  return (
    <span className={`inline-flex items-center gap-1 text-xs border rounded px-2 py-0.5 ${cls}`}>
      {row.region} {DIRECTION_LABELS[row.direction]} {(row.consensus_strength * 100).toFixed(0)}%
      <span className="text-neural-600">({row.n_findings})</span>
    </span>
  )
}

function MiniGraph({ datasetId: _datasetId, paperCount, clusterCount }: { datasetId: string; paperCount: number; clusterCount: number }) {
  // Static SVG schematic — not force-simulated
  const paperNodes = Array.from({ length: Math.min(paperCount, 4) }, (_, i) => ({
    cx: 40 + i * 60,
    cy: 30,
  }))
  const clusterNodes = Array.from({ length: Math.min(clusterCount, 6) }, (_, i) => ({
    cx: 20 + i * 46,
    cy: 80,
    color: i % 2 === 0 ? '#10b981' : '#ef4444',
  }))

  return (
    <svg width="280" height="110" className="overflow-visible">
      {/* Paper nodes */}
      {paperNodes.map((p, i) => (
        <g key={i}>
          <circle cx={p.cx} cy={p.cy} r={10} fill="#8b5cf620" stroke="#8b5cf6" strokeWidth={1} />
          <text x={p.cx} y={p.cy + 1} textAnchor="middle" dominantBaseline="middle" fill="#8b5cf6" fontSize={6}>P</text>
        </g>
      ))}
      {/* Cluster nodes */}
      {clusterNodes.map((c, i) => (
        <g key={i}>
          {paperNodes.slice(0, 2).map((p, j) => (
            <line key={j} x1={p.cx} y1={p.cy + 10} x2={c.cx} y2={c.cy - 6} stroke="#ffffff15" strokeWidth={0.5} />
          ))}
          <circle cx={c.cx} cy={c.cy} r={7} fill={`${c.color}20`} stroke={c.color} strokeWidth={1} />
        </g>
      ))}
      <text x={140} y={105} textAnchor="middle" fill="#374151" fontSize={7}>
        papers → finding clusters
      </text>
    </svg>
  )
}

export function RelatedFindingsPanel({ datasetId, brainRegions: _brainRegions, linkedPapers: _linkedPapers }: RelatedFindingsPanelProps) {
  const { data: neighborhood, isLoading } = useQuery({
    queryKey: ['dataset-neighborhood', datasetId],
    queryFn: () => fetchDatasetNeighborhood(datasetId),
    staleTime: 300_000,
    enabled: true,
  })

  if (isLoading) {
    return (
      <div className="mt-4 border border-neural-800/60 rounded-lg p-4 animate-pulse">
        <div className="h-3 w-32 bg-neural-800 rounded mb-2" />
        <div className="h-3 w-48 bg-neural-800 rounded" />
      </div>
    )
  }

  const consensusRows = neighborhood?.consensus_by_region ?? []
  const findings = neighborhood?.finding_clusters ?? []
  const papers = neighborhood?.linked_papers ?? []

  if (consensusRows.length === 0 && papers.length === 0) {
    return (
      <div className="mt-4 border border-neural-800/40 rounded-lg p-3">
        <p className="text-xs text-neural-600">No literature links found for this dataset yet.</p>
      </div>
    )
  }

  return (
    <div className="mt-4 border border-neural-800/60 rounded-lg p-4 bg-neural-950/50">
      <p className="text-xs uppercase tracking-wide text-neural-600 mb-3">Related Findings</p>

      {/* Consensus badges */}
      {consensusRows.length > 0 && (
        <div className="mb-3">
          <p className="text-xs text-neural-600 mb-1.5">Consensus by region</p>
          <div className="flex flex-wrap gap-1.5">
            {consensusRows.slice(0, 4).map((row) => (
              <ConsensusBadge key={`${row.region}-${row.direction}`} row={row} />
            ))}
          </div>
        </div>
      )}

      {/* Mini graph schematic */}
      {(papers.length > 0 || findings.length > 0) && (
        <div className="mb-3 overflow-hidden">
          <MiniGraph
            datasetId={datasetId}
            paperCount={papers.length}
            clusterCount={findings.length}
          />
        </div>
      )}

      {/* Finding snippets placeholder (top clusters only) */}
      {findings.slice(0, 2).map((f) => (
        <p key={`${f.region}-${f.direction}`} className="text-xs text-neural-500 italic mb-1">
          "{f.region}: {f.n_findings} findings toward {f.direction} (strength {(f.consensus_strength * 100).toFixed(0)}%)"
        </p>
      ))}

      {/* Deep-link */}
      <div className="mt-3">
        <Link
          to={`/graph?dataset=${encodeURIComponent(datasetId)}`}
          className="text-xs text-accent-cyan hover:text-white transition-colors"
        >
          View neighborhood in Knowledge Graph →
        </Link>
      </div>
    </div>
  )
}
