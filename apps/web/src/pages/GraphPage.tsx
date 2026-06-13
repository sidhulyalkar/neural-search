import { useRef, useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { getCompilationReport } from '../api/search'
import { DownloadIcon } from '../components/Icons'
import type { CompilationReport } from '../types'

// ---- Layout helpers --------------------------------------------------------

function polarToCartesian(cx: number, cy: number, r: number, angleDeg: number) {
  const rad = ((angleDeg - 90) * Math.PI) / 180
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) }
}

interface GraphNode {
  id: string
  label: string
  count: number
  ring: 'modality' | 'region' | 'species' | 'task'
  x: number
  y: number
  r: number
  searchQuery: string
}

function buildNodes(report: CompilationReport, cx: number, cy: number): GraphNode[] {
  const nodes: GraphNode[] = []

  const TOP_N = { modality: 10, region: 12, species: 4, task: 8 }
  const RINGS: Record<GraphNode['ring'], number> = { species: 80, modality: 170, task: 260, region: 360 }
  const MIN_R = 6
  const MAX_R = 22

  function addRing(
    ring: GraphNode['ring'],
    data: Record<string, number>,
    topN: number,
  ) {
    const sorted = Object.entries(data)
      .sort((a, b) => b[1] - a[1])
      .slice(0, topN)
    const max = sorted[0]?.[1] ?? 1
    const r = RINGS[ring]
    sorted.forEach(([id, count], i) => {
      const angle = (360 / sorted.length) * i
      const pos = polarToCartesian(cx, cy, r, angle)
      nodes.push({
        id: `${ring}:${id}`,
        label: id.replace(/_/g, ' '),
        count,
        ring,
        x: pos.x,
        y: pos.y,
        r: MIN_R + ((count / max) * (MAX_R - MIN_R)),
        searchQuery: `${ring === 'modality' ? '' : ring + ': '}${id.replace(/_/g, ' ')}`,
      })
    })
  }

  if (report.datasets_by_modality) addRing('modality', report.datasets_by_modality, TOP_N.modality)
  if (report.datasets_by_brain_region) addRing('region', report.datasets_by_brain_region, TOP_N.region)
  if (report.datasets_by_species) addRing('species', report.datasets_by_species, TOP_N.species)
  if (report.datasets_by_task) addRing('task', report.datasets_by_task, TOP_N.task)

  return nodes
}

const RING_COLORS: Record<GraphNode['ring'], string> = {
  modality: '#22d3ee',
  region: '#8b5cf6',
  species: '#10b981',
  task: '#f59e0b',
}

const RING_LABELS: Record<GraphNode['ring'], string> = {
  modality: 'Modality',
  region: 'Brain Region',
  species: 'Species',
  task: 'Task',
}

function slug(value: string) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'item'
}

function mermaidId(value: string) {
  return slug(value).replace(/-/g, '_')
}

function escapeMarkdownCell(value: string) {
  return value.replace(/\|/g, '\\|').replace(/\n/g, ' ')
}

function escapeMermaidLabel(value: string) {
  return value.replace(/"/g, '\\"')
}

function graphSearchUrl(query: string) {
  if (typeof window === 'undefined') return `/search?q=${encodeURIComponent(query)}`
  return `${window.location.origin}/search?q=${encodeURIComponent(query)}`
}

function downloadTextFile(text: string, filename: string, mimeType = 'text/markdown;charset=utf-8') {
  const blob = new Blob([text], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

function renderGraphTable(nodes: GraphNode[]) {
  const rows = nodes
    .slice()
    .sort((a, b) => a.ring.localeCompare(b.ring) || b.count - a.count || a.label.localeCompare(b.label))
    .map((node) => {
      const label = escapeMarkdownCell(node.label)
      const ring = RING_LABELS[node.ring]
      const search = graphSearchUrl(node.label)
      return `| ${label} | ${ring} | ${node.count} | [Search](${search}) |`
    })

  return [
    '| Concept | Dimension | Dataset count | Neural Search |',
    '| --- | --- | ---: | --- |',
    ...rows,
  ].join('\n')
}

function renderMermaidGraph(nodes: GraphNode[]) {
  const lines = [
    '```mermaid',
    'graph TD',
    '  corpus["Neural Search Corpus"]',
    ...Object.entries(RING_LABELS).map(([ring, label]) => `  ${ring}["${label}"]`),
    ...Object.keys(RING_LABELS).map((ring) => `  corpus --> ${ring}`),
  ]

  const seen = new Set<string>()
  nodes.forEach((node) => {
    let nodeId = `${node.ring}_${mermaidId(node.label)}`
    let suffix = 2
    while (seen.has(nodeId)) {
      nodeId = `${node.ring}_${mermaidId(node.label)}_${suffix}`
      suffix += 1
    }
    seen.add(nodeId)
    lines.push(`  ${nodeId}["${escapeMermaidLabel(node.label)} (${node.count})"]`)
    lines.push(`  ${node.ring} --> ${nodeId}`)
  })

  lines.push('```')
  return lines.join('\n')
}

function renderTopList(title: string, data: Record<string, number> | undefined) {
  if (!data) return ''
  const rows = Object.entries(data)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 12)
    .map(([key, count]) => `| ${escapeMarkdownCell(key.replace(/_/g, ' '))} | ${count} |`)

  return [`## ${title}`, '', '| Value | Datasets |', '| --- | ---: |', ...rows].join('\n')
}

function renderObsidianGraphNote(report: CompilationReport, nodes: GraphNode[]) {
  const generatedAt = new Date().toISOString()
  const sections = [
    '---',
    'type: neural_search_corpus_graph',
    `generated_at: ${generatedAt}`,
    `source_report_generated_at: ${report.generated_at}`,
    `total_datasets: ${report.total_datasets}`,
    'tags:',
    '  - neural-search',
    '  - knowledge-graph',
    '  - obsidian-import',
    '---',
    '',
    '# Neural Search Corpus Knowledge Graph',
    '',
    `Generated from Neural Search on ${generatedAt}.`,
    '',
    `Total datasets: ${report.total_datasets}`,
    '',
    '## Obsidian Graph View',
    '',
    renderMermaidGraph(nodes),
    '',
    '## Searchable Concepts',
    '',
    renderGraphTable(nodes),
    '',
    renderTopList('Top Modalities', report.datasets_by_modality),
    '',
    renderTopList('Top Brain Regions', report.datasets_by_brain_region),
    '',
    renderTopList('Top Species', report.datasets_by_species),
    '',
    renderTopList('Top Tasks', report.datasets_by_task),
    '',
    '## Archive Breakdown',
    '',
    renderTopList('Sources', report.datasets_by_source),
    '',
    '## Notes',
    '',
    '- Drop this Markdown file into an Obsidian vault to render the Mermaid graph and tables.',
    '- Search links open the same concept in the Neural Search frontend when the app is running.',
  ]

  return sections.join('\n')
}

// ---- Component -------------------------------------------------------------

export function GraphPage() {
  const navigate = useNavigate()
  const svgRef = useRef<SVGSVGElement>(null)
  const [hovered, setHovered] = useState<GraphNode | null>(null)
  const [svgSize, setSvgSize] = useState({ w: 800, h: 800 })
  const [exportStatus, setExportStatus] = useState<string | null>(null)

  const { data: report, isLoading, error } = useQuery<CompilationReport>({
    queryKey: ['compilation-report'],
    queryFn: getCompilationReport,
    retry: false,
  })

  useEffect(() => {
    const el = svgRef.current
    if (!el) return
    const obs = new ResizeObserver((entries) => {
      const { width } = entries[0].contentRect
      const size = Math.max(440, Math.min(width, 820))
      setSvgSize({ w: size, h: size })
    })
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  const cx = svgSize.w / 2
  const cy = svgSize.h / 2
  const nodes = report ? buildNodes(report, cx, cy) : []

  const exportObsidianNote = () => {
    if (!report || nodes.length === 0) return
    const markdown = renderObsidianGraphNote(report, nodes)
    downloadTextFile(markdown, 'neural-search-corpus-knowledge-graph.md')
    setExportStatus('Obsidian Markdown exported.')
  }

  return (
    <div className="max-w-5xl mx-auto px-6 lg:px-8 py-16">
      {/* Header */}
      <div className="mb-16 flex flex-col gap-6 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="mb-3">
            <span className="font-mono text-xs text-neural-600 tracking-widest uppercase">
              Corpus Map · {report?.total_datasets ?? '…'} datasets
            </span>
          </div>
          <h1 className="text-4xl sm:text-5xl font-extralight tracking-tight text-neural-100 mb-4">
            Corpus Visualization
          </h1>
          <p className="text-neural-500 text-sm max-w-xl leading-relaxed">
            Neural Search indexes datasets across DANDI and OpenNeuro. Each ring represents a
            concept dimension — click any node to search the corpus for datasets in that category.
          </p>
        </div>
        <div className="flex flex-col items-start sm:items-end gap-2">
          <button
            type="button"
            onClick={exportObsidianNote}
            disabled={!report || nodes.length === 0}
            className="inline-flex items-center gap-2 px-4 py-2 rounded border border-neural-700/70 text-sm text-neural-200 hover:border-accent-cyan/70 hover:text-white disabled:cursor-not-allowed disabled:opacity-40 transition-colors"
          >
            <DownloadIcon className="w-4 h-4" />
            Obsidian Markdown
          </button>
          {exportStatus && (
            <span className="text-xs text-accent-emerald" role="status">
              {exportStatus}
            </span>
          )}
        </div>
      </div>

      {/* Stats row */}
      {report && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-px bg-neural-800/30 border border-neural-800/30 rounded-lg overflow-hidden mb-16">
          <StatCell value={report.total_datasets} label="total datasets" />
          <StatCell value={Object.keys(report.datasets_by_modality || {}).length} label="modalities" />
          <StatCell value={Object.keys(report.datasets_by_brain_region || {}).length} label="brain regions" />
          <StatCell value={Object.keys(report.datasets_by_species || {}).length} label="species" />
        </div>
      )}

      {/* SVG visualization */}
      <div className="mb-12 flex justify-center">
        <div className="w-full max-w-[820px] relative">
          <svg
            ref={svgRef}
            width="100%"
            viewBox={`0 0 ${svgSize.w} ${svgSize.h}`}
            className="overflow-visible"
            style={{ height: svgSize.h }}
          >
            {/* Ring guides */}
            {([80, 170, 260, 360] as const).map((r) => (
              <circle
                key={r}
                cx={cx}
                cy={cy}
                r={r}
                fill="none"
                stroke="#1e3a4f"
                strokeWidth="1"
                strokeDasharray="3 6"
                opacity={0.4}
              />
            ))}

            {/* Spokes from center to nodes */}
            {nodes.map((node) => (
              <line
                key={`spoke-${node.id}`}
                x1={cx}
                y1={cy}
                x2={node.x}
                y2={node.y}
                stroke={RING_COLORS[node.ring]}
                strokeWidth={hovered?.id === node.id ? 1.5 : 0.5}
                opacity={hovered ? (hovered.id === node.id ? 0.5 : 0.08) : 0.15}
                className="transition-all duration-150"
              />
            ))}

            {/* Center node */}
            <circle cx={cx} cy={cy} r={18} fill="#0a1929" stroke="#22d3ee" strokeWidth={1.5} opacity={0.8} />
            <text x={cx} y={cy + 1} textAnchor="middle" dominantBaseline="middle" fill="#22d3ee" fontSize="6" fontFamily="monospace" letterSpacing="0.5">
              NS
            </text>

            {/* Concept nodes */}
            {nodes.map((node) => {
              const isHovered = hovered?.id === node.id
              const dimmed = hovered && !isHovered
              const color = RING_COLORS[node.ring]
              return (
                <g
                  key={node.id}
                  transform={`translate(${node.x},${node.y})`}
                  className="cursor-pointer"
                  onMouseEnter={() => setHovered(node)}
                  onMouseLeave={() => setHovered(null)}
                  onClick={() => navigate(`/search?q=${encodeURIComponent(node.label)}`)}
                  opacity={dimmed ? 0.2 : 1}
                  style={{ transition: 'opacity 0.15s' }}
                >
                  <circle
                    r={isHovered ? node.r + 3 : node.r}
                    fill={color}
                    fillOpacity={isHovered ? 0.25 : 0.1}
                    stroke={color}
                    strokeWidth={isHovered ? 1.5 : 0.8}
                    style={{ transition: 'r 0.1s' }}
                  />
                  {/* Label — only show if not too crowded or if hovered */}
                  {(isHovered || node.r > 12) && (
                    <text
                      y={node.r + 10}
                      textAnchor="middle"
                      fill={isHovered ? color : '#627d98'}
                      fontSize={isHovered ? 10 : 8}
                      fontFamily="Inter, sans-serif"
                      style={{ pointerEvents: 'none', transition: 'fill 0.1s' }}
                    >
                      {node.label.length > 18 ? node.label.slice(0, 16) + '…' : node.label}
                    </text>
                  )}
                </g>
              )
            })}

            {/* Hover tooltip */}
            {hovered && (() => {
              const tx = hovered.x > cx ? hovered.x - 10 : hovered.x + 10
              const anchor = hovered.x > cx ? 'end' : 'start'
              const ty = hovered.y > cy ? hovered.y - 16 : hovered.y + 26
              return (
                <g>
                  <text x={tx} y={ty} textAnchor={anchor} fill="white" fontSize="11" fontFamily="Inter, sans-serif" fontWeight="500">
                    {hovered.label}
                  </text>
                  <text x={tx} y={ty + 14} textAnchor={anchor} fill="#627d98" fontSize="9" fontFamily="Inter, sans-serif">
                    {hovered.count} datasets · click to search
                  </text>
                </g>
              )
            })()}
          </svg>
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap justify-center gap-6 mb-16">
        {(Object.entries(RING_COLORS) as [GraphNode['ring'], string][]).map(([ring, color]) => (
          <div key={ring} className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color, opacity: 0.7 }} />
            <span className="text-xs text-neural-500">{RING_LABELS[ring]}</span>
          </div>
        ))}
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-neural-600" />
          <span className="text-xs text-neural-500">node size ∝ dataset count</span>
        </div>
      </div>

      {/* Top modalities table */}
      {report && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-8 mb-16">
          <TopList
            title="Modalities"
            color="#22d3ee"
            data={report.datasets_by_modality}
            total={report.total_datasets}
            onRowClick={(key) => navigate(`/search?q=${encodeURIComponent(key.replace(/_/g, ' '))}`)}
          />
          <TopList
            title="Brain Regions"
            color="#8b5cf6"
            data={report.datasets_by_brain_region}
            total={report.total_datasets}
            onRowClick={(key) => navigate(`/search?q=${encodeURIComponent(key.replace(/_/g, ' '))}`)}
          />
          <TopList
            title="Species"
            color="#10b981"
            data={report.datasets_by_species}
            total={report.total_datasets}
            onRowClick={(key) => navigate(`/search?q=${encodeURIComponent(key)}`)}
          />
          <TopList
            title="Tasks"
            color="#f59e0b"
            data={report.datasets_by_task}
            total={report.total_datasets}
            onRowClick={(key) => navigate(`/search?q=${encodeURIComponent(key.replace(/_/g, ' '))}`)}
          />
        </div>
      )}

      {/* Source breakdown */}
      {report && (
        <div className="mb-16">
          <p className="text-xs text-neural-600 uppercase tracking-widest mb-4">By Archive</p>
          <div className="flex flex-wrap gap-4">
            {Object.entries(report.datasets_by_source || {})
              .sort((a, b) => b[1] - a[1])
              .map(([src, count]) => (
                <button
                  key={src}
                  onClick={() => navigate(`/search?q=${encodeURIComponent(src)}`)}
                  className="flex items-center gap-2 px-4 py-2 border border-neural-800/50 rounded hover:border-neural-600 transition-colors group"
                >
                  <span className="font-mono text-sm text-neural-200 group-hover:text-white transition-colors">
                    {src.toUpperCase()}
                  </span>
                  <span className="text-sm text-neural-600">{count}</span>
                </button>
              ))}
          </div>
        </div>
      )}

      {/* Loading / error */}
      {isLoading && (
        <div className="py-16 text-center">
          <span className="inline-flex items-center gap-3 text-neural-500 text-sm">
            <span className="w-4 h-4 border-2 border-neural-700 border-t-accent-cyan rounded-full animate-spin" />
            Loading corpus data…
          </span>
        </div>
      )}

      {error && (
        <div className="py-10 border border-neural-800/50 rounded-lg px-6">
          <p className="text-neural-400 text-sm mb-1">Could not load corpus data</p>
          <p className="text-neural-600 text-xs">Make sure the API is running: <code className="font-mono text-neural-400">make api</code></p>
        </div>
      )}
    </div>
  )
}

function StatCell({ value, label }: { value: number; label: string }) {
  return (
    <div className="bg-neural-950 px-5 py-4">
      <div className="text-2xl font-extralight text-neural-100 tabular-nums mb-0.5">{value}</div>
      <div className="text-xs text-neural-600">{label}</div>
    </div>
  )
}

function TopList({
  title,
  color,
  data,
  total,
  onRowClick,
}: {
  title: string
  color: string
  data: Record<string, number> | undefined
  total: number
  onRowClick: (key: string) => void
}) {
  if (!data) return null
  const sorted = Object.entries(data).sort((a, b) => b[1] - a[1]).slice(0, 8)
  const max = sorted[0]?.[1] ?? 1

  return (
    <div>
      <p className="text-xs uppercase tracking-widest mb-3" style={{ color }}>
        {title}
      </p>
      <div className="space-y-1">
        {sorted.map(([key, count]) => (
          <button
            key={key}
            onClick={() => onRowClick(key)}
            className="w-full flex items-center gap-3 group"
          >
            <div className="flex-1 flex items-center gap-2 min-w-0">
              {/* Bar */}
              <div className="h-1.5 rounded-full flex-shrink-0" style={{
                width: `${Math.max(4, (count / max) * 80)}px`,
                backgroundColor: color,
                opacity: 0.4,
              }} />
              <span className="text-xs text-neural-400 group-hover:text-neural-200 transition-colors truncate">
                {key.replace(/_/g, ' ')}
              </span>
            </div>
            <span className="text-xs text-neural-700 tabular-nums flex-shrink-0">
              {count}
            </span>
            <span className="text-xs text-neural-800 tabular-nums w-8 text-right flex-shrink-0">
              {((count / total) * 100).toFixed(0)}%
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}
