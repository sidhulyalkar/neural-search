import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchTopicTimeline } from '../../api/graph'
import type { TimelineYear } from '../../types/graph'

interface TimelinePanelProps {
  topicId: string
  topicLabel: string
}

const BAR_COLORS = {
  papers: '#22d3ee',
  findings: '#8b5cf6',
  datasets: '#10b981',
} as const

function YearBar({
  entry,
  maxVal,
  onHover,
  isHovered,
}: {
  entry: TimelineYear
  maxVal: number
  onHover: (entry: TimelineYear | null) => void
  isHovered: boolean
}) {
  const scale = maxVal > 0 ? 100 / maxVal : 0
  const paperH = Math.max(2, entry.n_papers * scale)
  const findingH = Math.max(2, entry.n_findings * scale)
  const datasetH = Math.max(2, entry.n_datasets * scale)

  return (
    <button
      type="button"
      className={`flex flex-col items-center gap-0.5 min-w-[28px] flex-shrink-0 cursor-pointer group outline-none ${
        isHovered ? 'opacity-100' : 'opacity-70 hover:opacity-100'
      }`}
      onMouseEnter={() => onHover(entry)}
      onMouseLeave={() => onHover(null)}
      onFocus={() => onHover(entry)}
      onBlur={() => onHover(null)}
    >
      <div className="flex flex-col-reverse items-center gap-px w-full h-[120px] justify-end">
        <div
          className="w-full rounded-t-sm transition-all"
          style={{ height: `${paperH}%`, backgroundColor: BAR_COLORS.papers }}
        />
        <div
          className="w-full transition-all"
          style={{ height: `${findingH}%`, backgroundColor: BAR_COLORS.findings }}
        />
        <div
          className="w-full rounded-t-sm transition-all"
          style={{ height: `${datasetH}%`, backgroundColor: BAR_COLORS.datasets }}
        />
      </div>
      <span
        className={`text-[9px] font-mono rotate-45 origin-left mt-1 whitespace-nowrap ${
          isHovered ? 'text-neural-200' : 'text-neural-600'
        }`}
      >
        {entry.year}
      </span>
    </button>
  )
}

function HoverTooltip({ entry }: { entry: TimelineYear }) {
  return (
    <div className="bg-neural-900 border border-neural-700 rounded p-3 text-xs space-y-2 min-w-[220px] max-w-[280px]">
      <div className="font-mono text-neural-300 font-medium">{entry.year}</div>
      <div className="flex gap-3 text-neural-400">
        <span style={{ color: BAR_COLORS.papers }}>{entry.n_papers} papers</span>
        <span style={{ color: BAR_COLORS.findings }}>{entry.n_findings} findings</span>
        <span style={{ color: BAR_COLORS.datasets }}>{entry.n_datasets} datasets</span>
      </div>
      {entry.top_papers.length > 0 && (
        <div className="space-y-1">
          <span className="text-neural-600 uppercase tracking-widest text-[9px]">Top papers</span>
          {entry.top_papers.slice(0, 2).map((p) => (
            <p key={p.id} className="text-neural-400 line-clamp-2 leading-snug">
              {p.title}
            </p>
          ))}
        </div>
      )}
      {entry.top_findings.length > 0 && (
        <div className="space-y-1">
          <span className="text-neural-600 uppercase tracking-widest text-[9px]">Top findings</span>
          {entry.top_findings.slice(0, 2).map((f) => (
            <p key={f.text} className="text-neural-500 italic line-clamp-1 leading-snug">
              "{f.text}"
            </p>
          ))}
        </div>
      )}
    </div>
  )
}

export function TimelinePanel({ topicId, topicLabel }: TimelinePanelProps) {
  const [hoveredEntry, setHoveredEntry] = useState<TimelineYear | null>(null)

  const { data: timeline, isLoading } = useQuery({
    queryKey: ['topic-timeline', topicId],
    queryFn: () => fetchTopicTimeline(topicId, 1990, 2025),
    staleTime: 300_000,
  })

  if (isLoading) {
    return (
      <div className="h-[280px] flex items-center justify-center text-neural-600 text-xs font-mono">
        <span className="w-3 h-3 border border-neural-700 border-t-accent-cyan rounded-full animate-spin mr-2" />
        Loading timeline…
      </div>
    )
  }

  if (!timeline || timeline.entries.length === 0) {
    return (
      <div className="h-[280px] flex items-center justify-center text-neural-600 text-xs font-mono text-center px-4">
        No timeline data — run ingestion to build
      </div>
    )
  }

  const maxVal = Math.max(
    ...timeline.entries.map((e) => Math.max(e.n_papers, e.n_findings, e.n_datasets)),
    1,
  )

  const yearRange = timeline.year_range
    ? `${timeline.year_range[0]}–${timeline.year_range[1]}`
    : '—'

  return (
    <div className="h-[280px] flex flex-col bg-neural-950 border-t border-neural-800/40">
      {/* Stat summary */}
      <div className="flex items-center gap-4 px-4 pt-3 pb-2 border-b border-neural-800/40 flex-shrink-0">
        <span className="text-xs font-mono text-neural-500 truncate">{topicLabel}</span>
        <div className="flex items-center gap-3 ml-auto text-[11px] font-mono flex-shrink-0">
          <span style={{ color: BAR_COLORS.papers }}>{timeline.total_papers} papers</span>
          <span style={{ color: BAR_COLORS.findings }}>{timeline.total_findings} findings</span>
          <span style={{ color: BAR_COLORS.datasets }}>{timeline.total_datasets} datasets</span>
          <span className="text-neural-600">{yearRange}</span>
        </div>
      </div>

      {/* Chart + tooltip row */}
      <div className="flex flex-1 overflow-hidden relative">
        {/* Bar chart */}
        <div className="flex-1 overflow-x-auto px-4 pt-3">
          <div className="flex items-end gap-1 h-[140px]">
            {timeline.entries.map((entry) => (
              <YearBar
                key={entry.year}
                entry={entry}
                maxVal={maxVal}
                onHover={setHoveredEntry}
                isHovered={hoveredEntry?.year === entry.year}
              />
            ))}
          </div>
        </div>

        {/* Tooltip pinned to right */}
        {hoveredEntry && (
          <div className="absolute right-4 top-2 z-10">
            <HoverTooltip entry={hoveredEntry} />
          </div>
        )}
      </div>

      {/* Methods row */}
      {hoveredEntry && hoveredEntry.methods_introduced.length > 0 && (
        <div className="px-4 pb-2 flex-shrink-0 border-t border-neural-800/30 pt-1">
          <span className="text-[9px] uppercase tracking-widest text-neural-600 mr-2">Methods</span>
          <div className="inline-flex flex-wrap gap-1 mt-0.5">
            {hoveredEntry.methods_introduced.map((m) => (
              <span
                key={m}
                className="text-[10px] font-mono bg-neural-900 border border-neural-800 text-neural-400 rounded px-1.5 py-0.5"
              >
                {m}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="px-4 pb-2 flex items-center gap-3 flex-shrink-0">
        {(Object.entries(BAR_COLORS) as [string, string][]).map(([key, color]) => (
          <span key={key} className="flex items-center gap-1 text-[10px] text-neural-600">
            <span className="w-2 h-2 rounded-sm inline-block" style={{ backgroundColor: color }} />
            {key}
          </span>
        ))}
      </div>
    </div>
  )
}
