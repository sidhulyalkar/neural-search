import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchTopics } from '../../api/graph'
import type { TopicSummary } from '../../types/graph'

interface TopicsPanelProps {
  activeSlug: string | null
  onTopicSelect: (slug: string) => void
}

function TopicCard({
  topic,
  isActive,
  onSelect,
}: {
  topic: TopicSummary
  isActive: boolean
  onSelect: () => void
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`w-full text-left px-2 py-2 rounded transition-colors ${
        isActive
          ? 'bg-neural-800/80 border border-neural-700'
          : 'hover:bg-neural-900/60 border border-transparent'
      }`}
    >
      <div className="flex items-center gap-2 mb-0.5">
        <div
          className="w-2 h-2 rounded-full flex-shrink-0"
          style={{ backgroundColor: topic.color || '#22d3ee' }}
        />
        <span
          className={`text-xs font-medium truncate ${
            isActive ? 'text-neural-100' : 'text-neural-300'
          }`}
        >
          {topic.label}
        </span>
      </div>
      {topic.description && (
        <p className="text-[10px] text-neural-600 line-clamp-1 pl-4 leading-snug">
          {topic.description}
        </p>
      )}
      <div className="flex gap-2 mt-1 pl-4">
        {topic.n_regions > 0 && (
          <span className="text-[9px] font-mono text-neural-700">{topic.n_regions} regions</span>
        )}
        {topic.n_tasks > 0 && (
          <span className="text-[9px] font-mono text-neural-700">{topic.n_tasks} tasks</span>
        )}
      </div>
    </button>
  )
}

export function TopicsPanel({ activeSlug, onTopicSelect }: TopicsPanelProps) {
  const [search, setSearch] = useState('')

  const { data: topics = [], isLoading } = useQuery<TopicSummary[]>({
    queryKey: ['topics-list'],
    queryFn: fetchTopics,
    staleTime: 300_000,
  })

  const filtered = search.trim()
    ? topics.filter(
        (t) =>
          t.label.toLowerCase().includes(search.toLowerCase()) ||
          t.description.toLowerCase().includes(search.toLowerCase()),
      )
    : topics

  return (
    <div className="space-y-2">
      <span className="block text-xs uppercase tracking-widest text-neural-600">Topics</span>

      <input
        type="text"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        placeholder="Filter topics…"
        className="w-full bg-neural-900 border border-neural-800 rounded px-2 py-1 text-xs text-neural-300 placeholder-neural-700 outline-none focus:border-neural-700 transition-colors font-mono"
      />

      {isLoading && (
        <div className="flex items-center gap-2 py-2 text-neural-600 text-xs">
          <span className="w-3 h-3 border border-neural-700 border-t-accent-cyan rounded-full animate-spin" />
          Loading…
        </div>
      )}

      {!isLoading && filtered.length === 0 && (
        <div className="text-xs text-neural-700 py-2">
          {search ? 'No matches' : 'No topics available'}
        </div>
      )}

      <div className="space-y-0.5">
        {filtered.map((topic) => (
          <TopicCard
            key={topic.id}
            topic={topic}
            isActive={activeSlug === topic.id}
            onSelect={() => onTopicSelect(topic.id)}
          />
        ))}
      </div>
    </div>
  )
}
