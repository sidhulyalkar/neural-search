import { useQuery } from '@tanstack/react-query'
import { fetchTopicAncestors } from '../../api/graph'
import type { FoundationalPaper } from '../../types/graph'

interface FoundationalPapersPanelProps {
  topicId: string
}

function PaperSkeleton() {
  return (
    <div className="bg-neural-900 border border-neural-800 rounded p-3 space-y-2 animate-pulse">
      <div className="h-3 bg-neural-800 rounded w-full" />
      <div className="h-3 bg-neural-800 rounded w-3/4" />
      <div className="flex gap-2 mt-1">
        <div className="h-2 bg-neural-800 rounded w-10" />
        <div className="h-2 bg-neural-800 rounded w-16" />
      </div>
    </div>
  )
}

function PaperCard({ paper }: { paper: FoundationalPaper }) {
  const doiUrl = paper.doi
    ? paper.doi.startsWith('http')
      ? paper.doi
      : `https://doi.org/${paper.doi}`
    : null

  return (
    <div className="bg-neural-900 border border-neural-800 hover:border-neural-700 rounded p-3 transition-colors space-y-1.5">
      <p className="text-xs text-neural-200 line-clamp-2 leading-snug font-medium">
        {paper.title}
      </p>
      <div className="flex items-center gap-2 flex-wrap">
        {paper.year !== null && (
          <span className="text-[10px] font-mono text-neural-600">{paper.year}</span>
        )}
        <span className="text-[10px] font-mono bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/20 rounded px-1.5 py-0.5">
          {paper.in_topic_citations} in-topic cit.
        </span>
        {paper.citation_count > 0 && (
          <span className="text-[10px] font-mono text-neural-600">
            {paper.citation_count} total cit.
          </span>
        )}
        {doiUrl && (
          <a
            href={doiUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-[10px] font-mono text-neural-600 hover:text-accent-cyan transition-colors ml-auto"
          >
            DOI →
          </a>
        )}
      </div>
    </div>
  )
}

export function FoundationalPapersPanel({ topicId }: FoundationalPapersPanelProps) {
  const { data: papers, isLoading } = useQuery({
    queryKey: ['topic-ancestors', topicId],
    queryFn: () => fetchTopicAncestors(topicId, 10),
    staleTime: 300_000,
  })

  const sorted = papers
    ? [...papers].sort((a, b) => b.in_topic_citations - a.in_topic_citations).slice(0, 10)
    : []

  return (
    <div className="space-y-2 py-2">
      <div className="text-[10px] uppercase tracking-widest text-neural-600 px-1">
        Foundational Papers
      </div>

      {isLoading && (
        <div className="space-y-2">
          {Array.from({ length: 4 }, (_, i) => (
            <PaperSkeleton key={i} />
          ))}
        </div>
      )}

      {!isLoading && sorted.length === 0 && (
        <div className="text-xs text-neural-600 font-mono text-center py-6">
          No foundational papers found
        </div>
      )}

      {!isLoading && sorted.length > 0 && (
        <div className="space-y-2">
          {sorted.map((paper) => (
            <PaperCard key={paper.id} paper={paper} />
          ))}
        </div>
      )}
    </div>
  )
}
