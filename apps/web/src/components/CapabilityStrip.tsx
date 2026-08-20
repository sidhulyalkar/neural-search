import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { getRuntimeStatus } from '../api/runtime'

const capabilityLabels: Record<string, string> = {
  real_dataset_corpus: 'corpus',
  dense_semantic_index: 'dense',
  knowledge_graph: 'KG',
  literature_findings: 'findings',
  paper_dataset_links: 'paper links',
  coverage_gap_boost: 'coverage',
}

export function CapabilityStrip() {
  const { data, isError } = useQuery({
    queryKey: ['runtime-status'],
    queryFn: () => getRuntimeStatus(),
    staleTime: 30_000,
    retry: 1,
  })

  if (isError) {
    return (
      <div className="border-t border-neural-800/40 py-1.5 text-xs text-neural-600">
        runtime capability state unavailable
      </div>
    )
  }
  if (!data) return null

  const keyCapabilities = data.capabilities.capabilities.filter(
    (item) => capabilityLabels[item.capability],
  )
  const healthClass =
    data.health === 'ready'
      ? 'text-accent-emerald'
      : data.health === 'degraded'
      ? 'text-amber-300'
      : 'text-red-300'

  return (
    <div className="border-t border-neural-800/40 py-1.5 flex items-center gap-3 overflow-x-auto text-xs">
      <Link to="/system" className={`font-mono flex-shrink-0 ${healthClass} hover:text-white`}>
        {data.profile.name} · {data.health}
      </Link>
      <span className="text-neural-700 flex-shrink-0">|</span>
      <span className="text-neural-500 flex-shrink-0">
        {data.capabilities.available}/{data.capabilities.total} capabilities
      </span>
      <div className="flex items-center gap-1.5">
        {keyCapabilities.map((item) => (
          <span
            key={item.capability}
            title={`${item.description}${item.version ? ` · ${item.version}` : ''}`}
            className={`px-1.5 py-0.5 rounded border flex-shrink-0 ${
              item.available
                ? 'border-accent-cyan/25 text-neural-300 bg-accent-cyan/5'
                : 'border-neural-800 text-neural-600 bg-neural-900'
            }`}
          >
            {item.available ? '●' : '○'} {capabilityLabels[item.capability]}
          </span>
        ))}
      </div>
    </div>
  )
}
