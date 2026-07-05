import { deriveReadinessItems, type Readiness } from '../../lib/sceneReadiness'
import type { SearchResultItem } from '../../types'

const READINESS_STYLE: Record<Readiness, string> = {
  'file-verified': 'text-accent-emerald border-accent-emerald/30 bg-accent-emerald/5',
  inferred: 'text-accent-cyan border-accent-cyan/30 bg-accent-cyan/5',
  'can generate': 'text-neural-400 border-dashed border-neural-700 bg-white/[0.02]',
  missing: 'text-neural-600 border-neural-800 bg-neural-900/40 line-through decoration-neural-700',
}

interface SceneReadinessStripProps {
  result: SearchResultItem
}

export function SceneReadinessStrip({ result }: SceneReadinessStripProps) {
  const items = deriveReadinessItems(result)

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {items.map((item) => (
        <span
          key={item.label}
          className={`text-[10px] uppercase tracking-wide border rounded px-1.5 py-0.5 ${READINESS_STYLE[item.readiness]}`}
        >
          {item.label}: {item.readiness}
        </span>
      ))}
    </div>
  )
}
