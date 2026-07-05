import { useState } from 'react'
import { describeMissingLayerAction } from '../../lib/missingLayerActions'
import { GlassPanel } from './GlassPanel'

interface WarningsDrawerProps {
  warnings: string[]
  missingRequirements: string[]
}

export function WarningsDrawer({ warnings, missingRequirements }: WarningsDrawerProps) {
  const [open, setOpen] = useState(false)
  const all = [...warnings, ...missingRequirements]
  if (all.length === 0) return null

  return (
    <GlassPanel tone="amber" className="p-4">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="w-full flex items-center justify-between text-left"
      >
        <span className="text-sm text-amber-300">
          {all.length} evidence gap{all.length > 1 ? 's' : ''} to review
        </span>
        <span className="text-xs text-amber-400/70">{open ? 'Hide' : 'Show'}</span>
      </button>
      {open && (
        <ul className="mt-3 space-y-1.5 text-xs text-amber-200/80">
          {all.map((item) => {
            const action = describeMissingLayerAction(item)
            return (
              <li key={item}>
                <span>⚠ {item}</span>
                {action && <span className="block text-amber-400/60 pl-4">→ {action}</span>}
              </li>
            )
          })}
        </ul>
      )}
    </GlassPanel>
  )
}
