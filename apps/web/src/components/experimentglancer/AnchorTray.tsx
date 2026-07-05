import type { ExperimentGlancerAnchor } from '../../api/experimentglancer'
import { GlassPanel } from './GlassPanel'

interface AnchorTrayProps {
  anchors: ExperimentGlancerAnchor[]
  activeAnchorId: string | null
  onJump: (anchor: ExperimentGlancerAnchor) => void
}

/** Every jump target the backend could justify for this scene (event
 * keyword matches, affordance-implied anchors, trial structure, dataset
 * overview) -- not just the one the scene opens on. Anchors without a
 * file-validated timestamp jump to a schematic position and say so. */
export function AnchorTray({ anchors, activeAnchorId, onJump }: AnchorTrayProps) {
  if (anchors.length === 0) return null

  return (
    <GlassPanel tone="neutral" className="p-3">
      <p className="text-xs uppercase tracking-widest text-neural-500 mb-2">Anchor tray</p>
      <div className="space-y-1">
        {anchors.map((anchor) => (
          <button
            key={anchor.anchor_id}
            type="button"
            onClick={() => onJump(anchor)}
            className={`w-full text-left rounded-lg px-2.5 py-1.5 border transition-colors ${
              activeAnchorId === anchor.anchor_id
                ? 'border-accent-violet/50 bg-accent-violet/10 text-accent-violet'
                : 'border-white/5 hover:border-white/15 text-neural-400 hover:text-neural-200'
            }`}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm">{anchor.label}</span>
              <span className="text-[10px] uppercase tracking-wide opacity-60">{anchor.kind}</span>
            </div>
            <p className="text-[11px] opacity-70 mt-0.5">
              {anchor.time !== null ? `t = ${anchor.time.toFixed(2)}s` : 'schematic — no file-validated time'}
            </p>
          </button>
        ))}
      </div>
    </GlassPanel>
  )
}
