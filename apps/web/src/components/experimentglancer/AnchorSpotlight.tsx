import type { CoordinateSpace, ExperimentGlancerAnchor } from '../../api/experimentglancer'
import { GlassPanel } from './GlassPanel'

interface AnchorSpotlightProps {
  anchor: ExperimentGlancerAnchor
  coordinateSpace: CoordinateSpace
}

export function AnchorSpotlight({ anchor, coordinateSpace }: AnchorSpotlightProps) {
  const { pre, post } = coordinateSpace.default_window
  const totalWindow = pre + post || 1
  const centerPercent = (pre / totalWindow) * 100

  return (
    <GlassPanel tone="violet" className="p-5">
      <div className="flex items-start justify-between gap-4 mb-4">
        <div>
          <p className="text-xs uppercase tracking-widest text-accent-violet/80 mb-1">Anchor</p>
          <h2 className="text-lg text-neural-100">{anchor.label}</h2>
        </div>
        <span className="text-xs font-mono text-neural-500 border border-neural-800 rounded px-2 py-1 flex-shrink-0">
          {coordinateSpace.clock}
        </span>
      </div>

      <p className="text-sm text-neural-400 leading-relaxed mb-4">{anchor.reason}</p>

      {anchor.evidence.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-4">
          {anchor.evidence.map((item) => (
            <span
              key={item}
              className="text-xs rounded px-2 py-0.5 border border-accent-violet/30 bg-accent-violet/5 text-accent-violet"
            >
              {item}
            </span>
          ))}
        </div>
      )}

      <div className="relative h-10 rounded-full bg-neural-900/60 border border-white/10 overflow-hidden">
        <div className="absolute inset-y-0 bg-accent-violet/10" style={{ left: 0, width: `${centerPercent}%` }} />
        <div
          className="absolute inset-y-0 bg-accent-violet/5"
          style={{ left: `${centerPercent}%`, right: 0 }}
        />
        <div
          className="absolute top-0 bottom-0 w-px bg-accent-violet shadow-[0_0_12px_2px_rgba(139,92,246,0.8)]"
          style={{ left: `${centerPercent}%` }}
        />
        <div className="absolute inset-0 flex items-center justify-between px-3 text-[10px] font-mono text-neural-500">
          <span>-{pre.toFixed(1)}s</span>
          <span className="text-accent-violet">
            {anchor.time !== null ? `t = ${anchor.time.toFixed(2)}s` : 'time unresolved'}
          </span>
          <span>+{post.toFixed(1)}s</span>
        </div>
      </div>

      {anchor.time === null && (
        <p className="mt-2 text-[11px] text-neural-600">
          No file-validated timestamp yet — position is schematic, not literal.
        </p>
      )}
    </GlassPanel>
  )
}
