import type { ExperimentGlancerLayer } from '../../api/experimentglancer'
import { CONTINUOUS_LAYER_KINDS, syntheticMarkPositions } from '../../lib/timelineSynthetic'
import { LayerKindIcon } from './layerIcons'

const TONE_COLOR: Record<string, string> = {
  cyan: '#22d3ee',
  amber: '#f59e0b',
  emerald: '#10b981',
  violet: '#8b5cf6',
  neutral: '#64748b',
}

const TONE_BY_TRACK: Record<string, string> = {
  timeline: 'cyan',
  behavior: 'amber',
  neural: 'emerald',
  model: 'violet',
  metadata: 'neutral',
}

interface TimelineTrackRowProps {
  layer: ExperimentGlancerLayer
  visible: boolean
  onToggleVisible: () => void
  /** Fixed bounds of the scene's full nominal window (independent of zoom) --
   * used to place synthetic marks at a stable absolute time. */
  fullWindowStart: number
  fullWindowEnd: number
  /** The currently visible (possibly zoomed/panned) time slice, used to
   * project absolute times to a horizontal percent. */
  visibleStart: number
  visibleEnd: number
  /** Horizontal percent of the synced cursor within this row's plot area,
   * or null when the cursor is outside the visible window. */
  cursorPercent: number | null
  onSeek: (fraction: number) => void
}

/** One layer's lane within the synchronized timeline. Discrete layers
 * (spikes, licks, events) render schematic tick marks positioned by a
 * deterministic seeded hash -- illustrative density, not real event times.
 * Continuous layers (LFP, pupil, wheel, pose, video) render as a band
 * instead, since ticks would misrepresent them either way. */
export function TimelineTrackRow({
  layer,
  visible,
  onToggleVisible,
  fullWindowStart,
  fullWindowEnd,
  visibleStart,
  visibleEnd,
  cursorPercent,
  onSeek,
}: TimelineTrackRowProps) {
  const tone = TONE_BY_TRACK[layer.display.track] ?? 'neutral'
  const color = layer.display.color ?? TONE_COLOR[tone]
  const isAvailable = layer.status === 'available'
  const isContinuous = CONTINUOUS_LAYER_KINDS.has(layer.kind)
  const fullWindow = fullWindowEnd - fullWindowStart || 1
  const visibleWindow = visibleEnd - visibleStart || 1

  const marks = isContinuous
    ? []
    : syntheticMarkPositions(layer.layer_id, layer.kind)
        .map((position) => fullWindowStart + position * fullWindow)
        .filter((time) => time >= visibleStart && time <= visibleEnd)
        .map((time) => ((time - visibleStart) / visibleWindow) * 100)

  return (
    <div className="flex items-center gap-2 py-1.5">
      <button
        type="button"
        onClick={onToggleVisible}
        className={`flex items-center gap-1.5 w-36 shrink-0 text-left text-xs transition-opacity ${
          visible ? 'opacity-100' : 'opacity-35'
        }`}
        title={visible ? 'Hide layer' : 'Show layer'}
      >
        <span style={{ color }}>
          <LayerKindIcon kind={layer.kind} className="w-3.5 h-3.5" />
        </span>
        <span className="truncate text-neural-300">{layer.label}</span>
      </button>

      <div
        className="relative flex-1 h-6 rounded bg-white/[0.03] border border-white/5 overflow-hidden cursor-crosshair"
        onClick={(event) => {
          const rect = event.currentTarget.getBoundingClientRect()
          onSeek((event.clientX - rect.left) / rect.width)
        }}
      >
        {cursorPercent !== null && (
          <div
            className="absolute inset-y-0 w-px bg-accent-violet shadow-[0_0_8px_1px_rgba(139,92,246,0.7)] z-10 pointer-events-none"
            style={{ left: `${cursorPercent}%` }}
          />
        )}
        {visible && isContinuous && (
          <div
            className="absolute inset-y-1.5 left-2 right-2 rounded-full"
            style={{
              backgroundColor: color,
              opacity: isAvailable ? 0.35 : 0.15,
              backgroundImage: isAvailable
                ? undefined
                : `repeating-linear-gradient(45deg, ${color}55 0 4px, transparent 4px 8px)`,
            }}
          />
        )}
        {visible &&
          !isContinuous &&
          marks.map((percent, index) => (
            <div
              key={index}
              className="absolute top-1 bottom-1 w-px"
              style={{ left: `${percent}%`, backgroundColor: color, opacity: isAvailable ? 0.85 : 0.4 }}
            />
          ))}
      </div>
    </div>
  )
}
