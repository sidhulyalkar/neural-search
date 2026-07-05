import { useMemo, useState } from 'react'
import type { ExperimentGlancerAnchor, ExperimentGlancerLayer, ExperimentGlancerScene } from '../../api/experimentglancer'
import { useControllableState } from '../../hooks/useControllableState'
import type { SceneCommandAction } from '../../lib/sceneCommandParser'
import { AnchorTray } from './AnchorTray'
import { GlassPanel } from './GlassPanel'
import { SceneCommandBar } from './SceneCommandBar'
import { TimelineTrackRow } from './TimelineTrackRow'

const TRACK_LABEL: Record<string, string> = {
  timeline: 'Timeline',
  behavior: 'Behavior',
  neural: 'Neural',
  model: 'Model',
  metadata: 'Metadata',
}

const MAX_ZOOM = 8
const TICK_COUNT = 6

function groupByTrack(layers: ExperimentGlancerLayer[]): Map<string, ExperimentGlancerLayer[]> {
  const byTrack = new Map<string, ExperimentGlancerLayer[]>()
  for (const layer of layers) {
    byTrack.set(layer.display.track, [...(byTrack.get(layer.display.track) ?? []), layer])
  }
  return byTrack
}

function clampWindow(start: number, end: number, bounds: [number, number]): [number, number] {
  const [lo, hi] = bounds
  const span = end - start
  if (span >= hi - lo) return [lo, hi]
  if (start < lo) return [lo, lo + span]
  if (end > hi) return [hi - span, hi]
  return [start, end]
}

interface SynchronizedTimelineProps {
  scene: ExperimentGlancerScene
  /** Supply both to share one cursor across two timelines (compare mode).
   * Omit for a normal, self-contained single-scene timeline. */
  cursorTime?: number
  onCursorTimeChange?: (time: number) => void
  /** Hides the command bar -- used in compare mode where a single shared
   * command bar sits above both timelines instead of one per pane. */
  hideCommandBar?: boolean
  onActiveAnchorChange?: (anchor: ExperimentGlancerAnchor | null) => void
}

/** The real, interactive timeline surface: a shared ruler and cursor across
 * every track, zoom that narrows the visible window, per-layer visibility
 * toggles, an anchor tray of every jump target the backend can justify, and
 * a small command-bar grammar for the same actions. Layer content is
 * schematic (seeded synthetic density, not real event times) until a
 * resolver reports file-derived evidence -- see `LayerConfidenceLegend` and
 * each layer's own status/evidence tier. */
export function SynchronizedTimeline({
  scene,
  cursorTime: controlledCursorTime,
  onCursorTimeChange,
  hideCommandBar = false,
  onActiveAnchorChange,
}: SynchronizedTimelineProps) {
  const { pre, post } = scene.coordinate_space.default_window
  const fullWindowStart = -pre || -2
  const fullWindowEnd = post || 5
  const bounds: [number, number] = [fullWindowStart, fullWindowEnd]

  const [zoom, setZoom] = useState(1)
  const [cursorTime, setCursorTime] = useControllableState(controlledCursorTime, onCursorTimeChange, 0)
  const [hiddenLayerIds, setHiddenLayerIds] = useState<Set<string>>(new Set())
  const [activeAnchorId, setActiveAnchorIdRaw] = useState<string | null>(scene.anchors[0]?.anchor_id ?? null)

  const setActiveAnchorId = (anchorId: string | null) => {
    setActiveAnchorIdRaw(anchorId)
    onActiveAnchorChange?.(anchorId ? scene.anchors.find((anchor) => anchor.anchor_id === anchorId) ?? null : null)
  }

  const layersByTrack = useMemo(() => groupByTrack(scene.layers), [scene.layers])

  const halfWindow = (fullWindowEnd - fullWindowStart) / (2 * zoom)
  const [visibleStart, visibleEnd] = clampWindow(cursorTime - halfWindow, cursorTime + halfWindow, bounds)
  const visibleSpan = visibleEnd - visibleStart || 1

  const cursorPercent =
    cursorTime >= visibleStart && cursorTime <= visibleEnd ? ((cursorTime - visibleStart) / visibleSpan) * 100 : null

  const ticks = Array.from({ length: TICK_COUNT + 1 }, (_, index) => {
    const time = visibleStart + (index / TICK_COUNT) * visibleSpan
    return { time, percent: (index / TICK_COUNT) * 100 }
  })

  const seekToFraction = (fraction: number) => {
    const clamped = Math.min(1, Math.max(0, fraction))
    const nextTime = visibleStart + clamped * visibleSpan
    setCursorTime(Math.min(fullWindowEnd, Math.max(fullWindowStart, nextTime)))
    setActiveAnchorId(null)
  }

  const toggleLayerVisible = (layerId: string) => {
    setHiddenLayerIds((current) => {
      const next = new Set(current)
      if (next.has(layerId)) next.delete(layerId)
      else next.add(layerId)
      return next
    })
  }

  const jumpToAnchor = (anchor: ExperimentGlancerAnchor) => {
    setCursorTime(anchor.time ?? 0)
    setActiveAnchorId(anchor.anchor_id)
    setZoom(1)
  }

  const handleCommandAction = (action: SceneCommandAction) => {
    switch (action.type) {
      case 'set_layer_visibility':
        setHiddenLayerIds((current) => {
          const next = new Set(current)
          for (const layerId of action.layerIds) {
            if (action.visible) next.delete(layerId)
            else next.add(layerId)
          }
          return next
        })
        break
      case 'jump_to_anchor':
        jumpToAnchor(action.anchor)
        break
      case 'offset_from_anchor':
        setCursorTime(Math.min(fullWindowEnd, Math.max(fullWindowStart, (action.anchor.time ?? 0) + action.seconds)))
        setActiveAnchorId(null)
        setZoom(1)
        break
      case 'reset_view':
        setZoom(1)
        setCursorTime(0)
        setActiveAnchorId(scene.anchors[0]?.anchor_id ?? null)
        setHiddenLayerIds(new Set())
        break
    }
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1fr_260px] gap-4 items-start">
      <GlassPanel tone="cyan" className="p-4">
        <div className="flex items-center justify-between mb-3 text-xs">
          <div className="flex items-center gap-2 text-neural-500">
            <span className="uppercase tracking-widest">Timeline</span>
            <span className="text-neural-400 font-mono">
              {activeAnchorId ? `t = ${cursorTime.toFixed(2)}s (schematic)` : `t = ${cursorTime.toFixed(2)}s`}
            </span>
          </div>
          <div className="flex items-center gap-1.5 text-neural-500">
            <button
              type="button"
              onClick={() => setZoom((value) => Math.max(1, value / 2))}
              disabled={zoom <= 1}
              className="w-6 h-6 rounded border border-white/10 hover:border-accent-cyan/50 hover:text-accent-cyan disabled:opacity-30 disabled:hover:border-white/10 disabled:hover:text-neural-500"
            >
              −
            </button>
            <span className="w-8 text-center font-mono">{zoom}×</span>
            <button
              type="button"
              onClick={() => setZoom((value) => Math.min(MAX_ZOOM, value * 2))}
              disabled={zoom >= MAX_ZOOM}
              className="w-6 h-6 rounded border border-white/10 hover:border-accent-cyan/50 hover:text-accent-cyan disabled:opacity-30 disabled:hover:border-white/10 disabled:hover:text-neural-500"
            >
              +
            </button>
          </div>
        </div>

        {/* Ruler */}
        <div className="flex items-center gap-2 mb-1">
          <div className="w-36 shrink-0" />
          <div
            className="relative flex-1 h-5 cursor-crosshair"
            onClick={(event) => {
              const rect = event.currentTarget.getBoundingClientRect()
              seekToFraction((event.clientX - rect.left) / rect.width)
            }}
          >
            {ticks.map(({ time, percent }) => (
              <div key={percent} className="absolute top-0 bottom-0 text-[10px] text-neural-600" style={{ left: `${percent}%` }}>
                <div className="w-px h-2 bg-neural-700" />
                <span className="absolute -translate-x-1/2 mt-2 font-mono">{time.toFixed(1)}s</span>
              </div>
            ))}
            {cursorPercent !== null && (
              <div
                className="absolute inset-y-0 w-px bg-accent-violet shadow-[0_0_8px_1px_rgba(139,92,246,0.7)] pointer-events-none"
                style={{ left: `${cursorPercent}%` }}
              />
            )}
          </div>
        </div>

        {/* Tracks */}
        <div className="mt-5 space-y-3">
          {scene.layout.tracks
            .filter((track) => (layersByTrack.get(track) ?? []).length > 0)
            .map((track) => (
              <div key={track}>
                <p className="text-[10px] uppercase tracking-widest text-neural-600 mb-0.5 pl-[9.5rem]">
                  {TRACK_LABEL[track] ?? track}
                </p>
                {(layersByTrack.get(track) ?? []).map((layer) => (
                  <TimelineTrackRow
                    key={layer.layer_id}
                    layer={layer}
                    visible={!hiddenLayerIds.has(layer.layer_id)}
                    onToggleVisible={() => toggleLayerVisible(layer.layer_id)}
                    fullWindowStart={fullWindowStart}
                    fullWindowEnd={fullWindowEnd}
                    visibleStart={visibleStart}
                    visibleEnd={visibleEnd}
                    cursorPercent={cursorPercent}
                    onSeek={seekToFraction}
                  />
                ))}
              </div>
            ))}
        </div>

        <p className="mt-4 text-[11px] text-neural-600">
          Tick marks are schematic (seeded, deterministic placement) until a layer's evidence tier is{' '}
          <span className="text-neural-400">file_derived</span> -- see the legend for what each status means.
        </p>

        {!hideCommandBar && (
          <SceneCommandBar anchors={scene.anchors} layers={scene.layers} onAction={handleCommandAction} />
        )}
      </GlassPanel>

      <AnchorTray anchors={scene.anchors} activeAnchorId={activeAnchorId} onJump={jumpToAnchor} />
    </div>
  )
}
