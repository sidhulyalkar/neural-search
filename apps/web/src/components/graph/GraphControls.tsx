import type { LayerMode, ViewMode } from '../../types/graph'

const VIEW_MODES: { value: ViewMode; label: string; icon: string }[] = [
  { value: 'galaxy', label: 'Galaxy', icon: '🌌' },
  { value: 'explorer', label: 'Explorer', icon: '🔭' },
  { value: '2d', label: '2D', icon: '📐' },
]

const LAYER_MODES: { value: LayerMode; label: string }[] = [
  { value: 'corpus', label: 'Corpus' },
  { value: 'consensus', label: 'Consensus' },
  { value: 'literature', label: 'Literature' },
  { value: 'bridge', label: 'Bridge' },
  { value: 'morphology', label: 'Morphology' },
  { value: 'topics', label: 'Topics' },
]

interface GraphControlsProps {
  viewMode: ViewMode
  layerMode: LayerMode
  onViewModeChange: (mode: ViewMode) => void
  onLayerModeChange: (mode: LayerMode) => void
  onLegendToggle: () => void
}

export function GraphControls({
  viewMode, layerMode, onViewModeChange, onLayerModeChange, onLegendToggle,
}: GraphControlsProps) {
  return (
    <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-2 bg-neural-950/80 backdrop-blur border border-neural-800/50 rounded-lg px-3 py-2">
      {/* View mode */}
      <div className="flex gap-1 border-r border-neural-800 pr-2 mr-1">
        {VIEW_MODES.map(({ value, label, icon }) => (
          <button
            key={value}
            type="button"
            onClick={() => onViewModeChange(value)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors ${
              viewMode === value
                ? 'bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/30'
                : 'text-neural-400 hover:text-neural-200 border border-transparent'
            }`}
          >
            <span>{icon}</span>
            {label}
          </button>
        ))}
      </div>

      {/* Layer mode */}
      <select
        value={layerMode}
        onChange={(e) => onLayerModeChange(e.target.value as LayerMode)}
        className="bg-neural-900 border border-neural-700 rounded px-2 py-1.5 text-xs text-neural-200 focus:outline-none focus:border-neural-500"
      >
        {LAYER_MODES.map(({ value, label }) => (
          <option key={value} value={value}>Layer: {label}</option>
        ))}
      </select>

      {/* Legend */}
      <button
        type="button"
        onClick={onLegendToggle}
        className="ml-1 text-xs text-neural-500 hover:text-neural-200 px-2 py-1.5 rounded border border-transparent hover:border-neural-700 transition-colors"
        title="Toggle legend"
      >
        ?
      </button>
    </div>
  )
}
