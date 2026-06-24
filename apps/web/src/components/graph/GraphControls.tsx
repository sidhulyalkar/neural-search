import type { LayerMode, ViewMode } from '../../types/graph'

const VIEW_MODES: { value: ViewMode; label: string; icon: string }[] = [
  { value: 'galaxy', label: 'Galaxy', icon: '🌌' },
  { value: 'explorer', label: 'Explorer', icon: '🔭' },
  { value: '2d', label: '2D', icon: '📐' },
]

const LAYER_MODES: { value: LayerMode; label: string; title: string }[] = [
  { value: 'corpus', label: 'Corpus', title: 'Dataset Corpus — 7,171 dataset records with metadata' },
  { value: 'consensus', label: 'Consensus', title: 'Consensus Findings — aggregated evidence across papers per region' },
  { value: 'literature', label: 'Literature', title: 'Literature Findings — extracted from ~12K Tier-C findings' },
  { value: 'bridge', label: 'Bridge', title: 'Paper-Dataset Bridges — 168 DOI-exact + 225 fuzzy paper links' },
  { value: 'morphology', label: 'Morphology', title: 'Morphology — structural connectivity and anatomy' },
  { value: 'validation', label: 'Validation', title: 'Validation / Qrels — 175 silver + 3 adjudicated qrels; gold benchmark pending' },
  { value: 'coverage_gaps', label: 'Gaps', title: 'Coverage Gaps — regions or tasks with fewer than 3 datasets' },
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

      {/* Layer mode — toggle buttons with tooltips */}
      <div className="flex gap-0.5 border-r border-neural-800 pr-2 mr-1">
        {LAYER_MODES.map(({ value, label, title }) => {
          const isNew = value === 'validation' || value === 'coverage_gaps'
          return (
            <button
              key={value}
              type="button"
              onClick={() => onLayerModeChange(value)}
              title={title}
              className={`relative px-2.5 py-1.5 rounded text-xs font-medium transition-colors ${
                layerMode === value
                  ? 'bg-accent-violet/20 text-accent-violet border border-accent-violet/30'
                  : 'text-neural-500 hover:text-neural-200 border border-transparent'
              }`}
            >
              {label}
              {isNew && (
                <span className="absolute -top-1 -right-0.5 w-1.5 h-1.5 rounded-full bg-accent-cyan" />
              )}
            </button>
          )
        })}
      </div>

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
