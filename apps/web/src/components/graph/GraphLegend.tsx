interface GraphLegendProps {
  onClose: () => void
}

const NODE_TYPES = [
  { color: '#f59e0b', label: 'Brain System' },
  { color: '#fcd34d', label: 'Region' },
  { color: '#10b981', label: 'Finding cluster (↑ increase)' },
  { color: '#ef4444', label: 'Finding cluster (↓ decrease)' },
  { color: '#8b5cf6', label: 'Finding cluster (correlation)' },
  { color: '#22d3ee', label: 'Dataset' },
  { color: '#8b5cf6', label: 'Paper' },
]

const EDGE_TYPES = [
  { color: '#10b981', label: 'Supports' },
  { color: '#ef4444', label: 'Contradicts' },
  { color: '#ffffff30', label: 'Dataset → Paper' },
  { color: '#fcd34d30', label: 'Region → System' },
]

export function GraphLegend({ onClose }: GraphLegendProps) {
  return (
    <div className="absolute top-16 right-4 z-30 bg-neural-950/95 backdrop-blur border border-neural-800/50 rounded-lg p-4 w-56">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs uppercase tracking-widest text-neural-600">Legend</span>
        <button type="button" onClick={onClose} className="text-neural-600 hover:text-neural-300 text-xs">✕</button>
      </div>

      <p className="text-xs text-neural-600 mb-2">Nodes</p>
      <div className="space-y-1.5 mb-4">
        {NODE_TYPES.map(({ color, label }) => (
          <div key={label} className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
            <span className="text-xs text-neural-500">{label}</span>
          </div>
        ))}
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full flex-shrink-0 bg-neural-500" />
          <span className="text-xs text-neural-500">Node size ∝ finding count</span>
        </div>
      </div>

      <p className="text-xs text-neural-600 mb-2">Edges</p>
      <div className="space-y-1.5">
        {EDGE_TYPES.map(({ color, label }) => (
          <div key={label} className="flex items-center gap-2">
            <div className="w-5 h-0.5 flex-shrink-0" style={{ backgroundColor: color }} />
            <span className="text-xs text-neural-500">{label}</span>
          </div>
        ))}
        <div className="flex items-center gap-2">
          <div className="w-5 h-0.5 flex-shrink-0 bg-neural-500" />
          <span className="text-xs text-neural-500">Edge width ∝ evidence count</span>
        </div>
      </div>
    </div>
  )
}
