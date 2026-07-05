import { useState } from 'react'

interface FilterPanelProps {
  regions: string[]
  species: string[]
  tasks: string[]
  onFiltersChange: (filters: { regions: string[]; species: string[]; tasks: string[] }) => void
}

export function FilterPanel({ regions, onFiltersChange }: FilterPanelProps) {
  const [regionInput, setRegionInput] = useState('')

  const addRegion = () => {
    const r = regionInput.trim().toLowerCase()
    if (r && !regions.includes(r)) {
      onFiltersChange({ regions: [...regions, r], species: [], tasks: [] })
    }
    setRegionInput('')
  }

  const removeRegion = (r: string) =>
    onFiltersChange({ regions: regions.filter((x) => x !== r), species: [], tasks: [] })

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-widest text-neural-600">Filters</span>
        {regions.length > 0 && (
          <button
            type="button"
            onClick={() => onFiltersChange({ regions: [], species: [], tasks: [] })}
            className="text-xs text-neural-600 hover:text-neural-300"
          >
            Clear
          </button>
        )}
      </div>

      <div>
        <span className="block text-xs text-neural-500 mb-1.5">Region</span>
        <div className="flex gap-1">
          <input
            value={regionInput}
            onChange={(e) => setRegionInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') { e.preventDefault(); addRegion() }
            }}
            placeholder="e.g. hippocampus"
            className="flex-1 bg-neural-900 border border-neural-700 rounded px-2 py-1.5 text-xs text-neural-200 placeholder-neural-600 focus:outline-none focus:border-neural-500"
          />
          <button
            type="button"
            onClick={addRegion}
            className="px-2 py-1.5 text-xs bg-neural-800 text-neural-200 rounded hover:bg-neural-700"
          >
            +
          </button>
        </div>
        {regions.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-1.5">
            {regions.map((r) => (
              <span
                key={r}
                className="inline-flex items-center gap-1 text-xs bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/30 rounded px-2 py-0.5"
              >
                {r}
                <button type="button" onClick={() => removeRegion(r)} className="hover:text-white">x</button>
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
