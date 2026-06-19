import { useState } from 'react'

const SPECIES_OPTIONS = ['mouse', 'rat', 'human', 'macaque', 'zebrafish']

interface FilterPanelProps {
  regions: string[]
  species: string[]
  tasks: string[]
  onFiltersChange: (filters: { regions: string[]; species: string[]; tasks: string[] }) => void
}

export function FilterPanel({ regions, species, tasks, onFiltersChange }: FilterPanelProps) {
  const [regionInput, setRegionInput] = useState('')

  const addRegion = () => {
    const r = regionInput.trim().toLowerCase()
    if (r && !regions.includes(r)) {
      onFiltersChange({ regions: [...regions, r], species, tasks })
    }
    setRegionInput('')
  }

  const removeRegion = (r: string) =>
    onFiltersChange({ regions: regions.filter((x) => x !== r), species, tasks })

  const toggleSpecies = (s: string) => {
    const next = species.includes(s) ? species.filter((x) => x !== s) : [...species, s]
    onFiltersChange({ regions, species: next, tasks })
  }

  const clearAll = () => onFiltersChange({ regions: [], species: [], tasks: [] })

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-widest text-neural-600">Filters</span>
        {(regions.length > 0 || species.length > 0 || tasks.length > 0) && (
          <button type="button" onClick={clearAll} className="text-xs text-neural-600 hover:text-neural-300">
            Clear
          </button>
        )}
      </div>

      {/* Region input */}
      <div>
        <span className="block text-xs text-neural-500 mb-1.5">Region</span>
        <div className="flex gap-1">
          <input
            value={regionInput}
            onChange={(e) => setRegionInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); addRegion() } }}
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
              <span key={r} className="inline-flex items-center gap-1 text-xs bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/30 rounded px-2 py-0.5">
                {r}
                <button type="button" onClick={() => removeRegion(r)} className="hover:text-white">×</button>
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Species */}
      <div>
        <span className="block text-xs text-neural-500 mb-1.5">Species</span>
        <div className="flex flex-wrap gap-1">
          {SPECIES_OPTIONS.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => toggleSpecies(s)}
              className={`text-xs rounded px-2 py-0.5 border transition-colors ${
                species.includes(s)
                  ? 'bg-accent-emerald/10 text-accent-emerald border-accent-emerald/30'
                  : 'text-neural-600 border-neural-800 hover:text-neural-300'
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
