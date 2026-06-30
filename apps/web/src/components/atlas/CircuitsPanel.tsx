import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { atlasApi, type Circuit } from '../../api/coverage'

function CircuitCard({
  circuit,
  isActive,
  onSelect,
  onRegionClick,
}: {
  circuit: Circuit
  isActive: boolean
  onSelect: () => void
  onRegionClick: (id: string, label: string) => void
}) {
  return (
    <div
      className={`border rounded-xl transition-all ${
        isActive
          ? 'border-neural-600 bg-neural-900/80'
          : 'border-neural-800 bg-neural-900/40 hover:border-neural-700'
      }`}
    >
      {/* Circuit header */}
      <button
        type="button"
        className="w-full text-left px-4 py-3 flex items-start gap-3"
        onClick={onSelect}
      >
        <div
          className="w-2.5 h-2.5 rounded-full flex-shrink-0 mt-1"
          style={{ backgroundColor: circuit.color }}
        />
        <div className="min-w-0">
          <p className="text-sm font-mono text-neural-100 leading-tight">{circuit.label}</p>
          <p className="text-[10px] text-neural-500 mt-0.5 leading-snug">{circuit.description}</p>
        </div>
        <span className="ml-auto text-neural-700 text-xs font-mono flex-shrink-0">{isActive ? '▲' : '▼'}</span>
      </button>

      {/* Circuit region chain */}
      {isActive && (
        <div className="px-4 pb-4 space-y-3">
          {/* Linear pathway visualization */}
          <div className="flex flex-col gap-1.5">
            {circuit.regions.map((region, idx) => (
              <div key={region.id} className="flex items-start gap-2">
                {/* Step connector */}
                <div className="flex flex-col items-center gap-0 flex-shrink-0 w-5">
                  <div
                    className="w-2 h-2 rounded-full border flex-shrink-0"
                    style={{ borderColor: circuit.color, backgroundColor: `${circuit.color}30` }}
                  />
                  {idx < circuit.regions.length - 1 && (
                    <div className="w-px flex-1 min-h-[12px]" style={{ backgroundColor: `${circuit.color}40` }} />
                  )}
                </div>

                {/* Region node */}
                <div className="flex-1 min-w-0 pb-1">
                  <button
                    type="button"
                    className="text-left w-full group"
                    onClick={() => onRegionClick(region.id, region.label)}
                  >
                    <span className="text-xs font-mono text-neural-200 group-hover:text-white transition-colors leading-tight block">
                      {region.label}
                    </span>
                  </button>
                  <span className="text-[9px] font-mono text-neural-600 leading-tight">
                    {region.role}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {/* Topic links */}
          {circuit.topics.length > 0 && (
            <div className="border-t border-neural-800 pt-2 mt-1">
              <span className="text-[9px] uppercase tracking-widest text-neural-600 block mb-1">Related Topics</span>
              <div className="flex flex-wrap gap-1">
                {circuit.topics.map((t) => (
                  <span
                    key={t}
                    className="text-[9px] font-mono px-1.5 py-0.5 rounded border border-neural-700 bg-neural-900 text-neural-500"
                  >
                    {t.replace(/_/g, ' ')}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export function CircuitsPanel({
  onRegionClick,
}: {
  onRegionClick: (id: string, label: string) => void
}) {
  const [activeCircuit, setActiveCircuit] = useState<string | null>('visual_pathway')

  const { data: circuits = [], isLoading } = useQuery({
    queryKey: ['atlas-circuits'],
    queryFn: atlasApi.circuits,
    staleTime: Infinity,
  })

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 py-8 justify-center text-neural-600 text-xs font-mono">
        <span className="w-3 h-3 border border-neural-700 border-t-accent-cyan rounded-full animate-spin" />
        Loading circuits…
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div className="flex items-baseline gap-2 mb-1">
        <span className="text-xs font-mono text-neural-400 uppercase tracking-wider">Functional Circuits</span>
        <span className="text-xs text-neural-700">{circuits.length} pathways</span>
      </div>
      <p className="text-[11px] text-neural-600 leading-snug">
        Major neuroscience circuits connecting brain regions into functional systems.
        Click any region node to view its datasets and connections.
      </p>
      <div className="space-y-2 mt-3">
        {circuits.map((circuit) => (
          <CircuitCard
            key={circuit.id}
            circuit={circuit}
            isActive={activeCircuit === circuit.id}
            onSelect={() => setActiveCircuit(activeCircuit === circuit.id ? null : circuit.id)}
            onRegionClick={onRegionClick}
          />
        ))}
      </div>
    </div>
  )
}
