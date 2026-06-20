import { useState } from 'react'

// Brain system hierarchy for top-level tree
const SYSTEMS: { id: string; label: string; regions: string[] }[] = [
  { id: 'hippocampal_formation', label: 'Hippocampal Formation',
    regions: ['hippocampus', 'ca1', 'ca3', 'dentate gyrus', 'entorhinal cortex'] },
  { id: 'prefrontal', label: 'Prefrontal Cortex',
    regions: ['prefrontal cortex', 'anterior cingulate cortex', 'orbitofrontal cortex'] },
  { id: 'basal_ganglia', label: 'Basal Ganglia',
    regions: ['striatum', 'nucleus accumbens', 'globus pallidus', 'substantia nigra'] },
  { id: 'cerebellum', label: 'Cerebellum', regions: ['cerebellum', 'purkinje cell layer'] },
  { id: 'brainstem', label: 'Brainstem',
    regions: ['brainstem', 'midbrain', 'pons', 'medulla', 'locus coeruleus'] },
  { id: 'sensory_cortex', label: 'Sensory Cortex',
    regions: ['visual cortex', 'auditory cortex', 'somatosensory cortex'] },
  { id: 'motor_cortex', label: 'Motor Cortex',
    regions: ['motor cortex', 'primary motor cortex'] },
  { id: 'limbic', label: 'Limbic System',
    regions: ['amygdala', 'thalamus', 'hypothalamus', 'insula'] },
]

interface OntologyTreePanelProps {
  onRegionSelect: (region: string) => void
}

export function OntologyTreePanel({ onRegionSelect }: OntologyTreePanelProps) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set())

  const toggle = (id: string) =>
    setExpanded((prev) => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })

  return (
    <div className="space-y-3">
      <span className="block text-xs uppercase tracking-widest text-neural-600">Ontology</span>
      <div className="space-y-0.5">
        {SYSTEMS.map((sys) => (
          <div key={sys.id}>
            <button
              type="button"
              onClick={() => toggle(sys.id)}
              className="w-full flex items-center gap-1.5 px-1.5 py-1 text-xs text-neural-400 hover:text-neural-200 rounded hover:bg-neural-900/50 transition-colors text-left"
            >
              <span className="text-neural-700 w-3 text-center flex-shrink-0">
                {expanded.has(sys.id) ? '▾' : '▸'}
              </span>
              {sys.label}
            </button>
            {expanded.has(sys.id) && (
              <div className="ml-4 space-y-0.5 mb-1">
                {sys.regions.map((region) => (
                  <button
                    key={region}
                    type="button"
                    onClick={() => onRegionSelect(region)}
                    className="w-full text-left px-2 py-0.5 text-xs text-neural-600 hover:text-accent-cyan hover:bg-accent-cyan/5 rounded transition-colors"
                  >
                    {region}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
