import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  fetchDisorders,
  fetchDisorder,
  fetchCircuitMatrix,
  fetchConcepts,
  fetchConcept,
  type DisorderDetail,
  type ConceptSummary,
} from '../api/kg'

type Tab = 'disorders' | 'circuits' | 'concepts'

const TYPE_COLORS: Record<string, string> = {
  psychotic_disorder: '#a78bfa',
  mood_disorder: '#60a5fa',
  anxiety_disorder: '#34d399',
  neurodevelopmental: '#f59e0b',
  neurodegenerative: '#f97316',
  addiction: '#ec4899',
  neurological: '#22d3ee',
  other: '#6b7280',
}

const BAND_COLORS: Record<string, string> = {
  gamma: '#f97316',
  beta: '#10b981',
  alpha: '#06b6d4',
  theta: '#3b82f6',
  delta: '#6366f1',
  spindle: '#a78bfa',
  ripple: '#ec4899',
}

function typeColor(t?: string) {
  return TYPE_COLORS[t ?? 'other'] ?? '#6b7280'
}

function bandColor(b?: string) {
  return BAND_COLORS[b ?? ''] ?? '#6b7280'
}

function TypeBadge({ type }: { type?: string }) {
  const label = (type ?? 'other').replace(/_/g, ' ')
  return (
    <span
      className="px-1.5 py-0.5 rounded text-xs font-mono"
      style={{ background: typeColor(type) + '25', color: typeColor(type), border: `1px solid ${typeColor(type)}40` }}
    >
      {label}
    </span>
  )
}

function CircuitTag({ circuit, active }: { circuit: string; active?: boolean }) {
  const label = circuit.replace(/_/g, ' ')
  return (
    <span
      className={`px-2 py-1 rounded text-xs font-mono transition-colors ${
        active ? 'bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/40' : 'bg-neural-800 text-neural-400 border border-neural-700'
      }`}
    >
      {label}
    </span>
  )
}

// ── Disorders tab ──────────────────────────────────────────────────────────────

function DisordersTab() {
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [typeFilter, setTypeFilter] = useState<string | null>(null)

  const { data, isLoading } = useQuery({
    queryKey: ['disorders', typeFilter],
    queryFn: () => fetchDisorders(typeFilter ? { disorder_type: typeFilter } : undefined),
    staleTime: Infinity,
  })

  const { data: detail } = useQuery({
    queryKey: ['disorder', selectedId],
    queryFn: () => fetchDisorder(selectedId!),
    enabled: selectedId !== null,
    staleTime: Infinity,
  })

  const allTypes = data ? Object.keys(data.by_type) : []

  if (isLoading) return <div className="text-neural-500 text-sm p-8">Loading disorders…</div>
  if (!data) return null

  return (
    <div className="flex gap-6 h-full">
      {/* Left: list */}
      <div className="w-72 flex-shrink-0 space-y-3">
        {/* Type filter */}
        <div className="flex flex-wrap gap-1.5">
          <button
            onClick={() => setTypeFilter(null)}
            className={`px-2 py-0.5 rounded text-xs transition-colors ${!typeFilter ? 'bg-neural-700 text-neural-100' : 'text-neural-500 hover:text-neural-300'}`}
          >
            all
          </button>
          {allTypes.map(t => (
            <button
              key={t}
              onClick={() => setTypeFilter(t === typeFilter ? null : t)}
              className="px-2 py-0.5 rounded text-xs transition-colors"
              style={{
                background: typeFilter === t ? typeColor(t) + '30' : 'transparent',
                color: typeColor(t),
                border: `1px solid ${typeColor(t)}${typeFilter === t ? '60' : '20'}`,
              }}
            >
              {t.replace(/_/g, ' ')}
            </button>
          ))}
        </div>

        <div className="space-y-1 overflow-y-auto max-h-[calc(100vh-280px)]">
          {data.disorders.map(d => (
            <button
              key={d.id}
              onClick={() => setSelectedId(d.id === selectedId ? null : d.id)}
              className={`w-full text-left px-3 py-2 rounded transition-colors ${
                selectedId === d.id
                  ? 'bg-neural-800 border border-neural-600'
                  : 'hover:bg-neural-900 border border-transparent'
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <span className="text-sm text-neural-200 leading-tight">{d.label}</span>
                {d.icd11 && <span className="text-xs text-neural-600 font-mono shrink-0">{d.icd11}</span>}
              </div>
              <div className="mt-1">
                <TypeBadge type={d.type} />
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Right: detail */}
      <div className="flex-1 min-w-0">
        {!detail ? (
          <div className="flex items-center justify-center h-64 text-neural-600 text-sm">
            Select a disorder to view its circuit disruptions and biomarkers
          </div>
        ) : (
          <DisorderDetailPanel detail={detail} />
        )}
      </div>
    </div>
  )
}

function DisorderDetailPanel({ detail }: { detail: DisorderDetail }) {
  return (
    <div className="space-y-6">
      <div>
        <div className="flex items-center gap-3 mb-1">
          <h2 className="text-xl font-semibold text-neural-100">{detail.label}</h2>
          <TypeBadge type={detail.type} />
          {detail.icd11 && <span className="text-xs text-neural-600 font-mono">ICD-11: {detail.icd11}</span>}
          {detail.dsm5 && <span className="text-xs text-neural-600 font-mono">DSM-5: {detail.dsm5}</span>}
        </div>
      </div>

      {/* Disrupted circuits */}
      <section>
        <h3 className="text-xs uppercase tracking-widest text-neural-500 mb-2">Disrupted Circuits</h3>
        <div className="flex flex-wrap gap-2">
          {detail.disrupted_circuits.map(c => (
            <CircuitTag key={c} circuit={c} active />
          ))}
        </div>
      </section>

      {/* Oscillation biomarkers */}
      {detail.oscillation_biomarkers.length > 0 && (
        <section>
          <h3 className="text-xs uppercase tracking-widest text-neural-500 mb-2">Oscillation Biomarkers</h3>
          <div className="space-y-2">
            {detail.oscillation_biomarkers.map((b, i) => (
              <div key={i} className="flex items-start gap-3 bg-neural-900 rounded px-3 py-2">
                <span
                  className="px-2 py-0.5 rounded text-xs font-mono font-medium shrink-0"
                  style={{ background: bandColor(b.band) + '25', color: bandColor(b.band) }}
                >
                  {b.band}
                </span>
                <div className="text-sm">
                  {b.direction && (
                    <span className={`text-xs font-mono mr-2 ${b.direction.includes('reduced') ? 'text-red-400' : 'text-emerald-400'}`}>
                      {b.direction.replace(/_/g, ' ')}
                    </span>
                  )}
                  {b.region && <span className="text-neural-500 text-xs mr-2">{b.region.replace(/_/g, ' ')}</span>}
                  {b.note && <span className="text-neural-400 text-xs">{b.note}</span>}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Species models */}
      {detail.species_models.length > 0 && (
        <section>
          <h3 className="text-xs uppercase tracking-widest text-neural-500 mb-2">Animal Models</h3>
          <div className="grid grid-cols-2 gap-2">
            {detail.species_models.map((m, i) => (
              <div key={i} className="bg-neural-900 rounded px-3 py-2 text-sm">
                <span className="text-neural-300 font-medium capitalize">{m.species}</span>
                <span className="text-neural-600 mx-1">·</span>
                <span className="text-neural-400 text-xs font-mono">{m.model}</span>
                {m.face_validity && (
                  <div className="text-xs text-neural-600 mt-0.5">{m.face_validity.replace(/_/g, ' ')}</div>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Topics */}
      {detail.topics.length > 0 && (
        <section>
          <h3 className="text-xs uppercase tracking-widest text-neural-500 mb-2">Related Topics</h3>
          <div className="flex flex-wrap gap-1.5">
            {detail.topics.map(t => (
              <span key={t} className="px-2 py-0.5 rounded text-xs bg-neural-800 text-neural-400 font-mono">
                {t.replace(/_/g, ' ')}
              </span>
            ))}
          </div>
        </section>
      )}

      {/* Key papers */}
      {detail.key_papers.length > 0 && (
        <section>
          <h3 className="text-xs uppercase tracking-widest text-neural-500 mb-2">Key Papers</h3>
          <ul className="space-y-1.5">
            {detail.key_papers.map((p, i) => (
              <li key={i} className="text-xs text-neural-400 bg-neural-900 rounded px-3 py-2">
                {p}
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* Diagnostic methods */}
      {detail.diagnostic_methods.length > 0 && (
        <section>
          <h3 className="text-xs uppercase tracking-widest text-neural-500 mb-2">Diagnostic Methods</h3>
          <div className="flex flex-wrap gap-1.5">
            {detail.diagnostic_methods.map(m => (
              <span key={m} className="px-2 py-0.5 rounded text-xs bg-neural-800 text-neural-400 font-mono">
                {m.replace(/_/g, ' ')}
              </span>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}

// ── Circuit matrix tab ─────────────────────────────────────────────────────────

function CircuitMatrixTab() {
  const [hoveredCircuit, setHoveredCircuit] = useState<string | null>(null)

  const { data, isLoading } = useQuery({
    queryKey: ['circuit-matrix'],
    queryFn: fetchCircuitMatrix,
    staleTime: Infinity,
  })

  if (isLoading) return <div className="text-neural-500 text-sm p-8">Loading circuit matrix…</div>
  if (!data) return null

  return (
    <div className="space-y-4">
      <p className="text-sm text-neural-500">
        Which circuits are disrupted across disorders? Each row = a circuit; coloured cells = disorders that disrupt it.
      </p>

      <div className="space-y-2 overflow-y-auto max-h-[calc(100vh-260px)]">
        {data.circuits.map(row => (
          <div
            key={row.circuit}
            className="flex items-center gap-3 group"
            onMouseEnter={() => setHoveredCircuit(row.circuit)}
            onMouseLeave={() => setHoveredCircuit(null)}
          >
            <div className="w-48 shrink-0">
              <span className={`text-xs font-mono transition-colors ${hoveredCircuit === row.circuit ? 'text-neural-100' : 'text-neural-500'}`}>
                {row.circuit.replace(/_/g, ' ')}
              </span>
            </div>
            <div className="flex flex-wrap gap-1 flex-1">
              {row.disorders.map(d => (
                <span
                  key={d.id}
                  className="px-1.5 py-0.5 rounded text-xs transition-all"
                  style={{
                    background: typeColor(d.type) + '20',
                    color: typeColor(d.type),
                    border: `1px solid ${typeColor(d.type)}30`,
                  }}
                  title={d.label}
                >
                  {d.label.length > 20 ? d.label.slice(0, 18) + '…' : d.label}
                </span>
              ))}
            </div>
            <div className="shrink-0 w-8 text-right">
              <span className="text-xs text-neural-700">{row.n_disorders}</span>
            </div>
          </div>
        ))}
      </div>

      <div className="flex flex-wrap gap-2 pt-2 border-t border-neural-800">
        {Object.entries(TYPE_COLORS).map(([type, color]) => (
          <div key={type} className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-sm" style={{ background: color }} />
            <span className="text-xs text-neural-600">{type.replace(/_/g, ' ')}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Concepts tab ───────────────────────────────────────────────────────────────

function ConceptsTab() {
  const [selectedId, setSelectedId] = useState<string | null>(null)

  const { data } = useQuery({
    queryKey: ['concepts'],
    queryFn: () => fetchConcepts(),
    staleTime: Infinity,
  })

  const { data: detail } = useQuery({
    queryKey: ['concept', selectedId],
    queryFn: () => fetchConcept(selectedId!),
    enabled: selectedId !== null,
    staleTime: Infinity,
  })

  if (!data) return <div className="text-neural-500 text-sm p-8">Loading concepts…</div>

  return (
    <div className="flex gap-6 h-full">
      {/* Left: list */}
      <div className="w-64 flex-shrink-0 space-y-1 overflow-y-auto max-h-[calc(100vh-260px)]">
        {data.concepts.map(c => (
          <button
            key={c.id}
            onClick={() => setSelectedId(c.id === selectedId ? null : c.id)}
            className={`w-full text-left px-3 py-2 rounded transition-colors ${
              selectedId === c.id
                ? 'bg-neural-800 border border-neural-600'
                : 'hover:bg-neural-900 border border-transparent'
            }`}
          >
            <div className="text-sm text-neural-200 leading-tight">{c.label}</div>
            {c.concept_type && (
              <div className="text-xs text-neural-600 font-mono mt-0.5">{c.concept_type.replace(/_/g, ' ')}</div>
            )}
          </button>
        ))}
      </div>

      {/* Right: detail */}
      <div className="flex-1 min-w-0">
        {!detail ? (
          <div className="flex items-center justify-center h-64 text-neural-600 text-sm">
            Select a concept to view its definition, formula, and predictions
          </div>
        ) : (
          <ConceptDetailPanel concept={detail} />
        )}
      </div>
    </div>
  )
}

function ConceptDetailPanel({ concept }: { concept: ConceptSummary & { narrower_resolved?: { id: string; label: string }[] } }) {
  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-semibold text-neural-100 mb-1">{concept.label}</h2>
        {concept.concept_type && (
          <span className="text-xs text-neural-500 font-mono">{concept.concept_type.replace(/_/g, ' ')}</span>
        )}
        {concept.aliases.length > 0 && (
          <p className="text-xs text-neural-600 mt-1">Also: {concept.aliases.slice(0, 4).join(' · ')}</p>
        )}
      </div>

      <section>
        <p className="text-sm text-neural-300 leading-relaxed">{concept.definition}</p>
      </section>

      {concept.formula && (
        <section>
          <h3 className="text-xs uppercase tracking-widest text-neural-500 mb-1">Formula</h3>
          <code className="block bg-neural-900 rounded px-3 py-2 text-xs text-accent-cyan font-mono break-all">
            {concept.formula}
          </code>
        </section>
      )}

      {concept.testable_predictions.length > 0 && (
        <section>
          <h3 className="text-xs uppercase tracking-widest text-neural-500 mb-2">Testable Predictions</h3>
          <ul className="space-y-1.5">
            {concept.testable_predictions.map((p, i) => (
              <li key={i} className="flex gap-2 text-sm">
                <span className="text-neural-600 shrink-0 mt-0.5">·</span>
                <span className="text-neural-400">{p}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {concept.related_methods.length > 0 && (
        <section>
          <h3 className="text-xs uppercase tracking-widest text-neural-500 mb-2">Related Methods</h3>
          <div className="flex flex-wrap gap-1.5">
            {concept.related_methods.map(m => (
              <span key={m} className="px-2 py-0.5 rounded text-xs bg-neural-800 text-neural-400 font-mono">
                {m.replace(/_/g, ' ')}
              </span>
            ))}
          </div>
        </section>
      )}

      {concept.related_regions.length > 0 && (
        <section>
          <h3 className="text-xs uppercase tracking-widest text-neural-500 mb-2">Related Regions</h3>
          <div className="flex flex-wrap gap-1.5">
            {concept.related_regions.map(r => (
              <span key={r} className="px-2 py-0.5 rounded text-xs bg-neural-800 text-accent-cyan/70 font-mono">
                {r.replace(/_/g, ' ')}
              </span>
            ))}
          </div>
        </section>
      )}

      {concept.broader_concept && (
        <section>
          <h3 className="text-xs uppercase tracking-widest text-neural-500 mb-1">Broader</h3>
          <span className="text-xs text-neural-400 font-mono">{concept.broader_concept.replace(/_/g, ' ')}</span>
        </section>
      )}

      {(concept.narrower_resolved?.length ?? 0) > 0 && (
        <section>
          <h3 className="text-xs uppercase tracking-widest text-neural-500 mb-2">Narrower Concepts</h3>
          <div className="flex flex-wrap gap-1.5">
            {concept.narrower_resolved!.map(n => (
              <span key={n.id} className="px-2 py-0.5 rounded text-xs bg-neural-800 text-neural-300 font-mono">
                {n.label}
              </span>
            ))}
          </div>
        </section>
      )}

      {concept.scholarpedia_url && (
        <section>
          <a
            href={concept.scholarpedia_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-accent-cyan/70 hover:text-accent-cyan transition-colors"
          >
            Scholarpedia article
          </a>
        </section>
      )}
    </div>
  )
}

// ── Page shell ─────────────────────────────────────────────────────────────────

const TABS: { id: Tab; label: string; desc: string }[] = [
  { id: 'disorders', label: 'Disorders', desc: '28 neuropsychiatric disorders with circuit mappings' },
  { id: 'circuits', label: 'Circuit Matrix', desc: 'Which circuits are disrupted across disorders' },
  { id: 'concepts', label: 'Theory', desc: '21 foundational concepts with formulas & predictions' },
]

export function DisorderMapPage() {
  const [activeTab, setActiveTab] = useState<Tab>('disorders')

  return (
    <div className="max-w-6xl mx-auto px-6 lg:px-8 py-10">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-neural-100 mb-1">Disorder & Theory Map</h1>
        <p className="text-sm text-neural-500">
          Neuropsychiatric disorder–circuit mappings, oscillation biomarkers, and foundational theoretical concepts.
        </p>
      </div>

      {/* Tab bar */}
      <div className="flex gap-0 mb-8 border-b border-neural-800">
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2.5 text-sm transition-colors border-b-2 -mb-px ${
              activeTab === tab.id
                ? 'border-accent-cyan text-neural-100'
                : 'border-transparent text-neural-500 hover:text-neural-300'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === 'disorders' && <DisordersTab />}
      {activeTab === 'circuits' && <CircuitMatrixTab />}
      {activeTab === 'concepts' && <ConceptsTab />}
    </div>
  )
}
