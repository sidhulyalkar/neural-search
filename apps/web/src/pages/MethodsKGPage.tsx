import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  fetchMethodCategories,
  fetchMethods,
  fetchMethodDetail,
  fetchOscillations,
  fetchHomologyGroups,
  fetchParadigms,
  fetchModalities,
  fetchCrossModalPairs,
  fetchModalityGroups,
  type MethodDetail,
  type OscillationSignature,
  type Modality,
  type CrossModalPair,
} from '../api/methods'

const CONFIDENCE_COLORS: Record<string, string> = {
  high: '#10b981',
  medium: '#f59e0b',
  low: '#ef4444',
}

const BAND_COLORS: Record<string, string | undefined> = {
  delta: '#6366f1',
  theta: '#3b82f6',
  alpha: '#06b6d4',
  beta: '#10b981',
  beta_low: '#34d399',
  beta_high: '#059669',
  low_gamma: '#f59e0b',
  high_gamma: '#ef4444',
  gamma: '#f97316',
  ripple: '#ec4899',
  ultra_fast: '#8b5cf6',
}

const SPECIES_COLORS: Record<string, string | undefined> = {
  mouse: '#f59e0b',
  rat: '#f97316',
  macaque: '#10b981',
  human: '#22d3ee',
  cat: '#a78bfa',
}

const CLASS_COLORS: Record<string, string | undefined> = {
  electrophysiology: '#22d3ee',
  hemodynamic: '#f59e0b',
  optical: '#10b981',
  structural: '#8b5cf6',
  molecular: '#f97316',
  behavioral: '#ec4899',
}

const INVASIVENESS_COLORS: Record<string, string | undefined> = {
  invasive: '#ef4444',
  semi_invasive: '#f59e0b',
  non_invasive: '#10b981',
  ex_vivo: '#8b5cf6',
}

type Tab = 'methods' | 'oscillations' | 'homology' | 'paradigms' | 'modalities'

export function MethodsKGPage() {
  const [activeTab, setActiveTab] = useState<Tab>('methods')
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [selectedMethod, setSelectedMethod] = useState<string | null>(null)
  const [selectedBand, setSelectedBand] = useState<string | null>(null)
  const [selectedSpecies, setSelectedSpecies] = useState<string | null>(null)
  const [confidenceFilter, setConfidenceFilter] = useState<string | null>(null)

  const { data: categories = [] } = useQuery({
    queryKey: ['method-categories'],
    queryFn: fetchMethodCategories,
    staleTime: Infinity,
  })

  const { data: methods = [] } = useQuery({
    queryKey: ['methods', selectedCategory],
    queryFn: () => fetchMethods(selectedCategory ? { category: selectedCategory } : undefined),
    staleTime: Infinity,
  })

  const { data: methodDetail } = useQuery({
    queryKey: ['method-detail', selectedMethod],
    queryFn: () => fetchMethodDetail(selectedMethod!),
    enabled: !!selectedMethod,
    staleTime: Infinity,
  })

  const { data: oscillationData } = useQuery({
    queryKey: ['oscillations', selectedBand, selectedSpecies],
    queryFn: () =>
      fetchOscillations({
        ...(selectedBand ? { band: selectedBand } : {}),
        ...(selectedSpecies ? { species: selectedSpecies } : {}),
      }),
    enabled: activeTab === 'oscillations',
    staleTime: Infinity,
  })

  const { data: homologyGroups = [] } = useQuery({
    queryKey: ['homology', confidenceFilter, selectedSpecies],
    queryFn: () =>
      fetchHomologyGroups({
        ...(confidenceFilter ? { confidence: confidenceFilter } : {}),
        ...(selectedSpecies ? { species: selectedSpecies } : {}),
      }),
    enabled: activeTab === 'homology',
    staleTime: Infinity,
  })

  const { data: paradigms = [] } = useQuery({
    queryKey: ['paradigms', selectedSpecies],
    queryFn: () => fetchParadigms(selectedSpecies ? { species: selectedSpecies } : undefined),
    enabled: activeTab === 'paradigms',
    staleTime: Infinity,
  })

  const [selectedModalityClass, setSelectedModalityClass] = useState<string | null>(null)
  const [selectedModality, setSelectedModality] = useState<string | null>(null)

  const { data: modalities = [] } = useQuery({
    queryKey: ['modalities', selectedModalityClass],
    queryFn: () =>
      fetchModalities(selectedModalityClass ? { modality_class: selectedModalityClass } : undefined),
    enabled: activeTab === 'modalities',
    staleTime: Infinity,
  })

  const { data: crossModalPairs = [] } = useQuery({
    queryKey: ['cross-modal-pairs', selectedModality],
    queryFn: () =>
      fetchCrossModalPairs(selectedModality ? { modality: selectedModality } : undefined),
    enabled: activeTab === 'modalities',
    staleTime: Infinity,
  })

  const { data: modalityGroups = [] } = useQuery({
    queryKey: ['modality-groups'],
    queryFn: fetchModalityGroups,
    enabled: activeTab === 'modalities',
    staleTime: Infinity,
  })

  return (
    <div className="min-h-screen bg-neural-bg text-neural-text p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">
            Neural Methods & Mathematical Foundations
          </h1>
          <p className="text-neural-muted max-w-2xl">
            Mathematical frameworks, analysis methods, cross-species homology, oscillatory
            signatures, and validated behavioral paradigms — the measurement layer of the KG.
          </p>
        </div>

        {/* Tab bar */}
        <div className="flex gap-1 mb-6 border-b border-neural-border">
          {(
            [
              { id: 'methods', label: 'Analysis Methods' },
              { id: 'oscillations', label: 'Oscillation Signatures' },
              { id: 'homology', label: 'Species Homology' },
              { id: 'paradigms', label: 'Cross-Species Paradigms' },
              { id: 'modalities', label: 'Recording Modalities' },
            ] as { id: Tab; label: string }[]
          ).map((t) => (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                activeTab === t.id
                  ? 'border-accent-cyan text-accent-cyan'
                  : 'border-transparent text-neural-muted hover:text-neural-text'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* ── Methods tab ─────────────────────────────────────────────────── */}
        {activeTab === 'methods' && (
          <div className="flex gap-6">
            {/* Left: categories + method list */}
            <div className="w-64 shrink-0 space-y-4">
              <div>
                <p className="text-xs text-neural-muted uppercase tracking-widest mb-2">
                  Categories
                </p>
                <div className="space-y-1">
                  <button
                    onClick={() => {
                      setSelectedCategory(null)
                      setSelectedMethod(null)
                    }}
                    className={`w-full text-left px-3 py-2 text-sm rounded ${
                      !selectedCategory
                        ? 'bg-accent-cyan/10 text-accent-cyan'
                        : 'text-neural-muted hover:text-neural-text hover:bg-neural-surface/50'
                    }`}
                  >
                    All ({methods.length})
                  </button>
                  {categories.map((cat) => (
                    <button
                      key={cat.id}
                      onClick={() => {
                        setSelectedCategory(cat.id)
                        setSelectedMethod(null)
                      }}
                      className={`w-full text-left px-3 py-2 text-sm rounded ${
                        selectedCategory === cat.id
                          ? 'bg-accent-cyan/10 text-accent-cyan'
                          : 'text-neural-muted hover:text-neural-text hover:bg-neural-surface/50'
                      }`}
                    >
                      {cat.label}
                      <span className="ml-1 text-xs opacity-60">({cat.count})</span>
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <p className="text-xs text-neural-muted uppercase tracking-widest mb-2">Methods</p>
                <div className="space-y-1 max-h-96 overflow-y-auto">
                  {methods.map((m) => (
                    <button
                      key={m.id}
                      onClick={() => setSelectedMethod(m.id)}
                      className={`w-full text-left px-3 py-1.5 text-xs rounded leading-tight ${
                        selectedMethod === m.id
                          ? 'bg-accent-emerald/10 text-accent-emerald'
                          : 'text-neural-muted hover:text-neural-text hover:bg-neural-surface/50'
                      }`}
                    >
                      {m.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Right: method detail */}
            <div className="flex-1">
              {selectedMethod && methodDetail ? (
                <MethodDetailPanel method={methodDetail} />
              ) : (
                <MethodGrid methods={methods} onSelect={setSelectedMethod} />
              )}
            </div>
          </div>
        )}

        {/* ── Oscillations tab ──────────────────────────────────────────── */}
        {activeTab === 'oscillations' && (
          <div>
            {/* Filters */}
            <div className="flex gap-4 mb-6 flex-wrap">
              <div>
                <p className="text-xs text-neural-muted mb-1">Frequency Band</p>
                <div className="flex gap-1 flex-wrap">
                  <button
                    onClick={() => setSelectedBand(null)}
                    className={`px-2 py-1 text-xs rounded ${!selectedBand ? 'bg-accent-cyan/20 text-accent-cyan' : 'bg-neural-surface text-neural-muted hover:text-white'}`}
                  >
                    All
                  </button>
                  {oscillationData?.frequency_bands.map((b) => (
                    <button
                      key={b.id}
                      onClick={() => setSelectedBand(b.id === selectedBand ? null : b.id)}
                      className={`px-2 py-1 text-xs rounded font-mono ${
                        selectedBand === b.id
                          ? 'text-white'
                          : 'bg-neural-surface text-neural-muted hover:text-white'
                      }`}
                      style={
                        selectedBand === b.id
                          ? { backgroundColor: BAND_COLORS[b.id] + '40', color: BAND_COLORS[b.id] }
                          : {}
                      }
                    >
                      {b.label}
                      <span className="ml-1 opacity-60">
                        {b.range_hz[0]}–{b.range_hz[1]}Hz
                      </span>
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <p className="text-xs text-neural-muted mb-1">Species</p>
                <div className="flex gap-1">
                  {['mouse', 'rat', 'human', 'macaque'].map((sp) => (
                    <button
                      key={sp}
                      onClick={() => setSelectedSpecies(sp === selectedSpecies ? null : sp)}
                      className="px-2 py-1 text-xs rounded"
                      style={{
                        backgroundColor:
                          selectedSpecies === sp ? SPECIES_COLORS[sp] + '30' : '#1e2a3a',
                        color: selectedSpecies === sp ? SPECIES_COLORS[sp] : '#94a3b8',
                        border: `1px solid ${selectedSpecies === sp ? SPECIES_COLORS[sp] : 'transparent'}`,
                      }}
                    >
                      {sp}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Oscillation grid */}
            <OscillationGrid signatures={oscillationData?.signatures ?? []} />
          </div>
        )}

        {/* ── Homology tab ──────────────────────────────────────────────── */}
        {activeTab === 'homology' && (
          <div>
            <div className="flex gap-4 mb-6">
              <div>
                <p className="text-xs text-neural-muted mb-1">Confidence</p>
                <div className="flex gap-1">
                  {['high', 'medium', 'low'].map((c) => (
                    <button
                      key={c}
                      onClick={() => setConfidenceFilter(c === confidenceFilter ? null : c)}
                      className="px-3 py-1 text-xs rounded capitalize font-medium"
                      style={{
                        backgroundColor:
                          confidenceFilter === c ? CONFIDENCE_COLORS[c] + '30' : '#1e2a3a',
                        color: confidenceFilter === c ? CONFIDENCE_COLORS[c] : '#94a3b8',
                        border: `1px solid ${confidenceFilter === c ? CONFIDENCE_COLORS[c] : 'transparent'}`,
                      }}
                    >
                      {c}
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <p className="text-xs text-neural-muted mb-1">Filter species</p>
                <div className="flex gap-1">
                  {['mouse', 'rat', 'human', 'macaque'].map((sp) => (
                    <button
                      key={sp}
                      onClick={() => setSelectedSpecies(sp === selectedSpecies ? null : sp)}
                      className="px-2 py-1 text-xs rounded"
                      style={{
                        backgroundColor:
                          selectedSpecies === sp ? SPECIES_COLORS[sp] + '30' : '#1e2a3a',
                        color: selectedSpecies === sp ? SPECIES_COLORS[sp] : '#94a3b8',
                        border: `1px solid ${selectedSpecies === sp ? SPECIES_COLORS[sp] : 'transparent'}`,
                      }}
                    >
                      {sp}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <HomologyTable groups={homologyGroups} />
          </div>
        )}

        {/* ── Paradigms tab ──────────────────────────────────────────────── */}
        {activeTab === 'paradigms' && (
          <div>
            <div className="flex gap-4 mb-6">
              <div>
                <p className="text-xs text-neural-muted mb-1">Species</p>
                <div className="flex gap-1">
                  {['mouse', 'rat', 'human', 'macaque'].map((sp) => (
                    <button
                      key={sp}
                      onClick={() => setSelectedSpecies(sp === selectedSpecies ? null : sp)}
                      className="px-2 py-1 text-xs rounded"
                      style={{
                        backgroundColor:
                          selectedSpecies === sp ? SPECIES_COLORS[sp] + '30' : '#1e2a3a',
                        color: selectedSpecies === sp ? SPECIES_COLORS[sp] : '#94a3b8',
                        border: `1px solid ${selectedSpecies === sp ? SPECIES_COLORS[sp] : 'transparent'}`,
                      }}
                    >
                      {sp}
                    </button>
                  ))}
                </div>
              </div>
            </div>
            <ParadigmGrid paradigms={paradigms} />
          </div>
        )}
        {/* ── Modalities tab ────────────────────────────────────────────── */}
        {activeTab === 'modalities' && (
          <ModalitiesTab
            modalities={modalities}
            crossModalPairs={crossModalPairs}
            modalityGroups={modalityGroups}
            selectedModalityClass={selectedModalityClass}
            setSelectedModalityClass={setSelectedModalityClass}
            selectedModality={selectedModality}
            setSelectedModality={setSelectedModality}
          />
        )}
      </div>
    </div>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────

function MethodGrid({
  methods,
  onSelect,
}: {
  methods: ReturnType<typeof fetchMethods> extends Promise<infer T> ? T : never
  onSelect: (id: string) => void
}) {
  return (
    <div className="grid grid-cols-2 gap-3">
      {methods.map((m) => (
        <button
          key={m.id}
          onClick={() => onSelect(m.id)}
          className="text-left p-4 rounded-lg bg-neural-surface border border-neural-border hover:border-accent-cyan/50 transition-colors group"
        >
          <p className="font-medium text-sm text-white group-hover:text-accent-cyan mb-1">
            {m.label}
          </p>
          {m.formula && (
            <p className="text-xs font-mono text-accent-emerald/70 mb-2 truncate">{m.formula}</p>
          )}
          <div className="flex flex-wrap gap-1 mt-2">
            {m.topics.slice(0, 3).map((t) => (
              <span key={t} className="px-1.5 py-0.5 text-xs rounded bg-neural-border/50 text-neural-muted">
                {t.replace(/_/g, ' ')}
              </span>
            ))}
          </div>
        </button>
      ))}
    </div>
  )
}

function MethodDetailPanel({ method }: { method: MethodDetail }) {
  const assumptionEntries = method.assumptions
    ? Object.entries(method.assumptions)
    : []

  return (
    <div className="space-y-6">
      <div>
        <div className="flex items-start justify-between mb-2">
          <h2 className="text-xl font-bold text-white">{method.label}</h2>
          <span className="text-xs px-2 py-1 rounded bg-accent-cyan/10 text-accent-cyan">
            {method.category_label}
          </span>
        </div>
        {method.aliases && method.aliases.length > 0 && (
          <p className="text-sm text-neural-muted mb-3">
            Also known as: {method.aliases.join(', ')}
          </p>
        )}
        {method.principle && (
          <p className="text-sm text-neural-text leading-relaxed border-l-2 border-accent-cyan/40 pl-3">
            {method.principle}
          </p>
        )}
      </div>

      {method.formula && (
        <div>
          <p className="text-xs text-neural-muted uppercase tracking-widest mb-2">Formula</p>
          <div className="bg-neural-bg rounded p-3 border border-neural-border">
            <p className="font-mono text-sm text-accent-emerald leading-relaxed">{method.formula}</p>
          </div>
        </div>
      )}

      {method.mathematical_basis && (
        <div>
          <p className="text-xs text-neural-muted uppercase tracking-widest mb-2">
            Mathematical Basis
          </p>
          <ul className="space-y-1">
            {method.mathematical_basis.map((b, i) => (
              <li key={i} className="text-sm text-neural-text flex gap-2">
                <span className="text-accent-violet shrink-0 mt-0.5">▸</span>
                <span>{b}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {assumptionEntries.length > 0 && (
        <div>
          <p className="text-xs text-neural-muted uppercase tracking-widest mb-2">
            Mathematical Assumptions
          </p>
          <div className="space-y-2">
            {assumptionEntries.map(([key, desc]) => (
              <div
                key={key}
                className="p-3 rounded bg-amber-500/5 border border-amber-500/20"
              >
                <p className="text-xs font-semibold text-amber-400 mb-1 capitalize">
                  {key.replace(/_/g, ' ')}
                </p>
                <p className="text-sm text-neural-text">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {method.limitations && (
        <div>
          <p className="text-xs text-neural-muted uppercase tracking-widest mb-2">Limitations</p>
          <ul className="space-y-1">
            {method.limitations.map((l, i) => (
              <li key={i} className="text-sm text-neural-text flex gap-2">
                <span className="text-rose-400 shrink-0 mt-0.5">⚠</span>
                <span>{l}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {method.key_papers && method.key_papers.length > 0 && (
        <div>
          <p className="text-xs text-neural-muted uppercase tracking-widest mb-2">Key Papers</p>
          <ul className="space-y-1">
            {method.key_papers.map((p, i) => (
              <li key={i} className="text-sm text-accent-cyan/80">
                {p}
              </li>
            ))}
          </ul>
        </div>
      )}

      {method.species_note && (
        <div className="p-3 rounded bg-accent-emerald/5 border border-accent-emerald/20">
          <p className="text-xs font-semibold text-accent-emerald mb-1">Species Notes</p>
          <p className="text-sm text-neural-text">{method.species_note}</p>
        </div>
      )}

      {method.topics.length > 0 && (
        <div>
          <p className="text-xs text-neural-muted uppercase tracking-widest mb-2">Related Topics</p>
          <div className="flex flex-wrap gap-2">
            {method.topics.map((t) => (
              <span
                key={t}
                className="px-2 py-1 text-xs rounded bg-accent-violet/10 text-accent-violet"
              >
                {t.replace(/_/g, ' ')}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function OscillationGrid({ signatures }: { signatures: OscillationSignature[] }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {signatures.map((sig, i) => {
        const bandColor = BAND_COLORS[sig.frequency_band] ?? '#94a3b8'
        return (
          <div
            key={`${sig.region_id}-${sig.frequency_band}-${i}`}
            className="p-4 rounded-lg bg-neural-surface border border-neural-border"
            style={{ borderLeftColor: bandColor, borderLeftWidth: 3 }}
          >
            <div className="flex items-start justify-between mb-2">
              <div>
                <p className="font-semibold text-white text-sm">
                  {sig.region_id.replace(/_/g, ' ')}
                </p>
                <span
                  className="text-xs font-mono px-1.5 py-0.5 rounded"
                  style={{ backgroundColor: bandColor + '20', color: bandColor }}
                >
                  {sig.frequency_band.replace(/_/g, ' ')}
                </span>
              </div>
              <div className="flex gap-1 flex-wrap justify-end">
                {sig.species.map((sp) => (
                  <span
                    key={sp}
                    className="text-xs px-1.5 py-0.5 rounded"
                    style={{
                      backgroundColor: (SPECIES_COLORS[sp] ?? '#1e2a3a') + '20',
                      color: SPECIES_COLORS[sp] ?? '#94a3b8',
                    }}
                  >
                    {sp}
                  </span>
                ))}
              </div>
            </div>
            <p className="text-sm text-neural-text mb-2">{sig.functional_role}</p>
            <p className="text-xs text-neural-muted italic">{sig.condition}</p>
            {sig.translational_significance && (
              <p className="text-xs text-accent-cyan mt-2 border-t border-neural-border pt-2">
                {sig.translational_significance}
              </p>
            )}
          </div>
        )
      })}
      {signatures.length === 0 && (
        <div className="col-span-2 text-center py-12 text-neural-muted">
          No oscillation signatures match the current filters.
        </div>
      )}
    </div>
  )
}

function HomologyTable({ groups }: { groups: ReturnType<typeof fetchHomologyGroups> extends Promise<infer T> ? T : never }) {
  return (
    <div className="space-y-4">
      {groups.map((group) => {
        const color = CONFIDENCE_COLORS[group.confidence] ?? '#94a3b8'
        return (
          <div
            key={group.group_id}
            className="p-4 rounded-lg bg-neural-surface border border-neural-border"
          >
            <div className="flex items-center gap-3 mb-2">
              <span
                className="text-xs font-bold px-2 py-0.5 rounded capitalize"
                style={{ backgroundColor: color + '20', color }}
              >
                {group.confidence}
              </span>
              <h3 className="font-semibold text-white">
                {group.group_id.replace(/_/g, ' ')}
              </h3>
              <span className="text-xs text-neural-muted">[{group.basis.join(', ')}]</span>
            </div>
            <p className="text-sm text-neural-text mb-3">{group.notes}</p>
            <div className="flex flex-wrap gap-2 mb-3">
              {group.members.map((m) => (
                <div
                  key={`${m.species}-${m.region_id}`}
                  className="px-3 py-1.5 rounded text-xs"
                  style={{
                    backgroundColor: (SPECIES_COLORS[m.species] ?? '#1e2a3a') + '15',
                    border: `1px solid ${(SPECIES_COLORS[m.species] ?? '#334155') + '40'}`,
                    color: SPECIES_COLORS[m.species] ?? '#94a3b8',
                  }}
                >
                  <span className="font-bold capitalize">{m.species}</span>
                  {' '}
                  <span className="opacity-80">{m.region_id.replace(/_/g, ' ')}</span>
                  {m.allen_acronym && (
                    <span className="font-mono ml-1 opacity-60">({m.allen_acronym})</span>
                  )}
                </div>
              ))}
            </div>
            {group.divergence && (
              <p className="text-xs text-amber-400/80 italic border-t border-neural-border pt-2">
                ⚠ {group.divergence}
              </p>
            )}
          </div>
        )
      })}
      {groups.length === 0 && (
        <div className="text-center py-12 text-neural-muted">
          No homology groups match the current filters.
        </div>
      )}
    </div>
  )
}

function ParadigmGrid({ paradigms }: { paradigms: ReturnType<typeof fetchParadigms> extends Promise<infer T> ? T : never }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {paradigms.map((p) => (
        <div
          key={p.id}
          className="p-4 rounded-lg bg-neural-surface border border-neural-border"
        >
          <h3 className="font-semibold text-white mb-1">{p.label}</h3>
          <p className="text-xs text-neural-muted mb-2 italic">{p.cognitive_construct.replace(/_/g, ' ')}</p>
          <p className="text-sm text-neural-text mb-3 leading-relaxed">{p.description}</p>

          {/* Species availability */}
          <div className="flex gap-1 flex-wrap mb-3">
            {p.species_available.map((sp) => (
              <span
                key={sp}
                className="text-xs px-2 py-0.5 rounded"
                style={{
                  backgroundColor: (SPECIES_COLORS[sp] ?? '#1e2a3a') + '20',
                  color: SPECIES_COLORS[sp] ?? '#94a3b8',
                }}
              >
                {sp}
              </span>
            ))}
          </div>

          {p.key_neural_signal && (
            <div className="p-2 rounded bg-accent-emerald/5 border border-accent-emerald/20 mb-2">
              <p className="text-xs font-medium text-accent-emerald">Key Signal</p>
              <p className="text-xs text-neural-text">{p.key_neural_signal.replace(/_/g, ' ')}</p>
            </div>
          )}

          {p.key_finding && (
            <p className="text-xs text-neural-muted italic border-t border-neural-border pt-2 mt-2">
              {p.key_finding}
            </p>
          )}
        </div>
      ))}
      {paradigms.length === 0 && (
        <div className="col-span-2 text-center py-12 text-neural-muted">
          No paradigms match the current filters.
        </div>
      )}
    </div>
  )
}

// ── Modalities tab ─────────────────────────────────────────────────────────

const MODALITY_CLASS_ORDER = [
  'electrophysiology',
  'hemodynamic',
  'optical',
  'structural',
  'molecular',
  'behavioral',
]

function ModalitiesTab({
  modalities,
  crossModalPairs,
  modalityGroups,
  selectedModalityClass,
  setSelectedModalityClass,
  selectedModality,
  setSelectedModality,
}: {
  modalities: Modality[]
  crossModalPairs: CrossModalPair[]
  modalityGroups: ReturnType<typeof fetchModalityGroups> extends Promise<infer T> ? T : never
  selectedModalityClass: string | null
  setSelectedModalityClass: (c: string | null) => void
  selectedModality: string | null
  setSelectedModality: (m: string | null) => void
}) {
  const [view, setView] = useState<'cards' | 'matrix'>('cards')

  return (
    <div>
      {/* View toggle */}
      <div className="flex items-center gap-4 mb-6">
        <div className="flex gap-1 p-1 bg-neural-surface rounded-lg">
          <button
            onClick={() => setView('cards')}
            className={`px-3 py-1 text-xs rounded ${view === 'cards' ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-neural-muted hover:text-white'}`}
          >
            Modality Cards
          </button>
          <button
            onClick={() => setView('matrix')}
            className={`px-3 py-1 text-xs rounded ${view === 'matrix' ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-neural-muted hover:text-white'}`}
          >
            Cross-Modal Matrix
          </button>
        </div>

        {view === 'cards' && (
          <div className="flex gap-1 flex-wrap">
            <button
              onClick={() => setSelectedModalityClass(null)}
              className={`px-2 py-1 text-xs rounded ${!selectedModalityClass ? 'bg-neural-border text-white' : 'text-neural-muted hover:text-white'}`}
            >
              All
            </button>
            {MODALITY_CLASS_ORDER.map((cls) => (
              <button
                key={cls}
                onClick={() => setSelectedModalityClass(cls === selectedModalityClass ? null : cls)}
                className="px-2 py-1 text-xs rounded capitalize"
                style={{
                  backgroundColor:
                    selectedModalityClass === cls ? (CLASS_COLORS[cls] ?? '#94a3b8') + '25' : '#1e2a3a',
                  color:
                    selectedModalityClass === cls ? CLASS_COLORS[cls] ?? '#94a3b8' : '#94a3b8',
                  border: `1px solid ${selectedModalityClass === cls ? (CLASS_COLORS[cls] ?? '#334155') : 'transparent'}`,
                }}
              >
                {cls.replace(/_/g, ' ')}
              </button>
            ))}
          </div>
        )}
      </div>

      {view === 'cards' ? (
        <div>
          {/* Modality group summary */}
          {!selectedModalityClass && modalityGroups.length > 0 && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
              {modalityGroups.map((g) => (
                <div
                  key={g.id}
                  className="p-3 rounded-lg bg-neural-surface border border-neural-border text-center"
                >
                  <p className="font-medium text-white text-sm mb-1">{g.label}</p>
                  <p className="text-xs text-neural-muted">{g.temporal_resolution_tier}</p>
                  <div className="flex gap-1 flex-wrap justify-center mt-2">
                    {g.members.slice(0, 4).map((m) => (
                      <span key={m} className="text-xs text-neural-muted font-mono opacity-60">
                        {m}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {modalities.map((mod) => (
              <ModalityCard
                key={mod.id}
                modality={mod}
                isSelected={selectedModality === mod.id}
                onSelect={() => setSelectedModality(selectedModality === mod.id ? null : mod.id)}
              />
            ))}
          </div>
        </div>
      ) : (
        <CrossModalMatrix
          pairs={crossModalPairs}
          modalities={modalities}
          selectedModality={selectedModality}
          onSelectModality={setSelectedModality}
        />
      )}
    </div>
  )
}

function ModalityCard({
  modality: mod,
  isSelected,
  onSelect,
}: {
  modality: Modality
  isSelected: boolean
  onSelect: () => void
}) {
  const classColor = CLASS_COLORS[mod.modality_class] ?? '#94a3b8'
  const invasivenessColor = INVASIVENESS_COLORS[mod.invasiveness] ?? '#94a3b8'

  return (
    <button
      onClick={onSelect}
      className="text-left p-4 rounded-lg bg-neural-surface border transition-colors w-full"
      style={{
        borderColor: isSelected ? classColor : 'var(--color-neural-border, #1e2a3a)',
        borderLeftWidth: 3,
        borderLeftColor: classColor,
      }}
    >
      <div className="flex items-start justify-between mb-2">
        <div>
          <h3 className="font-semibold text-white text-sm">{mod.label}</h3>
          <span
            className="text-xs px-1.5 py-0.5 rounded capitalize"
            style={{ backgroundColor: classColor + '20', color: classColor }}
          >
            {mod.modality_class.replace(/_/g, ' ')}
          </span>
        </div>
        <span
          className="text-xs px-2 py-0.5 rounded capitalize"
          style={{ backgroundColor: invasivenessColor + '20', color: invasivenessColor }}
        >
          {mod.invasiveness.replace(/_/g, ' ')}
        </span>
      </div>

      <p className="text-xs text-neural-muted mb-3">{mod.signal_origin}</p>

      {/* Resolution badges */}
      <div className="flex gap-2 mb-3">
        {mod.temporal_resolution_ms !== null && (
          <div className="text-center px-2 py-1 rounded bg-neural-bg">
            <p className="text-xs font-mono text-accent-cyan">{mod.temporal_resolution_ms}ms</p>
            <p className="text-xs text-neural-muted">temporal</p>
          </div>
        )}
        {mod.spatial_resolution_mm !== null && (
          <div className="text-center px-2 py-1 rounded bg-neural-bg">
            <p className="text-xs font-mono text-accent-violet">{mod.spatial_resolution_mm}mm</p>
            <p className="text-xs text-neural-muted">spatial</p>
          </div>
        )}
      </div>

      {/* Species */}
      <div className="flex gap-1 flex-wrap mb-2">
        {mod.species.map((sp) => (
          <span
            key={sp}
            className="text-xs px-1.5 py-0.5 rounded"
            style={{
              backgroundColor: (SPECIES_COLORS[sp] ?? '#1e2a3a') + '20',
              color: SPECIES_COLORS[sp] ?? '#94a3b8',
            }}
          >
            {sp}
          </span>
        ))}
      </div>

      {mod.cross_modal_value && (
        <p className="text-xs text-accent-emerald/80 border-t border-neural-border pt-2 mt-2 leading-relaxed">
          {mod.cross_modal_value}
        </p>
      )}
    </button>
  )
}

function CrossModalMatrix({
  pairs,
  modalities,
  selectedModality,
  onSelectModality,
}: {
  pairs: CrossModalPair[]
  modalities: Modality[]
  selectedModality: string | null
  onSelectModality: (id: string | null) => void
}) {
  const COMPAT_COLORS: Record<string, string> = {
    high: '#10b981',
    medium: '#f59e0b',
    low: '#ef4444',
  }

  const filteredPairs = selectedModality
    ? pairs.filter((p) => p.modality_a === selectedModality || p.modality_b === selectedModality)
    : pairs

  return (
    <div>
      {/* Modality selector for matrix */}
      <div className="mb-4">
        <p className="text-xs text-neural-muted mb-2">Filter pairs by modality</p>
        <div className="flex gap-1 flex-wrap">
          <button
            onClick={() => onSelectModality(null)}
            className={`px-2 py-1 text-xs rounded ${!selectedModality ? 'bg-neural-border text-white' : 'text-neural-muted hover:text-white'}`}
          >
            All pairs ({pairs.length})
          </button>
          {modalities.slice(0, 12).map((mod) => {
            const color = CLASS_COLORS[mod.modality_class] ?? '#94a3b8'
            return (
              <button
                key={mod.id}
                onClick={() => onSelectModality(selectedModality === mod.id ? null : mod.id)}
                className="px-2 py-1 text-xs rounded"
                style={{
                  backgroundColor: selectedModality === mod.id ? color + '25' : '#1e2a3a',
                  color: selectedModality === mod.id ? color : '#94a3b8',
                  border: `1px solid ${selectedModality === mod.id ? color : 'transparent'}`,
                }}
              >
                {mod.id}
              </button>
            )
          })}
        </div>
      </div>

      <div className="space-y-3">
        {filteredPairs.map((pair, i) => {
          const color = COMPAT_COLORS[pair.compatibility] ?? '#94a3b8'
          const aColor = CLASS_COLORS[
            modalities.find((m) => m.id === pair.modality_a)?.modality_class ?? ''
          ] ?? '#94a3b8'
          const bColor = CLASS_COLORS[
            modalities.find((m) => m.id === pair.modality_b)?.modality_class ?? ''
          ] ?? '#94a3b8'

          return (
            <div
              key={i}
              className="p-4 rounded-lg bg-neural-surface border border-neural-border"
              style={{ borderLeftColor: color, borderLeftWidth: 3 }}
            >
              {/* Pair header */}
              <div className="flex items-center gap-3 mb-2">
                <span
                  className="text-xs font-mono px-2 py-0.5 rounded"
                  style={{ backgroundColor: aColor + '20', color: aColor }}
                >
                  {pair.modality_a}
                </span>
                <span className="text-neural-muted text-xs">↔</span>
                <span
                  className="text-xs font-mono px-2 py-0.5 rounded"
                  style={{ backgroundColor: bColor + '20', color: bColor }}
                >
                  {pair.modality_b}
                </span>
                <span
                  className="text-xs px-2 py-0.5 rounded capitalize ml-auto"
                  style={{ backgroundColor: color + '20', color }}
                >
                  {pair.compatibility} compatibility
                </span>
              </div>

              <p className="text-sm text-neural-text mb-2">{pair.complementarity}</p>

              {pair.combination_methods && pair.combination_methods.length > 0 && (
                <div className="flex gap-1 flex-wrap">
                  {pair.combination_methods.map((m) => (
                    <span key={m} className="text-xs px-1.5 py-0.5 rounded bg-accent-violet/10 text-accent-violet font-mono">
                      {m}
                    </span>
                  ))}
                </div>
              )}

              {pair.neatlabs_relevance && (
                <p className="text-xs text-accent-cyan mt-2 border-t border-neural-border pt-2">
                  NEATLabs: {pair.neatlabs_relevance}
                </p>
              )}
            </div>
          )
        })}
        {filteredPairs.length === 0 && (
          <div className="text-center py-12 text-neural-muted">
            No cross-modal pairs match the current filter.
          </div>
        )}
      </div>
    </div>
  )
}
