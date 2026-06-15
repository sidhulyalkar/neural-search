import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { coverageApi, type SourceRate, type UncoveredRegion, type DarkPair } from '../api/coverage'

// ── Helpers ───────────────────────────────────────────────────────────────────

function pctColor(pct: number): string {
  if (pct >= 80) return 'text-emerald-400'
  if (pct >= 50) return 'text-yellow-400'
  if (pct >= 25) return 'text-orange-400'
  return 'text-red-400'
}

function PctBar({ pct }: { pct: number }) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-neural-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${
            pct >= 80 ? 'bg-emerald-500' : pct >= 50 ? 'bg-yellow-500' : pct >= 25 ? 'bg-orange-500' : 'bg-red-500'
          }`}
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
      </div>
      <span className={`text-xs font-mono w-12 text-right ${pctColor(pct)}`}>{pct.toFixed(1)}%</span>
    </div>
  )
}

// ── Sub-sections ──────────────────────────────────────────────────────────────

function SummaryPanel() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['coverage-summary'],
    queryFn: coverageApi.summary,
  })

  if (isLoading) return <div className="text-neural-500 text-sm">Loading…</div>
  if (error || !data) return <div className="text-red-400 text-sm">Failed to load coverage summary.</div>

  const dims = Object.entries(data.dimension_coverage).sort((a, b) => b[1].pct - a[1].pct)

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
      <div className="col-span-2 sm:col-span-3 flex items-baseline gap-3 mb-1">
        <span className="font-mono text-2xl text-white">{data.total_datasets.toLocaleString()}</span>
        <span className="text-neural-500 text-sm">datasets</span>
        <span className="text-neural-700 text-xs">·</span>
        <span className="font-mono text-neural-400">{data.total_entries.toLocaleString()}</span>
        <span className="text-neural-500 text-sm">coverage entries</span>
      </div>
      {dims.map(([dim, { datasets, pct }]) => (
        <div key={dim} className="bg-neural-900 border border-neural-800 rounded-lg p-4">
          <div className="text-neural-400 text-xs mb-1 capitalize">{dim.replace(/_/g, ' ')}</div>
          <div className="font-mono text-lg text-white mb-2">{datasets.toLocaleString()}</div>
          <PctBar pct={pct} />
        </div>
      ))}
    </div>
  )
}

function SourceRatesTable() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['coverage-source-rates'],
    queryFn: coverageApi.sourceRates,
  })
  const [sortKey, setSortKey] = useState<keyof SourceRate>('n_total')

  if (isLoading) return <div className="text-neural-500 text-sm">Loading…</div>
  if (error || !data) return <div className="text-red-400 text-sm">Failed to load source rates.</div>

  const sorted = [...data].sort((a, b) => (b[sortKey] as number) - (a[sortKey] as number))

  const cols: { key: keyof SourceRate; label: string }[] = [
    { key: 'source', label: 'Source' },
    { key: 'n_total', label: 'Datasets' },
    { key: 'regions_pct', label: 'Regions %' },
    { key: 'modalities_pct', label: 'Modalities %' },
    { key: 'species_pct', label: 'Species %' },
    { key: 'tasks_pct', label: 'Tasks %' },
  ]

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-neural-800">
            {cols.map(c => (
              <th
                key={c.key}
                className={`text-left py-2 px-3 text-neural-500 font-normal cursor-pointer hover:text-neural-200 transition-colors ${sortKey === c.key ? 'text-accent-cyan' : ''}`}
                onClick={() => setSortKey(c.key)}
              >
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map(row => (
            <tr key={row.source} className="border-b border-neural-800/40 hover:bg-neural-900/50">
              <td className="py-2 px-3 font-mono text-neural-200">{row.source}</td>
              <td className="py-2 px-3 font-mono text-neural-300">{row.n_total.toLocaleString()}</td>
              <td className="py-2 px-3">
                <span className={`font-mono ${pctColor(row.regions_pct)}`}>{row.regions_pct.toFixed(1)}%</span>
              </td>
              <td className="py-2 px-3">
                <span className={`font-mono ${pctColor(row.modalities_pct)}`}>{row.modalities_pct.toFixed(1)}%</span>
              </td>
              <td className="py-2 px-3">
                <span className={`font-mono ${pctColor(row.species_pct)}`}>{row.species_pct.toFixed(1)}%</span>
              </td>
              <td className="py-2 px-3">
                <span className={`font-mono ${pctColor(row.tasks_pct)}`}>{row.tasks_pct.toFixed(1)}%</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function UncoveredRegionsList() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['coverage-uncovered-regions'],
    queryFn: coverageApi.uncoveredRegions,
  })

  if (isLoading) return <div className="text-neural-500 text-sm">Loading…</div>
  if (error || !data) return <div className="text-red-400 text-sm">Failed to load uncovered regions.</div>
  if (data.length === 0) return <div className="text-emerald-400 text-sm">All ontology regions are covered!</div>

  return (
    <div className="space-y-2">
      {data.map((r: UncoveredRegion) => (
        <div key={r.id} className="flex items-start justify-between bg-neural-900 border border-neural-800 rounded-lg px-4 py-3">
          <div>
            <div className="text-neural-200 text-sm font-medium">{r.label}</div>
            <div className="text-neural-600 text-xs font-mono mt-0.5">{r.id}</div>
            {r.parents.length > 0 && (
              <div className="text-neural-600 text-xs mt-1">
                in: {r.parents.join(' › ')}
              </div>
            )}
          </div>
          <div className="text-right text-xs shrink-0 ml-4 space-y-1">
            {r.uberon_id && (
              <div className="font-mono text-neural-600">{r.uberon_id}</div>
            )}
            {r.allen_ccf_mouse_id && (
              <div className="font-mono text-neural-700">CCF {r.allen_ccf_mouse_id}</div>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}

function DarkPairsTable() {
  const [dimA, setDimA] = useState('brain_regions')
  const [dimB, setDimB] = useState('modalities')
  const { data, isLoading, error } = useQuery({
    queryKey: ['coverage-dark-pairs', dimA, dimB],
    queryFn: () => coverageApi.darkPairs({ dimA, dimB, limit: 20 }),
  })

  const dimOptions = ['brain_regions', 'modalities', 'species', 'tasks', 'recording_scales']

  return (
    <div>
      <div className="flex gap-3 mb-4 flex-wrap">
        <div className="flex items-center gap-2">
          <label className="text-neural-500 text-xs">Dim A</label>
          <select
            value={dimA}
            onChange={e => setDimA(e.target.value)}
            className="bg-neural-900 border border-neural-700 text-neural-200 text-xs rounded px-2 py-1"
          >
            {dimOptions.map(d => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-neural-500 text-xs">Dim B</label>
          <select
            value={dimB}
            onChange={e => setDimB(e.target.value)}
            className="bg-neural-900 border border-neural-700 text-neural-200 text-xs rounded px-2 py-1"
          >
            {dimOptions.map(d => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>
      </div>

      {isLoading && <div className="text-neural-500 text-sm">Loading…</div>}
      {error && <div className="text-red-400 text-sm">Failed to load dark pairs.</div>}
      {data && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-neural-800">
                <th className="text-left py-2 px-3 text-neural-500 font-normal">{dimA}</th>
                <th className="text-left py-2 px-3 text-neural-500 font-normal">{dimB}</th>
                <th className="text-right py-2 px-3 text-neural-500 font-normal">A datasets</th>
                <th className="text-right py-2 px-3 text-neural-500 font-normal">B datasets</th>
                <th className="text-right py-2 px-3 text-neural-500 font-normal">Opportunity</th>
              </tr>
            </thead>
            <tbody>
              {(data as DarkPair[]).map((row, i) => (
                <tr key={i} className="border-b border-neural-800/40 hover:bg-neural-900/50">
                  <td className="py-2 px-3 font-mono text-neural-300 text-xs">{row.a_value}</td>
                  <td className="py-2 px-3 font-mono text-neural-300 text-xs">{row.b_value}</td>
                  <td className="py-2 px-3 font-mono text-neural-500 text-right text-xs">{row.a_marginal}</td>
                  <td className="py-2 px-3 font-mono text-neural-500 text-right text-xs">{row.b_marginal}</td>
                  <td className="py-2 px-3 text-right">
                    <span className="font-mono text-accent-cyan text-xs">{row.opportunity_score}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

export function CoveragePage() {
  const [activeTab, setActiveTab] = useState<'summary' | 'sources' | 'uncovered' | 'gaps'>('summary')

  const tabs: { key: typeof activeTab; label: string }[] = [
    { key: 'summary', label: 'Summary' },
    { key: 'sources', label: 'By Source' },
    { key: 'uncovered', label: 'Uncovered Regions' },
    { key: 'gaps', label: 'Dark Pairs' },
  ]

  return (
    <div className="max-w-5xl mx-auto px-6 lg:px-8 py-12">
      <div className="mb-8">
        <h1 className="font-mono text-xl text-white mb-1">Coverage Ledger</h1>
        <p className="text-neural-500 text-sm">
          Corpus coverage across brain regions, modalities, species, and tasks — powered by the DuckDB gap matrix.
        </p>
      </div>

      <div className="flex gap-1 mb-8 border-b border-neural-800">
        {tabs.map(tab => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`px-4 py-2 text-sm transition-colors border-b-2 -mb-px ${
              activeTab === tab.key
                ? 'border-accent-cyan text-accent-cyan'
                : 'border-transparent text-neural-500 hover:text-neural-200'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === 'summary' && (
        <section>
          <h2 className="font-mono text-sm text-neural-400 uppercase tracking-wider mb-4">Dimension Coverage</h2>
          <SummaryPanel />
        </section>
      )}

      {activeTab === 'sources' && (
        <section>
          <h2 className="font-mono text-sm text-neural-400 uppercase tracking-wider mb-4">Per-Source Coverage Rates</h2>
          <p className="text-neural-600 text-xs mb-4">Click column headers to sort. Coverage = % of datasets with at least one label in that dimension.</p>
          <SourceRatesTable />
        </section>
      )}

      {activeTab === 'uncovered' && (
        <section>
          <h2 className="font-mono text-sm text-neural-400 uppercase tracking-wider mb-4">Zero-Coverage Ontology Regions</h2>
          <p className="text-neural-600 text-xs mb-4">These brain regions appear in the ontology but have no matching datasets in the corpus. Priority targets for corpus acquisition.</p>
          <UncoveredRegionsList />
        </section>
      )}

      {activeTab === 'gaps' && (
        <section>
          <h2 className="font-mono text-sm text-neural-400 uppercase tracking-wider mb-4">Dark Dimension Pairs</h2>
          <p className="text-neural-600 text-xs mb-4">Dimension combinations with zero observed co-occurrence. High opportunity score = both values are individually common but never paired.</p>
          <DarkPairsTable />
        </section>
      )}
    </div>
  )
}
