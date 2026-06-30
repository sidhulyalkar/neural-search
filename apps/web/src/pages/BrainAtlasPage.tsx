import { useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { coverageApi, atlasApi, type RegionCount } from '../api/coverage'
import { BrainSchematic } from '../components/atlas/BrainSchematic'
import { AllenHierarchyPanel } from '../components/atlas/AllenHierarchyPanel'
import { RegionDetailPanel } from '../components/atlas/RegionDetailPanel'
import { CircuitsPanel } from '../components/atlas/CircuitsPanel'

// ── Anatomical grouping ───────────────────────────────────────────────────────
const ATLAS_GROUPS: Record<string, string[]> = {
  'Visual System': [
    'visual_cortex', 'v1', 'v2', 'v4', 'area_mt', 'inferior_temporal_cortex',
    'lateral_geniculate',
  ],
  'Frontal / Motor': [
    'prefrontal_cortex', 'dlpfc', 'ofc', 'vlpfc', 'anterior_cingulate_cortex',
    'motor_cortex', 'm1', 'premotor_cortex', 'supplementary_motor_area',
    'orbitofrontal_cortex',
  ],
  'Somatosensory': [
    'somatosensory_cortex', 's1', 's2', 'barrel_cortex',
  ],
  'Parietal / Temporal': [
    'parietal_cortex', 'posterior_parietal_cortex', 'temporal_cortex',
    'auditory_cortex', 'inferior_parietal_lobule',
  ],
  'Hippocampal Fm.': [
    'hippocampus', 'ca1', 'ca3', 'dentate_gyrus', 'entorhinal_cortex',
    'subiculum', 'parahippocampal_cortex',
  ],
  'Cingulate / Insula': [
    'cingulate_cortex', 'insular_cortex', 'retrosplenial_cortex',
    'medial_prefrontal_cortex',
  ],
  'Basal Ganglia': [
    'striatum', 'caudate_nucleus', 'putamen', 'nucleus_accumbens',
    'globus_pallidus', 'subthalamic_nucleus', 'basal_ganglia',
  ],
  'Thalamus': [
    'thalamus', 'mediodorsal_thalamus', 'pulvinar', 'medial_geniculate',
    'ventral_posterior_thalamus',
  ],
  'Amygdala': [
    'amygdala', 'lateral_amygdala', 'basal_amygdala', 'central_amygdala',
    'bed_nucleus_stria_terminalis',
  ],
  'Hypothalamus': [
    'hypothalamus', 'lateral_hypothalamus', 'paraventricular_nucleus',
  ],
  'Brainstem': [
    'midbrain', 'superior_colliculus', 'inferior_colliculus',
    'substantia_nigra', 'ventral_tegmental_area', 'periaqueductal_gray',
    'pons', 'locus_coeruleus', 'raphe_nucleus', 'medulla',
  ],
  'Cerebellum': [
    'cerebellum', 'cerebellar_cortex', 'deep_cerebellar_nuclei',
  ],
}

// ── Colour helpers ─────────────────────────────────────────────────────────────
function tileClasses(n: number, selected: boolean, highlighted: boolean): string {
  const base = 'relative border rounded px-2 py-1.5 text-left transition-all cursor-pointer'
  const ring = selected ? ' ring-2 ring-accent-cyan ring-offset-1 ring-offset-neural-950' : ''
  const glow = highlighted && !selected ? ' ring-1 ring-accent-cyan/40 ring-offset-1 ring-offset-neural-950' : ''

  if (n === 0) return `${base}${ring}${glow} bg-neural-900 border-neural-800 text-neural-600 cursor-default`
  if (n < 5)   return `${base}${ring}${glow} bg-red-950 border-red-800/50 text-red-300 hover:bg-red-900/60`
  if (n < 30)  return `${base}${ring}${glow} bg-orange-950 border-orange-700/50 text-orange-300 hover:bg-orange-900/60`
  if (n < 100) return `${base}${ring}${glow} bg-yellow-950 border-yellow-600/50 text-yellow-300 hover:bg-yellow-900/60`
  if (n < 400) return `${base}${ring}${glow} bg-emerald-950 border-emerald-700/50 text-emerald-300 hover:bg-emerald-900/60`
  return `${base}${ring}${glow} bg-emerald-900 border-emerald-500/70 text-emerald-100 hover:bg-emerald-800`
}

// ── Sub-components ─────────────────────────────────────────────────────────────
function RegionTile({
  label,
  nDatasets,
  selected,
  highlighted,
  onClick,
}: {
  regionId: string
  label: string
  nDatasets: number
  selected: boolean
  highlighted: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={nDatasets > 0 ? onClick : undefined}
      title={`${label} — ${nDatasets} datasets`}
      className={tileClasses(nDatasets, selected, highlighted)}
    >
      <div className="text-xs font-medium leading-tight truncate">{label}</div>
      <div className="text-xs font-mono opacity-70 mt-0.5">{nDatasets}</div>
    </button>
  )
}

function Legend() {
  return (
    <div className="flex items-center gap-3 flex-wrap">
      <span className="text-xs text-neural-500">Coverage:</span>
      {[
        { label: '0', cls: 'bg-neural-900 border-neural-800' },
        { label: '1–4', cls: 'bg-red-950 border-red-800/50' },
        { label: '5–29', cls: 'bg-orange-950 border-orange-700/50' },
        { label: '30–99', cls: 'bg-yellow-950 border-yellow-600/50' },
        { label: '100–399', cls: 'bg-emerald-950 border-emerald-700/50' },
        { label: '400+', cls: 'bg-emerald-900 border-emerald-500/70' },
      ].map((e) => (
        <div key={e.label} className="flex items-center gap-1">
          <div className={`w-3 h-3 rounded border ${e.cls}`} />
          <span className="text-xs text-neural-500">{e.label}</span>
        </div>
      ))}
    </div>
  )
}

// ── Atlas Stats panel ─────────────────────────────────────────────────────────
function AtlasStatsPanel() {
  const { data, isLoading } = useQuery({
    queryKey: ['atlas-coverage'],
    queryFn: atlasApi.coverage,
    staleTime: 1000 * 60 * 5,
  })

  if (isLoading) {
    return <div className="text-neural-500 text-sm py-4">Loading atlas stats…</div>
  }

  if (!data) {
    return <div className="text-neural-600 text-sm py-4 italic">No data available.</div>
  }

  const pct = ((data.total_ontology_mapped / data.total_mouse_structures) * 100).toFixed(1)

  return (
    <div className="space-y-4">
      {/* Summary card */}
      <div className="bg-neural-900/60 border border-neural-800 rounded-lg p-4">
        <div className="text-xs font-mono text-neural-500 uppercase tracking-wider mb-2">
          Allen CCF Coverage Summary
        </div>
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-mono text-white">{data.total_ontology_mapped}</span>
          <span className="text-neural-500 text-sm">
            / {data.total_mouse_structures.toLocaleString()} mouse structures mapped
          </span>
        </div>
        <div className="mt-1">
          <div className="h-1.5 bg-neural-800 rounded-full overflow-hidden w-full">
            <div
              className="h-full bg-accent-cyan rounded-full"
              style={{ width: `${Math.min(parseFloat(pct), 100)}%` }}
            />
          </div>
          <span className="text-xs text-neural-600 font-mono mt-1 inline-block">{pct}% coverage</span>
        </div>
      </div>

      {/* By-level breakdown */}
      <div className="bg-neural-900/60 border border-neural-800 rounded-lg overflow-hidden">
        <div className="px-4 py-2 border-b border-neural-800">
          <span className="text-xs font-mono text-neural-500 uppercase tracking-wider">
            Coverage by Hierarchy Level
          </span>
        </div>
        <table className="w-full text-xs font-mono">
          <thead>
            <tr className="border-b border-neural-800">
              <th className="text-left px-4 py-2 text-neural-600 font-normal">Level</th>
              <th className="text-right px-4 py-2 text-neural-600 font-normal">Mapped</th>
              <th className="text-right px-4 py-2 text-neural-600 font-normal">Total</th>
              <th className="text-right px-4 py-2 text-neural-600 font-normal">Coverage</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(data.by_level)
              .sort(([a], [b]) => Number(a) - Number(b))
              .map(([level, stats]) => {
                const levelPct = stats.total > 0
                  ? ((stats.mapped / stats.total) * 100).toFixed(0)
                  : '0'
                return (
                  <tr key={level} className="border-b border-neural-800/50 hover:bg-neural-800/30">
                    <td className="px-4 py-2 text-neural-400">{level}</td>
                    <td className="px-4 py-2 text-right text-emerald-400">{stats.mapped}</td>
                    <td className="px-4 py-2 text-right text-neural-500">{stats.total}</td>
                    <td className="px-4 py-2 text-right text-neural-600">{levelPct}%</td>
                  </tr>
                )
              })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Allen coverage summary card (shown in heatmap tab) ────────────────────────
function AllenCoverageBadge() {
  const { data } = useQuery({
    queryKey: ['atlas-coverage'],
    queryFn: atlasApi.coverage,
    staleTime: 1000 * 60 * 5,
  })

  if (!data) return null

  const pct = ((data.total_ontology_mapped / data.total_mouse_structures) * 100).toFixed(1)

  return (
    <div className="bg-neural-900/40 border border-neural-800 rounded-lg px-4 py-2 mb-4 flex items-center gap-3 flex-wrap">
      <span className="text-xs text-neural-500">Allen CCF Coverage</span>
      <span className="font-mono text-sm text-white">
        {data.total_ontology_mapped}
        <span className="text-neural-600">/{data.total_mouse_structures.toLocaleString()}</span>
      </span>
      <span className="text-xs text-neural-600">mouse structures mapped</span>
      <span className="text-xs font-mono text-accent-cyan ml-auto">{pct}%</span>
    </div>
  )
}

// ── Page ───────────────────────────────────────────────────────────────────────
type ActiveTab = 'heatmap' | 'circuits' | 'hierarchy' | 'stats'

export function BrainAtlasPage() {
  const [selected, setSelected] = useState<{ id: string; label: string } | null>(null)
  const [activeTab, setActiveTab] = useState<ActiveTab>('heatmap')
  const [selectedLobe, setSelectedLobe] = useState<string | null>(null)
  const [lobeRegionIds, setLobeRegionIds] = useState<string[] | null>(null)

  const groupRefs = useRef<Record<string, HTMLDivElement | null>>({})

  const { data: counts, isLoading } = useQuery({
    queryKey: ['region-counts'],
    queryFn: coverageApi.regionCounts,
  })

  const countMap = new Map<string, { n: number; label: string }>(
    (counts ?? []).map((r: RegionCount) => [r.region_id, { n: r.n_datasets, label: r.region_label }])
  )

  const coveredRegions = counts?.filter((r) => r.n_datasets > 0).length ?? 0
  const totalInAtlas = Object.values(ATLAS_GROUPS).flat().length

  // Set of highlighted region IDs when a lobe is selected in heatmap mode
  const highlightedRegionIds = new Set(lobeRegionIds ?? [])

  function handleTileClick(id: string, label: string) {
    setSelected((prev) => (prev?.id === id ? null : { id, label }))
  }

  function handleLobeClick(lobeId: string, regionIds: string[]) {
    if (selectedLobe === lobeId) {
      setSelectedLobe(null)
      setLobeRegionIds(null)
      return
    }
    setSelectedLobe(lobeId)
    setLobeRegionIds(regionIds)

    if (activeTab === 'heatmap') {
      // Find which atlas group overlaps most with these regionIds and scroll to it
      const regionSet = new Set(regionIds)
      let bestGroup = ''
      let bestOverlap = 0
      for (const [groupLabel, groupRegions] of Object.entries(ATLAS_GROUPS)) {
        const overlap = groupRegions.filter((r) => regionSet.has(r)).length
        if (overlap > bestOverlap) {
          bestOverlap = overlap
          bestGroup = groupLabel
        }
      }
      if (bestGroup && groupRefs.current[bestGroup]) {
        groupRefs.current[bestGroup]?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }
    }
  }

  function handleAllenRegionSelect(regionId: string) {
    const info = countMap.get(regionId)
    setSelected({ id: regionId, label: info?.label ?? regionId.replace(/_/g, ' ') })
  }

  const TABS: { id: ActiveTab; label: string }[] = [
    { id: 'heatmap', label: 'Coverage Heatmap' },
    { id: 'circuits', label: 'Circuits' },
    { id: 'hierarchy', label: 'Allen Hierarchy' },
    { id: 'stats', label: 'Atlas Stats' },
  ]

  return (
    <div className="max-w-7xl mx-auto px-6 lg:px-8 py-8">
      {/* Page header */}
      <div className="mb-5">
        <h1 className="font-mono text-2xl text-white mb-1">Brain Atlas</h1>
        <p className="text-neural-500 text-sm">
          Dataset coverage across brain regions. Click any region to explore its datasets.
        </p>
      </div>

      {/* Two-column layout */}
      <div className="flex gap-6 items-start">
        {/* ── Left column: schematic + hierarchy ── */}
        <div className="w-80 flex-shrink-0 hidden lg:flex flex-col gap-4 sticky top-24">
          {/* Brain schematic — always visible */}
          <BrainSchematic
            countMap={countMap}
            selectedLobe={selectedLobe}
            onLobeClick={handleLobeClick}
          />

          {/* Allen hierarchy tree */}
          <div className="bg-neural-900/50 border border-neural-800 rounded-xl p-3 max-h-80 overflow-hidden flex flex-col">
            <AllenHierarchyPanel
              onRegionSelect={handleAllenRegionSelect}
              countMap={countMap}
              filterLobeRegionIds={activeTab === 'hierarchy' ? lobeRegionIds : null}
            />
          </div>
        </div>

        {/* ── Right column: tabs + content ── */}
        <div className="flex-1 min-w-0">
          {/* Stats + legend row */}
          {counts && (
            <div className="flex items-center gap-6 mb-4 flex-wrap text-sm">
              <span>
                <span className="font-mono text-white">{coveredRegions}</span>
                <span className="text-neural-500 ml-1">/ {totalInAtlas} regions covered</span>
              </span>
              <span className="text-neural-700 hidden sm:inline">·</span>
              <Legend />
            </div>
          )}

          {/* Tab row */}
          <div className="flex gap-1 mb-4 bg-neural-900/40 border border-neural-800 rounded-lg p-1 w-fit">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={`px-3 py-1.5 rounded text-xs font-mono transition-colors ${
                  activeTab === tab.id
                    ? 'bg-neural-800 text-white border border-neural-700'
                    : 'text-neural-500 hover:text-neural-300'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Loading state */}
          {isLoading && activeTab === 'heatmap' && (
            <div className="text-neural-500 text-sm">Loading atlas…</div>
          )}

          {/* ── Heatmap tab ── */}
          {activeTab === 'heatmap' && (
            <>
              <AllenCoverageBadge />

              {/* Lobe filter active notice */}
              {selectedLobe && (
                <div className="flex items-center gap-2 mb-3 text-xs text-accent-cyan border border-accent-cyan/30 bg-accent-cyan/5 rounded-lg px-3 py-2">
                  <span>Highlighting regions in selected lobe</span>
                  <button
                    type="button"
                    onClick={() => { setSelectedLobe(null); setLobeRegionIds(null) }}
                    className="ml-auto text-neural-500 hover:text-neural-300"
                  >
                    Clear
                  </button>
                </div>
              )}

              <div className="flex gap-6 items-start">
                {/* Tile grid */}
                <div className="flex-1 min-w-0">
                  {Object.entries(ATLAS_GROUPS).map(([groupLabel, regionIds]) => {
                    const regions = regionIds.map((rid) => {
                      const info = countMap.get(rid)
                      return {
                        id: rid,
                        label: info?.label ?? rid.replace(/_/g, ' '),
                        n: info?.n ?? 0,
                      }
                    })
                    const groupTotal = regions.reduce((s, r) => s + r.n, 0)

                    return (
                      <div
                        key={groupLabel}
                        className="mb-5"
                        ref={(el) => { groupRefs.current[groupLabel] = el }}
                      >
                        <div className="flex items-baseline gap-2 mb-2">
                          <span className="text-xs font-mono text-neural-400 uppercase tracking-wider">
                            {groupLabel}
                          </span>
                          <span className="text-xs text-neural-700">
                            {groupTotal.toLocaleString()} datasets
                          </span>
                        </div>
                        <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-5 xl:grid-cols-6 gap-1.5">
                          {regions.map((r) => (
                            <RegionTile
                              key={r.id}
                              regionId={r.id}
                              label={r.label}
                              nDatasets={r.n}
                              selected={selected?.id === r.id}
                              highlighted={highlightedRegionIds.has(r.id)}
                              onClick={() => handleTileClick(r.id, r.label)}
                            />
                          ))}
                        </div>
                      </div>
                    )
                  })}
                </div>

                {/* Side panel: selected region */}
                <div className="w-72 flex-shrink-0 hidden md:block">
                  <div className="sticky top-24 bg-neural-900/50 border border-neural-800 rounded-xl p-4 max-h-[80vh] overflow-y-auto">
                    {selected ? (
                      <RegionDetailPanel
                        regionId={selected.id}
                        regionLabel={selected.label}
                        onRegionClick={handleTileClick}
                      />
                    ) : (
                      <div className="text-center py-12">
                        <div className="text-neural-600 text-sm">
                          Click a brain region to explore its datasets, hierarchy, and research topics
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </>
          )}

          {/* ── Circuits tab ── */}
          {activeTab === 'circuits' && (
            <div className="flex gap-6 items-start">
              <div className="flex-1 min-w-0">
                <CircuitsPanel onRegionClick={handleTileClick} />
              </div>
              <div className="w-72 flex-shrink-0 hidden md:block">
                <div className="sticky top-24 bg-neural-900/50 border border-neural-800 rounded-xl p-4 max-h-[80vh] overflow-y-auto">
                  {selected ? (
                    <RegionDetailPanel
                      regionId={selected.id}
                      regionLabel={selected.label}
                      onRegionClick={handleTileClick}
                    />
                  ) : (
                    <div className="text-center py-12">
                      <div className="text-neural-600 text-sm">
                        Click any region node in a circuit to see its datasets and connections
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ── Hierarchy tab ── */}
          {activeTab === 'hierarchy' && (
            <div className="flex gap-6 items-start">
              <div className="flex-1 min-w-0 bg-neural-900/50 border border-neural-800 rounded-xl p-4 min-h-[60vh] flex flex-col">
                <AllenHierarchyPanel
                  onRegionSelect={handleAllenRegionSelect}
                  countMap={countMap}
                  filterLobeRegionIds={lobeRegionIds}
                />
              </div>

              {/* Side panel */}
              <div className="w-72 flex-shrink-0 hidden md:block">
                <div className="sticky top-24 bg-neural-900/50 border border-neural-800 rounded-xl p-4 max-h-[80vh] overflow-y-auto">
                  {selected ? (
                    <RegionDetailPanel
                      regionId={selected.id}
                      regionLabel={selected.label}
                      onRegionClick={handleTileClick}
                    />
                  ) : (
                    <div className="text-center py-12">
                      <div className="text-neural-600 text-sm">
                        Click a region in the tree to explore datasets
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ── Stats tab ── */}
          {activeTab === 'stats' && <AtlasStatsPanel />}
        </div>
      </div>

      {/* Mobile: selected region (below main content) */}
      {selected && (
        <div className="md:hidden mt-6 bg-neural-900/50 border border-neural-800 rounded-xl p-4">
          <RegionDetailPanel
            regionId={selected.id}
            regionLabel={selected.label}
            onRegionClick={handleTileClick}
          />
        </div>
      )}
    </div>
  )
}
